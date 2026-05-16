# Reconstruct: Session 5 commits (LOST — never pushed anywhere)

**For:** Linux Claude
**From:** Windows Claude, 2026-05-16
**Source:** Memory at `~/.claude/projects/C--Users-alexx-Documents-mmy-business/memory/brain_encoder_current_status.md` lines 78-179 + direct audit of current `train_commercial.py` at `HEAD = 3d86f21`.

## What's lost

Four commits committed somewhere on 2026-04-12 but never pushed and not in any clone:

| SHA | What | Files |
|---|---|---|
| `68ba8c9` | P0-2 collective-op deadlock fix | `train_commercial.py` |
| `f69daf6` | P1-3 (val_pearson broadcast) + P1-4 (rolling latest checkpoint) + P1-5 (atomic write) | `train_commercial.py`, `setup_runpod.sh` |
| `33d9f37` | EVAL_PROTOCOL.md + eval.py + `--test-split` arg + 3-way `random_split` | new files + `train_commercial.py` |
| `3406a6e` | README.md + HUGGINGFACE_MODEL_CARD.md | new files |

**Priority for RunPod launch:** `68ba8c9` and `f69daf6` are **required** (deadlock blocker + checkpoint resilience). `33d9f37` and `3406a6e` are post-training docs — can defer.

The fixes below are reconstructed from the audit memo + direct grep against the current code. **Audit them before committing** — they're high-confidence but not eyeballed against a running cluster.

---

## Bug map — current `train_commercial.py` (at HEAD = `3d86f21`)

I verified these by reading the actual file. Line numbers below are exact for current HEAD.

### Bug 1: load_training_state deadlock (lines 548-553)

```python
548:     if args.resume is not None and is_main:
549:         resume_dir = Path(args.resume)
550:         start_epoch, global_step, best_val_pearson = load_training_state(
551:             checkpoint_dir=resume_dir,
552:             accelerator=accelerator,
553:         )
```

`load_training_state` calls `accelerator.load_state(...)` at line 250 — collective op. Only rank 0 enters → ranks 1+ never reach the internal barrier → deadlock.

### Bug 2: save_training_state deadlock (lines 723-739)

```python
723:                 # LoRA adapter checkpoint when val improves (ADR-002)
724:                 if val_pearson_r > best_val_pearson:
725:                     best_val_pearson = val_pearson_r
726:                     unwrapped = accelerator.unwrap_model(model)
727:                     save_lora_adapters(unwrapped, best_lora_dir)
728:                     save_training_state(
729:                         epoch=epoch,
730:                         global_step=global_step,
731:                         best_val_pearson=best_val_pearson,
732:                         checkpoint_dir=best_lora_dir,
733:                         accelerator=accelerator,
734:                     )
```

This entire block is inside `if is_main:` (starts line 704). `save_training_state` calls `accelerator.save_state(...)` at line 296 — collective op. **Same deadlock.** Will hang on the first epoch with a val improvement, i.e. epoch 0.

### Bug 3: val_pearson reduce uses mean (line 701)

```python
699:             # Broadcast val_pearson_r to all processes for consistent early-stop decision
700:             val_r_tensor = torch.tensor(val_pearson_r, device=accelerator.device)
701:             val_r_tensor = accelerator.reduce(val_r_tensor, reduction="mean")
702:             val_pearson_r = float(val_r_tensor.item())
```

Only rank 0 sets `val_pearson_r` to a real value (line 685-686, inside `if is_main:`). Other ranks hold `val_pearson_r = 0.0` (line 684). `reduce(mean)` on 2 GPUs gives half-value on non-main → broken early-stop logic.

### Bug 4: only saves on improvement (no `latest/` rolling)

Lines 723-739 save only when `val_pearson_r > best_val_pearson`. Long plateau + pod preemption = lose all progress since last best.

### Bug 5: non-atomic checkpoint write

`save_training_state` at line 286-296 writes directly into `checkpoint_dir`. Crash mid-`accelerator.save_state` → torn `accelerate_state/` → next resume crashes on `load_state`.

---

## Reconstruction patches

Apply in this order. Stage as **one commit** since they're tightly coupled (the test for one is a side effect of another).

### Patch A — add `shutil` import

`train_commercial.py` line 22 area:

```diff
 import argparse
 import json
 import logging
 import os
+import shutil
 import sys
 from pathlib import Path
```

### Patch B — fix `save_training_state` (P0-2 + P1-5)

Replace lines 267-304 (the entire `save_training_state` function) with:

```python
def save_training_state(
    epoch: int,
    global_step: int,
    best_val_pearson: float,
    checkpoint_dir: Path,
    accelerator: Accelerator,
) -> None:
    """
    Persist training state for RunPod pod restart resumption.

    Multi-GPU safe:
      - `accelerator.save_state()` is a collective op — ALL ranks must enter.
      - JSON write + rename are rank-0 only.
      - Atomic: writes to `<dir>_tmp/`, then rank 0 renames after barrier.

    Writes into checkpoint_dir:
      training_state.json   — epoch, global_step, best_val_pearson (scalars)
      accelerate_state/     — optimizer + scheduler via accelerator.save_state()
    """
    tmp_dir = checkpoint_dir.parent / (checkpoint_dir.name + "_tmp")

    # Clean up tmp from any prior crash, then ensure parent exists.
    if accelerator.is_main_process:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # JSON scalars — main only.
    if accelerator.is_main_process:
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_pearson": best_val_pearson,
        }
        with open(tmp_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

    # COLLECTIVE — all ranks enter. Save into tmp.
    accelerator.save_state(str(tmp_dir / "accelerate_state"))

    # Atomic rename — main only, after every rank finished writing.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        tmp_dir.rename(checkpoint_dir)
        logger.info(
            "Training state saved: epoch=%d  global_step=%d  best_pearson=%.4f → %s",
            epoch,
            global_step,
            best_val_pearson,
            checkpoint_dir,
        )
    accelerator.wait_for_everyone()
```

### Patch C — fix `load_training_state` call site (P0-2)

Replace lines 548-553 with:

```python
    # load_training_state contains a collective op (accelerator.load_state).
    # ALL ranks must enter — main reads JSON, all enter load_state together.
    if args.resume is not None:
        resume_dir = Path(args.resume)
        start_epoch, global_step, best_val_pearson = load_training_state(
            checkpoint_dir=resume_dir,
            accelerator=accelerator,
        )
```

Then inside `load_training_state` itself (lines 220-264), wrap the JSON read in `is_main_process` and broadcast scalars. Replace lines 232-264 with:

```python
    json_path = checkpoint_dir / "training_state.json"
    accel_state_path = checkpoint_dir / "accelerate_state"

    if not checkpoint_dir.exists() or not json_path.exists():
        if accelerator.is_main_process:
            logger.info("No checkpoint found at %s — starting fresh.", checkpoint_dir)
        return 0, 0, float("-inf")

    # JSON read — main only, then broadcast.
    if accelerator.is_main_process:
        with open(json_path) as f:
            state = json.load(f)
        resumed_epoch = int(state["epoch"]) + 1
        resumed_global_step = int(state["global_step"])
        resumed_best_val_pearson = float(state["best_val_pearson"])
    else:
        resumed_epoch = 0
        resumed_global_step = 0
        resumed_best_val_pearson = float("-inf")

    # accelerator.load_state is a COLLECTIVE op — all ranks must enter.
    if accel_state_path.exists():
        accelerator.load_state(str(accel_state_path))
        if accelerator.is_main_process:
            logger.info(
                "Resumed from checkpoint: next_epoch=%d  global_step=%d  best_pearson=%.4f",
                resumed_epoch,
                resumed_global_step,
                resumed_best_val_pearson,
            )
    else:
        if accelerator.is_main_process:
            logger.warning(
                "training_state.json found but accelerate_state/ missing at %s — "
                "scalars restored but optimizer/scheduler reset to initial state.",
                checkpoint_dir,
            )

    return resumed_epoch, resumed_global_step, resumed_best_val_pearson
```

(Caller already does `accelerator.reduce(..., reduction="max")` for the scalars at lines 556-564 — that broadcast still works correctly.)

### Patch D — fix val_pearson reduce (P1-3)

`train_commercial.py` line 701:

```diff
             val_r_tensor = torch.tensor(val_pearson_r, device=accelerator.device)
-            val_r_tensor = accelerator.reduce(val_r_tensor, reduction="mean")
+            # Only rank 0 holds a real value; others are 0.0. max propagates rank 0's value.
+            val_r_tensor = accelerator.reduce(val_r_tensor, reduction="max")
             val_pearson_r = float(val_r_tensor.item())
```

### Patch E — best/latest rolling save + collective save site (P0-2 finish + P1-4)

This is the big one. Replace lines 723-739 with:

```python
            # ----------------------------------------------------------------
            # Checkpointing — must be OUTSIDE is_main because save_training_state
            # contains a collective op. We broadcast is_new_best so all ranks
            # take the same branch.
            # ----------------------------------------------------------------
            if is_main:
                is_new_best = val_pearson_r > best_val_pearson
            else:
                is_new_best = False

        # ↑ close the `if is_main:` block from line 704 here.

        # Broadcast is_new_best to all ranks (reduce(sum) > 0 means any rank saw improvement).
        is_new_best_t = torch.tensor(int(is_new_best), device=accelerator.device)
        is_new_best_t = accelerator.reduce(is_new_best_t, reduction="sum")
        is_new_best_all = is_new_best_t.item() > 0

        # ALWAYS save latest/ checkpoint every epoch (P1-4 — protects against
        # plateau + preemption). Collective op — all ranks enter.
        latest_lora_dir = checkpoint_dir / "latest"
        if accelerator.is_main_process:
            unwrapped_latest = accelerator.unwrap_model(model)
            save_lora_adapters(unwrapped_latest, latest_lora_dir)
        save_training_state(
            epoch=epoch,
            global_step=global_step,
            best_val_pearson=best_val_pearson,
            checkpoint_dir=latest_lora_dir,
            accelerator=accelerator,
        )

        # Save best/ only on improvement. Collective op — all ranks enter.
        if is_new_best_all:
            if accelerator.is_main_process:
                best_val_pearson = val_pearson_r  # already broadcast via reduce(max) earlier
                unwrapped_best = accelerator.unwrap_model(model)
                save_lora_adapters(unwrapped_best, best_lora_dir)
            # Broadcast updated best_val_pearson via reduce(max) so all ranks agree.
            bv = torch.tensor(best_val_pearson, device=accelerator.device)
            bv = accelerator.reduce(bv, reduction="max")
            best_val_pearson = float(bv.item())
            save_training_state(
                epoch=epoch,
                global_step=global_step,
                best_val_pearson=best_val_pearson,
                checkpoint_dir=best_lora_dir,
                accelerator=accelerator,
            )
            if accelerator.is_main_process:
                logger.info(
                    "New best val_pearson=%.4f — LoRA adapters + training state saved to %s",
                    best_val_pearson,
                    best_lora_dir,
                )

        # Early stop decision — main computes, all ranks reduce.
        if accelerator.is_main_process:
            if monitor is not None and monitor.should_stop(val_pearson_r, threshold=args.early_stop_threshold):
                logger.info(
                    "Early stop triggered: val_pearson=%.4f >= threshold=%.4f",
                    val_pearson_r,
                    args.early_stop_threshold,
                )
                stopped_early = True
```

**Note on the block structure:** the existing `if is_main:` opened at line 704 needs to close before the checkpoint block. The patch above shows where to close it (the `# ↑ close the if is_main: block from line 704 here.` comment). The metric logging / epoch_metrics still goes inside `if is_main:` — only the checkpoint save needs to come out.

Also add `latest_lora_dir` declaration near line 538:

```diff
     checkpoint_dir = Path(args.checkpoint_dir)
     best_lora_dir = checkpoint_dir / "best"
+    latest_lora_dir = checkpoint_dir / "latest"
```

(Actually it's declared inline above — either works. Pick one and be consistent.)

### Patch F — setup_runpod.sh prefer `latest/`

Replace lines 156-163 of `setup_runpod.sh`:

```diff
 RESUME_ARG=""
-if [[ -f "${CHECKPOINT_DIR}/best/training_state.json" ]]; then
-    RESUME_ARG="--resume ${CHECKPOINT_DIR}/best"
-    echo "    Resuming from checkpoint: ${CHECKPOINT_DIR}/best"
-else
-    echo "    No checkpoint found — starting fresh"
-fi
+if [[ -f "${CHECKPOINT_DIR}/latest/training_state.json" ]]; then
+    RESUME_ARG="--resume ${CHECKPOINT_DIR}/latest"
+    echo "    Resuming from latest checkpoint: ${CHECKPOINT_DIR}/latest"
+elif [[ -f "${CHECKPOINT_DIR}/best/training_state.json" ]]; then
+    RESUME_ARG="--resume ${CHECKPOINT_DIR}/best"
+    echo "    Resuming from best checkpoint (no latest/ found): ${CHECKPOINT_DIR}/best"
+else
+    echo "    No checkpoint found — starting fresh"
+fi
```

---

## Verify before committing

```bash
# 1. Syntax check
python -c "import ast; ast.parse(open('train_commercial.py').read()); print('OK')"
bash -n setup_runpod.sh && echo "shell OK"

# 2. Imports resolve
python -c "import train_commercial" 2>&1 | head -20
# If this fails because torch/accelerate aren't installed on Linux,
# fall back to ast-only verification.

# 3. Grep sanity — these should all return matches
grep -n "reduction=\"max\"" train_commercial.py    # val_pearson + best_pearson broadcasts
grep -n "wait_for_everyone" train_commercial.py    # atomic barriers
grep -n "import shutil" train_commercial.py        # shutil for rmtree/rename
grep -n "latest_lora_dir\|latest/training_state.json" train_commercial.py setup_runpod.sh

# 4. Run unit tests if any cover save/load
pytest tests/ -k "checkpoint or save or load or resume" -v 2>&1 | tail -20

# 5. Ideally: launch a tiny 2-process dry-run on Linux to verify no deadlock
# (only if you have a 2-GPU Linux host; otherwise this only validates on RunPod)
# accelerate launch --num_processes 2 train_commercial.py --dry-run --data-path /tmp/fake.h5
```

---

## Commit + push

```bash
git add train_commercial.py setup_runpod.sh
git status   # confirm ONLY those two files staged

git commit -m "Reconstruct lost Session 5 commits: P0-2 deadlock + P1-3/4/5 resilience

Original commits 68ba8c9 / f69daf6 were committed locally on 2026-04-12 but
never pushed and lost on device cleanup. Reconstructing from memory + direct
audit of current code.

P0-2 (deadlock blocker for 2×A100):
  - accelerator.save_state / load_state are collective ops; previously wrapped
    in is_main → non-main ranks never hit internal wait_for_everyone barrier
    → rank 0 hangs forever on first checkpoint save (epoch 0).
  - All ranks now enter save/load; JSON I/O guarded by is_main_process;
    is_new_best broadcast via reduce(sum); best_val_pearson via reduce(max).

P1-3 (val_pearson broadcast):
  - reduce(mean) on a value only rank 0 holds → half-value on non-main ranks
    → broken early-stop logic. Swapped to reduce(max).

P1-4 (rolling latest/ checkpoint):
  - Was: only saved on val improvement. Long plateau + pod preemption = total
    loss. Now: latest/ saved every epoch, best/ on improvement.
  - setup_runpod.sh prefers latest/ over best/ on resume.

P1-5 (atomic checkpoint write):
  - Was: direct write into checkpoint_dir. Crash mid-save → torn dir → next
    resume crashes on load_state. Now: write to <dir>_tmp/, barrier, rank-0
    rename. Safe under preemption + sigkill."

git push origin master
```

After push, `origin/master` should have 5 commits ahead of where it started. Verify:

```bash
git log origin/master --oneline | head -10
# Expect: a reconstruction commit, then 3d86f21, a58fb1b, b08591b, 9825b04
```

---

## What about 33d9f37 (eval) and 3406a6e (docs)?

**Defer.** These don't block RunPod launch:

- `EVAL_PROTOCOL.md` + `eval.py` + `--test-split` arg + 3-way `random_split`: needed AFTER training completes to score the model. Can reconstruct between launch and training completion (~hours to days of buffer).
- `README.md` + `HUGGINGFACE_MODEL_CARD.md`: post-publish docs. Reconstruct before the Twitter thread goes out.

Memory has the structure: EVAL_PROTOCOL.md = pre-registered eval (80/10/10 split, Pearson r, 10 ROIs, 4 baselines, bootstrap CI), eval.py = offline script with `--dry-run` support, `train_commercial.py` gains `--test-split` arg + uses `torch.utils.data.random_split` 3-way. Can be reconstructed cleanly when needed.

---

## RunPod launch gate

Only proceed past this point when **ALL** of these are true:

- [ ] `git log origin/master | head -10` shows the reconstruction commit
- [ ] `grep "reduction=\"max\"" train_commercial.py` returns lines around the val_pearson broadcast AND the best_pearson broadcast
- [ ] `grep "latest/training_state.json" setup_runpod.sh` returns a hit
- [ ] `grep "wait_for_everyone" train_commercial.py` returns at least 2 hits (load + save + early-stop sync)
- [ ] Tiny smoke test on RunPod: launch with `--num-epochs 1 --batch-size 1`, verify the first epoch's checkpoint save completes (doesn't hang)
- [ ] WANDB_API_KEY exported in the pod env

If any unchecked → don't launch. Burning $3/hr on a deadlocked pod is the failure mode this whole reconstruction prevents.

---

## Audit note for Linux Claude

I wrote these patches against the file as it sits at `3d86f21` on Windows. Linux's HEAD is identical (we verified). Apply directly. If anything diverges (you've made other changes Windows didn't see), reconcile manually before pushing — Patch E's block structure is the most fragile because the `if is_main:` close needs to land in the right spot.

The save_lora_adapters function (line 198) wasn't audited for collective-op issues. It does `peft_model.save_pretrained()` which is typically rank-0-safe but check if your version of peft+accelerate is. If unsure: leave inside `is_main_process` (current behavior) — it's not a collective op so no deadlock risk.
