# Real Training Plan — Wire Up Real Stimulus Data

**Status:** Pipeline validated end-to-end on RTX 5090 (2026-05-17). Training currently uses `torch.randn` placeholders for video/audio inputs — model cannot learn from noise. This document is the next-session entry point for replacing placeholders with real aligned stimulus data.

**Goal:** Pearson r > 0.23 on held-out test split (ADR-005 early-stop threshold). Currently r ≈ 0 because inputs are random noise.

---

## What's already working (don't redo)

- HDF5 dataloader → model forward → loss → backward → optimizer → checkpoint → WandB (validated 2026-05-17)
- 287 subjects of real BOLD data on RunPod NV `serious_green_wolverine` in EU-RO-1
- Reconstructed Session 5 commits on origin (`f8a391a`): P0-2 deadlock fix, P1-3/4/5 resilience, rolling latest checkpoint, atomic write
- RTX 5090 Blackwell support in setup_runpod.sh (auto-detects sm_120, installs cu128 nightly)
- DDP `find_unused_parameters=True` for inactive recon/context heads
- Defensive try/except around corrupt HDF5 chunks
- Subject-without-bold skip in dataloader
- Video tensor shape fixed to (T, C, H, W) for VideoMAE

---

## Phase 1 — Stimulus discovery (≤30 min, local PC, no GPU)

Both training datasets are auditory narratives. **No video in source data.** Video backbone gets zeros.

### ds002345 — Narratives (OpenNeuro)

- 269 subjects, multiple stories (`task-pieman`, `task-tunnel`, etc.)
- Stimuli: `.wav` audio files in `stimuli/` folder (one per story/task)
- Sample rate: typically 44.1 kHz, mono
- Timing: each subject heard ~10-30 min of spoken story per run
- Event timing: `sub-XXX/func/sub-XXX_task-Y_run-N_events.tsv`
- TR: 1.5s

**Where to find:**
- OpenNeuro: https://openneuro.org/datasets/ds002345
- Or check D:\brain-encoder-data\ds002345\stimuli\ if not deleted
- Use OpenNeuro CLI: `openneuro-py download --dataset ds002345 --include 'stimuli/*' --include '**/*_events.tsv'`

### ds003688 — Mother Of All Comparisons

- 18 subjects, audio narrative listening
- Same general structure — stimuli + events.tsv
- TR may differ; check `dataset_description.json`

**First action of next session:**
```bash
# On your PC, after downloading just stimuli/ + events.tsv from OpenNeuro:
python -c "
import os
import wave
for stim in os.listdir('ds002345/stimuli'):
    if stim.endswith('.wav'):
        with wave.open(f'ds002345/stimuli/{stim}', 'rb') as w:
            print(f'{stim}: {w.getnchannels()}ch, {w.getframerate()}Hz, {w.getnframes()/w.getframerate():.1f}s')
"
```

---

## Phase 2 — Decision: video path

**Decided:** Zero-out video, keep VideoMAE backbone.

Rationale: minimal model changes, lets the encoder still be applied to future video+audio datasets without re-architecting. VideoMAE will produce embeddings for zero input — they collapse to a constant vector, but gradients still flow.

Code change is one line in `train_commercial.py` (Phase 5 below).

Alternative considered: drop VideoMAE entirely — rejected for now, defer until Phase 6 results.

---

## Phase 3 — Build audio aligner (~2-3 hrs, local PC)

Create `tools/align_audio.py`. Input: `(dataset_id, subject_id, run_id, t_bold)` → output: `np.ndarray[80000]` of audio samples at 16 kHz.

### Alignment math

For BOLD volume at TR index `t_bold`:
- BOLD onset time: `bold_t = t_bold * TR`  (TR = 1.5 for ds002345, check ds003688)
- HRF lag: 5.0 seconds (canonical hemodynamic response peak)
- Audio window center: `audio_center = bold_t - hrf_lag`
- Audio window: `[audio_center - 2.5, audio_center + 2.5]` (5s total)
- Sample at 16 kHz → 80000 samples
- Edge cases:
  - `audio_center < 2.5` → zero-pad start
  - `audio_center + 2.5 > stimulus_duration` → zero-pad end
  - Inter-run silence → zeros

### Event timing lookup

Each `sub-XXX_task-Y_run-N_events.tsv` has columns: `onset`, `duration`, `stim_file`, etc.
- Map (subject, run) → stimulus file
- Stimulus duration determines available audio window

### Resampling

Source audio likely 44.1 kHz mono. Use `librosa.resample` or `scipy.signal.resample_poly` to 16 kHz.

### Smoke test

Before writing into h5, verify on one subject:
- Load `sub-001_task-pieman_run-1_events.tsv`
- For t_bold = 50 (= 75s into run, 70s before HRF lag), extract audio
- Plot waveform — should be speech, not silence

---

## Phase 4 — Reprocess h5 with audio (~4-8 hrs, CPU pod or Linux PC)

Two strategies — pick **Strategy A**.

### Strategy A — Add audio dataset alongside bold (recommended)

For each subject group `{ds_id}/{sub_id}`:
- Existing: `bold` shape `(n_tp, 64, 64, 48)`
- Add: `audio` shape `(n_tp, 80000)` float32

Storage estimate:
- Per timepoint: 80000 × 4 bytes = 320 KB uncompressed
- 273 subjects × ~1500 timepoints × 320 KB = ~130 GB uncompressed
- With gzip-6 + shuffle (real speech compresses well): expect ~30-50 GB
- NV has 300 GB - 236 GB BOLD = 64 GB free → fits, but tight

Compression flags (already in `data_pipeline.py` at commit 3d86f21):
```python
grp.create_dataset(
    "audio",
    data=audio_fp32,  # keep float32; fp16 loses speech detail
    chunks=(min(n_tp, 100), 80000),
    compression="gzip",
    compression_opts=6,
    shuffle=True,
)
```

### Strategy B — Load on-the-fly from .wav (NOT recommended)

Slower (~50× slower data loading) unless heavily cached. Skip.

### Run location

Option 1: Linux PC at home — slow but free. ~4-8 hrs.

Option 2: RunPod CPU pod in EU-RO-1 with NV attached — fast, ~$2 cost. **Recommended.**
- Upload stimuli + events to NV first
- Run preprocessing script on NV
- Same pattern as the gdown rclone flow we used to upload the h5

---

## Phase 5 — Patch dataloader (≤30 min, local)

In `train_commercial.py`, `BoldWindowDataset.__getitem__`:

```python
# REPLACE these placeholder lines:
video = torch.randn(self.WINDOW_SIZE, 3, 224, 224, dtype=torch.float32)
audio = torch.randn(80000, dtype=torch.float32)

# WITH:
video = torch.zeros(self.WINDOW_SIZE, 3, 224, 224, dtype=torch.float32)
audio_key = f"{dataset_id}/{subject_id}/audio"
audio = torch.from_numpy(
    f[audio_key][t_centre].astype("float32")
)
```

Also update `BoldWindowDataset.__init__` to verify both `bold` AND `audio` exist per subject. Skip subjects missing either.

Commit the change. Push to origin.

---

## Phase 6 — Smoke test on small subset (≤1 hr GPU, ~$2-3)

**Goal:** confirm real signal emerges. Don't burn money on the full dataset until this passes.

### Setup

- Deploy 1× RTX 5090 in EU-RO-1 with NV attached
- Set `WANDB_API_KEY` as pod env var
- Modify `BoldWindowDataset.__init__` temporarily to limit to first 10 subjects
- Launch with `--num-epochs 2 --batch-size 4 --num-workers 0`

### Success criteria

- Training loss decreases over epoch 0 (not just oscillates around 1.0)
- `val_pearson_r > 0.05` by end of epoch 2
- No new errors

### Failure modes

If r stays at ~0:
- Audio alignment may be wrong (HRF lag, run timing)
- Verify by plotting predicted vs target BOLD for one voxel across time
- Check that audio actually contains speech (not silence) at sampled timepoints

---

## Phase 7 — Full training (~12-24 hrs GPU, ~$30-50)

If Phase 6 hits r > 0.05:

- Revert the subject-limit hack in dataloader
- Deploy 1× RTX 5090 in EU-RO-1 with NV
- `--num-epochs 50 --batch-size 4 --num-workers 0`
- Early-stop kicks in at r ≥ 0.23 (ADR-005)
- Top up RunPod balance before launch (~$50 buffer recommended)

### Watch for

- Training Pearson r climbing past 0.10 by epoch 3-5
- Val Pearson r climbing past 0.15 by epoch 10
- Loss curves stable (no NaNs, no sudden spikes)
- LoRA `best/` checkpoint updates every few epochs
- `latest/` checkpoint updates every epoch (preemption resilience)

### Cost monitoring

Burn rate: $1.49/hr (1× A100 SXM) or $0.99/hr (1× RTX 5090).
Top up RunPod before balance hits $5 — pod is killed at $0 with no grace.

---

## Reference: RunPod / data infrastructure

### Existing assets

- **NV**: `serious_green_wolverine` in EU-RO-1, 300 GB, contains `/workspace/data/superior_brain_data.h5` (236 GB, 287 subjects)
- **GitHub**: github.com/alexxenn/commercial-brain-encoder (HEAD: f8a391a)
- **WandB project**: `commercial-brain-encoder` (alexxen-alexxen-dev workspace)

### Known h5 issues (v1 file)

- Some BOLD chunks have corrupt gzip data — dataloader has try/except → zero tensor
- v1 file is fp32, gzip-1 — should be reprocessed at some point with fp16 + gzip-6 + shuffle for ~5× smaller files
- 14 subjects have no `bold` (only stimulus metadata) — dataloader skips them

### When to reprocess the h5

Reprocessing is a 12-24 hr job (needs raw BOLD on disk). Worth it if:
- You hit corrupt-chunk warnings on >5% of reads
- You need to halve storage cost ($16/mo → $5/mo)
- You add new datasets (ds000113, ds001499) and want them in the same file

Phase 4 of THIS plan adds audio to the existing h5 (no BOLD reprocessing needed). Full BOLD reprocessing is a separate task — not blocking real training.

### Cost so far (2026-05-17 session)

- ~$5 on debugging the multi-GPU stack
- ~$3 on the CPU pod for rclone download
- ~$0.50 NV storage for the day
- Pipeline validation: complete

### Cost projection for real training

- Phase 4 preprocessing (CPU pod): ~$2
- Phase 6 smoke test (1× GPU, 1 hr): ~$1-3
- Phase 7 full training (1× GPU, 12-24 hr): ~$15-30
- **Total to first Pearson r > 0.23**: ~$20-35 if smoke test passes first try

---

## Code changes committed this session

`setup_runpod.sh`:
- GPU-aware PyTorch install (Blackwell detection → cu128 nightly; else cu121 stable)
- Removed `zero_stage` and `gradient_accumulation_steps` from accelerate config heredoc (rejected by accelerate 1.x)

`train_commercial.py`:
- `DistributedDataParallelKwargs(find_unused_parameters=True)` for DDP (inactive recon/context heads)
- Skip subjects without `bold` in dataloader iteration
- `try/except OSError` around bold reads for corrupt gzip chunks
- Video tensor shape `(C, T, H, W)` → `(T, C, H, W)` in both real and synthetic datasets

`tools/fix_h5_attrs.py`:
- One-time migration script to add `n_timepoints` attribute to existing h5 files
- Run via `python tools/fix_h5_attrs.py /workspace/data/superior_brain_data.h5`

---

## What to do RIGHT NOW (next session opening)

```bash
# 1. Verify the v1 h5 is still on the NV
# 2. Download stimuli folder from OpenNeuro for ds002345 to your local PC
# 3. Inspect the .wav files (codec, sample rate, duration)
# 4. Inspect one events.tsv file to understand timing
# 5. Decide: pre-process on your PC (free, slow) or CPU pod (cheap, fast)
```

Don't launch any GPU pod until Phase 4 preprocessing is complete and verified.
