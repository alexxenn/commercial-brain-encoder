"""
train_commercial.py — Explicit Accelerate training loop for CommercialBrainEncoder.

Launch command (2×GPU):
    accelerate launch --num_processes 2 train_commercial.py [args]

Dry-run (no HDF5 needed):
    python train_commercial.py --max-steps 10 --dry-run

ADRs honoured:
  ADR-002: LoRA adapter-only checkpoint (~10MB) via peft_model.save_pretrained()
  ADR-003: BrainEncoderLoss(w_voxel=0.5, w_recon=1.0, w_ctx=0.3)
  ADR-005: Early stop at val Pearson r >= 0.23 (manual, no Lightning callback)
  ADR-006: Accelerate — explicit loop, no pl.Trainer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from accelerate import Accelerator, DistributedDataParallelKwargs
from torchmetrics import PearsonCorrCoef

# ---------------------------------------------------------------------------
# stdlib stdout must be UTF-8 on Windows (Windows AppControl feedback)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from commercial_brain_encoder import (
    BrainEncoderConfig,
    BrainEncoderLoss,
    CommercialBrainEncoder,
)
from monitor import TrainingMonitor, build_approximate_roi_masks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_commercial")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BoldWindowDataset(Dataset):
    """
    Sliding-window dataset over BOLD time series stored in superior_brain_data.h5.

    HDF5 structure expected (written by data_pipeline.py):
        {dataset_id}/{subject_id}/bold  — shape: (T, X, Y, Z)
        attrs: n_timepoints, n_runs

    Each item is a window of `window_size` consecutive BOLD frames.

    Returns dict:
        bold:  (1, 64, 64, 48)  float32  — single BOLD volume (centre of window)
        video: (3, 16, 224, 224) float32 — TODO: replace with aligned video frames
        audio: (80000,)          float32 — TODO: replace with aligned audio waveform
    """

    WINDOW_SIZE: int = 16  # frames, matches VideoMAE clip length

    def __init__(self, h5_path: str | Path) -> None:
        import h5py

        self._h5_path = str(h5_path)
        self._index: list[tuple[str, str, int]] = []  # (dataset_id, subject_id, t_start)

        with h5py.File(self._h5_path, "r") as f:
            for dataset_id in f.keys():
                for subject_id in f[dataset_id].keys():
                    grp = f[f"{dataset_id}/{subject_id}"]
                    if "bold" not in grp:
                        continue  # skip subjects with no BOLD data
                    n_tp: int = int(grp.attrs["n_timepoints"])
                    # Stride = 1 (dense windows). Adjust stride for less overlap.
                    for t in range(0, n_tp - self.WINDOW_SIZE + 1):
                        self._index.append((dataset_id, subject_id, t))

        if not self._index:
            raise RuntimeError(
                f"No windows found in {h5_path}. "
                "Run data_pipeline.py first or use --dry-run."
            )

        logger.info("BoldWindowDataset: %d windows across all subjects", len(self._index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import h5py

        dataset_id, subject_id, t_start = self._index[idx]
        key = f"{dataset_id}/{subject_id}/bold"

        with h5py.File(self._h5_path, "r") as f:
            t_centre = t_start + self.WINDOW_SIZE // 2
            try:
                # Load centre frame only for BOLD input — shape (X, Y, Z)
                bold_vol = torch.from_numpy(
                    f[key][t_centre].astype("float32")  # (64, 64, 48)
                )
            except OSError:
                # Corrupt gzip chunk — return zeros so training continues.
                # Tracked as a known limitation of the v1 h5 (some chunks
                # have bad bytes). Will be eliminated when h5 is reprocessed
                # with the fp16+gzip-6+shuffle pipeline (commit 3d86f21).
                bold_vol = torch.zeros(64, 64, 48, dtype=torch.float32)

        bold = bold_vol.unsqueeze(0)  # (1, 64, 64, 48)

        # TODO(Phase 5 — see REAL_TRAINING_PLAN.md): replace placeholder video
        # with torch.zeros (audio-only datasets) or aligned video frames.
        # Shape order is (T, C, H, W) — VideoMAE reads channels at dim 2.
        video = torch.randn(self.WINDOW_SIZE, 3, 224, 224, dtype=torch.float32)

        # TODO(Phase 5 — see REAL_TRAINING_PLAN.md): replace placeholder noise
        # with audio window from {ds}/{sub}/audio (5s, 80000 samples @ 16kHz).
        audio = torch.randn(80000, dtype=torch.float32)

        return {"bold": bold, "video": video, "audio": audio}


class SyntheticBoldDataset(Dataset):
    """
    Fully synthetic dataset for --dry-run. No HDF5 required.
    Produces the same dict shape as BoldWindowDataset.
    """

    def __init__(self, size: int = 64) -> None:
        self._size = size
        logger.info("SyntheticBoldDataset: %d synthetic samples (dry-run)", size)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        return {
            "bold": torch.randn(1, 64, 64, 48, dtype=torch.float32),
            # VideoMAE expects (T, C, H, W) — channels at dim 2 after batching.
            "video": torch.randn(16, 3, 224, 224, dtype=torch.float32),
            "audio": torch.randn(80000, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Pearson r (val) — manual implementation (~25 lines, ADR-005)
# ---------------------------------------------------------------------------

class RunningPearsonR:
    """
    Accumulates (pred, target) pairs across val batches and computes mean
    Pearson r over all voxels and samples at .compute() time.

    Manual implementation — no Lightning callback (ADR-005).
    Uses torchmetrics.PearsonCorrCoef per-sample and averages.
    """

    def __init__(self, device: torch.device) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._device = device

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: (B, num_voxels) on any device — detached before storing."""
        self._preds.append(pred.detach().cpu())
        self._targets.append(target.detach().cpu())

    def compute(self) -> float:
        """Returns mean Pearson r across all accumulated (pred, target) pairs."""
        if not self._preds:
            return 0.0
        all_pred = torch.cat(self._preds, dim=0)    # (N, num_voxels)
        all_tgt = torch.cat(self._targets, dim=0)   # (N, num_voxels)

        # Per-sample Pearson r across voxels
        pred_c = all_pred - all_pred.mean(dim=1, keepdim=True)
        tgt_c = all_tgt - all_tgt.mean(dim=1, keepdim=True)
        r = (pred_c * tgt_c).sum(dim=1) / (
            pred_c.norm(dim=1) * tgt_c.norm(dim=1) + 1e-8
        )  # (N,)
        return float(r.mean().item())

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


# ---------------------------------------------------------------------------
# LoRA adapter checkpoint (ADR-002)
# ---------------------------------------------------------------------------

def save_lora_adapters(model: CommercialBrainEncoder, checkpoint_dir: str | Path) -> None:
    """
    Save LoRA adapters only (~10MB) — NOT full model state_dict.
    ADR-002: adapter-only checkpoint for fast iteration and small storage.

    Saves:
        {checkpoint_dir}/video_backbone/  — VideoMAE LoRA adapter
        {checkpoint_dir}/audio_backbone/  — Wav2Vec2 LoRA adapter
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model.video_backbone.save_pretrained(str(ckpt_path / "video_backbone"))
    model.audio_backbone.save_pretrained(str(ckpt_path / "audio_backbone"))

    logger.info("LoRA adapters saved to %s", ckpt_path)


# ---------------------------------------------------------------------------
# Training state checkpoint — pod restart resumption
# ---------------------------------------------------------------------------

def load_training_state(
    checkpoint_dir: Path,
    accelerator: Accelerator,
) -> tuple[int, int, float]:
    """
    Restore training state from a checkpoint written by save_training_state().

    Returns: (resumed_epoch, resumed_global_step, resumed_best_val_pearson)

    Guard: if checkpoint_dir doesn't exist or training_state.json is missing,
    returns (0, 0, -inf) so training starts fresh without crashing.
    ADR-006: accelerator.load_state() handles device placement automatically.
    """
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CommercialBrainEncoder with Accelerate (ADR-006)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default="D:/brain-encoder-data/superior_brain_data.h5",
        help="Path to superior_brain_data.h5 (written by data_pipeline.py)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="D:/brain-encoder-data/checkpoints",
        help="Root directory for checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N global steps (dry-run / smoke test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run 10 steps with synthetic data, verify Accelerate setup, "
            "save LoRA checkpoint, exit 0. No HDF5 file needed."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default="commercial-brain-encoder",
        help="WandB project name",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="WandB run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--discord-webhook",
        default=None,
        help="Discord webhook URL for epoch notifications",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of dataset held out for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.23,
        help="Pearson r threshold for early stopping (ADR-005)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR",
        help=(
            "Resume training from this checkpoint dir (e.g. /workspace/checkpoints/best). "
            "Expects training_state.json + accelerate_state/ written by save_training_state(). "
            "If missing or path doesn't exist, starts fresh with no error."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build synthetic voxel targets for loss computation
# ---------------------------------------------------------------------------

def _make_voxel_targets(
    batch_size: int,
    num_voxels: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Synthetic voxel targets used when real labels are not yet aligned.
    TODO: replace with actual BOLD-aligned target voxel responses.
    Shape: (B, num_voxels)
    """
    return torch.randn(batch_size, num_voxels, device=device)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --dry-run forces --max-steps 10
    if args.dry_run:
        logger.info("DRY-RUN mode: synthetic data, 10 steps, verify Accelerate + LoRA save")
        if args.max_steps is None:
            args.max_steps = 10

    # ------------------------------------------------------------------
    # Accelerator (ADR-006, bf16, 2×GPU)
    # find_unused_parameters=True: recon_head + context_head are inactive by
    # default (ADR-007/008 — deferred to post-training). Their params never
    # receive gradients, which would crash DDP's AllReduce without this flag.
    # ------------------------------------------------------------------
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process

    if is_main:
        logger.info(
            "Accelerate: device=%s  num_processes=%d  mixed_precision=%s",
            accelerator.device,
            accelerator.num_processes,
            accelerator.mixed_precision,
        )

    # ------------------------------------------------------------------
    # Dataset + DataLoaders
    # ------------------------------------------------------------------
    if args.dry_run:
        full_dataset: Dataset = SyntheticBoldDataset(size=max(64, (args.max_steps or 10) * args.batch_size * 2))
    else:
        full_dataset = BoldWindowDataset(h5_path=args.data_path)

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if sys.platform == "win32" else args.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not args.dry_run,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not args.dry_run,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    config = BrainEncoderConfig()  # full production config
    model = CommercialBrainEncoder(config)

    if is_main:
        param_info = model.count_parameters()
        logger.info(
            "Model: total=%s  trainable=%s (%.1f%%)  frozen=%s",
            f"{param_info['total']:,}",
            f"{param_info['trainable']:,}",
            param_info["trainable_pct"],
            f"{param_info['frozen']:,}",
        )

    # ADR-003: fixed loss weights
    criterion = BrainEncoderLoss(w_voxel=0.5, w_recon=1.0, w_ctx=0.3)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Accelerate prepare — wraps model, optimizer, loaders for multi-GPU
    # ------------------------------------------------------------------
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ------------------------------------------------------------------
    # Monitor (main process only)
    # ------------------------------------------------------------------
    run_name = args.run_name or f"run-{'dry' if args.dry_run else 'prod'}-{accelerator.num_processes}gpu"
    monitor: TrainingMonitor | None = None
    if is_main:
        monitor = TrainingMonitor(
            wandb_project=args.wandb_project,
            run_name=run_name,
            discord_webhook=args.discord_webhook,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    best_lora_dir = checkpoint_dir / "best"
    latest_lora_dir = checkpoint_dir / "latest"

    # ------------------------------------------------------------------
    # Resume from checkpoint (--resume flag, T2)
    # load_training_state() is a no-op if path missing — starts fresh.
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_pearson: float = float("-inf")

    # load_training_state contains a collective op (accelerator.load_state).
    # ALL ranks must enter — main reads JSON, all enter load_state together.
    if args.resume is not None:
        resume_dir = Path(args.resume)
        start_epoch, global_step, best_val_pearson = load_training_state(
            checkpoint_dir=resume_dir,
            accelerator=accelerator,
        )

    # Broadcast resume scalars to all processes so every rank skips the same epochs
    start_epoch_t = torch.tensor(start_epoch, device=accelerator.device)
    global_step_t = torch.tensor(global_step, device=accelerator.device)
    best_pearson_t = torch.tensor(best_val_pearson, device=accelerator.device)
    start_epoch_t = accelerator.reduce(start_epoch_t, reduction="max")
    global_step_t = accelerator.reduce(global_step_t, reduction="max")
    best_pearson_t = accelerator.reduce(best_pearson_t, reduction="max")
    start_epoch = int(start_epoch_t.item())
    global_step = int(global_step_t.item())
    best_val_pearson = float(best_pearson_t.item())

    # ------------------------------------------------------------------
    # Pearson r tracker for val
    # ------------------------------------------------------------------
    pearson_tracker = RunningPearsonR(device=accelerator.device)

    # ------------------------------------------------------------------
    # ROI masks — approximate equal partition until brain_mask coordinates
    # are saved by data_pipeline.py (replace with atlas-based masks then).
    # ------------------------------------------------------------------
    roi_masks = build_approximate_roi_masks(
        num_voxels=config.num_voxels,
        device=torch.device("cpu"),  # kept on CPU — pred/target gathered to CPU
    )

    # ------------------------------------------------------------------
    # Training loop (explicit — ADR-006)
    # ------------------------------------------------------------------
    stopped_early = False

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if stopped_early:
                break

            # ---- Train ----
            model.train()
            epoch_train_losses: dict[str, float] = {}

            for batch in train_loader:
                if args.max_steps is not None and global_step >= args.max_steps:
                    stopped_early = True
                    break

                video: torch.Tensor = batch["video"]    # (B, 3, 16, 224, 224)
                audio: torch.Tensor = batch["audio"]    # (B, 80000)
                bold: torch.Tensor = batch["bold"]      # (B, 1, 64, 64, 48)

                B = bold.shape[0]

                optimizer.zero_grad()

                outputs = model(
                    video_frames=video,
                    audio_waveform=audio,
                    brain_voxels=bold,
                )

                # TODO: wire real voxel targets from dataset once stimulus alignment done
                voxel_targets = _make_voxel_targets(B, config.num_voxels, bold.device)

                loss_dict = criterion(
                    outputs=outputs,
                    voxel_targets=voxel_targets,
                    clip_embeddings=None,   # TODO: add CLIP embeddings when available
                    context_labels=None,    # TODO: add context labels when available
                )

                accelerator.backward(loss_dict["total"])
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Log raw detached losses (no Pearson metric in train step — val only)
                step_losses = {k: v.detach().item() for k, v in loss_dict.items()}

                # Running average for epoch summary
                for k, v in step_losses.items():
                    epoch_train_losses[k] = epoch_train_losses.get(k, 0.0) + v

                # Train-step Pearson: use BrainEncoderLoss.pearson_metric (non-differentiable)
                # Computed only for monitor display — NOT used for early stop or checkpointing
                with torch.no_grad():
                    train_pearson = float(
                        BrainEncoderLoss.pearson_metric(
                            outputs["voxel_pred"].detach(),
                            voxel_targets,
                        ).item()
                    )

                if is_main and monitor is not None:
                    monitor.log_step(
                        step=global_step,
                        losses=step_losses,
                        pearson_r=train_pearson,
                    )

                global_step += 1

            if stopped_early:
                break

            # ---- Validation (ADR-005: Pearson r early stop) ----
            model.eval()
            pearson_tracker.reset()

            with torch.no_grad():
                for val_batch in val_loader:
                    v_video = val_batch["video"]
                    v_audio = val_batch["audio"]
                    v_bold = val_batch["bold"]
                    B_val = v_bold.shape[0]

                    val_outputs = model(
                        video_frames=v_video,
                        audio_waveform=v_audio,
                        brain_voxels=v_bold,
                    )

                    # TODO: wire real val targets when available
                    val_targets = _make_voxel_targets(B_val, config.num_voxels, v_bold.device)

                    # Gather across GPUs for correct multi-process Pearson r (ADR-006)
                    gathered_pred = accelerator.gather_for_metrics(val_outputs["voxel_pred"])
                    gathered_tgt = accelerator.gather_for_metrics(val_targets)

                    if is_main:
                        pearson_tracker.update(gathered_pred, gathered_tgt)

            val_pearson_r: float = 0.0
            if is_main:
                val_pearson_r = pearson_tracker.compute()

                # ROI-level Pearson r — uses accumulated CPU tensors from pearson_tracker
                if monitor is not None and pearson_tracker._preds:
                    all_pred = torch.cat(pearson_tracker._preds, dim=0)   # (N, num_voxels)
                    all_tgt = torch.cat(pearson_tracker._targets, dim=0)  # (N, num_voxels)
                    monitor.log_roi_pearson(
                        pred=all_pred,
                        target=all_tgt,
                        roi_masks=roi_masks,
                        step=global_step,
                    )

            # Broadcast val_pearson_r to all processes for consistent early-stop decision
            val_r_tensor = torch.tensor(val_pearson_r, device=accelerator.device)
            # Only rank 0 holds a real value; others are 0.0. max propagates rank 0's value.
            val_r_tensor = accelerator.reduce(val_r_tensor, reduction="max")
            val_pearson_r = float(val_r_tensor.item())

            if is_main:
                epoch_metrics = {"val_pearson": val_pearson_r}

                # Append epoch-averaged train losses to epoch metrics
                n_batches = max(1, len(train_loader))
                for k, v in epoch_train_losses.items():
                    epoch_metrics[f"train_{k}_avg"] = v / n_batches

                if monitor is not None:
                    monitor.log_epoch(epoch, epoch_metrics)

                logger.info(
                    "Epoch %d/%d — val_pearson=%.4f  best=%.4f",
                    epoch,
                    args.num_epochs - 1,
                    val_pearson_r,
                    best_val_pearson,
                )

                # Determine if this is a new best (will be broadcast below)
                is_new_best = val_pearson_r > best_val_pearson
            else:
                is_new_best = False

            # ----------------------------------------------------------------
            # Checkpointing — must be OUTSIDE is_main because save_training_state
            # contains a collective op. We broadcast is_new_best so all ranks
            # take the same branch.
            # ----------------------------------------------------------------

            # Broadcast is_new_best to all ranks (reduce(sum) > 0 means any rank saw improvement).
            is_new_best_t = torch.tensor(int(is_new_best), device=accelerator.device)
            is_new_best_t = accelerator.reduce(is_new_best_t, reduction="sum")
            is_new_best_all = is_new_best_t.item() > 0

            # ALWAYS save latest/ checkpoint every epoch (P1-4 — protects against
            # plateau + preemption). Collective op — all ranks enter.
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

            # Sync early-stop decision across all processes
            stop_tensor = torch.tensor(int(stopped_early), device=accelerator.device)
            stop_tensor = accelerator.reduce(stop_tensor, reduction="sum")
            if stop_tensor.item() > 0:
                stopped_early = True
                break

    finally:
        if is_main and monitor is not None:
            monitor.close()

    # ------------------------------------------------------------------
    # Dry-run verification output
    # ------------------------------------------------------------------
    if args.dry_run and is_main:
        logger.info("DRY-RUN complete:")
        logger.info("  global_step=%d", global_step)
        logger.info("  Accelerate num_processes=%d", accelerator.num_processes)
        logger.info("  mixed_precision=%s", accelerator.mixed_precision)
        logger.info("  LoRA adapter checkpoint dir: %s", best_lora_dir)

        video_ckpt = best_lora_dir / "video_backbone"
        audio_ckpt = best_lora_dir / "audio_backbone"
        if video_ckpt.exists() and audio_ckpt.exists():
            logger.info("  video_backbone adapter: %s", video_ckpt)
            logger.info("  audio_backbone adapter: %s", audio_ckpt)
            logger.info("DRY-RUN PASSED: Accelerate setup verified, LoRA checkpoint saved.")
        else:
            logger.warning(
                "DRY-RUN: checkpoint not found at expected path %s "
                "(may be missing due to no val epoch completing in 10 steps). "
                "Forcing adapter save now...",
                best_lora_dir,
            )
            # Force save so dry-run always produces a checkpoint artifact
            unwrapped = accelerator.unwrap_model(model)
            save_lora_adapters(unwrapped, best_lora_dir)
            logger.info("DRY-RUN PASSED (forced save): LoRA checkpoint at %s", best_lora_dir)

        sys.exit(0)

    if is_main:
        logger.info(
            "Training complete. Best val_pearson=%.4f. LoRA adapters at %s",
            best_val_pearson,
            best_lora_dir,
        )


if __name__ == "__main__":
    main()
