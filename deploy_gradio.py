"""
deploy_gradio.py — Gradio demo for CommercialBrainEncoder.

Accepts an fMRI NIfTI upload, runs inference with zero stimulus tensors
(no video/audio at demo time), and returns:
  1. 3-plane max-projection heatmap of the input brain volume (PIL Image)
  2. PSNR of voxel_pred vs. a zero baseline (float, dB)
  3. Top context class label (str)

NOTE: video_frames and audio_waveform are passed as zeros at demo time.
      This means the model decodes from brain signal alone — voxel_pred
      and context_logits are driven entirely by the brain_encoder pathway.

Usage:
  python deploy_gradio.py --demo            # launch locally
  python deploy_gradio.py --share --demo    # launch with public URL
  python deploy_gradio.py --checkpoint checkpoints/best/ --port 7861 --demo
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Logging — no print() anywhere in this file
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("brain_encoder.demo")

# Force non-interactive matplotlib backend before any figure calls
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_BYTES: int = 500 * 1024 * 1024  # 500 MB
VALID_EXTENSIONS: frozenset[str] = frozenset({".nii", ".nii.gz"})
TARGET_SHAPE: tuple[int, int, int] = (64, 64, 48)  # (D, H, W) matching BrainEncoderConfig

CONTEXT_LABELS: list[str] = [
    "Visual-Static",
    "Visual-Dynamic",
    "Auditory-Speech",
    "Auditory-Music",
    "Auditory-Noise",
    "Visual+Auditory",
    "Motor",
    "Cognitive",
    "Emotional",
    "Rest",
]

# Module-level model cache — loaded once, reused across requests
_model_cache: Optional[torch.nn.Module] = None
_model_config_cache: Optional[object] = None

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load CommercialBrainEncoder from checkpoint directory.

    Tries ``CommercialBrainEncoder.from_pretrained()`` first; falls back to
    loading ``config.pt`` + ``model.pt`` (state_dict) manually.

    Args:
        checkpoint_path: Directory containing saved checkpoint files.

    Returns:
        Model in eval mode on CPU (or CUDA if available).

    Raises:
        FileNotFoundError: If no recognisable checkpoint files are found.
        RuntimeError: If state_dict loading fails.
    """
    # Local import to avoid loading heavy backbones at module import time
    from commercial_brain_encoder import BrainEncoderConfig, CommercialBrainEncoder

    ckpt_dir = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model from '%s' onto %s", ckpt_dir, device)

    # --- Strategy 1: from_pretrained() if the classmethod exists ---
    if hasattr(CommercialBrainEncoder, "from_pretrained"):
        log.info("Using CommercialBrainEncoder.from_pretrained()")
        model: CommercialBrainEncoder = CommercialBrainEncoder.from_pretrained(str(ckpt_dir))
        model.to(device).eval()
        return model

    # --- Strategy 2: manual config + state_dict load ---
    config_path = ckpt_dir / "config.pt"
    weights_path = ckpt_dir / "model.pt"

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Pass --checkpoint <path> pointing to a directory with model.pt"
        )

    # Load config (may not exist if training was never run — use defaults)
    if config_path.exists():
        cfg: BrainEncoderConfig = torch.load(str(config_path), map_location="cpu")
        log.info("Loaded BrainEncoderConfig from %s", config_path)
    else:
        log.warning(
            "config.pt not found at '%s' — using default BrainEncoderConfig", config_path
        )
        cfg = BrainEncoderConfig()

    model = CommercialBrainEncoder(config=cfg)

    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location="cpu")
        # Handle DataParallel / DDP wrapper prefix
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys in state_dict (%d): %s …", len(missing), missing[:5])
        if unexpected:
            log.warning(
                "Unexpected keys in state_dict (%d): %s …", len(unexpected), unexpected[:5]
            )
        log.info("Loaded state_dict from %s", weights_path)
    else:
        log.warning(
            "model.pt not found at '%s' — running with random weights (demo/test only)",
            weights_path,
        )

    model.to(device).eval()
    return model


def get_model(checkpoint_path: str) -> torch.nn.Module:
    """Return cached model, loading from disk on first call.

    Args:
        checkpoint_path: Passed through to ``_load_model`` on first call only.

    Returns:
        Cached ``CommercialBrainEncoder`` in eval mode.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model(checkpoint_path)
        log.info("Model cached — subsequent calls skip disk load")
    return _model_cache


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


def _validate_nifti_upload(
    original_filename: str,
    tmp_path: str,
) -> None:
    """Validate uploaded file before nibabel touches it.

    Checks (in order):
      1. Extension — .nii or .nii.gz only (from browser-reported name)
      2. File size — must be ≤ MAX_FILE_BYTES
      3. Magic bytes — prevents extension-spoofing attacks:
           .nii.gz → first 2 bytes must be gzip magic 0x1f 0x8b
           .nii    → bytes 344-347 must be NIfTI magic b'n+1\\0' or b'ni1\\0'

    Args:
        original_filename: Original name as reported by the browser/Gradio
            (used for extension check — the tempfile path is not trusted).
        tmp_path: Filesystem path Gradio wrote the upload to.

    Raises:
        ValueError: On extension mismatch, size violation, or magic byte failure.
    """
    fname = original_filename.lower()
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        raise ValueError(
            f"Invalid file type: '{original_filename}'. "
            "Only .nii and .nii.gz files are accepted."
        )

    file_size = os.path.getsize(tmp_path)
    if file_size > MAX_FILE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File too large: {size_mb:.1f} MB. Maximum allowed is 500 MB."
        )

    # Magic byte check — reject files that lie about their extension
    with open(tmp_path, "rb") as fh:
        if fname.endswith(".nii.gz"):
            magic = fh.read(2)
            if magic != b"\x1f\x8b":
                raise ValueError(
                    "File does not appear to be a valid gzip archive. "
                    "Ensure the file is a genuine .nii.gz NIfTI."
                )
        else:  # .nii (uncompressed)
            # NIfTI-1 magic is at byte offset 344 — read just enough
            if file_size < 348:
                raise ValueError(
                    "File too small to be a valid NIfTI-1 file (< 348 bytes)."
                )
            fh.seek(344)
            nifti_magic = fh.read(4)
            if nifti_magic not in (b"n+1\x00", b"ni1\x00"):
                raise ValueError(
                    "File does not contain a valid NIfTI-1 magic signature. "
                    "Ensure the file is a genuine uncompressed .nii NIfTI."
                )

    log.info(
        "File validation passed: '%s' (%.1f MB)",
        original_filename,
        file_size / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Preprocessing — mirrors data_pipeline.py exactly
# ---------------------------------------------------------------------------


def _normalize_bold_single(volume: np.ndarray) -> np.ndarray:
    """Voxel-wise z-score for a single 3D volume (D, H, W).

    Replicates ``normalize_bold`` from data_pipeline.py for a single frame
    rather than a time series: normalise by spatial mean/std across the volume.

    Args:
        volume: Float32 array of shape (D, H, W).

    Returns:
        Z-scored array, same shape. Zero-variance voxels → 0.
    """
    mean = volume.mean()
    std = volume.std()
    if std < 1e-8:
        std = 1.0
    return (volume - mean) / std


def _resample_to_standard(
    volume: np.ndarray,
    target_shape: tuple[int, int, int] = TARGET_SHAPE,
) -> np.ndarray:
    """Resample 3D volume to target spatial shape via linear zoom.

    Replicates ``resample_to_standard`` from data_pipeline.py for a single
    3D volume.

    Args:
        volume: Float32 array of arbitrary spatial shape (D, H, W).
        target_shape: Desired output shape (D, H, W).

    Returns:
        Resampled float32 array of shape ``target_shape``.
    """
    factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return ndimage.zoom(volume, factors, order=1).astype(np.float32)


def preprocess_nifti(nii_path: str) -> torch.Tensor:
    """Load and preprocess a NIfTI fMRI volume for model inference.

    Pipeline:
      1. Load with nibabel → float32 NumPy array
      2. If 4-D (T, D, H, W), take the temporal mean (TR-averaged volume)
      3. Voxel-wise z-score normalisation
      4. Resample to (64, 64, 48)
      5. Return as (1, 1, 64, 64, 48) float32 tensor  [B, C, D, H, W]

    Args:
        nii_path: Filesystem path to the NIfTI file (already validated).

    Returns:
        Tensor of shape (1, 1, 64, 64, 48).

    Raises:
        ValueError: If the loaded data has an unexpected number of dimensions.
    """
    img = nib.load(nii_path)
    data: np.ndarray = img.get_fdata(dtype=np.float32)

    log.info("Loaded NIfTI shape: %s", data.shape)

    if data.ndim == 3:
        volume: np.ndarray = data  # (D, H, W)
    elif data.ndim == 4:
        # Temporal mean — demo doesn't need individual TRs
        volume = data.mean(axis=-1)  # (D, H, W)
        log.info("4-D volume: took temporal mean over %d TRs", data.shape[-1])
    else:
        raise ValueError(
            f"Unexpected NIfTI dimensionality: {data.ndim}D (shape {data.shape}). "
            "Expected 3-D or 4-D brain volume."
        )

    volume = _normalize_bold_single(volume)
    volume = _resample_to_standard(volume, TARGET_SHAPE)

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    log.info("Preprocessed tensor shape: %s", tuple(tensor.shape))
    return tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    brain_tensor: torch.Tensor,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Run forward pass with zero stimulus tensors.

    video_frames and audio_waveform are zeros — no stimulus is available at
    demo time. The model will produce outputs driven by the brain pathway.

    Args:
        brain_tensor: Preprocessed fMRI tensor, shape (1, 1, 64, 64, 48).
        model: Loaded CommercialBrainEncoder in eval mode.

    Returns:
        Raw output dict from model.forward() (all tensors on CPU).
    """
    device = next(model.parameters()).device
    brain_tensor = brain_tensor.to(device)

    # Zero stimulus — documented clearly in UI
    # VideoMAE expects (B, C, T, H, W) = (1, 3, 16, 224, 224)
    video_frames = torch.zeros(1, 3, 16, 224, 224, device=device)
    # Wav2Vec2 expects (B, L) = (1, 80000) — 5s at 16kHz
    audio_waveform = torch.zeros(1, 80_000, device=device)

    with torch.no_grad():
        outputs = model(
            video_frames=video_frames,
            audio_waveform=audio_waveform,
            brain_voxels=brain_tensor,
        )

    # Move everything to CPU for downstream NumPy ops
    return {k: v.cpu() for k, v in outputs.items()}


# ---------------------------------------------------------------------------
# Output computation
# ---------------------------------------------------------------------------


def compute_psnr_vs_zero(voxel_pred: torch.Tensor) -> float:
    """Compute PSNR of voxel_pred against an all-zeros reference.

    Since ground-truth BOLD responses are unavailable at demo time, we compare
    against a zero baseline to quantify the signal magnitude. Higher PSNR means
    the model predicts larger, more structured activations above baseline.

    PSNR = 10 * log10(MAX^2 / MSE),  MAX = max(|voxel_pred|)

    Args:
        voxel_pred: Tensor of shape (B, num_voxels).

    Returns:
        PSNR in dB as a Python float. Returns 0.0 if pred is all zeros.
    """
    pred = voxel_pred[0].float()  # (num_voxels,)
    mse = pred.pow(2).mean().item()
    if mse < 1e-12:
        log.warning("voxel_pred is effectively zero — PSNR undefined, returning 0.0")
        return 0.0
    max_val = pred.abs().max().item()
    if max_val < 1e-12:
        return 0.0
    psnr = 10.0 * math.log10(max_val**2 / mse)
    return round(psnr, 3)


def get_top_context_label(context_logits: torch.Tensor) -> str:
    """Map top-1 predicted class index to a human-readable label.

    Args:
        context_logits: Tensor of shape (B, num_context_classes).

    Returns:
        Label string from CONTEXT_LABELS.
    """
    idx = int(context_logits[0].argmax().item())
    if idx < len(CONTEXT_LABELS):
        label = CONTEXT_LABELS[idx]
    else:
        label = f"Class-{idx}"
    log.info("Top context prediction: index=%d label='%s'", idx, label)
    return label


def render_heatmap(brain_tensor: torch.Tensor) -> Image.Image:
    """Render 3-plane max-projection of a brain volume as a PIL Image.

    Planes rendered: axial (z), sagittal (x), coronal (y).
    Each plane shows the max-projection along its perpendicular axis,
    then normalised to [0, 1] and coloured with the 'hot' colormap.

    Args:
        brain_tensor: Tensor of shape (1, 1, D, H, W) — preprocessed volume.

    Returns:
        PIL Image (RGB) containing the three projections side-by-side.
    """
    vol: np.ndarray = brain_tensor[0, 0].numpy()  # (D, H, W)

    # Three max-projections
    axial = vol.max(axis=2)      # project along W → (D, H)
    sagittal = vol.max(axis=0)   # project along D → (H, W)
    coronal = vol.max(axis=1)    # project along H → (D, W)

    def _norm(arr: np.ndarray) -> np.ndarray:
        """Normalise array to [0, 1], handling flat arrays."""
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-8:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    planes = [
        (_norm(axial), "Axial (max-proj)"),
        (_norm(sagittal), "Sagittal (max-proj)"),
        (_norm(coronal), "Coronal (max-proj)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#1a1a2e")
    colormap = "hot"

    for ax, (plane, title) in zip(axes, planes):
        ax.imshow(plane.T, cmap=colormap, origin="lower", aspect="auto")
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")

    fig.suptitle(
        "fMRI Max-Projection Heatmap",
        color="white",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    # Render to PIL without touching the filesystem
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(height, width, 3)
    pil_image = Image.fromarray(buf)
    plt.close(fig)

    return pil_image


# ---------------------------------------------------------------------------
# Gradio inference handler
# ---------------------------------------------------------------------------


def predict(
    nifti_file: Optional[gr.File],
    checkpoint_path: str,
) -> tuple[Optional[Image.Image], str, str]:
    """Gradio inference callback.

    Args:
        nifti_file: Gradio File object from the upload component.
            ``nifti_file.name`` is the tempfile path; ``nifti_file.orig_name``
            is the original browser filename.
        checkpoint_path: Text input value pointing to checkpoint directory.

    Returns:
        Tuple of (heatmap PIL Image | None, psnr_str, context_label_str).
        Returns (None, error_msg, "") on any validation or runtime error.
    """
    if nifti_file is None:
        return None, "No file uploaded. Please upload a .nii or .nii.gz file.", ""

    # Gradio >=4 passes a file-like dict or a gr.File object.
    # Handle both interfaces safely.
    if hasattr(nifti_file, "name"):
        tmp_path: str = nifti_file.name
        # Gradio 4.x: original filename is in .orig_name; 3.x may differ
        orig_name: str = getattr(nifti_file, "orig_name", None) or os.path.basename(tmp_path)
    else:
        # Gradio returns a plain file path string for some versions
        tmp_path = str(nifti_file)
        orig_name = os.path.basename(tmp_path)

    log.info("Upload received: orig='%s' tmp='%s'", orig_name, tmp_path)

    # --- Security: validate BEFORE nibabel ---
    try:
        _validate_nifti_upload(orig_name, tmp_path)
    except ValueError as exc:
        log.warning("Validation failed: %s", exc)
        return None, f"Validation error: {exc}", ""

    # --- Preprocessing ---
    try:
        brain_tensor = preprocess_nifti(tmp_path)
    except Exception as exc:
        log.error("Preprocessing failed: %s", exc, exc_info=True)
        return None, f"Preprocessing error: {exc}", ""

    # --- Model ---
    try:
        model = get_model(checkpoint_path.strip() or "checkpoints/best/")
    except Exception as exc:
        log.error("Model load failed: %s", exc, exc_info=True)
        return None, f"Model load error: {exc}", ""

    # --- Inference ---
    try:
        outputs = run_inference(brain_tensor, model)
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        return None, f"Inference error: {exc}", ""

    # --- Outputs ---
    psnr = compute_psnr_vs_zero(outputs["voxel_pred"])
    context_label = get_top_context_label(outputs["context_logits"])
    heatmap = render_heatmap(brain_tensor)

    psnr_str = f"{psnr:.3f} dB (vs. zero baseline)"
    log.info("Inference complete — PSNR=%.3f dB  context='%s'", psnr, context_label)

    return heatmap, psnr_str, context_label


# ---------------------------------------------------------------------------
# Gradio interface definition
# ---------------------------------------------------------------------------


def build_interface(default_checkpoint: str) -> gr.Blocks:
    """Construct and return the Gradio Blocks interface.

    Args:
        default_checkpoint: Pre-filled checkpoint path in the text box.

    Returns:
        Configured gr.Blocks object (not yet launched).
    """
    with gr.Blocks(
        title="CommercialBrainEncoder Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # CommercialBrainEncoder — fMRI Inference Demo
            **Upload a brain fMRI volume** (.nii or .nii.gz, max 500 MB) to run inference.

            > **Note on stimulus inputs:** At demo time, video and audio inputs are set to
            > zeros — no external stimulus is provided. Voxel predictions and context
            > classification are therefore driven by the brain signal alone via the
            > `BrainVoxelEncoder` pathway. For full multi-modal inference, integrate
            > with a matched video/audio loader.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                nifti_input = gr.File(
                    label="fMRI NIfTI File (.nii / .nii.gz, max 500 MB)",
                    file_types=[".nii", ".gz"],
                    type="filepath",
                )
                checkpoint_input = gr.Textbox(
                    label="Checkpoint Path",
                    value=default_checkpoint,
                    placeholder="checkpoints/best/",
                    info="Directory containing model.pt (and optionally config.pt).",
                )
                run_btn = gr.Button("Run Inference", variant="primary")

            with gr.Column(scale=3):
                heatmap_output = gr.Image(
                    label="Brain Activation Heatmap (3-plane max-projection)",
                    type="pil",
                )

        with gr.Row():
            psnr_output = gr.Textbox(
                label="PSNR vs. Zero Baseline",
                interactive=False,
                info="Measures predicted signal magnitude. Not a ground-truth comparison.",
            )
            context_output = gr.Textbox(
                label="Predicted Context Class",
                interactive=False,
                info="Top-1 class from context_logits (10 classes).",
            )

        gr.Markdown(
            """
            ---
            **Context classes (10):** Visual-Static · Visual-Dynamic · Auditory-Speech ·
            Auditory-Music · Auditory-Noise · Visual+Auditory · Motor · Cognitive ·
            Emotional · Rest

            **PSNR note:** Compared against a zero vector (no ground truth available).
            A higher value indicates larger predicted activations above baseline.

            **License:** MIT — CommercialBrainEncoder is clean-room IP.
            """
        )

        run_btn.click(
            fn=predict,
            inputs=[nifti_input, checkpoint_input],
            outputs=[heatmap_output, psnr_output, context_output],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with checkpoint, port, demo, share attributes.
    """
    parser = argparse.ArgumentParser(
        description="CommercialBrainEncoder Gradio demo server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/",
        help="Path to checkpoint directory (model.pt + optional config.pt).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port for Gradio server.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch the Gradio interface immediately (required to start server).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable Gradio public URL (tunnelled). Off by default for security.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — build interface and optionally launch.

    Exits with code 0 after launch returns (Gradio blocks until Ctrl-C).
    Does nothing if --demo is not passed (safe import behaviour).
    """
    args = parse_args()

    if not args.demo:
        log.info(
            "No --demo flag provided. Interface built but not launched. "
            "Pass --demo to start the server."
        )
        return

    log.info(
        "Launching CommercialBrainEncoder demo | checkpoint='%s' port=%d share=%s",
        args.checkpoint,
        args.port,
        args.share,
    )

    demo = build_interface(default_checkpoint=args.checkpoint)
    demo.launch(
        server_port=args.port,
        share=args.share,  # False by default — only True when --share is passed
        show_error=True,
    )


if __name__ == "__main__":
    main()
