"""
tests/test_model.py — Unit tests for CommercialBrainEncoder.

All tests use session-scoped `model_small` + `small_config` fixtures.
Runtime target: <60s on CPU (enforced by small_config dimensions).
No TRIBE v2 / MindEye / BrainBench imports — clean-room only.
"""

import ast
import os
from typing import Set

import pytest
import torch

from commercial_brain_encoder import BrainEncoderConfig, CommercialBrainEncoder, PerceiverResampler


# ---------------------------------------------------------------------------
# Required output keys (subset — forward() may include additional keys such
# as `stimulus_repr` when called without brain_voxels, and `brain_cls` /
# `brain_tokens` when brain_voxels is provided).
# ---------------------------------------------------------------------------

_REQUIRED_OUTPUT_KEYS: Set[str] = {"voxel_pred", "recon_emb", "context_logits"}
_FORBIDDEN_IMPORTS: Set[str] = {"tribe", "mindeye", "brainbench"}


# ---------------------------------------------------------------------------
# 1. Output keys
# ---------------------------------------------------------------------------


def test_forward_output_keys(
    model_small: CommercialBrainEncoder,
    dummy_video: torch.Tensor,
    dummy_audio: torch.Tensor,
) -> None:
    """forward() output must contain voxel_pred, recon_emb, context_logits."""
    with torch.no_grad():
        outputs = model_small(dummy_video, dummy_audio)

    assert _REQUIRED_OUTPUT_KEYS.issubset(outputs.keys()), (
        f"Missing keys: {_REQUIRED_OUTPUT_KEYS - outputs.keys()}"
    )


# ---------------------------------------------------------------------------
# 2. voxel_pred shape
# ---------------------------------------------------------------------------


def test_voxel_pred_shape(
    model_small: CommercialBrainEncoder,
    small_config: BrainEncoderConfig,
    dummy_video: torch.Tensor,
    dummy_audio: torch.Tensor,
) -> None:
    """voxel_pred shape must be (B=2, num_voxels=256)."""
    B: int = dummy_video.shape[0]
    with torch.no_grad():
        outputs = model_small(dummy_video, dummy_audio)

    assert outputs["voxel_pred"].shape == (B, small_config.num_voxels), (
        f"Expected ({B}, {small_config.num_voxels}), got {outputs['voxel_pred'].shape}"
    )


# ---------------------------------------------------------------------------
# 3. recon_emb unit normalisation
# ---------------------------------------------------------------------------


def test_recon_emb_unit_normalized(
    model_small: CommercialBrainEncoder,
    dummy_video: torch.Tensor,
    dummy_audio: torch.Tensor,
) -> None:
    """recon_emb must be L2-unit-normalised along the last dimension."""
    with torch.no_grad():
        outputs = model_small(dummy_video, dummy_audio)

    recon_emb: torch.Tensor = outputs["recon_emb"]
    norm: torch.Tensor = recon_emb.norm(dim=-1)
    ones: torch.Tensor = torch.ones_like(norm)

    assert torch.allclose(norm, ones, atol=1e-5), (
        f"recon_emb is not unit-normalised; norms: {norm.tolist()}"
    )


# ---------------------------------------------------------------------------
# 4. No NaN in outputs
# ---------------------------------------------------------------------------


def test_no_nan_in_outputs(
    model_small: CommercialBrainEncoder,
    dummy_video: torch.Tensor,
    dummy_audio: torch.Tensor,
    dummy_bold: torch.Tensor,
) -> None:
    """None of voxel_pred, recon_emb, context_logits may contain NaN."""
    with torch.no_grad():
        outputs = model_small(dummy_video, dummy_audio, dummy_bold)

    for key in ("voxel_pred", "recon_emb", "context_logits"):
        tensor: torch.Tensor = outputs[key]
        assert not torch.isnan(tensor).any(), (
            f"NaN detected in outputs['{key}']"
        )


# ---------------------------------------------------------------------------
# 5. brain_voxels is optional
# ---------------------------------------------------------------------------


def test_brain_voxels_optional(
    model_small: CommercialBrainEncoder,
    dummy_video: torch.Tensor,
    dummy_audio: torch.Tensor,
) -> None:
    """forward() must not crash when brain_voxels is None or omitted."""
    with torch.no_grad():
        # Explicitly pass None
        outputs_none = model_small(dummy_video, dummy_audio, brain_voxels=None)
        # Omit entirely
        outputs_omit = model_small(dummy_video, dummy_audio)

    assert _REQUIRED_OUTPUT_KEYS.issubset(outputs_none.keys())
    assert _REQUIRED_OUTPUT_KEYS.issubset(outputs_omit.keys())

    # brain_cls must NOT appear when no BOLD provided
    assert "brain_cls" not in outputs_none, (
        "brain_cls should be absent when brain_voxels=None"
    )
    assert "brain_cls" not in outputs_omit, (
        "brain_cls should be absent when brain_voxels is omitted"
    )


# ---------------------------------------------------------------------------
# 6. PerceiverResampler output shape
# ---------------------------------------------------------------------------


def test_perceiver_resampler_output_shape(
    small_config: BrainEncoderConfig,
    device: torch.device,
) -> None:
    """PerceiverResampler output shape must be (B, num_latents, dim)."""
    B: int = 2
    seq_len: int = 20
    dim: int = small_config.dim
    num_latents: int = small_config.num_video_latents

    resampler = PerceiverResampler(
        dim=dim,
        num_latents=num_latents,
        depth=small_config.perceiver_depth,
        num_heads=small_config.perceiver_heads,
    ).to(device)
    resampler.eval()

    x: torch.Tensor = torch.randn(B, seq_len, dim, device=device)
    with torch.no_grad():
        out: torch.Tensor = resampler(x)

    assert out.shape == (B, num_latents, dim), (
        f"Expected ({B}, {num_latents}, {dim}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 7. context_logits shape
# ---------------------------------------------------------------------------


def test_context_logits_shape(
    model_small: CommercialBrainEncoder,
    small_config: BrainEncoderConfig,
    dummy_video: torch.Tensor,
    dummy_audio: torch.Tensor,
) -> None:
    """context_logits shape must be (B=2, num_context_classes=10)."""
    B: int = dummy_video.shape[0]
    with torch.no_grad():
        outputs = model_small(dummy_video, dummy_audio)

    assert outputs["context_logits"].shape == (B, small_config.num_context_classes), (
        f"Expected ({B}, {small_config.num_context_classes}), "
        f"got {outputs['context_logits'].shape}"
    )


# ---------------------------------------------------------------------------
# 8. No forbidden imports (AST compile-time check)
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    """commercial_brain_encoder.py must not import from tribe, mindeye, or brainbench."""
    source_path: str = os.path.join(
        os.path.dirname(__file__), "..", "commercial_brain_encoder.py"
    )
    with open(source_path, "r", encoding="utf-8") as fh:
        source: str = fh.read()

    tree: ast.Module = ast.parse(source, filename="commercial_brain_encoder.py")

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root: str = alias.name.split(".")[0].lower()
                if root in _FORBIDDEN_IMPORTS:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                root = node.module.split(".")[0].lower()
                if root in _FORBIDDEN_IMPORTS:
                    violations.append(f"from {node.module} import ...")

    assert not violations, (
        f"Forbidden imports found in commercial_brain_encoder.py: {violations}"
    )
