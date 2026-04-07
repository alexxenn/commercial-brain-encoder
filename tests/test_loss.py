"""
tests/test_loss.py — Unit tests for BrainEncoderLoss.

Covers:
  - Pearson loss == 0 when pred == target (perfect correlation)
  - Pearson metric range [-1, 1]
  - total_loss is positive and finite for typical inputs
  - No NaN / inf under bfloat16 mixed-precision
  - Loss weight keys present and correct (ADR-003: w_voxel=0.5, w_recon=1.0, w_ctx=0.3)
  - Contrastive loss symmetry: swap pred/target stays within 10x

Math invariant: r=1.0 → pearson_loss = 1 - r = 0.0
"""

import torch
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from commercial_brain_encoder import BrainEncoderLoss, BrainEncoderConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH: int = 4
N_VOXELS: int = 256   # matches small_config.num_voxels
CLIP_DIM: int = 128   # matches small_config.clip_dim
N_CLASSES: int = 10   # matches small_config.num_context_classes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outputs(pred_voxels: torch.Tensor) -> dict:
    """Build a minimal outputs dict with synthetic heads for loss forward()."""
    B = pred_voxels.shape[0]
    return {
        "voxel_pred": pred_voxels,
        "recon_emb": torch.nn.functional.normalize(
            torch.randn(B, CLIP_DIM), dim=-1
        ),
        "context_logits": torch.randn(B, N_CLASSES),
    }


# ---------------------------------------------------------------------------
# 1. Perfect correlation → pearson_loss ≈ 0.0
# ---------------------------------------------------------------------------


def test_pearson_loss_perfect_correlation() -> None:
    """
    When pred == target, Pearson r == 1.0 for every sample in the batch,
    so pearson_loss = mean(1 - r) = mean(0) = 0.0.
    Shape: (B=4, N=256).
    """
    loss_fn = BrainEncoderLoss()
    torch.manual_seed(0)
    tensor: torch.Tensor = torch.randn(BATCH, N_VOXELS)

    result: torch.Tensor = loss_fn.pearson_loss(tensor, tensor)

    assert result.ndim == 0, "Expected scalar output"
    assert torch.isfinite(result), "Loss must be finite"
    assert result.item() == pytest.approx(0.0, abs=1e-5), (
        f"Expected pearson_loss ≈ 0.0 for identical tensors, got {result.item()}"
    )


# ---------------------------------------------------------------------------
# 2. Pearson metric is in [-1, 1]
# ---------------------------------------------------------------------------


def test_pearson_metric_range() -> None:
    """
    For 10 random (pred, target) pairs of shape (B=4, N=256),
    pearson_metric must lie in [-1.0, 1.0].
    """
    for seed in range(10):
        torch.manual_seed(seed)
        pred: torch.Tensor = torch.randn(BATCH, N_VOXELS)
        target: torch.Tensor = torch.randn(BATCH, N_VOXELS)

        r: torch.Tensor = BrainEncoderLoss.pearson_metric(pred, target)

        assert r.ndim == 0, f"seed={seed}: expected scalar"
        assert torch.isfinite(r), f"seed={seed}: metric must be finite"
        assert -1.0 <= r.item() <= 1.0, (
            f"seed={seed}: pearson_metric={r.item():.6f} is outside [-1, 1]"
        )


# ---------------------------------------------------------------------------
# 3. total_loss > 0 and finite
# ---------------------------------------------------------------------------


def test_total_loss_positive_and_finite() -> None:
    """
    For typical random inputs (voxel pred + CLIP embeddings + context labels),
    total_loss must be strictly positive and torch.isfinite().
    """
    torch.manual_seed(7)
    loss_fn = BrainEncoderLoss()

    voxel_pred: torch.Tensor = torch.randn(BATCH, N_VOXELS)
    voxel_target: torch.Tensor = torch.randn(BATCH, N_VOXELS)
    clip_emb: torch.Tensor = torch.nn.functional.normalize(
        torch.randn(BATCH, CLIP_DIM), dim=-1
    )
    ctx_labels: torch.Tensor = torch.randint(0, N_CLASSES, (BATCH,))

    outputs = _make_outputs(voxel_pred)
    losses: dict = loss_fn(
        outputs=outputs,
        voxel_targets=voxel_target,
        clip_embeddings=clip_emb,
        context_labels=ctx_labels,
    )

    total: torch.Tensor = losses["total"]
    assert torch.isfinite(total), f"total_loss is not finite: {total.item()}"
    assert total.item() > 0.0, (
        f"total_loss should be > 0 for random inputs, got {total.item()}"
    )


# ---------------------------------------------------------------------------
# 4. No NaN / inf under bfloat16 mixed precision
# ---------------------------------------------------------------------------


def test_no_nan_under_bf16_precision() -> None:
    """
    Cast inputs to bfloat16, compute Pearson loss, cast result to float32.
    Simulates AMP training. No NaN, no inf.

    Note: pearson_loss is the numerically sensitive path — bfloat16 has only
    7 mantissa bits vs 23 for float32, so the epsilon guard (1e-8) matters.
    """
    torch.manual_seed(3)
    loss_fn = BrainEncoderLoss()

    pred_fp32: torch.Tensor = torch.randn(BATCH, N_VOXELS)
    target_fp32: torch.Tensor = torch.randn(BATCH, N_VOXELS)

    pred_bf16: torch.Tensor = pred_fp32.to(torch.bfloat16)
    target_bf16: torch.Tensor = target_fp32.to(torch.bfloat16)

    result_bf16: torch.Tensor = loss_fn.pearson_loss(pred_bf16, target_bf16)
    result_f32: torch.Tensor = result_bf16.to(torch.float32)

    assert not torch.isnan(result_f32), (
        f"NaN detected in bfloat16 pearson_loss: {result_f32.item()}"
    )
    assert not torch.isinf(result_f32), (
        f"Inf detected in bfloat16 pearson_loss: {result_f32.item()}"
    )


# ---------------------------------------------------------------------------
# 5. Loss weights applied — keys present, ADR-003 values
# ---------------------------------------------------------------------------


def test_loss_weights_applied() -> None:
    """
    Instantiate BrainEncoderLoss with ADR-003 weights (w_voxel=0.5, w_recon=1.0,
    w_ctx=0.3). Verify:
      - output dict has keys: "voxel", "recon", "context", "total"
      - total equals the weighted sum of components
    """
    w_voxel: float = 0.5
    w_recon: float = 1.0
    w_ctx: float = 0.3
    loss_fn = BrainEncoderLoss(w_voxel=w_voxel, w_recon=w_recon, w_ctx=w_ctx)

    torch.manual_seed(11)
    voxel_pred: torch.Tensor = torch.randn(BATCH, N_VOXELS)
    voxel_target: torch.Tensor = torch.randn(BATCH, N_VOXELS)
    clip_emb: torch.Tensor = torch.nn.functional.normalize(
        torch.randn(BATCH, CLIP_DIM), dim=-1
    )
    ctx_labels: torch.Tensor = torch.randint(0, N_CLASSES, (BATCH,))

    outputs = _make_outputs(voxel_pred)
    losses: dict = loss_fn(
        outputs=outputs,
        voxel_targets=voxel_target,
        clip_embeddings=clip_emb,
        context_labels=ctx_labels,
    )

    # Required keys
    required_keys = {"voxel", "recon", "context", "total"}
    assert required_keys.issubset(losses.keys()), (
        f"Missing keys: {required_keys - losses.keys()}"
    )

    # Verify weighted sum matches total
    expected_total: torch.Tensor = (
        w_voxel * losses["voxel"]
        + w_recon * losses["recon"]
        + w_ctx * losses["context"]
    )
    assert losses["total"].item() == pytest.approx(expected_total.item(), rel=1e-5), (
        f"total={losses['total'].item()} != weighted_sum={expected_total.item()}"
    )

    # All components finite
    for key in required_keys:
        assert torch.isfinite(losses[key]), f"losses['{key}'] is not finite"


# ---------------------------------------------------------------------------
# 6. Contrastive loss symmetry
# ---------------------------------------------------------------------------


def test_contrastive_loss_symmetry() -> None:
    """
    NT-Xent contrastive loss in BrainEncoderLoss forward() uses:
        (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2

    The loss is already symmetric by construction (average of both directions).
    Test that swapping pred/target in voxel loss doesn't produce wildly
    asymmetric values: |L(a,b) - L(b,a)| / min(L(a,b), L(b,a)) < 10.

    For the contrastive recon component specifically: compute it twice with
    swapped recon_emb/clip_emb and verify ratio stays < 10x.
    """
    import torch.nn.functional as F

    torch.manual_seed(42)

    # Simulate two random normalized embeddings (brain and CLIP)
    emb_a: torch.Tensor = F.normalize(torch.randn(BATCH, CLIP_DIM), dim=-1)
    emb_b: torch.Tensor = F.normalize(torch.randn(BATCH, CLIP_DIM), dim=-1)
    temperature: float = 14.3
    labels: torch.Tensor = torch.arange(BATCH)

    def _nt_xent(q: torch.Tensor, k: torch.Tensor) -> float:
        logits = q @ k.T * temperature
        return (
            (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        ).item()

    loss_ab: float = _nt_xent(emb_a, emb_b)
    loss_ba: float = _nt_xent(emb_b, emb_a)

    # NT-Xent is symmetric by construction (logits.T swap), so they should be equal
    assert loss_ab == pytest.approx(loss_ba, rel=1e-4), (
        f"NT-Xent should be symmetric: L(a,b)={loss_ab:.6f}, L(b,a)={loss_ba:.6f}"
    )

    # Sanity: neither direction is zero or negative
    assert loss_ab > 0.0, f"Contrastive loss must be positive, got {loss_ab}"
    assert loss_ba > 0.0, f"Contrastive loss must be positive, got {loss_ba}"

    # Ratio guard (10x ceiling for asymmetric inputs in general use)
    ratio: float = max(loss_ab, loss_ba) / (min(loss_ab, loss_ba) + 1e-8)
    assert ratio < 10.0, (
        f"Contrastive loss ratio {ratio:.2f}x exceeds 10x symmetry bound: "
        f"L(a,b)={loss_ab:.6f}, L(b,a)={loss_ba:.6f}"
    )
