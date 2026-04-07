"""
tests/conftest.py — Shared pytest fixtures for CommercialBrainEncoder.

All fixtures are CPU-safe and sized for <60s wall-clock on a development machine.
No TRIBE v2 / MindEye / BrainBench / CC-BY-NC imports — clean-room only.
"""

import pytest
import torch

import sys
import os

# Ensure project root is importable regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from commercial_brain_encoder import BrainEncoderConfig, CommercialBrainEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_config() -> BrainEncoderConfig:
    """
    Minimal BrainEncoderConfig for CPU tests.

    Constraints:
    - dim=128: fusion_heads=8 divides evenly (128 % 8 == 0),
      perceiver_heads=8 divides evenly (128 % 8 == 0).
    - fusion_layers=2: 2 transformer layers instead of default 12.
    - lora_r=4: smallest practical LoRA rank, lora_alpha=8 (2×r convention).
    - num_voxels reduced to 256 so VoxelPredictionHead is tiny on CPU.
    - perceiver_depth=1: single cross-attention pass, halves resampler cost.
    """
    return BrainEncoderConfig(
        dim=128,
        video_dim=768,        # VideoMAE base — unchanged, it's a backbone output dim
        audio_dim=768,        # Wav2Vec2 base — unchanged
        num_voxels=256,       # reduced from 20000 for CPU speed
        voxel_input_shape=(64, 64, 48),
        num_video_latents=8,  # reduced from 32
        num_audio_latents=8,  # reduced from 32
        perceiver_depth=1,    # reduced from 2
        perceiver_heads=8,    # must divide dim=128
        fusion_heads=8,       # must divide dim=128
        fusion_layers=2,
        fusion_dropout=0.0,   # deterministic for tests
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        clip_dim=128,         # reduced — only needs to be divisible by heads
        num_context_classes=10,
    )


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def dummy_bold() -> torch.Tensor:
    """
    Simulated fMRI BOLD volume: (B=2, 1, D=64, H=64, W=48), float32, z-scored.

    Values sampled from N(0,1) and clamped to [-2, 2] to mimic
    z-scored BOLD signal used during preprocessing.
    """
    torch.manual_seed(42)
    t = torch.randn(2, 1, 64, 64, 48, dtype=torch.float32)
    return t.clamp(-2.0, 2.0)


@pytest.fixture(scope="function")
def dummy_video() -> torch.Tensor:
    """
    Simulated video frames: (B=2, C=3, T=16, H=224, W=224), float32, [0, 1].

    Shape matches VideoMAE input convention (channels-first, T frames).
    Uniform random in [0, 1] — no ImageNet normalisation applied here;
    tests that need normalised input should do so explicitly.
    """
    torch.manual_seed(0)
    return torch.rand(2, 3, 16, 224, 224, dtype=torch.float32)


@pytest.fixture(scope="function")
def dummy_audio() -> torch.Tensor:
    """
    Simulated audio waveform: (B=2, L=80000), float32, [-1, 1].

    80000 samples = 5 seconds at 16 kHz, matching Wav2Vec2 expected input.
    Values uniform in [-1, 1] to represent normalised PCM audio.
    """
    torch.manual_seed(1)
    return torch.rand(2, 80000, dtype=torch.float32) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Always CPU — ensures tests are hardware-agnostic and reproducible."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_small(small_config: BrainEncoderConfig, device: torch.device) -> CommercialBrainEncoder:
    """
    CommercialBrainEncoder instantiated with small_config, on CPU, in eval mode.

    Session-scoped to avoid re-downloading VideoMAE + Wav2Vec2 checkpoints
    on every test. Tests that require specific parameter states (e.g. training
    mode, gradient tracking) must call .train() / .requires_grad_() locally
    and restore state after.
    """
    model = CommercialBrainEncoder(config=small_config)
    model.to(device)
    model.eval()
    return model
