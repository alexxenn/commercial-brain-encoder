#!/bin/bash
# setup_runpod.sh — RunPod launch script for CommercialBrainEncoder (2×A100)
# Usage: bash setup_runpod.sh
# Idempotent: safe to run twice. set -euo pipefail aborts on any error.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. GPU HEALTH CHECK
# Abort immediately if fewer than 2 CUDA GPUs are visible.
# ---------------------------------------------------------------------------
echo "==> Checking GPU availability..."
python -c "
import torch
count = torch.cuda.device_count()
assert count >= 2, f'Need 2 GPUs, got {count}'
for i in range(count):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'  GPU check passed: {count} devices')
"

# ---------------------------------------------------------------------------
# 1. ENV VAR VALIDATION
# Fail hard on required vars; warn and continue on optional ones.
# ---------------------------------------------------------------------------
echo "==> Validating environment variables..."

# WANDB_API_KEY — required; training monitor initialises wandb at startup
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY not set. Export it before running this script."
    echo "  export WANDB_API_KEY=<your-key>"
    exit 1
fi
echo "  WANDB_API_KEY: set"

# DISCORD_WEBHOOK — optional; passed to --discord-webhook arg in train_commercial.py
if [[ -z "${DISCORD_WEBHOOK:-}" ]]; then
    echo "  WARNING: DISCORD_WEBHOOK not set — epoch notifications disabled"
else
    echo "  DISCORD_WEBHOOK: set"
fi

# DATA_PATH — path to superior_brain_data.h5 written by data_pipeline.py
DATA_PATH="${DATA_PATH:-/workspace/data/superior_brain_data.h5}"
echo "  DATA_PATH: ${DATA_PATH}"

# CHECKPOINT_DIR — LoRA adapter checkpoints land here (ADR-002: ~10MB per save)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR}"

# ---------------------------------------------------------------------------
# 2. DIRECTORY SETUP
# Create data and checkpoint dirs so the script is safe to run before data
# is uploaded, and so save_lora_adapters() never fails on mkdir.
# ---------------------------------------------------------------------------
echo "==> Creating directories..."
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "$(dirname "${DATA_PATH}")"

# ---------------------------------------------------------------------------
# 3. PIP INSTALL — PINNED DEPS
# Install PyTorch ecosystem first via the CUDA 12.1 index to guarantee
# torchvision/torchaudio binary ABI compatibility with torch.
# Subsequent packages use the default PyPI index.
# pip install --upgrade is idempotent — re-running upgrades in-place.
# ---------------------------------------------------------------------------
echo "==> Installing PyTorch ecosystem (CUDA 12.1)..."
pip install --upgrade \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    torchaudio>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing HuggingFace + training deps..."
pip install --upgrade \
    transformers>=4.38.0 \
    peft>=0.9.0 \
    accelerate>=0.27.0

echo "==> Installing metrics and monitoring..."
pip install --upgrade \
    torchmetrics>=1.3.0 \
    wandb>=0.16.0 \
    rich>=13.7.0

echo "==> Installing data pipeline deps..."
pip install --upgrade \
    h5py>=3.10.0 \
    nibabel>=5.2.0 \
    scipy>=1.12.0 \
    tqdm>=4.66.0 \
    numpy>=1.26.0

echo "==> Installing inference/demo dep..."
pip install --upgrade \
    gradio>=4.20.0

# ---------------------------------------------------------------------------
# 4. HUGGINGFACE MODEL CACHE — pre-download backbones
# commercial_brain_encoder.py pulls MCG-NJU/videomae-base and
# facebook/wav2vec2-base on first CommercialBrainEncoder() instantiation.
# Doing it here gives a clear download log separate from training output.
# HF_HOME defaults to ~/.cache/huggingface; override via env var if needed.
# ---------------------------------------------------------------------------
echo "==> Pre-downloading HuggingFace model weights..."
python -c "
from transformers import VideoMAEModel, Wav2Vec2Model
print('  Downloading MCG-NJU/videomae-base...')
VideoMAEModel.from_pretrained('MCG-NJU/videomae-base', ignore_mismatched_sizes=True)
print('  Downloading facebook/wav2vec2-base...')
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
print('  Model weights cached.')
"

# ---------------------------------------------------------------------------
# 5. ACCELERATE CONFIG
# Written inline so the user never needs to run `accelerate config`.
# Overwritten on every run (idempotent). bf16 matches ADR-006.
# num_processes=2 targets the 2×A100 pod.
# ---------------------------------------------------------------------------
echo "==> Writing accelerate_config.yaml..."
cat > accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
# bf16 — ADR-006: A100 supports bf16 natively; better dynamic range than fp16
mixed_precision: bf16
zero_stage: 0
gradient_accumulation_steps: 1
EOF
echo "  accelerate_config.yaml written."

# ---------------------------------------------------------------------------
# 6. WANDB LOGIN
# Authenticate non-interactively using the key validated above.
# --relogin ensures stale cached credentials are replaced.
# ---------------------------------------------------------------------------
echo "==> Authenticating WandB..."
wandb login --relogin "${WANDB_API_KEY}"

# ---------------------------------------------------------------------------
# 7. LAUNCH TRAINING
# accelerate launch handles DDP process spawning (2 processes, one per GPU).
# All CLI args map 1:1 to argparse flags in train_commercial.py.
# DISCORD_WEBHOOK is passed only if set; empty string would confuse the
# argparse optional string check so we branch here.
# ---------------------------------------------------------------------------
echo "==> Launching training on 2×A100..."
echo "    DATA_PATH:      ${DATA_PATH}"
echo "    CHECKPOINT_DIR: ${CHECKPOINT_DIR}"

DISCORD_ARG=""
if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
    DISCORD_ARG="--discord-webhook ${DISCORD_WEBHOOK}"
fi

# shellcheck disable=SC2086  # DISCORD_ARG intentionally word-splits to nothing when empty
RESUME_ARG=""
if [[ -f "${CHECKPOINT_DIR}/best/training_state.json" ]]; then
    RESUME_ARG="--resume ${CHECKPOINT_DIR}/best"
    echo "    Resuming from checkpoint: ${CHECKPOINT_DIR}/best"
else
    echo "    No checkpoint found — starting fresh"
fi

# shellcheck disable=SC2086  # RESUME_ARG and DISCORD_ARG intentionally word-split when empty
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 2 \
    train_commercial.py \
    --data-path "${DATA_PATH}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --num-epochs 50 \
    --batch-size 4 \
    --num-workers 4 \
    --wandb-project "commercial-brain-encoder" \
    ${RESUME_ARG} \
    ${DISCORD_ARG}

echo "==> Training complete. LoRA adapters at ${CHECKPOINT_DIR}/best/"
