# RunPod Launch Guide — CommercialBrainEncoder

## Before you start (do these on your machine)

- [ ] All datasets processed into `superior_brain_data.h5`
- [ ] HDF5 verified (shapes correct, no missing subjects)
- [ ] Latest code pushed to GitHub
- [ ] WandB API key ready → get it at wandb.ai/authorize
- [ ] Discord webhook URL ready (optional — epoch notifications)

---

## Step 1 — Create RunPod account

1. Go to **runpod.io** → Sign up
2. Add a payment method (credit card)
3. Add credits — $25 covers ~2 full training runs

---

## Step 2 — Rent the pod

1. Click **GPU Pods** → **+ Deploy**
2. Search for **A100 SXM** → select **80 GB VRAM**
3. Select **On-Demand** (not Spot — avoids interruption mid-training)
4. Template: choose **RunPod PyTorch 2.1** (has CUDA 12.1 pre-installed)
5. **Container disk**: set to **30 GB** (for code + pip packages + model weights)
6. **Volume disk**: set to **20 GB** mounted at `/workspace/data` (for HDF5 + checkpoints)
7. Click **Deploy** → wait ~2 min for pod to start

---

## Step 3 — Connect to the pod

1. In RunPod dashboard, click your pod → **Connect**
2. Click **Start Web Terminal** (or use SSH — RunPod shows the SSH command)

---

## Step 4 — Upload the HDF5

From your machine (in PowerShell or the RunPod web file manager):

**Option A — RunPod web file manager (easiest):**
1. In the pod dashboard click **Files**
2. Navigate to `/workspace/data/`
3. Upload `D:\brain-encoder-data\superior_brain_data.h5`

**Option B — rsync via SSH (faster for large files):**
```bash
# RunPod shows your SSH command in the Connect panel, e.g.:
# ssh root@<ip> -p <port> -i ~/.ssh/id_rsa

rsync -avz --progress \
  -e "ssh -p <PORT>" \
  "D:/brain-encoder-data/superior_brain_data.h5" \
  root@<IP>:/workspace/data/
```

The HDF5 is ~7-15GB depending on how many datasets you processed — upload takes 5-15 min.

---

## Step 5 — Clone the code

In the pod terminal:

```bash
cd /workspace
git clone https://github.com/alexxenn/commercial-brain-encoder brain-encoder
cd brain-encoder
```

---

## Step 6 — Set environment variables

```bash
export WANDB_API_KEY=<your-key-from-wandb.ai/authorize>
export DISCORD_WEBHOOK=<your-webhook-url>   # optional

# Confirm data path (should already be correct default)
export DATA_PATH=/workspace/data/superior_brain_data.h5
export CHECKPOINT_DIR=/workspace/checkpoints
```

---

## Step 7 — Run the setup + training script

```bash
bash setup_runpod.sh
```

This will automatically:
1. Check GPU is available
2. Install all pip dependencies
3. Download VideoMAE + Wav2Vec2 weights from HuggingFace (~2 GB, one-time)
4. Write accelerate config for single A100
5. Authenticate WandB
6. Launch training (50 epochs, batch_size=4)

**Expected setup time:** ~10 min before training starts
**Expected training time:** 8-15 hours depending on dataset size

---

## Step 8 — Monitor training

**WandB dashboard (primary):**
- Go to wandb.ai → your project `commercial-brain-encoder`
- Watch: `train/pearson_r`, `epoch/val_pearson`, `roi/*/pearson_r`
- Target: `epoch/val_pearson` ≥ 0.23 → early stop triggers automatically

**Discord (if webhook set):**
- Epoch-end notification with val_pearson + all metrics

**Pod terminal:**
- Rich 4-panel live display showing step-level metrics

---

## Step 9 — Download checkpoints when done

Training saves LoRA adapters (~10 MB) to `/workspace/checkpoints/best/`.

```bash
# From your machine:
rsync -avz \
  -e "ssh -p <PORT>" \
  root@<IP>:/workspace/checkpoints/ \
  "D:/brain-encoder-data/checkpoints/"
```

---

## Step 10 — Destroy the pod

**IMPORTANT: do this as soon as training finishes or you will keep paying.**

1. RunPod dashboard → your pod → **Stop** → **Terminate**
2. Confirm termination
3. Verify it no longer shows as running

The `/workspace/data/` volume can be kept separately (small monthly cost) so you don't re-upload the HDF5 next time.

---

## Cost estimate

| Item | Cost |
|---|---|
| A100 SXM 80GB on-demand | $1.49/hr |
| 10h training run | ~$15 |
| 15h training run | ~$22 |
| Setup time (~10 min) | ~$0.25 |

---

## If training gets interrupted

```bash
# Re-run the script — it will resume from the last checkpoint
# (pip install is idempotent, wandb login is idempotent)
bash setup_runpod.sh
```

Note: `train_commercial.py` does not currently have checkpoint resumption — it restarts from epoch 0. If interrupted mid-run, the LoRA adapters from the best epoch so far are still saved at `/workspace/checkpoints/best/`.
