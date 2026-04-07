"""
data_pipeline.py — Commercial Brain Encoder Dataset Pipeline
Downloads + preprocesses OpenNeuro datasets → superior_brain_data.h5

Datasets:
  1. ds003688  iEEG+fMRI+video (OpenNeuro, CC0)
  2. ds004499  movie fMRI (OpenNeuro, CC-BY)
  3. NOD       57K images→fMRI (CC-BY)
  4. IBC       50+ contexts fMRI (CC-BY)
  5. FastMRI   clinical brain MRI (CC0)

Runtime estimate: 12-20h download + 4-6h preprocessing on fast connection
Output: superior_brain_data.h5 (~50-200GB depending on subset)

Usage:
  python data_pipeline.py --datasets ds003688,ds004499 --output-dir data/ --max-subjects 20
"""

import os
import subprocess
import logging
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASETS = {
    "ds003688": {
        "name": "iEEG+fMRI+Video",
        "openneuro_id": "ds003688",
        "license": "CC0",
        "modalities": ["bold", "ieeg"],
        "has_video": True,
        "tr": 1.5,  # repetition time in seconds
        "approx_gb": 40,
    },
    "ds004499": {
        "name": "Movie fMRI",
        "openneuro_id": "ds004499",
        "license": "CC-BY-4.0",
        "modalities": ["bold"],
        "has_video": True,
        "tr": 1.0,
        "approx_gb": 380,
    },
    "ibc": {
        "name": "Individual Brain Charting",
        "download_url": "https://openneuro.org/datasets/ds002685",
        "license": "CC-BY-4.0",
        "modalities": ["bold"],
        "has_video": False,
        "tr": 2.0,
        "approx_gb": 150,
    },
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def check_aws_cli() -> bool:
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_datalad() -> bool:
    # Try CLI first, then python -m datalad (common on Windows where Scripts/ not in PATH)
    for cmd in [["datalad", "--version"], ["python", "-m", "datalad", "--version"]]:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False


def check_boto3() -> bool:
    try:
        import boto3  # noqa: F401
        from botocore.config import Config  # noqa: F401
        return True
    except ImportError:
        return False


def download_openneuro_boto3(
    dataset_id: str,
    output_dir: Path,
    subjects: Optional[list] = None,
) -> None:
    """
    Download from OpenNeuro S3 using boto3 anonymous access.
    Preferred on Windows — no CLI required, boto3 is already installed.
    Bucket: openneuro.org (us-east-1)
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    output_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )
    bucket = "openneuro.org"
    prefix = f"{dataset_id}/"

    # List all keys under prefix (or filtered by subjects)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys_to_download: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            # Filter to subjects if specified
            if subjects:
                rel = key[len(prefix):]  # e.g. "sub-01/func/..."
                sub_folder = rel.split("/")[0]  # "sub-01"
                sub_id = sub_folder.replace("sub-", "")
                if sub_id not in subjects and sub_folder not in subjects:
                    continue
            keys_to_download.append(key)

    log.info(f"{dataset_id}: {len(keys_to_download)} files to download → {output_dir}")

    for key in tqdm(keys_to_download, desc=f"boto3 {dataset_id}"):
        rel_path = key[len(prefix):]
        local_path = output_dir / rel_path
        if local_path.exists():
            continue  # resume support
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))

    log.info(f"Download complete: {dataset_id} → {output_dir}")


def download_openneuro_aws(dataset_id: str, output_dir: Path, subjects: Optional[list] = None):
    """
    Download from OpenNeuro S3 bucket via AWS CLI.
    NOTE: OpenNeuro S3 paths are: s3://openneuro.org/{dataset_id}/
    Use --no-sign-request for anonymous access.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects:
        for sub in subjects:
            sub_path = f"sub-{sub.lstrip('sub-')}"
            s3_path = f"s3://openneuro.org/{dataset_id}/{sub_path}/"
            local_path = output_dir / sub_path
            log.info(f"Downloading {dataset_id}/{sub_path} from S3...")
            cmd = [
                "aws", "s3", "sync", s3_path, str(local_path),
                "--no-sign-request",
                "--include", "*.nii.gz",
                "--include", "*.tsv",
                "--include", "*.json",
            ]
            subprocess.run(cmd, check=True)
    else:
        s3_path = f"s3://openneuro.org/{dataset_id}/"
        log.warning(f"Downloading FULL dataset {dataset_id} — this may be 100+ GB")
        cmd = ["aws", "s3", "sync", s3_path, str(output_dir), "--no-sign-request"]
        subprocess.run(cmd, check=True)

    log.info(f"Download complete: {dataset_id} → {output_dir}")


def download_openneuro_datalad(dataset_id: str, output_dir: Path):
    """DataLad download (preferred — lazy fetch, only get what you need)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/OpenNeuroDatasets/{dataset_id}"
    log.info(f"Installing DataLad dataset {dataset_id}...")
    subprocess.run(["datalad", "install", "-s", url, str(output_dir)], check=True)
    # Fetch only BOLD runs
    subprocess.run(
        ["datalad", "get", "-J", "4", "**/*bold*.nii.gz"],
        cwd=str(output_dir), check=True
    )


def estimate_download_time(gb: float, speed_mbps: float = 100.0) -> str:
    hours = (gb * 1024) / (speed_mbps * 60 * 60 / 8)
    return f"{hours:.1f}h at {speed_mbps}Mbps"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_bold(nii_path: Path) -> np.ndarray:
    """Load and validate BOLD fMRI volume. Returns (T, X, Y, Z) float32."""
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        data = data[..., np.newaxis].transpose(3, 0, 1, 2)
    elif data.ndim == 4:
        data = data.transpose(3, 0, 1, 2)  # (T, X, Y, Z)
    else:
        raise ValueError(f"Unexpected BOLD shape: {data.shape}")
    return data


def normalize_bold(bold: np.ndarray) -> np.ndarray:
    """
    Voxel-wise z-score normalization.
    Each voxel time series: mean=0, std=1.
    NaN/zero-variance voxels → 0.
    """
    mean = bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (bold - mean) / std


def resample_to_standard(bold: np.ndarray, target_shape=(64, 64, 48)) -> np.ndarray:
    """
    Resample spatial dims to standard shape using zoom.
    Input: (T, X, Y, Z) → Output: (T, *target_shape)
    """
    T = bold.shape[0]
    factors = [target_shape[i] / bold.shape[i+1] for i in range(3)]
    resampled = np.stack([
        ndimage.zoom(bold[t], factors, order=1)
        for t in range(T)
    ])
    return resampled


def compute_temporal_snr(bold: np.ndarray) -> float:
    """tSNR = mean(bold_mean) / std(bold_std). Quality metric."""
    mean_img = bold.mean(axis=0)
    std_img = bold.std(axis=0)
    mask = std_img > 1e-8
    tsnr = (mean_img[mask] / std_img[mask]).mean()
    return float(tsnr)


def hrf_convolve(signal: np.ndarray, tr: float = 2.0) -> np.ndarray:
    """
    Convolve stimulus signal with canonical HRF.
    Used for aligning stimulus events to BOLD response.
    """
    from scipy.signal import fftconvolve
    # Canonical double-gamma HRF (SPM-style)
    t = np.arange(0, 32, tr)
    hrf = (
        (t ** 6) * np.exp(-t) / 720
        - 0.35 * (t ** 16) * np.exp(-t) / 1.2e13
    )
    hrf /= hrf.max()
    convolved = fftconvolve(signal, hrf, mode='full')[:len(signal)]
    return convolved

# ---------------------------------------------------------------------------
# Subject processor
# ---------------------------------------------------------------------------

class SubjectProcessor:
    def __init__(self, subject_dir: Path, dataset_config: dict, target_shape=(64, 64, 48)):
        self.subject_dir = subject_dir
        self.config = dataset_config
        self.target_shape = target_shape
        self.subject_id = subject_dir.name

    def find_bold_runs(self) -> list[Path]:
        return sorted(self.subject_dir.rglob("*task-*_bold.nii.gz"))

    def process(self) -> Optional[dict]:
        runs = self.find_bold_runs()
        if not runs:
            log.warning(f"{self.subject_id}: no BOLD runs found, skipping")
            return None

        all_bold = []
        all_tsnr = []
        processed_runs = []

        for run_path in runs:
            try:
                bold = load_bold(run_path)
                tsnr = compute_temporal_snr(bold)

                if tsnr < 15.0:
                    log.warning(f"{self.subject_id}/{run_path.name}: tSNR={tsnr:.1f} < 15, skipping")
                    continue

                bold = normalize_bold(bold)
                bold = resample_to_standard(bold, self.target_shape)
                all_bold.append(bold)
                all_tsnr.append(tsnr)
                processed_runs.append(run_path.name)
                log.info(f"  {self.subject_id}/{run_path.name}: shape={bold.shape} tSNR={tsnr:.1f}")

            except Exception as e:
                log.error(f"{self.subject_id}/{run_path.name}: {e}")
                continue

        if not all_bold:
            return None

        concatenated = np.concatenate(all_bold, axis=0)

        return {
            "bold": concatenated,           # (T_total, X, Y, Z)
            "tsnr_values": all_tsnr,
            "tsnr_mean": float(np.mean(all_tsnr)),
            "n_timepoints": concatenated.shape[0],
            "n_runs": len(processed_runs),
            "runs": processed_runs,
        }

# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_to_h5(output_path: Path, dataset_id: str, subject_id: str, data: dict):
    with h5py.File(str(output_path), "a") as f:
        grp_path = f"{dataset_id}/{subject_id}"
        if grp_path in f:
            del f[grp_path]
        grp = f.require_group(grp_path)
        grp.create_dataset("bold", data=data["bold"], compression="gzip", compression_opts=4)
        grp.attrs["tsnr_mean"] = data["tsnr_mean"]
        grp.attrs["n_timepoints"] = data["n_timepoints"]
        grp.attrs["n_runs"] = data["n_runs"]
        grp.attrs["runs"] = json.dumps(data["runs"])

# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------

def print_dataset_stats(h5_path: Path) -> None:
    sep = "=" * 60
    log.info(sep)
    log.info("SUPERIOR BRAIN DATA — DATASET STATISTICS")
    log.info(sep)
    total_subjects = 0
    total_timepoints = 0

    with h5py.File(str(h5_path), "r") as f:
        for dataset_id in f.keys():
            subjects = list(f[dataset_id].keys())
            ds_timepoints = sum(f[f"{dataset_id}/{s}"].attrs["n_timepoints"] for s in subjects)
            tsnr_vals = [f[f"{dataset_id}/{s}"].attrs["tsnr_mean"] for s in subjects]
            log.info(f"  {dataset_id}:")
            log.info(f"    Subjects:    {len(subjects)}")
            log.info(f"    Timepoints:  {ds_timepoints:,} ({ds_timepoints * 2 / 3600:.1f}h at TR=2s)")
            log.info(f"    tSNR:        {np.mean(tsnr_vals):.1f} ± {np.std(tsnr_vals):.1f}")
            total_subjects += len(subjects)
            total_timepoints += ds_timepoints

    log.info("  TOTAL:")
    log.info(f"    Subjects:    {total_subjects}")
    log.info(f"    Timepoints:  {total_timepoints:,} (~{total_timepoints * 2 / 3600:.0f}h fMRI)")
    log.info(f"    File size:   {h5_path.stat().st_size / 1e9:.1f}GB")
    log.info(sep)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Commercial Brain Encoder Data Pipeline")
    parser.add_argument("--datasets", default="ds003688,ds004499",
                        help="Comma-separated dataset IDs to process")
    parser.add_argument("--data-dir", default="D:/brain-encoder-data", help="Root data directory")
    parser.add_argument("--output", default="D:/brain-encoder-data/superior_brain_data.h5", help="Output HDF5 path")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit subjects per dataset (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, process existing data")
    parser.add_argument("--target-shape", default="64,64,48",
                        help="Target voxel shape X,Y,Z")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    target_shape = tuple(int(x) for x in args.target_shape.split(","))
    dataset_ids = [d.strip() for d in args.datasets.split(",")]

    # Check tools — boto3 preferred on Windows (no CLI needed)
    has_boto3 = check_boto3()
    has_aws = check_aws_cli()
    has_datalad = check_datalad()
    if not has_boto3 and not has_aws and not has_datalad:
        log.error("No download method available. Run: pip install boto3")
        return
    if has_boto3:
        log.info("Using boto3 for download (anonymous S3 access — no credentials needed)")
    elif has_datalad:
        log.info("Using DataLad for download")
    else:
        log.info("Using AWS CLI for download")

    for dataset_id in dataset_ids:
        config = DATASETS.get(dataset_id)
        if not config:
            log.warning(f"Unknown dataset: {dataset_id}, skipping")
            continue

        dataset_dir = data_dir / dataset_id

        # --- DOWNLOAD ---
        if not args.skip_download:
            approx_gb = config["approx_gb"]
            log.info(f"Dataset {dataset_id}: ~{approx_gb}GB, "
                     f"est. {estimate_download_time(approx_gb)}")

            subjects = None
            if args.max_subjects:
                subjects = [str(i).zfill(2) for i in range(1, args.max_subjects + 1)]
                log.info(f"Limiting to {args.max_subjects} subjects")

            if has_boto3:
                download_openneuro_boto3(dataset_id, dataset_dir, subjects)
            elif has_datalad:
                download_openneuro_datalad(dataset_id, dataset_dir)
            else:
                download_openneuro_aws(dataset_id, dataset_dir, subjects)

        # --- PROCESS ---
        subject_dirs = sorted(dataset_dir.glob("sub-*"))
        if args.max_subjects:
            subject_dirs = subject_dirs[:args.max_subjects]

        log.info(f"Processing {len(subject_dirs)} subjects from {dataset_id}...")

        for sub_dir in tqdm(subject_dirs, desc=f"{dataset_id}"):
            processor = SubjectProcessor(sub_dir, config, target_shape)
            result = processor.process()
            if result is not None:
                write_to_h5(output_path, dataset_id, sub_dir.name, result)

    print_dataset_stats(output_path)
    log.info(f"Pipeline complete → {output_path}")


if __name__ == "__main__":
    main()
