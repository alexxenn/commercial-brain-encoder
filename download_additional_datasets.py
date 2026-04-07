"""
download_additional_datasets.py — Download priority datasets for CommercialBrainEncoder

Run this WHILE data_pipeline.py --skip-download processes ds003688.
Two terminals, running simultaneously.

Priority order (by value/size ratio):
  1. ds000113 — StudyForrest (22GB, 15 subjects, 2h movie, CC-BY-4.0) ← START HERE
  2. ds002685 — IBC (150GB, 12 subjects, 50 tasks, CC-BY-4.0) ← optional, large

Usage:
  python download_additional_datasets.py --dataset ds000113
  python download_additional_datasets.py --dataset ds002685
  python download_additional_datasets.py --all-priority   # ds000113 only (safe default)
"""

import sys
import os

# Windows UTF-8 fix
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import logging
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry — all CC0 or CC-BY, commercially verified
# Excludes: ds003688 (already downloaded), ds004499 (380GB, defer)
# ---------------------------------------------------------------------------

ADDITIONAL_DATASETS = {
    "ds000113": {
        "name": "StudyForrest — Forrest Gump fMRI",
        "openneuro_id": "ds000113",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 22,
        "subjects": 15,
        "timepoints_per_subject": 7200,  # 8 runs × ~15min × TR=2.0s
        "why": "2h naturalistic movie watching — 12x more timepoints than ds003688",
        "tr": 2.0,
    },
    "ds002345": {
        "name": "Narratives — fMRI during spoken stories",
        "openneuro_id": "ds002345",
        "license": "CC0",
        "commercial_verified": True,
        "approx_gb": 30,   # 345 subjects × ~90MB average
        "subjects": 345,   # largest subject count of any open fMRI dataset
        "timepoints_per_subject": 1500,  # varies by story (~10-20min each)
        "why": "345 subjects — by far the biggest signal for audio encoder generalisation",
        "tr": 1.5,
    },
    "ds006642": {
        "name": "NNDb-3T+ — Back to the Future fMRI",
        "openneuro_id": "ds006642",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 25,
        "subjects": 40,
        "timepoints_per_subject": 6000,  # full-length movie ~2h at TR=1.0s
        "why": "Full movie watching with eye-tracking + physio — perfect video+audio paradigm",
        "tr": 1.0,
    },
    "ds004848": {
        "name": "Game of Thrones fMRI",
        "openneuro_id": "ds004848",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 15,
        "subjects": 73,   # 28 prosopagnosics + 45 healthy controls
        "timepoints_per_subject": 800,
        "why": "73 subjects, audiovisual clips, face/scene localizers included",
        "tr": 2.0,
    },
    "ds001499": {
        "name": "BOLD5000 — 5000 scenes fMRI",
        "openneuro_id": "ds001499",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 18,
        "subjects": 4,
        "timepoints_per_subject": 5254,  # one trial per image
        "why": "5,254 diverse images (SUN/COCO/ImageNet) — bridges brain and computer vision",
        "tr": 2.0,
    },
    "ds004192": {
        "name": "THINGS-fMRI",
        "openneuro_id": "ds004192",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 40,
        "subjects": 4,
        "timepoints_per_subject": 26107,  # 26K unique images
        "why": "26K images × 720 semantic categories — best for visual object encoding",
        "tr": 1.5,
    },
    "ds003643": {
        "name": "Le Petit Prince fMRI Corpus",
        "openneuro_id": "ds003643",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 20,
        "subjects": 49,   # English speakers (+ 35 Chinese + 28 French in separate datasets)
        "timepoints_per_subject": 3000,
        "why": "Multilingual audiobook with NLP word-level annotations — rich audio encoder training",
        "tr": 1.5,
    },
    "ds002685": {
        "name": "Individual Brain Charting (IBC)",
        "openneuro_id": "ds002685",
        "license": "CC-BY-4.0",
        "commercial_verified": True,
        "approx_gb": 150,
        "subjects": 12,
        "timepoints_per_subject": 50000,
        "why": "50+ cognitive tasks — adds task diversity beyond movie watching",
        "tr": 2.0,
    },
}

# Tier 1: download these now (high value, manageable size, perfect modality match)
PRIORITY_DATASETS = ["ds002345", "ds006642", "ds004848"]

# Tier 2: after Tier 1 finishes (also great, slightly less urgent)
TIER2_DATASETS = ["ds001499", "ds004192", "ds003643"]

# Large datasets: opt-in only
LARGE_DATASETS = ["ds002685"]  # 150GB IBC


def estimate_time(gb: float, speed_mbps: float = 100.0) -> str:
    hours = (gb * 1024) / (speed_mbps * 3600 / 8)
    if hours < 1:
        return f"{hours * 60:.0f}min at {speed_mbps}Mbps"
    return f"{hours:.1f}h at {speed_mbps}Mbps"


def download_openneuro_boto3(
    dataset_id: str,
    output_dir: Path,
    max_subjects: int | None = None,
) -> None:
    """Anonymous S3 download from OpenNeuro. No credentials needed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )
    bucket = "openneuro.org"
    prefix = f"{dataset_id}/"

    log.info(f"Listing files for {dataset_id}...")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            if max_subjects is not None:
                rel = key[len(prefix):]
                top = rel.split("/")[0]
                if top.startswith("sub-"):
                    sub_num = int(top.replace("sub-", "").lstrip("0") or "0")
                    if sub_num > max_subjects:
                        continue
            keys.append(key)

    log.info(f"{dataset_id}: {len(keys)} files to download")

    downloaded = 0
    skipped = 0
    for key in tqdm(keys, desc=dataset_id):
        rel_path = key[len(prefix):]
        local_path = output_dir / rel_path
        if local_path.exists():
            skipped += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        downloaded += 1

    log.info(f"Done: {downloaded} downloaded, {skipped} skipped (already existed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download additional fMRI datasets")
    parser.add_argument(
        "--dataset",
        choices=list(ADDITIONAL_DATASETS.keys()),
        help="Specific dataset to download",
    )
    parser.add_argument(
        "--all-priority",
        action="store_true",
        help=f"Download Tier 1 datasets: {PRIORITY_DATASETS}",
    )
    parser.add_argument(
        "--tier2",
        action="store_true",
        help=f"Download Tier 2 datasets: {TIER2_DATASETS}",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/brain-encoder-data",
        help="Root output directory (default: D:/brain-encoder-data)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit subjects (for testing)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets with details",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable additional datasets (all CC-BY/CC0, commercially verified):\n")
        for key, cfg in ADDITIONAL_DATASETS.items():
            print(f"  {key}: {cfg['name']}")
            print(f"    License:   {cfg['license']}")
            print(f"    Size:      ~{cfg['approx_gb']}GB ({estimate_time(cfg['approx_gb'])})")
            print(f"    Subjects:  {cfg['subjects']}")
            print(f"    Why:       {cfg['why']}")
            print()
        return

    targets: list[str] = []
    if args.dataset:
        targets = [args.dataset]
    elif args.all_priority:
        targets = PRIORITY_DATASETS
    elif args.tier2:
        targets = TIER2_DATASETS
    else:
        parser.print_help()
        print("\nTip: Run with --list to see available datasets")
        return

    output_root = Path(args.output_dir)

    for dataset_id in targets:
        cfg = ADDITIONAL_DATASETS[dataset_id]
        log.info(
            f"Starting download: {cfg['name']} | ~{cfg['approx_gb']}GB | "
            f"est. {estimate_time(cfg['approx_gb'])}"
        )
        log.info(f"Why this dataset: {cfg['why']}")
        out_dir = output_root / dataset_id
        download_openneuro_boto3(
            dataset_id=cfg["openneuro_id"],
            output_dir=out_dir,
            max_subjects=args.max_subjects,
        )
        log.info(f"Finished: {dataset_id} → {out_dir}")


if __name__ == "__main__":
    main()
