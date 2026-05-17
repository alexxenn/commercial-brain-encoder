"""
One-time data migration: add `n_timepoints` attribute to each subject group
in an existing superior_brain_data.h5.

The streaming data_pipeline.py (commit a58fb1b) writes this attribute on every
new h5 it produces. But existing v1 files processed by older versions of the
pipeline lack it, and the dataloader at train_commercial.py:90 requires it.

Usage:
    python tools/fix_h5_attrs.py /path/to/superior_brain_data.h5
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py


def fix(h5_path: Path) -> None:
    print(f"=== Inspecting {h5_path} ===")
    with h5py.File(h5_path, "r") as f:
        sample_ds = next(iter(f.keys()))
        sample_sub = next(iter(f[sample_ds].keys()))
        grp = f[f"{sample_ds}/{sample_sub}"]
        print(f"  Sample {sample_ds}/{sample_sub} attrs: {dict(grp.attrs)}")

    print("\n=== Writing n_timepoints ===")
    with h5py.File(h5_path, "a") as f:
        updated = 0
        skipped = 0
        for ds_id in f.keys():
            for sub_id in f[ds_id].keys():
                grp = f[f"{ds_id}/{sub_id}"]
                if "bold" in grp:
                    grp.attrs["n_timepoints"] = grp["bold"].shape[0]
                    updated += 1
                else:
                    skipped += 1
        print(f"  Updated: {updated} subjects")
        print(f"  Skipped (no bold): {skipped} subjects")

    print("\n=== Verifying ===")
    with h5py.File(h5_path, "r") as f:
        with_attr = 0
        total = 0
        for ds_id in f.keys():
            for sub_id in f[ds_id].keys():
                total += 1
                if "n_timepoints" in f[f"{ds_id}/{sub_id}"].attrs:
                    with_attr += 1
        print(f"  {with_attr} / {total} subjects have n_timepoints")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    fix(Path(sys.argv[1]))
