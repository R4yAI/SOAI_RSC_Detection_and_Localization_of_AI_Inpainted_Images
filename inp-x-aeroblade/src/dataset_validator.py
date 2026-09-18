import os
import sys
import glob
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_original_image(
    mask_stem: str,
    dataset_name: str,
    orig_dir: Path,
    orig_files_map: Dict[str, Path],
) -> Optional[Path]:
    """
    Robustly resolves the original image path for a given mask stem and dataset.
    Handles CelebAHQ, CityScapes, OpenImages, and SUN_RGBD naming variations.
    """
    # 1. Exact match with mask_stem
    if mask_stem in orig_files_map:
        return orig_files_map[mask_stem]

    # 2. Known dataset-specific patterns
    if dataset_name == "CelebAHQ":
        # e.g., '10052_hair' -> '10052'
        base_id = mask_stem.split("_")[0]
        if base_id in orig_files_map:
            return orig_files_map[base_id]

    elif dataset_name == "CityScapes":
        # e.g., 'aachen_000000_000019_instance000' -> 'aachen_000000_000019'
        if "_instance" in mask_stem:
            base_id = mask_stem.split("_instance")[0]
            if base_id in orig_files_map:
                return orig_files_map[base_id]
            # Try with _leftImg8bit suffix (found in train-data)
            left_id = f"{base_id}_leftImg8bit"
            if left_id in orig_files_map:
                return orig_files_map[left_id]

    elif dataset_name == "OpenImages":
        # e.g., '0017d9757c6f4793_m07mhn_6681e819' -> '0017d9757c6f4793' (16-char hex ID)
        base_id = mask_stem.split("_")[0]
        if base_id in orig_files_map:
            return orig_files_map[base_id]

    elif dataset_name == "SUN_RGBD":
        # e.g., '10_1__0000217-000007239456_008' -> '10_1__0000217-000007239456'
        base_id = mask_stem.rsplit("_", 1)[0]
        if base_id in orig_files_map:
            return orig_files_map[base_id]

    # 3. Fallback: longest matching prefix among available original files
    best_match = None
    longest_len = 0
    for orig_stem, path in orig_files_map.items():
        if mask_stem.startswith(orig_stem) and len(orig_stem) > longest_len:
            longest_len = len(orig_stem)
            best_match = path
            
    return best_match


def verify_exchange_integrity(
    orig_path: Path,
    std_path: Path,
    inpx_path: Path,
    mask_path: Path,
    tolerance: int = 2,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool, Tuple[int, int]]:
    """
    Numerically verifies INP-X exchange image construction:
    - Outside mask: inpainting_exchange should match original.
    - Inside mask: inpainting_exchange should match standard_inpainting.
    Returns: (max_err_out, frac_err_out, max_err_in, frac_err_in, is_valid, (width, height))
    """
    try:
        orig = np.array(Image.open(orig_path).convert("RGB"))
        std = np.array(Image.open(std_path).convert("RGB"))
        inpx = np.array(Image.open(inpx_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception as e:
        return None, None, None, None, False, (0, 0)

    # Shape checks
    if not (orig.shape == std.shape == inpx.shape):
        return None, None, None, None, False, (0, 0)
    if mask.shape != orig.shape[:2]:
        return None, None, None, None, False, (0, 0)

    height, width = orig.shape[:2]

    # Binarize mask: 1 = inpaint region, 0 = authentic background
    mask_bin = (mask > 127).astype(np.uint8)[..., None]
    outside_mask = 1 - mask_bin

    # Outside mask verification (INP-X vs Original)
    diff_out = np.abs(inpx.astype(np.int16) - orig.astype(np.int16)) * outside_mask
    max_err_out = float(diff_out.max())
    out_pixels = max(1, int(outside_mask.sum() * 3))
    frac_err_out = float((diff_out > tolerance).sum() / out_pixels)

    # Inside mask verification (INP-X vs Standard Inpainted)
    diff_in = np.abs(inpx.astype(np.int16) - std.astype(np.int16)) * mask_bin
    max_err_in = float(diff_in.max())
    in_pixels = max(1, int(mask_bin.sum() * 3))
    frac_err_in = float((diff_in > tolerance).sum() / in_pixels)

    return max_err_out, frac_err_out, max_err_in, frac_err_in, True, (width, height)


def validate_and_generate_manifests(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config.get("dataset_root", os.environ.get("INP_X_ROOT", "./mock_data/INP_X_ROOT")))
    split = config.get("split", "test-data")
    manifest_out = Path(config.get("manifest_dir", "./manifests"))
    manifest_out.mkdir(parents=True, exist_ok=True)
    num_shards = config.get("num_shards", 10)
    seed = config.get("seed", 42)

    split_dir = dataset_root / split
    if not split_dir.exists():
        # Check if dataset_root itself is the split directory
        if (dataset_root / "data" / "standard_inpainting").exists():
            split_dir = dataset_root
        else:
            print(f"Error: Split directory does not exist: {split_dir}")
            print(f"Please ensure dataset_root '{dataset_root}' contains '{split}' directory.")
            return

    print(f"============================================================")
    print(f"INP-X Dataset Validator & Manifest Generator")
    print(f"Dataset Root : {dataset_root.resolve()}")
    print(f"Target Split : {split} ({split_dir.resolve()})")
    print(f"Manifest Dir : {manifest_out.resolve()}")
    print(f"============================================================\n")

    data_dir = split_dir / "data"
    std_dir = data_dir / "standard_inpainting"
    inpx_dir = data_dir / "inpainting_exchange"
    orig_dir = data_dir / "originals"
    masks_dir = split_dir / "masks"

    if not std_dir.exists():
        print(f"Error: Standard inpainting directory not found at: {std_dir}")
        return

    # 1. Discover all dataset subfolders
    dataset_names = [p.name for p in std_dir.iterdir() if p.is_dir()]
    print(f"Found source datasets: {dataset_names}")

    # 2. Index all original images per dataset for fast and accurate lookup
    originals_index: Dict[str, Dict[str, Path]] = {}
    for ds_name in dataset_names:
        ds_orig_dir = orig_dir / ds_name
        originals_index[ds_name] = {}
        if ds_orig_dir.exists():
            for f in ds_orig_dir.glob("*.jpg"):
                originals_index[ds_name][f.stem] = f
            for f in ds_orig_dir.glob("*.png"):
                originals_index[ds_name][f.stem] = f
        print(f"  Indexed {len(originals_index[ds_name])} original images for {ds_name}")

    # 3. Find all standard inpainting images
    std_files = sorted(list(std_dir.glob("*/*.jpg")) + list(std_dir.glob("*/*.png")))
    print(f"\nTotal standard inpainting candidate files: {len(std_files)}")

    if len(std_files) == 0:
        print("No image candidates found. Exiting.")
        return

    records = []
    missing_counts = {"exchange": 0, "mask": 0, "original": 0}
    dimension_mismatches = 0

    for std_path in tqdm(std_files, desc=f"Validating {split}"):
        dataset_name = std_path.parent.name
        stem = std_path.stem

        # Filename convention: <mask_stem>_<dataset>_<generator>
        # e.g., '10052_hair_CelebAHQ_OpenJourney' -> mask_stem='10052_hair', generator='OpenJourney'
        tag = f"_{dataset_name}_"
        if tag in stem:
            mask_stem, generator = stem.split(tag, 1)
        else:
            # Fallback if naming differs
            parts = stem.split("_")
            generator = parts[-1]
            mask_stem = "_".join(parts[:-1])

        sample_id = f"{split}_{dataset_name}_{generator}_{mask_stem}"

        # Expected paired paths
        # Exchange image has '_simple.jpg' suffix
        inpx_path = inpx_dir / dataset_name / f"{stem}_simple{std_path.suffix}"
        if not inpx_path.exists():
            # Try without _simple as fallback
            inpx_path = inpx_dir / dataset_name / std_path.name
        if not inpx_path.exists():
            missing_counts["exchange"] += 1
            continue

        # Mask path is in masks/<dataset>_masks/<mask_stem>.jpg
        mask_path = masks_dir / f"{dataset_name}_masks" / f"{mask_stem}{std_path.suffix}"
        if not mask_path.exists():
            # Try png/jpg alternate extension
            alt_ext = ".png" if std_path.suffix == ".jpg" else ".jpg"
            mask_path = masks_dir / f"{dataset_name}_masks" / f"{mask_stem}{alt_ext}"
        if not mask_path.exists():
            # Try direct masks/<dataset>/ directory
            mask_path = masks_dir / dataset_name / f"{mask_stem}{std_path.suffix}"
        if not mask_path.exists():
            missing_counts["mask"] += 1
            continue

        # Original image lookup
        orig_path = find_original_image(
            mask_stem=mask_stem,
            dataset_name=dataset_name,
            orig_dir=orig_dir / dataset_name,
            orig_files_map=originals_index.get(dataset_name, {}),
        )
        if orig_path is None or not orig_path.exists():
            missing_counts["original"] += 1
            continue

        # Exchange verification & Dimension checks
        max_err_out, frac_err_out, max_err_in, frac_err_in, is_valid, dims = verify_exchange_integrity(
            orig_path=orig_path,
            std_path=std_path,
            inpx_path=inpx_path,
            mask_path=mask_path,
        )

        if not is_valid:
            dimension_mismatches += 1
            continue

        records.append({
            "sample_id": sample_id,
            "split": split,
            "source_dataset": dataset_name,
            "source_generator": generator,
            "mask_stem": mask_stem,
            "original_path": str(orig_path.resolve()),
            "standard_inpainting_path": str(std_path.resolve()),
            "inpainting_exchange_path": str(inpx_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "width": dims[0],
            "height": dims[1],
            "max_err_out": max_err_out,
            "frac_err_out": frac_err_out,
            "max_err_in": max_err_in,
            "frac_err_in": frac_err_in,
        })

    print(f"\n============================================================")
    print(f"Validation Audit Results for '{split}'")
    print(f"============================================================")
    print(f"Successfully Validated Triplets : {len(records)}")
    print(f"Missing Inpainting Exchange     : {missing_counts['exchange']}")
    print(f"Missing Manipulation Masks      : {missing_counts['mask']}")
    print(f"Missing Original Images         : {missing_counts['original']}")
    print(f"Dimension/Shape Mismatches      : {dimension_mismatches}")
    print(f"============================================================\n")

    if len(records) == 0:
        print("No complete triplets validated.")
        return

    df = pd.DataFrame(records)

    # Sort deterministically
    df = df.sort_values("sample_id").reset_index(drop=True)

    print("Breakdown by Source Dataset and Generator:")
    print(df.groupby(["source_dataset", "source_generator"]).size().to_frame("count").to_string())
    print("\nDimensions observed:")
    print(df.groupby(["width", "height"]).size().to_frame("count").to_string())
    print("\nExchange Error Statistics:")
    print(df[["max_err_out", "frac_err_out", "max_err_in", "frac_err_in"]].describe().to_string())

    # 4. Save Full Manifest
    full_manifest_path = manifest_out / f"manifest_{split}_full.csv"
    df.to_csv(full_manifest_path, index=False)
    print(f"\nSaved full manifest -> {full_manifest_path}")

    # 5. Deterministic Sharding for Restartability
    shard_size = int(np.ceil(len(df) / num_shards))
    for i in range(num_shards):
        shard_df = df.iloc[i * shard_size : (i + 1) * shard_size]
        if len(shard_df) > 0:
            shard_path = manifest_out / f"manifest_{split}_shard_{i:02d}.csv"
            shard_df.to_csv(shard_path, index=False)
    print(f"Generated {num_shards} deterministic shards for restartability.")

    # 6. Generate Stratified Pilot Manifest (up to 10 samples per dataset x generator cell)
    pilot_df = (
        df.groupby(["source_dataset", "source_generator"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 10), random_state=seed))
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    pilot_manifest_path = manifest_out / f"manifest_{split}_pilot.csv"
    pilot_df.to_csv(pilot_manifest_path, index=False)
    print(f"Saved stratified pilot manifest ({len(pilot_df)} samples) -> {pilot_manifest_path}")

    # 7. Generate Smoke Test Manifest (20 samples, stratified across available groups)
    n_per_group = max(1, 20 // len(df.groupby(["source_dataset", "source_generator"])))
    smoke_df = (
        df.groupby(["source_dataset", "source_generator"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), n_per_group), random_state=seed))
        .head(20)
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    smoke_manifest_path = manifest_out / f"manifest_{split}_smoke.csv"
    smoke_df.to_csv(smoke_manifest_path, index=False)
    print(f"Saved smoke test manifest ({len(smoke_df)} samples) -> {smoke_manifest_path}")

    # 8. Record SHA256 hashes for all generated manifests
    hashes = {}
    for p in [full_manifest_path, pilot_manifest_path, smoke_manifest_path]:
        hashes[p.name] = get_file_sha256(p)
    for p in sorted(manifest_out.glob(f"manifest_{split}_shard_*.csv")):
        hashes[p.name] = get_file_sha256(p)

    hashes_file = manifest_out / f"manifest_{split}_hashes.json"
    with open(hashes_file, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"\nRecorded manifest SHA256 hashes -> {hashes_file}")
    for name, h in hashes.items():
        print(f"  {name:30s} : {h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INP-X Dataset Validator and Manifest Builder")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()
    validate_and_generate_manifests(args.config)
