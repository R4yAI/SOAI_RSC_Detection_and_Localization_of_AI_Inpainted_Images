import os
import glob
import pandas as pd
import numpy as np
import yaml
import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse



def get_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def verify_exchange(orig_path, std_path, inpx_path, mask_path):
    # Returns (max_err_out, frac_err_out, max_err_in, frac_err_in, same_shape)
    try:
        orig = np.array(Image.open(orig_path).convert('RGB'))
        std = np.array(Image.open(std_path).convert('RGB'))
        inpx = np.array(Image.open(inpx_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
    except Exception as e:
        return None, None, None, None, False
    
    if not (orig.shape == std.shape == inpx.shape):
        return None, None, None, None, False
    if mask.shape != orig.shape[:2]:
        return None, None, None, None, False
        
    mask_bin = (mask > 127).astype(np.uint8)[..., None]
    
    # Outside mask (mask == 0): INP-X should match original
    outside_mask = 1 - mask_bin
    diff_out = np.abs(inpx.astype(np.int16) - orig.astype(np.int16)) * outside_mask
    max_err_out = diff_out.max()
    frac_err_out = (diff_out > 2).sum() / max(1, outside_mask.sum() * 3) # Allow small JPEG tolerance
    
    # Inside mask (mask == 1): INP-X should match standard
    diff_in = np.abs(inpx.astype(np.int16) - std.astype(np.int16)) * mask_bin
    max_err_in = diff_in.max()
    frac_err_in = (diff_in > 2).sum() / max(1, mask_bin.sum() * 3)
    
    return max_err_out, frac_err_out, max_err_in, frac_err_in, True

def generate_manifests(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_root = Path(config.get('dataset_root', os.environ.get('INP_X_ROOT', './mock_data/INP_X_ROOT')))
    manifest_out = Path(config.get('manifest_dir', './manifests'))
    manifest_out.mkdir(parents=True, exist_ok=True)
    
    # Find all standard inpainting images
    std_images = list(dataset_root.glob('data/standard-inpainting/**/*.png'))
    std_images += list(dataset_root.glob('data/standard-inpainting/**/*.jpg'))
    
    records = []
    
    print(f"Found {len(std_images)} standard inpainting candidates.")
    
    for std_path in tqdm(std_images, desc="Validating Dataset"):
        # Infer dataset and generator from path
        # Assume path is: data/standard-inpainting/<dataset>/<generator>/<image>
        # or data/standard-inpainting/<dataset>/<image>
        rel_path = std_path.relative_to(dataset_root / 'data' / 'standard-inpainting')
        parts = rel_path.parts
        dataset_name = parts[0]
        
        if len(parts) >= 3:
            generator = parts[1]
            img_name = parts[-1]
            subpath = Path(*parts[1:])
        else:
            # Fallback if generator isn't a directory
            generator = "unknown"
            img_name = parts[-1]
            subpath = Path(img_name)
            
        sample_id = f"{dataset_name}_{generator}_{std_path.stem}"
        
        # Build paths for corresponding files
        orig_path = dataset_root / 'data' / 'originals' / dataset_name / img_name
        if not orig_path.exists():
            orig_path = dataset_root / 'data' / 'originals' / dataset_name / generator / img_name
            
        inpx_path = dataset_root / 'data' / 'inpainting-exchange' / dataset_name / subpath
        mask_path = dataset_root / 'masks' / dataset_name / img_name
        if not mask_path.exists():
            mask_path = dataset_root / 'masks' / dataset_name / generator / img_name
            
        # Check existence
        exists = {
            'orig': orig_path.exists(),
            'std': std_path.exists(),
            'inpx': inpx_path.exists(),
            'mask': mask_path.exists()
        }
        
        if not all(exists.values()):
            print(f"Missing files for {sample_id}: {exists}")
            continue
            
        # Verify Exchange & Dimensions
        max_err_out, frac_err_out, max_err_in, frac_err_in, shape_ok = verify_exchange(orig_path, std_path, inpx_path, mask_path)
        
        if not shape_ok:
            print(f"Shape mismatch for {sample_id}")
            continue
            
        with Image.open(orig_path) as img:
            width, height = img.size
            
        records.append({
            'sample_id': sample_id,
            'source_dataset': dataset_name,
            'source_generator': generator,
            'original_path': str(orig_path.resolve()),
            'standard_inpainting_path': str(std_path.resolve()),
            'inpainting_exchange_path': str(inpx_path.resolve()),
            'mask_path': str(mask_path.resolve()),
            'width': width,
            'height': height,
            'max_err_out': max_err_out,
            'frac_err_out': frac_err_out,
            'max_err_in': max_err_in,
            'frac_err_in': frac_err_in
        })
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No valid triplets found.")
        return
        
    # Sort deterministically
    df = df.sort_values('sample_id').reset_index(drop=True)
    
    print("\n--- Dataset Audit ---")
    print(f"Total Valid Triplets: {len(df)}")
    print("By Dataset & Generator:")
    print(df.groupby(['source_dataset', 'source_generator']).size().to_string())
    
    # Save Full Manifest
    full_path = manifest_out / 'manifest_full.csv'
    df.to_csv(full_path, index=False)
    
    # Sharding logic: 10 chunks for restartability
    num_shards = 10
    shard_size = len(df) // num_shards + 1
    for i in range(num_shards):
        shard = df.iloc[i*shard_size : (i+1)*shard_size]
        if len(shard) > 0:
            shard.to_csv(manifest_out / f'manifest_full_shard_{i:02d}.csv', index=False)
    
    # Create Stratified Pilot (120 samples: 4 datasets * 3 generators * 10 samples)
    try:
        pilot_df = df.groupby(['source_dataset', 'source_generator'], group_keys=False).apply(lambda x: x.sample(min(len(x), 10), random_state=42))
        pilot_df = pilot_df.sort_values('sample_id').reset_index(drop=True)
        pilot_path = manifest_out / 'manifest_pilot.csv'
        pilot_df.to_csv(pilot_path, index=False)
    except Exception as e:
        print("Could not create stratified pilot:", e)
        pilot_df = df.head(120)
        pilot_path = manifest_out / 'manifest_pilot.csv'
        pilot_df.to_csv(pilot_path, index=False)
        
    # Create Smoke Test (20 samples)
    smoke_df = df.head(20)
    smoke_path = manifest_out / 'manifest_smoke.csv'
    smoke_df.to_csv(smoke_path, index=False)
    
    print("\n--- Manifest Hashes ---")
    for p in [full_path, pilot_path, smoke_path]:
        print(f"{p.name}: {get_hash(p)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    generate_manifests(args.config)
