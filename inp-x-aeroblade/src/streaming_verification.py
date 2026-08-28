import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.v2 as tf
from tqdm import tqdm

# Add aeroblade to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "aeroblade" / "src"))

from aeroblade.distances import _PatchedLPIPS
from diffusers import AutoPipelineForImage2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from torchvision.transforms.v2.functional import to_pil_image
from scipy.stats import pearsonr

def load_ae(repo_id="runwayml/stable-diffusion-v1-5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForImage2Image.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    ae = pipe.vae.to(device)
    ae.eval()
    return ae, device

def setup_lpips(net="vgg", device="cuda"):
    model = _PatchedLPIPS(spatial=True, net=net).to(device)
    model.eval()
    return model

def compute_upstream_path(img_orig_tensor, ae, lpips_model, generator, device, tmpdir, name):
    # img_orig_tensor is [1, 3, H, W] in [0, 1] range
    
    # Encode -> Decode
    with torch.no_grad():
        img_input = img_orig_tensor.to(device, dtype=ae.dtype) * 2.0 - 1.0
        latents = retrieve_latents(ae.encode(img_input), generator=generator)
        dec = ae.decode(latents.to(ae.dtype), return_dict=False)[0]
        rec = (dec / 2 + 0.5).clamp(0, 1)
    
    # Upstream saves to PNG
    rec_path = Path(tmpdir) / f"{name}_rec.png"
    to_pil_image(rec[0]).save(rec_path)
    
    # Upstream loads via ImageFolder transform
    transform = tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])
    
    rec_loaded = transform(Image.open(rec_path).convert("RGB")).unsqueeze(0).to(device)
    orig_loaded = img_orig_tensor.to(device) # Assume original is loaded from disk similarly
    
    with torch.no_grad():
        sum_batch, layers_batch = lpips_model(orig_loaded, rec_loaded, retPerLayer=True, normalize=True)
        
    # We want layer 2 (index 2 in layers_batch)
    # LPIPS returns a list: [layer0, layer1, layer2, ...]
    layer_idx = 2
    out_tensor = layers_batch[layer_idx]
    
    # Postprocess without negating (positive score)
    # Mean over spatial pooling if not spatial, but we have spatial=True
    # Upsample to original size (upstream default if output_size is set)
    _, _, H, W = orig_loaded.shape
    out_upsampled = torch.nn.functional.interpolate(
        out_tensor.to(dtype=torch.float32),
        size=(H, W),
        mode="bilinear",
        antialias=True
    ).to(dtype=torch.float16)
    
    return out_upsampled[0, 0].cpu().numpy()

def compute_streaming_path(img_orig_tensor, ae, lpips_model, generator, device):
    with torch.no_grad():
        img_input = img_orig_tensor.to(device, dtype=ae.dtype) * 2.0 - 1.0
        latents = retrieve_latents(ae.encode(img_input), generator=generator)
        dec = ae.decode(latents.to(ae.dtype), return_dict=False)[0]
        rec_streaming = (dec / 2 + 0.5).clamp(0, 1).to(torch.float32) # Stay in FP32
        
        orig_loaded = img_orig_tensor.to(device)
        sum_batch, layers_batch = lpips_model(orig_loaded, rec_streaming, retPerLayer=True, normalize=True)
        
    layer_idx = 2
    out_tensor = layers_batch[layer_idx]
    
    _, _, H, W = orig_loaded.shape
    out_upsampled = torch.nn.functional.interpolate(
        out_tensor.to(dtype=torch.float32),
        size=(H, W),
        mode="bilinear",
        antialias=True
    ).to(dtype=torch.float16)
    
    return out_upsampled[0, 0].cpu().numpy()

def main():
    manifest_path = Path("manifests/manifest_smoke.csv")
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    
    print("Loading AE and LPIPS models...")
    ae, device = load_ae()
    lpips_model = setup_lpips(device=device)
    
    transform = tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])
    
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying Streaming vs Upstream"):
            img_path = row['standard_inpainting_path']
            
            # Load original image
            img_pil = Image.open(img_path).convert("RGB")
            
            # Check dimensions (multiple of 8)
            # AEROBLADE native doesn't pad. We'll pass it as is to see if it crashes.
            img_orig_tensor = transform(img_pil).unsqueeze(0)
            
            generator_upstream = torch.Generator(device=device).manual_seed(42)
            upstream_map = compute_upstream_path(img_orig_tensor, ae, lpips_model, generator_upstream, device, tmpdir, row['sample_id'])
            
            generator_streaming = torch.Generator(device=device).manual_seed(42)
            streaming_map = compute_streaming_path(img_orig_tensor, ae, lpips_model, generator_streaming, device)
            
            # Compare numerically
            mae = np.abs(upstream_map.astype(np.float32) - streaming_map.astype(np.float32)).mean()
            max_diff = np.abs(upstream_map.astype(np.float32) - streaming_map.astype(np.float32)).max()
            
            corr, _ = pearsonr(upstream_map.flatten(), streaming_map.flatten())
            
            results.append({
                'sample_id': row['sample_id'],
                'mae': mae,
                'max_diff': max_diff,
                'correlation': corr
            })
            
    res_df = pd.DataFrame(results)
    print("\n--- Streaming Verification Results ---")
    print(res_df.describe())
    
    res_df.to_csv("manifests/streaming_verification_results.csv", index=False)
    print("\nDetailed results saved to manifests/streaming_verification_results.csv")

if __name__ == "__main__":
    main()
