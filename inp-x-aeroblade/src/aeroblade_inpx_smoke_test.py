"""
AEROBLADE × INP-X smoke-test experiment.

Runs the AEROBLADE reconstruction-error pipeline on a frozen INP-X smoke
manifest for three matched conditions:

    original image
    standard inpainting
    INP-X inpainting exchange

The script deliberately keeps the experiment self-contained in one file.
It reuses AEROBLADE's `_PatchedLPIPS` and Diffusers latent-retrieval logic,
while streaming the decoded reconstruction directly into LPIPS instead of
writing/reloading PNGs.

Outputs are written under --output-dir and include:
    config.yaml
    heatmaps/*.npy
    reconstructions/*.png        (optional)
    visualizations/*.png
    sample_metrics.csv
    detection_metrics.csv
    localization_metrics.csv
    evidence_metrics.csv
    run_summary.csv

The smoke test is intended as a controlled, diagnostic experiment rather
than a final benchmark. Threshold-dependent metrics are explicitly marked
with their thresholding rule; AP/AUC metrics are threshold-free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tf
from PIL import Image
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
import yaml

# ---------------------------------------------------------------------------
# AEROBLADE imports
# ---------------------------------------------------------------------------
# The repository is expected at:
#     <project_root>/external/aeroblade/src
# Adjust --aeroblade-src if your checkout uses a different layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AEROBLADE_SRC = PROJECT_ROOT / "external" / "aeroblade" / "src"

if str(DEFAULT_AEROBLADE_SRC) not in sys.path:
    sys.path.insert(0, str(DEFAULT_AEROBLADE_SRC))

from aeroblade.distances import _PatchedLPIPS  # noqa: E402
from diffusers import AutoPipelineForImage2Image  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (  # noqa: E402
    retrieve_latents,
)


CONDITIONS = {
    "original": "original_path",
    "standard": "standard_inpainting_path",
    "inpx": "inpainting_exchange_path",
}


def parse_args() -> argparse.Namespace:
    """Parse experiment settings while keeping all important parameters explicit."""
    parser = argparse.ArgumentParser(
        description="Run the AEROBLADE × INP-X controlled smoke test."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/manifest_test-data_smoke.csv"),
        help="Frozen INP-X manifest containing the smoke-test triplets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aeroblade/smoke_test"),
        help="Root directory for all reproducible experiment outputs.",
    )
    parser.add_argument(
        "--ae-repo-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face Diffusers model whose VAE is used as the forensic AE.",
    )
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=["vgg", "alex", "squeeze"],
        help="LPIPS backbone used by AEROBLADE.",
    )
    parser.add_argument(
        "--lpips-layer",
        type=int,
        default=2,
        help="Zero-based spatial LPIPS layer index returned by AEROBLADE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for VAE latent sampling for every condition.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save decoded AE reconstructions as PNG files for visual inspection.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging. By default, process the whole manifest.",
    )
    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Save diagnostic image/mask/heatmap panels.",
    )
    parser.add_argument(
        "--heatmap-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for saved heatmaps.",
    )
    parser.add_argument(
        "--aeroblade-src",
        type=Path,
        default=DEFAULT_AEROBLADE_SRC,
        help="Path to AEROBLADE's src directory.",
    )
    return parser.parse_args()


def setup_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create the complete output tree so the experiment never relies on pre-existing folders."""
    paths = {
        "root": output_dir,
        "heatmaps": output_dir / "heatmaps",
        "reconstructions": output_dir / "reconstructions",
        "visualizations": output_dir / "visualizations",
        "metrics": output_dir / "metrics",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_config(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    """Persist the exact experiment configuration alongside the numerical results."""
    config = {
        "experiment": "aeroblade_x_inpx_smoke_test",
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(paths["root"].resolve()),
        "forensic_autoencoder": args.ae_repo_id,
        "lpips_distance": "LPIPS",
        "lpips_backbone": args.lpips_net,
        "lpips_spatial_layer_index": args.lpips_layer,
        "latent_sampling_seed": args.seed,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "save_reconstructions": args.save_reconstructions,
        "heatmap_dtype": args.heatmap_dtype,
        "heatmap_orientation": "positive_reconstruction_error",
        "upsampling": "bilinear_antialias",
        "input_range": "[0,1] -> [-1,1] for VAE",
        "streaming_reconstruction": True,
        "thresholded_localization_rule": "per-image Otsu",
        "pixel_metrics": [
            "pixel_f1",
            "dice",
            "iou",
            "pixel_ap",
            "pixel_roc_auc",
            "boundary_f1",
        ],
        "image_metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ],
    }
    with open(paths["root"] / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def load_ae(repo_id: str, device: str):
    """Load the forensic autoencoder using the same Diffusers construction as the validation script."""
    pipe = AutoPipelineForImage2Image.from_pretrained(
        repo_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )
    ae = pipe.vae.to(device)
    ae.eval()
    return ae


def setup_lpips(net: str, device: str):
    """Instantiate AEROBLADE's spatial LPIPS implementation without modifying its model weights."""
    model = _PatchedLPIPS(spatial=True, net=net).to(device)
    model.eval()
    return model


def load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load one RGB image using AEROBLADE's [0,1] preprocessing convention."""
    transform = tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])
    with Image.open(path) as image:
        image = image.convert("RGB")
        return transform(image).unsqueeze(0)


def load_mask(path: Path, expected_hw: Tuple[int, int]) -> np.ndarray:
    """Load and binarize an INP-X mask, verifying that it matches the image geometry."""
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))

    if mask.shape != expected_hw:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {expected_hw}: {path}"
        )

    return (mask > 127).astype(np.uint8)


def reconstruct_and_score(
    image_tensor: torch.Tensor,
    ae,
    lpips_model,
    device: str,
    seed: int,
    layer_idx: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Reconstruct one image through the AE and return its positive spatial LPIPS error.

    The implementation follows AEROBLADE's encode -> sampled latent -> decode
    sequence and `[0,1] <-> [-1,1]` conversion. Unlike the upstream disk path,
    the decoded tensor is sent directly to spatial LPIPS, avoiding PNG
    quantization. The returned heatmap is deliberately NOT negated: larger
    values therefore mean larger reconstruction error / stronger anomaly
    evidence.
    """
    image_tensor = image_tensor.to(device)
    ae_input = image_tensor.to(dtype=ae.dtype) * 2.0 - 1.0

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        latents = retrieve_latents(ae.encode(ae_input), generator=generator)
        decoded = ae.decode(latents.to(ae.dtype), return_dict=False)[0]
        reconstruction = (decoded / 2.0 + 0.5).clamp(0, 1).float()

        _, layer_outputs = lpips_model(
            image_tensor,
            reconstruction,
            retPerLayer=True,
            normalize=True,
        )

        spatial_error = layer_outputs[layer_idx]

        _, _, height, width = image_tensor.shape
        spatial_error = torch.nn.functional.interpolate(
            spatial_error.float(),
            size=(height, width),
            mode="bilinear",
            antialias=True,
        )

    heatmap = spatial_error[0, 0].detach().cpu().numpy().astype(np.float32)
    return heatmap, reconstruction[0].detach().cpu()


def save_heatmap(
    heatmap: np.ndarray,
    paths: Dict[str, Path],
    sample_id: str,
    condition: str,
    ae_name: str,
    lpips_net: str,
    lpips_layer: int,
    dtype: str,
) -> Path:
    """Save a heatmap with all parameters needed to identify the experiment later."""
    safe_ae = ae_name.replace("/", "__")
    filename = (
        f"sample-{sample_id}"
        f"__condition-{condition}"
        f"__ae-{safe_ae}"
        f"__distance-LPIPS"
        f"__backbone-{lpips_net}"
        f"__layer-{lpips_layer}"
        f"__orientation-positive_error.npy"
    )
    output = paths["heatmaps"] / filename
    np.save(output, heatmap.astype(np.float16 if dtype == "float16" else np.float32))
    return output


def save_reconstruction(
    reconstruction: torch.Tensor,
    paths: Dict[str, Path],
    sample_id: str,
    condition: str,
    ae_name: str,
) -> Path:
    """Save an optional reconstruction using a descriptive, collision-resistant filename."""
    safe_ae = ae_name.replace("/", "__")
    output = (
        paths["reconstructions"]
        / f"sample-{sample_id}__condition-{condition}__ae-{safe_ae}__reconstruction.png"
    )
    image = tf.ToPILImage()(reconstruction.float().clamp(0, 1))
    image.save(output)
    return output


def boundary_map(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Extract a one-pixel morphological boundary around the positive mask region."""
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(mask.astype(bool), structure=structure, iterations=radius)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=structure, iterations=radius)
    return np.logical_xor(dilated, eroded)


def classify_heatmap_otsu(heatmap: np.ndarray) -> Tuple[np.ndarray, float]:
    """Threshold a heatmap with Otsu's between-class-variance criterion."""
    values = heatmap.astype(np.float64).ravel()
    lo, hi = float(values.min()), float(values.max())

    if hi <= lo:
        return np.zeros_like(heatmap, dtype=np.uint8), lo

    hist, edges = np.histogram(values, bins=256, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight = hist.astype(np.float64)
    cumulative = np.cumsum(weight)
    cumulative_mean = np.cumsum(weight * centers)
    total = cumulative[-1]
    total_mean = cumulative_mean[-1]

    denominator = cumulative * (total - cumulative)
    numerator = (total_mean * cumulative - cumulative_mean) ** 2
    score = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    threshold = float(centers[int(np.argmax(score))])
    prediction = (heatmap >= threshold).astype(np.uint8)
    return prediction, threshold


def safe_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute precision, recall, F1 and IoU while handling degenerate masks safely."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union if union else 1.0 if not y_true.any() else 0.0

    return float(precision), float(recall), float(f1), float(iou)


def compute_localization_metrics(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute threshold-free and Otsu-thresholded pixel localization metrics for one map."""
    y_true = mask.astype(np.uint8).ravel()
    scores = heatmap.astype(np.float64).ravel()

    if np.unique(y_true).size == 2:
        pixel_auc = float(roc_auc_score(y_true, scores))
        pixel_ap = float(average_precision_score(y_true, scores))
    else:
        pixel_auc = float("nan")
        pixel_ap = float("nan")

    prediction, threshold = classify_heatmap_otsu(heatmap)
    precision, recall, f1, iou = safe_binary_metrics(
        y_true,
        prediction.ravel(),
    )

    gt_boundary = boundary_map(mask)
    pred_boundary = boundary_map(prediction)
    boundary_precision, boundary_recall, boundary_f1, _ = safe_binary_metrics(
        gt_boundary.ravel().astype(np.uint8),
        pred_boundary.ravel().astype(np.uint8),
    )

    return {
        "pixel_ap": pixel_ap,
        "pixel_roc_auc": pixel_auc,
        "pixel_precision_otsu": precision,
        "pixel_recall_otsu": recall,
        "pixel_f1_otsu": f1,
        "dice_otsu": f1,
        "iou_otsu": iou,
        "boundary_f1_otsu": boundary_f1,
        "otsu_threshold": threshold,
    }


def compute_evidence_metrics(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Measure where reconstruction-error energy is concentrated relative to the ground-truth mask."""
    mask_bool = mask.astype(bool)
    boundary = boundary_map(mask)
    near_context = ndimage.binary_dilation(mask_bool, iterations=10) & ~mask_bool
    global_background = ~mask_bool

    total = float(np.sum(heatmap)) + 1e-12
    inside = float(np.sum(heatmap[mask_bool])) if mask_bool.any() else 0.0
    boundary_energy = float(np.sum(heatmap[boundary])) if boundary.any() else 0.0
    context_energy = float(np.sum(heatmap[near_context])) if near_context.any() else 0.0
    background_energy = float(np.sum(heatmap[global_background])) if global_background.any() else 0.0

    return {
        "mean_error_inside": float(np.mean(heatmap[mask_bool])) if mask_bool.any() else np.nan,
        "mean_error_boundary": float(np.mean(heatmap[boundary])) if boundary.any() else np.nan,
        "mean_error_near_context": float(np.mean(heatmap[near_context])) if near_context.any() else np.nan,
        "mean_error_background": float(np.mean(heatmap[global_background])) if global_background.any() else np.nan,
        "energy_fraction_inside": inside / total,
        "energy_fraction_boundary": boundary_energy / total,
        "energy_fraction_near_context": context_energy / total,
        "energy_fraction_background": background_energy / total,
        "inside_background_contrast": (
            float(np.mean(heatmap[mask_bool]) / (np.mean(heatmap[global_background]) + 1e-12))
            if mask_bool.any() and global_background.any()
            else np.nan
        ),
    }


def save_visualization(
    image_paths: Dict[str, Path],
    mask: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    output: Path,
) -> None:
    """Save a compact diagnostic panel for qualitative alignment and signal inspection."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for ax, condition in zip(axes[:3], ["original", "standard", "inpx"]):
        with Image.open(image_paths[condition]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(condition)
        ax.axis("off")

    axes[3].imshow(mask, cmap="gray")
    axes[3].set_title("ground-truth mask")
    axes[3].axis("off")

    for ax, condition in zip(axes[4:], ["original", "standard", "inpx"]):
        im = ax.imshow(heatmaps[condition])
        # ax.imshow(mask, alpha=0.25)  //Uncomment this if you want to see the mask plotted above the heatmap
        ax.set_title(f"{condition}: positive LPIPS error")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[7].axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def image_level_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
) -> Dict[str, float]:
    """Compute image-level detection metrics from reconstruction-error scores."""
    scores = np.asarray(list(scores), dtype=np.float64)
    labels = np.asarray(list(labels), dtype=np.uint8)

    if np.unique(labels).size < 2:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "threshold": np.nan,
        }

    # For the smoke-test diagnostic report, use the midpoint between the
    # class means. This is descriptive, not a trained detector.
    negative_mean = scores[labels == 0].mean()
    positive_mean = scores[labels == 1].mean()
    threshold = float((negative_mean + positive_mean) / 2.0)
    predictions = (scores >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(
            2 * precision_score(labels, predictions, zero_division=0)
            * recall_score(labels, predictions, zero_division=0)
            / max(
                precision_score(labels, predictions, zero_division=0)
                + recall_score(labels, predictions, zero_division=0),
                1e-12,
            )
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    """Run the complete smoke experiment and persist maps, reconstructions, metrics, and metadata."""
    args = parse_args()

    if not args.aeroblade_src.exists():
        raise FileNotFoundError(f"AEROBLADE source directory not found: {args.aeroblade_src}")

    # Put the requested repository source first in the import path.
    sys.path.insert(0, str(args.aeroblade_src))

    paths = setup_output_dirs(args.output_dir)
    save_config(args, paths)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    required_columns = {
        "sample_id",
        "source_dataset",
        "source_generator",
        "original_path",
        "standard_inpainting_path",
        "inpainting_exchange_path",
        "mask_path",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Manifest: {args.manifest}")
    print(f"Samples: {len(df)}")
    print(f"AE: {args.ae_repo_id}")
    print(f"LPIPS: {args.lpips_net}, spatial layer {args.lpips_layer}")
    print(f"Seed: {args.seed}")

    ae = load_ae(args.ae_repo_id, device)
    lpips_model = setup_lpips(args.lpips_net, device)

    sample_records = []
    localization_records = []
    evidence_records = []
    detection_scores = []
    detection_labels = []

    started = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="AEROBLADE × INP-X smoke test"):
        sample_id = str(row["sample_id"])

        image_paths = {
            condition: Path(row[path_column])
            for condition, path_column in CONDITIONS.items()
        }
        mask_path = Path(row["mask_path"])

        for path in [*image_paths.values(), mask_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing input for {sample_id}: {path}")

        # All three conditions must preserve the same geometry.
        with Image.open(image_paths["original"]) as image:
            expected_hw = (image.height, image.width)

        mask = load_mask(mask_path, expected_hw)
        heatmaps = {}

        for condition, image_path in image_paths.items():
            image_tensor = load_rgb_tensor(image_path)

            if tuple(image_tensor.shape[-2:]) != expected_hw:
                raise ValueError(
                    f"Geometry mismatch for {sample_id}/{condition}: "
                    f"{tuple(image_tensor.shape[-2:])} vs {expected_hw}"
                )

            heatmap, reconstruction = reconstruct_and_score(
                image_tensor=image_tensor,
                ae=ae,
                lpips_model=lpips_model,
                device=device,
                seed=args.seed,
                layer_idx=args.lpips_layer,
            )
            heatmaps[condition] = heatmap

            save_heatmap(
                heatmap=heatmap,
                paths=paths,
                sample_id=sample_id,
                condition=condition,
                ae_name=args.ae_repo_id,
                lpips_net=args.lpips_net,
                lpips_layer=args.lpips_layer,
                dtype=args.heatmap_dtype,
            )

            if args.save_reconstructions:
                save_reconstruction(
                    reconstruction=reconstruction,
                    paths=paths,
                    sample_id=sample_id,
                    condition=condition,
                    ae_name=args.ae_repo_id,
                )

        # The same scalar score used for image-level detection is the mean
        # positive reconstruction error over the complete spatial field.
        for condition, heatmap in heatmaps.items():
            score = float(np.mean(heatmap))
            label = 0 if condition == "original" else 1
            detection_scores.append(score)
            detection_labels.append(label)

            loc = compute_localization_metrics(heatmap, mask)
            evidence = compute_evidence_metrics(heatmap, mask)

            common = {
                "sample_id": sample_id,
                "source_dataset": row["source_dataset"],
                "source_generator": row["source_generator"],
                "condition": condition,
                "width": expected_hw[1],
                "height": expected_hw[0],
                "image_score_mean_error": score,
            }

            localization_records.append({**common, **loc})
            evidence_records.append({**common, **evidence})

            sample_records.append(
                {
                    **common,
                    "heatmap_min": float(np.min(heatmap)),
                    "heatmap_mean": float(np.mean(heatmap)),
                    "heatmap_std": float(np.std(heatmap)),
                    "heatmap_max": float(np.max(heatmap)),
                }
            )

        if args.visualizations:
            visualization_name = (
                f"sample-{sample_id}"
                f"__ae-{args.ae_repo_id.replace('/', '__')}"
                f"__distance-LPIPS"
                f"__backbone-{args.lpips_net}"
                f"__layer-{args.lpips_layer}.png"
            )
            save_visualization(
                image_paths=image_paths,
                mask=mask,
                heatmaps=heatmaps,
                output=paths["visualizations"] / visualization_name,
            )

    # -----------------------------------------------------------------------
    # Persist per-sample / per-condition tables.
    # -----------------------------------------------------------------------
    sample_df = pd.DataFrame(sample_records)
    loc_df = pd.DataFrame(localization_records)
    evidence_df = pd.DataFrame(evidence_records)

    sample_df.to_csv(paths["metrics"] / "sample_metrics.csv", index=False)
    loc_df.to_csv(paths["metrics"] / "localization_metrics.csv", index=False)
    evidence_df.to_csv(paths["metrics"] / "evidence_metrics.csv", index=False)

    # Detection is deliberately computed across all original vs manipulated
    # condition observations. The condition-specific rows make the comparison
    # directly inspectable, while the aggregate row answers "does the score
    # separate authentic from manipulated images?".
    detection = image_level_metrics(detection_scores, detection_labels)
    detection_row = {
        "experiment": "aeroblade_x_inpx_smoke_test",
        "ae": args.ae_repo_id,
        "lpips_backbone": args.lpips_net,
        "lpips_layer": args.lpips_layer,
        "score": "mean_positive_spatial_reconstruction_error",
        "positive_class": "standard_or_inpx",
        "threshold_rule": "midpoint_between_negative_and_positive_class_means",
        "n_images": len(detection_scores),
        **detection,
    }

    # Also report standard-vs-original and INP-X-vs-original separately.
    for manipulated_condition in ["standard", "inpx"]:
        subset = sample_df[sample_df["condition"].isin(["original", manipulated_condition])]
        scores = subset["image_score_mean_error"].to_numpy()
        labels = (subset["condition"] == manipulated_condition).astype(np.uint8).to_numpy()
        row_metrics = image_level_metrics(scores, labels)
        detection_rows = {
            "experiment": "aeroblade_x_inpx_smoke_test",
            "ae": args.ae_repo_id,
            "lpips_backbone": args.lpips_net,
            "lpips_layer": args.lpips_layer,
            "score": "mean_positive_spatial_reconstruction_error",
            "positive_class": manipulated_condition,
            "threshold_rule": "midpoint_between_negative_and_positive_class_means",
            "n_images": len(scores),
            **row_metrics,
        }
        detection_rows["comparison"] = f"{manipulated_condition}_vs_original"
        if "comparison" not in detection_row:
            detection_row["comparison"] = "standard_and_inpx_vs_original"

        if "detection_rows_list" not in locals():
            detection_rows_list = []
        detection_rows_list.append(detection_rows)

    detection_df = pd.DataFrame(detection_rows_list)
    detection_df.to_csv(paths["metrics"] / "detection_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # Standard -> INP-X degradation and retention.
    # Retention is defined here as:
    #     INP-X performance / Standard performance
    # This is most meaningful for bounded higher-is-better metrics such as AP,
    # AUC, Dice and IoU, and should be interpreted cautiously on a tiny smoke
    # test. We therefore calculate it mechanically and preserve both source
    # values in the report.
    # -----------------------------------------------------------------------
    loc_summary = (
        loc_df.groupby("condition")[
            ["pixel_ap", "pixel_roc_auc", "dice_otsu", "iou_otsu", "boundary_f1_otsu"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )

    std_row = loc_summary[loc_summary["condition"] == "standard"]
    inpx_row = loc_summary[loc_summary["condition"] == "inpx"]

    summary_rows = []
    if not std_row.empty and not inpx_row.empty:
        for metric in ["pixel_ap", "pixel_roc_auc", "dice_otsu", "iou_otsu", "boundary_f1_otsu"]:
            standard_value = float(std_row.iloc[0][metric])
            inpx_value = float(inpx_row.iloc[0][metric])
            summary_rows.append(
                {
                    "metric": metric,
                    "standard_mean": standard_value,
                    "inpx_mean": inpx_value,
                    "degradation_absolute": inpx_value - standard_value,
                    "retention_ratio": (
                        inpx_value / standard_value
                        if np.isfinite(standard_value) and standard_value != 0
                        else np.nan
                    ),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        paths["metrics"] / "retention_degradation_metrics.csv",
        index=False,
    )

    elapsed = time.time() - started
    run_summary = pd.DataFrame(
        [
            {
                "experiment": "aeroblade_x_inpx_smoke_test",
                "manifest": str(args.manifest.resolve()),
                "n_samples": len(df),
                "n_condition_images": len(sample_df),
                "ae": args.ae_repo_id,
                "lpips_backbone": args.lpips_net,
                "lpips_layer": args.lpips_layer,
                "seed": args.seed,
                "device": device,
                "elapsed_seconds": elapsed,
                "heatmap_count": len(list(paths["heatmaps"].glob("*.npy"))),
                "reconstruction_count": len(list(paths["reconstructions"].glob("*.png"))),
            }
        ]
    )
    run_summary.to_csv(paths["metrics"] / "run_summary.csv", index=False)

    with open(paths["logs"] / "completed.json", "w") as f:
        json.dump(
            {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "manifest": str(args.manifest.resolve()),
            },
            f,
            indent=2,
        )

    print("\n============================================================")
    print("AEROBLADE × INP-X SMOKE TEST COMPLETE")
    print("============================================================")
    print(f"Samples processed : {len(df)}")
    print(f"Heatmaps          : {paths['heatmaps']}")
    print(f"Metrics           : {paths['metrics']}")
    print(f"Visualizations    : {paths['visualizations']}")
    print(f"Elapsed           : {elapsed / 60:.2f} min")
    print("============================================================")


if __name__ == "__main__":
    main()