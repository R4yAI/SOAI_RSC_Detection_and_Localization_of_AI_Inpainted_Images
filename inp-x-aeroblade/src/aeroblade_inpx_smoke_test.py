"""
AEROBLADE x INP-X High-Throughput Multi-Triplet Batched Experiment.

Optimized & Mathematically Calibrated Pipeline:
1. Multi-triplet GPU batching: Evaluates B triplets simultaneously (B * 3 images per forward pass).
2. Channels-last memory layout & FP16 autocast on Tensor Cores.
3. Multi-worker prefetched PyTorch DataLoader for zero GPU starvation.
4. Option A (AEROBLADE alignment): Negated reconstruction error (-E) as anomaly score.
5. Inverted/calibrated thresholding for spatial localization (Otsu on -E, plus IoU_max/Dice_max).
6. Removed internal Matplotlib plotting loop; plotting is decoupled to smoke_test_visualization.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AEROBLADE_SRC = PROJECT_ROOT / "external" / "aeroblade" / "src"

if str(DEFAULT_AEROBLADE_SRC) not in sys.path:
    sys.path.insert(0, str(DEFAULT_AEROBLADE_SRC))

from aeroblade.distances import _PatchedLPIPS  # noqa: E402
from diffusers import AutoPipelineForImage2Image  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (  # noqa: E402
    retrieve_latents,
)

CONDITIONS = ["original", "standard", "inpx"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optimized AEROBLADE x INP-X multi-triplet batched experiment."
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
        "--triplet-batch-size",
        type=int,
        default=4,
        help="Number of triplets to batch simultaneously (e.g. 4 triplets = 12 images per VAE pass).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes for background image prefetching.",
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


# ---------------------------------------------------------------------------
# Dataset & DataLoader for zero GPU starvation
# ---------------------------------------------------------------------------
class InpXTripletDataset(data.Dataset):
    """Prefetches matched INP-X triplets with background multi-threading."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.transform = tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"])

        orig_path = Path(row["original_path"])
        std_path = Path(row["standard_inpainting_path"])
        inpx_path = Path(row["inpainting_exchange_path"])
        mask_path = Path(row["mask_path"])

        for p in [orig_path, std_path, inpx_path, mask_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing file for {sample_id}: {p}")

        with Image.open(orig_path) as im:
            orig_img = im.convert("RGB")
            hw = (orig_img.height, orig_img.width)

        with Image.open(std_path) as im:
            std_img = im.convert("RGB")
        with Image.open(inpx_path) as im:
            inpx_img = im.convert("RGB")
        with Image.open(mask_path) as im:
            mask_img = im.convert("L")

        if (std_img.height, std_img.width) != hw or (inpx_img.height, inpx_img.width) != hw:
            raise ValueError(f"Shape mismatch in triplet {sample_id}")
        if (mask_img.height, mask_img.width) != hw:
            raise ValueError(f"Shape mismatch in mask {sample_id}")

        orig_t = self.transform(orig_img)
        std_t = self.transform(std_img)
        inpx_t = self.transform(inpx_img)
        mask_arr = (np.asarray(mask_img) > 127).astype(np.uint8)

        return {
            "sample_id": sample_id,
            "source_dataset": str(row["source_dataset"]),
            "source_generator": str(row["source_generator"]),
            # Stacked triplet: [3, 3, H, W] in [0, 1]
            "tensors": torch.stack([orig_t, std_t, inpx_t], dim=0),
            "mask": torch.from_numpy(mask_arr),
            "h": hw[0],
            "w": hw[1],
        }


def setup_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "heatmaps": output_dir / "heatmaps",
        "reconstructions": output_dir / "reconstructions",
        "metrics": output_dir / "metrics",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_config(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
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
        "triplet_batch_size": args.triplet_batch_size,
        "save_reconstructions": args.save_reconstructions,
        "heatmap_dtype": args.heatmap_dtype,
        "score_convention": "AEROBLADE native negative error (-E)",
        "anomaly_map_convention": "A(x,y) = -E(x,y) (Option A)",
        "upsampling": "bilinear_antialias",
        "input_range": "[0,1] -> [-1,1] for VAE",
        "streaming_reconstruction": True,
        "thresholded_localization_rules": [
            "per-image Otsu on -E",
            "swept optimal F1/IoU (IoU_max)",
        ],
    }
    with open(paths["root"] / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def load_ae(repo_id: str, device: str):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        repo_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )
    ae = pipe.vae.to(device)
    ae.eval()
    if device == "cuda":
        ae = ae.to(memory_format=torch.channels_last)
    return ae


def setup_lpips(net: str, device: str):
    model = _PatchedLPIPS(spatial=True, net=net).to(device)
    model.eval()
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def boundary_map(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(mask.astype(bool), structure=structure, iterations=radius)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=structure, iterations=radius)
    return np.logical_xor(dilated, eroded)


def classify_anomaly_otsu(anomaly_map: np.ndarray) -> Tuple[np.ndarray, float]:
    values = anomaly_map.astype(np.float64).ravel()
    lo, hi = float(values.min()), float(values.max())

    if hi <= lo:
        return np.zeros_like(anomaly_map, dtype=np.uint8), lo

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
    prediction = (anomaly_map >= threshold).astype(np.uint8)
    return prediction, threshold


def compute_optimal_threshold_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    n_quantiles: int = 50,
) -> Tuple[float, float, float]:
    if not y_true.any() or y_true.all():
        return float("nan"), float("nan"), float("nan")

    quantiles = np.linspace(0.02, 0.98, n_quantiles)
    thresholds = np.quantile(anomaly_scores, quantiles)

    best_iou = 0.0
    best_f1 = 0.0
    best_th = float(thresholds[0])

    y_bool = y_true.astype(bool)
    n_pos = y_bool.sum()

    for th in thresholds:
        pred_bool = anomaly_scores >= th
        intersection = np.logical_and(y_bool, pred_bool).sum()
        union = np.logical_or(y_bool, pred_bool).sum()

        if union == 0:
            continue

        iou = intersection / union
        pred_pos = pred_bool.sum()
        precision = intersection / pred_pos if pred_pos > 0 else 0.0
        recall = intersection / n_pos
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if iou > best_iou:
            best_iou = float(iou)
            best_f1 = float(f1)
            best_th = float(th)

    return best_iou, best_f1, best_th


def safe_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union if union else 1.0 if not y_true.any() else 0.0

    return float(precision), float(recall), float(f1), float(iou)


def compute_localization_metrics(
    raw_heatmap: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    y_true = mask.astype(np.uint8).ravel()
    anomaly_map = -raw_heatmap
    scores = anomaly_map.astype(np.float64).ravel()

    if np.unique(y_true).size == 2:
        pixel_auc = float(roc_auc_score(y_true, scores))
        pixel_ap = float(average_precision_score(y_true, scores))
        iou_max, dice_max, optimal_th = compute_optimal_threshold_metrics(y_true, scores)
    else:
        pixel_auc = float("nan")
        pixel_ap = float("nan")
        iou_max = float("nan")
        dice_max = float("nan")
        optimal_th = float("nan")

    prediction, threshold = classify_anomaly_otsu(anomaly_map)
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
        "iou_max": iou_max,
        "dice_max": dice_max,
        "optimal_threshold": optimal_th,
    }


def compute_evidence_metrics(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    mask_bool = mask.astype(bool)
    boundary = boundary_map(mask)
    near_context = ndimage.binary_dilation(mask_bool, iterations=10) & ~mask_bool
    global_background = ~mask_bool

    total = float(np.sum(heatmap)) + 1e-12
    inside = float(np.sum(heatmap[mask_bool])) if mask_bool.any() else 0.0
    boundary_energy = float(np.sum(heatmap[boundary])) if boundary.any() else 0.0
    context_energy = float(np.sum(heatmap[near_context])) if near_context.any() else 0.0
    background_energy = float(np.sum(heatmap[global_background])) if global_background.any() else 0.0

    mean_inside = float(np.mean(heatmap[mask_bool])) if mask_bool.any() else np.nan
    mean_bg = float(np.mean(heatmap[global_background])) if global_background.any() else np.nan

    return {
        "mean_error_inside": mean_inside,
        "mean_error_boundary": float(np.mean(heatmap[boundary])) if boundary.any() else np.nan,
        "mean_error_near_context": float(np.mean(heatmap[near_context])) if near_context.any() else np.nan,
        "mean_error_background": mean_bg,
        "energy_fraction_inside": inside / total,
        "energy_fraction_boundary": boundary_energy / total,
        "energy_fraction_near_context": context_energy / total,
        "energy_fraction_background": background_energy / total,
        "inside_background_contrast": (mean_inside / (mean_bg + 1e-12)) if (mask_bool.any() and global_background.any()) else np.nan,
        "anomaly_contrast": (mean_bg / (mean_inside + 1e-12)) if (mask_bool.any() and global_background.any()) else np.nan,
    }


def image_level_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)

    if np.unique(labels).size < 2:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "threshold": np.nan,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }

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


# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if not args.aeroblade_src.exists():
        raise FileNotFoundError(f"AEROBLADE source directory not found: {args.aeroblade_src}")

    sys.path.insert(0, str(args.aeroblade_src))

    paths = setup_output_dirs(args.output_dir)
    save_config(args, paths)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"============================================================")
    print(f"AEROBLADE x INP-X Fast Multi-Triplet Batched Pipeline")
    print(f"Device              : {device}")
    print(f"Total Samples       : {len(df)} triplets ({len(df) * 3} images)")
    print(f"Triplet Batch Size  : {args.triplet_batch_size} ({args.triplet_batch_size * 3} images/pass)")
    print(f"Forensic AE         : {args.ae_repo_id}")
    print(f"LPIPS               : {args.lpips_net}, spatial layer {args.lpips_layer}")
    print(f"DataLoader Workers  : {args.num_workers}")
    print(f"Score Convention    : Option A (-E, higher = more anomalous)")
    print(f"============================================================\n")

    dataset = InpXTripletDataset(df)
    loader = data.DataLoader(
        dataset,
        batch_size=args.triplet_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    ae = load_ae(args.ae_repo_id, device)
    lpips_model = setup_lpips(args.lpips_net, device)

    sample_records = []
    localization_records = []
    evidence_records = []

    detection_scores = []
    detection_labels = []

    started = time.time()

    for batch in tqdm(loader, desc="Evaluating Batches"):
        b_samples = len(batch["sample_id"])
        sample_ids = batch["sample_id"]
        source_datasets = batch["source_dataset"]
        source_generators = batch["source_generator"]
        masks = batch["mask"].numpy()  # [B, H, W]
        heights = batch["h"].numpy()
        widths = batch["w"].numpy()

        # batch["tensors"] is [B, 3, 3, H, W] -> flatten to [B * 3, 3, H, W]
        flat_tensors = batch["tensors"].view(b_samples * 3, 3, heights[0], widths[0])
        flat_tensors = flat_tensors.to(device, non_blocking=True)
        if device == "cuda":
            flat_tensors = flat_tensors.to(memory_format=torch.channels_last)

        generator = torch.Generator(device=device).manual_seed(args.seed)

        # -------------------------------------------------------------------
        # Batched Forward Pass: All B * 3 images evaluated in a single step!
        # -------------------------------------------------------------------
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                ae_input = flat_tensors.to(dtype=ae.dtype) * 2.0 - 1.0
                latents = retrieve_latents(ae.encode(ae_input), generator=generator)
                decoded = ae.decode(latents.to(ae.dtype), return_dict=False)[0]
                reconstructions = (decoded / 2.0 + 0.5).clamp(0, 1).float()

                _, layer_outputs = lpips_model(
                    flat_tensors,
                    reconstructions,
                    retPerLayer=True,
                    normalize=True,
                )

                spatial_error = layer_outputs[args.lpips_layer]  # [B * 3, 1, h, w]
                spatial_error = torch.nn.functional.interpolate(
                    spatial_error.float(),
                    size=(int(heights[0]), int(widths[0])),
                    mode="bilinear",
                    antialias=True,
                )

        # Reshape spatial maps back to [B, 3, H, W]
        spatial_maps = spatial_error[:, 0].view(b_samples, 3, int(heights[0]), int(widths[0])).detach().cpu().numpy().astype(np.float32)

        # Optional reconstruction saving
        if args.save_reconstructions:
            safe_ae = args.ae_repo_id.replace("/", "__")
            flat_recs = reconstructions.detach().cpu()
            for b_idx in range(b_samples):
                s_id = sample_ids[b_idx]
                for c_idx, condition in enumerate(CONDITIONS):
                    rec_img = tf.ToPILImage()(flat_recs[b_idx * 3 + c_idx].clamp(0, 1))
                    rec_path = paths["reconstructions"] / f"sample-{s_id}__condition-{condition}__ae-{safe_ae}__reconstruction.png"
                    rec_img.save(rec_path)

        # Process metrics for each triplet in this batch
        safe_ae = args.ae_repo_id.replace("/", "__")
        for b_idx in range(b_samples):
            s_id = sample_ids[b_idx]
            s_ds = source_datasets[b_idx]
            s_gen = source_generators[b_idx]
            mask = masks[b_idx]
            hw = (int(heights[b_idx]), int(widths[b_idx]))

            heatmaps = {
                "original": spatial_maps[b_idx, 0],
                "standard": spatial_maps[b_idx, 1],
                "inpx": spatial_maps[b_idx, 2],
            }

            for condition in CONDITIONS:
                heatmap = heatmaps[condition]

                # Save heatmap .npy
                npy_path = paths["heatmaps"] / (
                    f"sample-{s_id}__condition-{condition}__ae-{safe_ae}"
                    f"__distance-LPIPS__backbone-{args.lpips_net}__layer-{args.lpips_layer}"
                    f"__orientation-positive_error.npy"
                )
                np.save(npy_path, heatmap.astype(np.float16 if args.heatmap_dtype == "float16" else np.float32))

                mean_error = float(np.mean(heatmap))
                aeroblade_score = -mean_error
                label = 0 if condition == "original" else 1

                detection_scores.append(aeroblade_score)
                detection_labels.append(label)

                common = {
                    "sample_id": s_id,
                    "source_dataset": s_ds,
                    "source_generator": s_gen,
                    "condition": condition,
                    "width": hw[1],
                    "height": hw[0],
                    "image_score_mean_error": mean_error,
                    "image_score_aeroblade": aeroblade_score,
                }

                loc = compute_localization_metrics(heatmap, mask)
                evidence = compute_evidence_metrics(heatmap, mask)

                localization_records.append({**common, **loc})
                evidence_records.append({**common, **evidence})

                sample_records.append(
                    {
                        **common,
                        "heatmap_min": float(np.min(heatmap)),
                        "heatmap_mean": mean_error,
                        "heatmap_std": float(np.std(heatmap)),
                        "heatmap_max": float(np.max(heatmap)),
                    }
                )

    # -----------------------------------------------------------------------
    # Persist metrics tables
    # -----------------------------------------------------------------------
    sample_df = pd.DataFrame(sample_records)
    loc_df = pd.DataFrame(localization_records)
    evidence_df = pd.DataFrame(evidence_records)

    sample_df.to_csv(paths["metrics"] / "sample_metrics.csv", index=False)
    loc_df.to_csv(paths["metrics"] / "localization_metrics.csv", index=False)
    evidence_df.to_csv(paths["metrics"] / "evidence_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # Detection metrics
    # -----------------------------------------------------------------------
    detection_rows_list = []
    for manipulated_condition in ["standard", "inpx"]:
        subset = sample_df[sample_df["condition"].isin(["original", manipulated_condition])]
        scores = subset["image_score_aeroblade"].to_numpy()
        labels = (subset["condition"] == manipulated_condition).astype(np.uint8).to_numpy()
        row_metrics = image_level_metrics(scores, labels)
        row = {
            "experiment": "aeroblade_x_inpx_smoke_test",
            "ae": args.ae_repo_id,
            "lpips_backbone": args.lpips_net,
            "lpips_layer": args.lpips_layer,
            "score": "aeroblade_negative_mean_error",
            "positive_class": manipulated_condition,
            "threshold_rule": "midpoint_between_negative_and_positive_class_means",
            "n_images": len(scores),
            "comparison": f"{manipulated_condition}_vs_original",
            **row_metrics,
        }
        detection_rows_list.append(row)

    detection_df = pd.DataFrame(detection_rows_list)
    detection_df.to_csv(paths["metrics"] / "detection_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # Retention & Degradation
    # -----------------------------------------------------------------------
    loc_summary = (
        loc_df.groupby("condition")[
            ["pixel_ap", "pixel_roc_auc", "dice_otsu", "iou_otsu", "boundary_f1_otsu", "iou_max", "dice_max"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )

    std_row = loc_summary[loc_summary["condition"] == "standard"]
    inpx_row = loc_summary[loc_summary["condition"] == "inpx"]

    summary_rows = []
    if not std_row.empty and not inpx_row.empty:
        for metric in ["pixel_ap", "pixel_roc_auc", "dice_otsu", "iou_otsu", "boundary_f1_otsu", "iou_max", "dice_max"]:
            std_val = float(std_row.iloc[0][metric])
            inpx_val = float(inpx_row.iloc[0][metric])
            summary_rows.append(
                {
                    "metric": metric,
                    "standard_mean": std_val,
                    "inpx_mean": inpx_val,
                    "degradation_absolute": inpx_val - std_val,
                    "retention_ratio": (inpx_val / std_val) if (np.isfinite(std_val) and std_val != 0) else np.nan,
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
                "triplet_batch_size": args.triplet_batch_size,
                "elapsed_seconds": elapsed,
                "triplet_batching": True,
                "channels_last": True,
                "fp16_autocast": True,
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
    print("AEROBLADE x INP-X MULTI-TRIPLET SMOKE TEST COMPLETE")
    print("============================================================")
    print(f"Triplets evaluated : {len(df)}")
    print(f"Total images       : {len(df) * 3}")
    print(f"Triplet batch size : {args.triplet_batch_size} ({args.triplet_batch_size * 3} images/step)")
    print(f"Elapsed time       : {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print(f"Speed              : {len(df) * 3 / max(elapsed, 1e-6):.2f} images/sec")
    print(f"Heatmaps           : {paths['heatmaps']}")
    print(f"Metrics            : {paths['metrics']}")
    print("============================================================")


if __name__ == "__main__":
    main()
