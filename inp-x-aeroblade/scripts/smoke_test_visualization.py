#!/usr/bin/env python3
"""
Visualize AEROBLADE × INP-X stress-test results.

This script consumes the CSV artifacts produced by the AEROBLADE × INP-X
evaluation pipeline and generates reproducible figures for:

1. Qualitative heatmap inspection:
   - Original / Standard / INP-X images
   - reconstruction-error heatmaps
   - heatmap overlays
   - ground-truth masks
   - Standard − INP-X difference maps

2. Reconstruction-error statistics:
   - mean error
   - heatmap standard deviation
   - heatmap maximum
   - paired per-sample condition comparisons

3. Spatial evidence:
   - regional mean reconstruction error
   - regional error energy fractions
   - inside/background contrast

4. Localization:
   - Pixel AP
   - Pixel ROC-AUC
   - Dice
   - IoU
   - Boundary F1

5. Detection:
   - Accuracy
   - Precision
   - Recall
   - F1
   - ROC-AUC
   - PR-AUC

6. INP-X degradation / retention:
   - Standard vs INP-X
   - absolute degradation
   - retention ratio
   - paired sample-level changes when available

The script is intentionally designed to remain useful when moving from the
20-sample smoke test to the pilot and full benchmark.

Expected directory structure:

    <experiment_root>/
    ├── metrics/
    │   ├── sample_metrics.csv
    │   ├── evidence_metrics.csv
    │   ├── localization_metrics.csv
    │   ├── detection_metrics.csv
    │   ├── retention_degradation_metrics.csv
    │   └── run_summary.csv
    │
    └── heatmaps/
        ├── ...
        └── ...

The exact heatmap directory can be supplied through --heatmaps-dir.

Example:

    python scripts/visualize_smoke_test.py \
        --metrics-dir metrics \
        --heatmaps-dir results/heatmaps \
        --output-dir visualizations/aeroblade_x_inpx_smoke_test

Dependencies:

    pandas
    numpy
    matplotlib
    seaborn
    Pillow

No model inference is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONDITION_ORDER = ["original", "standard", "inpx"]

CONDITION_LABELS = {
    "original": "Original",
    "standard": "Standard Inpainting",
    "inpx": "INP-X",
}

REGION_ORDER = [
    "inside",
    "boundary",
    "near_context",
    "background",
]

REGION_LABELS = {
    "inside": "Inside Mask",
    "boundary": "Boundary",
    "near_context": "Near Context",
    "background": "Background",
}

LOCALIZATION_METRICS = [
    ("pixel_ap", "Pixel AP"),
    ("pixel_roc_auc", "Pixel ROC-AUC"),
    ("dice_otsu", "Dice"),
    ("iou_otsu", "IoU"),
    ("boundary_f1_otsu", "Boundary F1"),
]

DETECTION_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

REGIONAL_ERROR_COLUMNS = {
    "inside": "mean_error_inside",
    "boundary": "mean_error_boundary",
    "near_context": "mean_error_near_context",
    "background": "mean_error_background",
}

REGIONAL_ENERGY_COLUMNS = {
    "inside": "energy_fraction_inside",
    "boundary": "energy_fraction_boundary",
    "near_context": "energy_fraction_near_context",
    "background": "energy_fraction_background",
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling input and output locations."""
    parser = argparse.ArgumentParser(
        description="Visualize AEROBLADE × INP-X evaluation results."
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Directory containing the evaluation CSV files.",
    )

    parser.add_argument(
        "--heatmaps-dir",
        type=Path,
        default=Path("results/heatmaps"),
        help="Directory containing saved reconstruction-error heatmaps.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/aeroblade_x_inpx_smoke_test"),
        help="Directory where generated figures are written.",
    )

    parser.add_argument(
        "--n-qualitative",
        type=int,
        default=10,
        help="Number of samples for qualitative heatmap panels.",
    )

    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional explicit sample IDs for qualitative visualization.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI used when saving figures.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )

    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create the visualization directory tree and return its subdirectories."""
    directories = {
        "root": output_dir,
        "qualitative": output_dir / "qualitative",
        "distributions": output_dir / "distributions",
        "evidence": output_dir / "evidence",
        "localization": output_dir / "localization",
        "detection": output_dir / "detection",
        "retention": output_dir / "retention",
        "summary": output_dir / "summary",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def load_csv(metrics_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Load a metrics CSV if it exists, returning None when unavailable."""
    path = metrics_dir / filename

    if not path.exists():
        print(f"[WARN] Missing metrics file: {path}")
        return None

    dataframe = pd.read_csv(path)
    print(f"[INFO] Loaded {filename}: {len(dataframe):,} rows")

    return dataframe


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int,
    show: bool = False,
) -> None:
    """Save a matplotlib figure with tight layout and optionally display it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)


def condition_sort_key(condition: str) -> int:
    """Return the canonical plotting order for an experiment condition."""
    try:
        return CONDITION_ORDER.index(condition)
    except ValueError:
        return len(CONDITION_ORDER)


def sort_conditions(values: Iterable[str]) -> list[str]:
    """Sort condition names using the experiment's canonical ordering."""
    return sorted(set(values), key=condition_sort_key)


def sanitize_filename(value: str) -> str:
    """Convert an arbitrary identifier into a filesystem-safe filename."""
    value = str(value)
    value = re.sub(r"[^\w\-.]+", "_", value)
    return value.strip("_")


def get_experiment_metadata(
    run_summary: Optional[pd.DataFrame],
) -> dict[str, str]:
    """Extract experiment configuration metadata from the run summary CSV."""
    if run_summary is None or run_summary.empty:
        return {}

    row = run_summary.iloc[0]

    metadata = {}

    for column in [
        "experiment",
        "ae",
        "lpips_backbone",
        "lpips_layer",
        "seed",
        "device",
    ]:
        if column in row.index:
            metadata[column] = str(row[column])

    return metadata


def experiment_title(metadata: dict[str, str]) -> str:
    """Build a concise configuration-aware title for generated figures."""
    if not metadata:
        return "AEROBLADE × INP-X"

    ae = metadata.get("ae", "Unknown AE")
    backbone = metadata.get("lpips_backbone", "Unknown backbone")
    layer = metadata.get("lpips_layer", "?")

    return (
        "AEROBLADE × INP-X\n"
        f"AE: {ae} | LPIPS: {backbone} layer {layer}"
    )


# ---------------------------------------------------------------------------
# Heatmap discovery and loading
# ---------------------------------------------------------------------------

def discover_heatmap_files(
    heatmaps_dir: Path,
) -> dict[tuple[str, str], Path]:
    """
    Discover saved heatmap files and map them to (sample_id, condition).

    The function intentionally supports several common filename conventions.
    It first attempts to identify condition tokens such as 'original',
    'standard', and 'inpx', then uses the remaining filename as the sample ID.
    """
    mapping: dict[tuple[str, str], Path] = {}

    if not heatmaps_dir.exists():
        print(f"[WARN] Heatmap directory does not exist: {heatmaps_dir}")
        return mapping

    valid_extensions = {".npy", ".npz"}

    files = [
        path
        for path in heatmaps_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]

    print(f"[INFO] Discovered {len(files):,} heatmap files.")

    for path in files:
        stem = path.stem
        stem_lower = stem.lower()

        condition = None

        for candidate in ["original", "standard", "inpx"]:
            if re.search(
                rf"(^|[_\-]){re.escape(candidate)}($|[_\-])",
                stem_lower,
            ):
                condition = candidate
                break

        if condition is None:
            continue

        # Remove common condition tokens from the filename to recover
        # the sample identifier. This is intentionally conservative.
        sample_id = stem

        for token in [
            "_original",
            "-original",
            "_standard",
            "-standard",
            "_inpx",
            "-inpx",
        ]:
            sample_id = sample_id.replace(token, "")

        sample_id = sample_id.strip("_-")

        mapping[(sample_id, condition)] = path

    return mapping


def load_heatmap(path: Path) -> np.ndarray:
    """
    Load a reconstruction-error heatmap from NPY or NPZ format.

    NPZ files are expected to contain either an array named 'heatmap' or a
    single array. The returned array is squeezed to remove singleton axes.
    """
    if path.suffix.lower() == ".npy":
        array = np.load(path)

    elif path.suffix.lower() == ".npz":
        archive = np.load(path)

        if "heatmap" in archive:
            array = archive["heatmap"]
        elif len(archive.files) == 1:
            array = archive[archive.files[0]]
        else:
            raise ValueError(
                f"Cannot determine heatmap array in {path}. "
                f"Available keys: {archive.files}"
            )
    else:
        raise ValueError(f"Unsupported heatmap format: {path}")

    array = np.asarray(array).squeeze()

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D heatmap, got shape {array.shape} from {path}"
        )

    return array.astype(np.float32)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def find_image_for_sample(
    row: pd.Series,
    condition: str,
) -> Optional[Path]:
    """Resolve the image path for a sample-condition row in sample metrics."""
    column_map = {
        "original": "original_path",
        "standard": "standard_inpainting_path",
        "inpx": "inpainting_exchange_path",
    }

    column = column_map.get(condition)

    if column is None or column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    path = Path(str(value))

    return path if path.exists() else None


def find_mask_for_sample(row: pd.Series) -> Optional[Path]:
    """Resolve the ground-truth mask path from a metrics row when available."""
    if "mask_path" not in row.index:
        return None

    value = row["mask_path"]

    if pd.isna(value):
        return None

    path = Path(str(value))

    return path if path.exists() else None


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image as an RGB float array in the [0, 1] range."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    """Load a binary ground-truth mask and normalize it to boolean values."""
    mask = Image.open(path).convert("L")
    array = np.asarray(mask)

    return array > 127


# ---------------------------------------------------------------------------
# Qualitative heatmap visualization
# ---------------------------------------------------------------------------

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Normalize a heatmap and blend it visually over an RGB image."""
    finite = np.isfinite(heatmap)

    if not finite.any():
        normalized = np.zeros_like(heatmap)
    else:
        minimum = np.nanmin(heatmap)
        maximum = np.nanmax(heatmap)

        if maximum > minimum:
            normalized = (heatmap - minimum) / (maximum - minimum)
        else:
            normalized = np.zeros_like(heatmap)

    cmap = plt.get_cmap("inferno")
    heatmap_rgb = cmap(normalized)[..., :3]

    if heatmap_rgb.shape[:2] != image.shape[:2]:
        heatmap_rgb = np.asarray(
            Image.fromarray(
                np.uint8(np.clip(heatmap_rgb, 0, 1) * 255)
            ).resize(
                (image.shape[1], image.shape[0]),
                Image.Resampling.BILINEAR,
            )
        ).astype(np.float32) / 255.0

    return (
        (1.0 - alpha) * image
        + alpha * heatmap_rgb
    ).clip(0, 1)


def plot_comparative_sample(
    sample_id: str,
    sample_metrics: pd.DataFrame,
    heatmap_mapping: dict[tuple[str, str], Path],
    output_path: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> bool:
    """
    Generate a multi-row qualitative panel for one sample.

    Rows contain source images, reconstruction-error heatmaps, overlays, and
    the ground-truth mask. Standard and INP-X are directly comparable while
    the original image acts as an authentic control.
    """
    rows = sample_metrics[
        sample_metrics["sample_id"].astype(str) == str(sample_id)
    ]

    if rows.empty:
        return False

    row_lookup = {
        str(row["condition"]): row
        for _, row in rows.iterrows()
    }

    conditions = [
        condition
        for condition in CONDITION_ORDER
        if condition in row_lookup
    ]

    if not conditions:
        return False

    n_columns = len(conditions)

    fig, axes = plt.subplots(
        4,
        n_columns,
        figsize=(5 * n_columns, 16),
        squeeze=False,
    )

    source_dataset = str(rows.iloc[0].get("source_dataset", "Unknown"))
    source_generator = str(rows.iloc[0].get("source_generator", "Unknown"))

    fig.suptitle(
        experiment_title(metadata)
        + f"\nSample: {sample_id} | Dataset: {source_dataset} | "
        f"Generator: {source_generator}",
        fontsize=14,
    )

    mask = None

    for column, condition in enumerate(conditions):
        row = row_lookup[condition]

        image_path = find_image_for_sample(row, condition)
        heatmap_path = heatmap_mapping.get((str(sample_id), condition))

        image = None
        heatmap = None

        if image_path is not None:
            image = load_rgb_image(image_path)

        if heatmap_path is not None:
            heatmap = load_heatmap(heatmap_path)

        if mask is None:
            mask_path = find_mask_for_sample(row)

            if mask_path is not None:
                mask = load_mask(mask_path)

        label = CONDITION_LABELS.get(condition, condition)

        axes[0, column].set_title(label, fontsize=12)

        if image is not None:
            axes[0, column].imshow(image)
        else:
            axes[0, column].text(
                0.5,
                0.5,
                "Image unavailable",
                ha="center",
                va="center",
            )

        axes[0, column].set_ylabel("Input Image")

        if heatmap is not None:
            axes[1, column].imshow(
                heatmap,
                cmap="inferno",
            )
            axes[1, column].set_title(
                f"{label}\nReconstruction Error"
            )
        else:
            axes[1, column].text(
                0.5,
                0.5,
                "Heatmap unavailable",
                ha="center",
                va="center",
            )

        axes[1, column].set_ylabel("Error Heatmap")

        if image is not None and heatmap is not None:
            axes[2, column].imshow(
                overlay_heatmap(image, heatmap)
            )
        else:
            axes[2, column].text(
                0.5,
                0.5,
                "Overlay unavailable",
                ha="center",
                va="center",
            )

        axes[2, column].set_ylabel("Overlay")

        if mask is not None:
            axes[3, column].imshow(
                mask,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        else:
            axes[3, column].text(
                0.5,
                0.5,
                "Mask unavailable",
                ha="center",
                va="center",
            )

        axes[3, column].set_ylabel("Ground Truth Mask")

        for row_index in range(4):
            axes[row_index, column].axis("off")

    save_figure(fig, output_path, dpi=dpi, show=show)

    return True


def plot_difference_map(
    sample_id: str,
    heatmap_mapping: dict[tuple[str, str], Path],
    output_path: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> bool:
    """
    Visualize Standard − INP-X reconstruction-error differences.

    Positive values indicate regions where Standard Inpainting produces larger
    reconstruction error than INP-X. A symmetric diverging color scale is used
    so both directions of change remain visible.
    """
    standard_path = heatmap_mapping.get((str(sample_id), "standard"))
    inpx_path = heatmap_mapping.get((str(sample_id), "inpx"))

    if standard_path is None or inpx_path is None:
        return False

    standard = load_heatmap(standard_path)
    inpx = load_heatmap(inpx_path)

    if standard.shape != inpx.shape:
        print(
            f"[WARN] Cannot compare heatmaps with different shapes for "
            f"{sample_id}: {standard.shape} vs {inpx.shape}"
        )
        return False

    difference = standard - inpx

    maximum = np.nanmax(np.abs(difference))

    if not np.isfinite(maximum) or maximum == 0:
        maximum = 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    axes[0].imshow(standard, cmap="inferno")
    axes[0].set_title("Standard Inpainting\nReconstruction Error")

    axes[1].imshow(inpx, cmap="inferno")
    axes[1].set_title("INP-X\nReconstruction Error")

    axes[2].imshow(
        difference,
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )
    axes[2].set_title(
        "Standard − INP-X\nReconstruction Error Difference"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        experiment_title(metadata)
        + f"\nSample: {sample_id}",
        fontsize=14,
    )

    save_figure(fig, output_path, dpi=dpi, show=show)

    return True


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_paired_metric(
    dataframe: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot paired per-sample values across experimental conditions."""
    if metric not in dataframe.columns:
        return

    pivot = dataframe.pivot_table(
        index="sample_id",
        columns="condition",
        values=metric,
        aggfunc="first",
    )

    conditions = [
        condition
        for condition in CONDITION_ORDER
        if condition in pivot.columns
    ]

    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for sample_id, row in pivot.iterrows():
        values = [
            row.get(condition, np.nan)
            for condition in conditions
        ]

        valid = [
            np.isfinite(value)
            for value in values
        ]

        if sum(valid) >= 2:
            ax.plot(
                range(len(conditions)),
                values,
                marker="o",
                alpha=0.35,
                linewidth=1,
            )

    means = pivot[conditions].mean()

    ax.plot(
        range(len(conditions)),
        means.values,
        marker="o",
        linewidth=3,
        label="Mean",
    )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [
            CONDITION_LABELS.get(condition, condition)
            for condition in conditions
        ]
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()

    save_figure(fig, output_path, dpi=dpi, show=show)


def plot_condition_boxplot(
    dataframe: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str,
    dpi: int,
    show: bool,
) -> None:
    """Plot a distribution of a metric separately for each condition."""
    if metric not in dataframe.columns:
        return

    conditions = sort_conditions(dataframe["condition"].dropna())

    values = [
        dataframe.loc[
            dataframe["condition"] == condition,
            metric,
        ].dropna().values
        for condition in conditions
    ]

    if not any(len(value) > 0 for value in values):
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
        values,
        labels=[
            CONDITION_LABELS.get(condition, condition)
            for condition in conditions
        ],
        showmeans=True,
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, output_path, dpi=dpi, show=show)


# ---------------------------------------------------------------------------
# Spatial evidence plots
# ---------------------------------------------------------------------------

def plot_regional_errors(
    evidence: pd.DataFrame,
    output_path: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Compare mean reconstruction error across spatial evidence regions."""
    available = [
        region
        for region, column in REGIONAL_ERROR_COLUMNS.items()
        if column in evidence.columns
    ]

    if not available:
        return

    long_records = []

    for region in available:
        column = REGIONAL_ERROR_COLUMNS[region]

        subset = evidence[
            ["condition", column]
        ].copy()

        subset["region"] = region
        subset["value"] = subset[column]

        long_records.append(
            subset[["condition", "region", "value"]]
        )

    long_df = pd.concat(long_records, ignore_index=True)

    conditions = sort_conditions(long_df["condition"].dropna())

    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(available))
    width = 0.8 / max(1, len(conditions))

    for index, condition in enumerate(conditions):
        means = []

        for region in available:
            values = long_df.loc[
                (long_df["condition"] == condition)
                & (long_df["region"] == region),
                "value",
            ]

            means.append(values.mean())

        positions = (
            x
            - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            means,
            width=width,
            label=CONDITION_LABELS.get(condition, condition),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [REGION_LABELS[region] for region in available]
    )

    ax.set_ylabel("Mean Reconstruction Error")
    ax.set_title(
        experiment_title(metadata)
        + "\nSpatial Distribution of Reconstruction Error"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, output_path, dpi=dpi, show=show)


def plot_regional_energy(
    evidence: pd.DataFrame,
    output_path: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Compare reconstruction-error energy fractions across spatial regions."""
    available = [
        region
        for region, column in REGIONAL_ENERGY_COLUMNS.items()
        if column in evidence.columns
    ]

    if not available:
        return

    conditions = sort_conditions(evidence["condition"].dropna())

    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(available))
    width = 0.8 / max(1, len(conditions))

    for index, condition in enumerate(conditions):
        means = []

        for region in available:
            column = REGIONAL_ENERGY_COLUMNS[region]

            values = evidence.loc[
                evidence["condition"] == condition,
                column,
            ]

            means.append(values.mean())

        positions = (
            x
            - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            means,
            width=width,
            label=CONDITION_LABELS.get(condition, condition),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [REGION_LABELS[region] for region in available]
    )

    ax.set_ylabel("Fraction of Reconstruction-Error Energy")
    ax.set_title(
        experiment_title(metadata)
        + "\nSpatial Distribution of Reconstruction-Error Energy"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, output_path, dpi=dpi, show=show)


def plot_inside_background_contrast(
    evidence: pd.DataFrame,
    output_path: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Visualize the inside-mask versus background reconstruction contrast."""
    metric = "inside_background_contrast"

    if metric not in evidence.columns:
        return

    conditions = sort_conditions(evidence["condition"].dropna())

    data = [
        evidence.loc[
            evidence["condition"] == condition,
            metric,
        ].dropna().values
        for condition in conditions
    ]

    if not any(len(values) for values in data):
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
        data,
        labels=[
            CONDITION_LABELS.get(condition, condition)
            for condition in conditions
        ],
        showmeans=True,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Equal Inside / Background Error",
    )

    ax.set_ylabel("Inside / Background Error Contrast")
    ax.set_title(
        experiment_title(metadata)
        + "\nSpatial Reconstruction-Error Contrast"
    )

    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    save_figure(fig, output_path, dpi=dpi, show=show)


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

def plot_localization_metrics(
    localization: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Generate condition-comparison plots for all available localization metrics."""
    for metric, label in LOCALIZATION_METRICS:
        if metric not in localization.columns:
            continue

        plot_condition_boxplot(
            localization,
            metric,
            label,
            output_dir / f"{metric}_by_condition.png",
            experiment_title(metadata)
            + f"\n{label} by Condition",
            dpi,
            show,
        )

    available = [
        (metric, label)
        for metric, label in LOCALIZATION_METRICS
        if metric in localization.columns
    ]

    if not available:
        return

    conditions = sort_conditions(localization["condition"].dropna())

    means = pd.DataFrame(
        {
            label: [
                localization.loc[
                    localization["condition"] == condition,
                    metric,
                ].mean()
                for condition in conditions
            ]
            for metric, label in available
        },
        index=[
            CONDITION_LABELS.get(condition, condition)
            for condition in conditions
        ],
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(available))
    width = 0.8 / max(1, len(conditions))

    for index, condition in enumerate(conditions):
        values = means.loc[
            CONDITION_LABELS.get(condition, condition)
        ].values

        positions = (
            x
            - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            values,
            width=width,
            label=CONDITION_LABELS.get(condition, condition),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _, label in available],
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0)

    ax.set_title(
        experiment_title(metadata)
        + "\nLocalization Performance by Condition"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir / "localization_metrics_comparison.png",
        dpi,
        show,
    )


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def plot_detection_metrics(
    detection: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Visualize image-level detection metrics for each comparison."""
    if detection.empty:
        return

    available = [
        (metric, label)
        for metric, label in DETECTION_METRICS
        if metric in detection.columns
    ]

    if not available:
        return

    comparisons = (
        detection["comparison"].dropna().unique()
        if "comparison" in detection.columns
        else []
    )

    if len(comparisons) == 0:
        comparisons = ["all"]

    for comparison in comparisons:
        if comparison == "all":
            subset = detection
        else:
            subset = detection[
                detection["comparison"] == comparison
            ]

        if subset.empty:
            continue

        values = [
            subset[metric].mean()
            for metric, _ in available
        ]

        labels = [
            label
            for _, label in available
        ]

        fig, ax = plt.subplots(figsize=(11, 6))

        bars = ax.bar(
            np.arange(len(labels)),
            values,
        )

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(
            labels,
            rotation=25,
            ha="right",
        )

        ax.set_ylabel("Score")
        ax.set_ylim(0, max(1.0, max(values) * 1.15))

        comparison_label = comparison.replace("_", " ").title()

        ax.set_title(
            experiment_title(metadata)
            + f"\nImage-Level Detection — {comparison_label}"
        )

        ax.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        filename = (
            "detection_"
            + sanitize_filename(str(comparison))
            + ".png"
        )

        save_figure(
            fig,
            output_dir / filename,
            dpi,
            show,
        )


# ---------------------------------------------------------------------------
# Retention / degradation
# ---------------------------------------------------------------------------

def plot_retention_degradation(
    retention: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Plot Standard-to-INP-X performance degradation and retention."""
    if retention is None or retention.empty:
        return

    required = {
        "metric",
        "standard_mean",
        "inpx_mean",
        "degradation_absolute",
        "retention_ratio",
    }

    missing = required - set(retention.columns)

    if missing:
        print(
            "[WARN] Retention CSV is missing columns:",
            sorted(missing),
        )
        return

    metrics = retention["metric"].astype(str).tolist()

    # ---------------------------------------------------------------
    # Absolute performance comparison
    # ---------------------------------------------------------------

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        retention["standard_mean"],
        width,
        label="Standard Inpainting",
    )

    ax.bar(
        x + width / 2,
        retention["inpx_mean"],
        width,
        label="INP-X",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)

    ax.set_title(
        experiment_title(metadata)
        + "\nStandard vs INP-X Localization Performance"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir / "standard_vs_inpx_performance.png",
        dpi,
        show,
    )

    # ---------------------------------------------------------------
    # Retention ratio
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        x,
        retention["retention_ratio"],
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="100% Retention",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylabel("Retention Ratio")
    ax.set_ylim(
        0,
        max(
            1.0,
            retention["retention_ratio"].max() * 1.15,
        ),
    )

    ax.set_title(
        experiment_title(metadata)
        + "\nINP-X Performance Retention"
    )

    for bar, value in zip(
        bars,
        retention["retention_ratio"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    save_figure(
        fig,
        output_dir / "inpx_retention.png",
        dpi,
        show,
    )

    # ---------------------------------------------------------------
    # Absolute degradation
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        x,
        retention["degradation_absolute"],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylabel(
        "INP-X − Standard"
    )

    ax.set_title(
        experiment_title(metadata)
        + "\nINP-X Absolute Performance Degradation"
    )

    for bar, value in zip(
        bars,
        retention["degradation_absolute"],
    ):
        offset = 0.01 if value >= 0 else -0.03

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir / "inpx_absolute_degradation.png",
        dpi,
        show,
    )


# ---------------------------------------------------------------------------
# Paired Standard → INP-X analysis
# ---------------------------------------------------------------------------

def plot_paired_standard_inpx(
    localization: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """
    Visualize per-sample Standard → INP-X changes for localization metrics.

    Each line represents the same underlying sample under the two inpainting
    conditions. This reveals whether aggregate degradation is consistent
    across samples or driven by a small number of failures.
    """
    if localization is None or localization.empty:
        return

    if "condition" not in localization.columns:
        return

    subset = localization[
        localization["condition"].isin(["standard", "inpx"])
    ].copy()

    if subset.empty:
        return

    for metric, label in LOCALIZATION_METRICS:
        if metric not in subset.columns:
            continue

        pivot = subset.pivot_table(
            index="sample_id",
            columns="condition",
            values=metric,
            aggfunc="first",
        )

        if not {"standard", "inpx"}.issubset(pivot.columns):
            continue

        valid = pivot[["standard", "inpx"]].dropna()

        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        for _, row in valid.iterrows():
            ax.plot(
                [0, 1],
                [row["standard"], row["inpx"]],
                alpha=0.35,
                marker="o",
                linewidth=1,
            )

        mean_standard = valid["standard"].mean()
        mean_inpx = valid["inpx"].mean()

        ax.plot(
            [0, 1],
            [mean_standard, mean_inpx],
            marker="o",
            linewidth=3,
            label="Mean",
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                "Standard Inpainting",
                "INP-X",
            ]
        )

        ax.set_ylabel(label)
        ax.set_title(
            experiment_title(metadata)
            + f"\nPaired Standard → INP-X Change: {label}"
        )

        ax.grid(alpha=0.25)
        ax.legend()

        save_figure(
            fig,
            output_dir / f"paired_standard_inpx_{metric}.png",
            dpi,
            show,
        )


def plot_paired_sample_error_change(
    sample_metrics: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """Plot paired sample-level reconstruction-error changes across conditions."""
    metric = "image_score_mean_error"

    if metric not in sample_metrics.columns:
        return

    pivot = sample_metrics.pivot_table(
        index="sample_id",
        columns="condition",
        values=metric,
        aggfunc="first",
    )

    conditions = [
        condition
        for condition in ["original", "standard", "inpx"]
        if condition in pivot.columns
    ]

    if len(conditions) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in pivot.iterrows():
        values = [
            row.get(condition, np.nan)
            for condition in conditions
        ]

        if sum(np.isfinite(values)) >= 2:
            ax.plot(
                range(len(conditions)),
                values,
                marker="o",
                alpha=0.35,
                linewidth=1,
            )

    means = pivot[conditions].mean()

    ax.plot(
        range(len(conditions)),
        means.values,
        marker="o",
        linewidth=3,
        label="Mean",
    )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [
            CONDITION_LABELS.get(condition, condition)
            for condition in conditions
        ]
    )

    ax.set_ylabel("Mean Spatial Reconstruction Error")
    ax.set_title(
        experiment_title(metadata)
        + "\nPaired Reconstruction-Error Change"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    save_figure(
        fig,
        output_dir / "paired_mean_reconstruction_error.png",
        dpi,
        show,
    )


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def create_summary_figure(
    sample_metrics: Optional[pd.DataFrame],
    localization: Optional[pd.DataFrame],
    retention: Optional[pd.DataFrame],
    output_path: Path,
    metadata: dict[str, str],
    dpi: int,
    show: bool,
) -> None:
    """
    Create a compact overview figure combining the principal smoke-test results.

    The figure intentionally emphasizes Standard versus INP-X, while retaining
    Original as the authentic control where sample-level reconstruction error
    is available.
    """
    fig = plt.figure(figsize=(15, 10))

    grid = fig.add_gridspec(
        2,
        2,
        hspace=0.35,
        wspace=0.25,
    )

    # ---------------------------------------------------------------
    # Panel 1: mean reconstruction error
    # ---------------------------------------------------------------

    ax = fig.add_subplot(grid[0, 0])

    if (
        sample_metrics is not None
        and "image_score_mean_error" in sample_metrics.columns
    ):
        means = (
            sample_metrics
            .groupby("condition")["image_score_mean_error"]
            .mean()
        )

        conditions = [
            condition
            for condition in CONDITION_ORDER
            if condition in means.index
        ]

        ax.bar(
            range(len(conditions)),
            [
                means[condition]
                for condition in conditions
            ],
        )

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [
                CONDITION_LABELS.get(condition, condition)
                for condition in conditions
            ],
            rotation=20,
            ha="right",
        )

        ax.set_ylabel("Mean Reconstruction Error")
        ax.set_title("Image-Level Reconstruction Error")

    else:
        ax.text(
            0.5,
            0.5,
            "No sample metrics available",
            ha="center",
            va="center",
        )

    ax.grid(axis="y", alpha=0.25)

    # ---------------------------------------------------------------
    # Panel 2: localization
    # ---------------------------------------------------------------

    ax = fig.add_subplot(grid[0, 1])

    if (
        localization is not None
        and "pixel_ap" in localization.columns
    ):
        means = (
            localization
            .groupby("condition")["pixel_ap"]
            .mean()
        )

        conditions = [
            condition
            for condition in ["standard", "inpx"]
            if condition in means.index
        ]

        ax.bar(
            range(len(conditions)),
            [
                means[condition]
                for condition in conditions
            ],
        )

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [
                CONDITION_LABELS.get(condition, condition)
                for condition in conditions
            ]
        )

        ax.set_ylabel("Pixel AP")
        ax.set_title("Localization: Pixel AP")

    else:
        ax.text(
            0.5,
            0.5,
            "No localization metrics available",
            ha="center",
            va="center",
        )

    ax.grid(axis="y", alpha=0.25)

    # ---------------------------------------------------------------
    # Panel 3: retention
    # ---------------------------------------------------------------

    ax = fig.add_subplot(grid[1, 0])

    if (
        retention is not None
        and "retention_ratio" in retention.columns
    ):
        ax.bar(
            retention["metric"],
            retention["retention_ratio"],
        )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Retention Ratio")
        ax.set_title("INP-X Retention")

        ax.tick_params(
            axis="x",
            rotation=20,
        )

    else:
        ax.text(
            0.5,
            0.5,
            "No retention metrics available",
            ha="center",
            va="center",
        )

    ax.grid(axis="y", alpha=0.25)

    # ---------------------------------------------------------------
    # Panel 4: ROC-AUC localization
    # ---------------------------------------------------------------

    ax = fig.add_subplot(grid[1, 1])

    if (
        localization is not None
        and "pixel_roc_auc" in localization.columns
    ):
        means = (
            localization
            .groupby("condition")["pixel_roc_auc"]
            .mean()
        )

        conditions = [
            condition
            for condition in ["standard", "inpx"]
            if condition in means.index
        ]

        ax.bar(
            range(len(conditions)),
            [
                means[condition]
                for condition in conditions
            ],
        )

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [
                CONDITION_LABELS.get(condition, condition)
                for condition in conditions
            ]
        )

        ax.set_ylabel("Pixel ROC-AUC")
        ax.set_ylim(0, 1)
        ax.set_title("Localization: Pixel ROC-AUC")

    else:
        ax.text(
            0.5,
            0.5,
            "No localization metrics available",
            ha="center",
            va="center",
        )

    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        experiment_title(metadata)
        + "\nSmoke-Test Experimental Summary",
        fontsize=16,
    )

    save_figure(
        fig,
        output_path,
        dpi,
        show,
    )


# ---------------------------------------------------------------------------
# Metadata / summary generation
# ---------------------------------------------------------------------------

def write_visualization_summary(
    output_dir: Path,
    metadata: dict[str, str],
    generated_files: list[Path],
    sample_metrics: Optional[pd.DataFrame],
) -> None:
    """Write a machine-readable record of the generated visualization run."""
    summary = {
        "experiment": metadata.get(
            "experiment",
            "aeroblade_x_inpx",
        ),
        "ae": metadata.get("ae"),
        "lpips_backbone": metadata.get("lpips_backbone"),
        "lpips_layer": metadata.get("lpips_layer"),
        "seed": metadata.get("seed"),
        "device": metadata.get("device"),
        "n_samples": (
            int(sample_metrics["sample_id"].nunique())
            if sample_metrics is not None
            and "sample_id" in sample_metrics.columns
            else None
        ),
        "conditions": (
            sort_conditions(sample_metrics["condition"].dropna())
            if sample_metrics is not None
            and "condition" in sample_metrics.columns
            else []
        ),
        "n_figures": len(generated_files),
        "figures": [
            str(path.relative_to(output_dir))
            for path in generated_files
            if path.exists()
        ],
    }

    path = output_dir / "summary" / "visualization_summary.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(f"[INFO] Wrote visualization summary: {path}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Load experimental artifacts and execute the complete visualization suite."""
    args = parse_args()

    metrics_dir = args.metrics_dir
    heatmaps_dir = args.heatmaps_dir
    output_dir = args.output_dir

    directories = ensure_output_dirs(output_dir)

    print("=" * 72)
    print("AEROBLADE × INP-X Visualization")
    print("=" * 72)
    print(f"Metrics directory : {metrics_dir}")
    print(f"Heatmaps directory: {heatmaps_dir}")
    print(f"Output directory  : {output_dir}")
    print()

    # ---------------------------------------------------------------
    # Load all available metrics
    # ---------------------------------------------------------------

    sample_metrics = load_csv(
        metrics_dir,
        "sample_metrics.csv",
    )

    evidence_metrics = load_csv(
        metrics_dir,
        "evidence_metrics.csv",
    )

    localization_metrics = load_csv(
        metrics_dir,
        "localization_metrics.csv",
    )

    detection_metrics = load_csv(
        metrics_dir,
        "detection_metrics.csv",
    )

    retention_metrics = load_csv(
        metrics_dir,
        "retention_degradation_metrics.csv",
    )

    run_summary = load_csv(
        metrics_dir,
        "run_summary.csv",
    )

    metadata = get_experiment_metadata(run_summary)

    print()
    print("Experiment configuration:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------------
    # Discover heatmaps
    # ---------------------------------------------------------------

    heatmap_mapping = discover_heatmap_files(heatmaps_dir)

    generated_files: list[Path] = []

    # ---------------------------------------------------------------
    # Qualitative sample selection
    # ---------------------------------------------------------------

    if sample_metrics is not None and not sample_metrics.empty:

        available_sample_ids = (
            sample_metrics["sample_id"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if args.sample_ids:
            selected_sample_ids = [
                sample_id
                for sample_id in args.sample_ids
                if sample_id in available_sample_ids
            ]

            missing_requested = set(args.sample_ids) - set(
                selected_sample_ids
            )

            for sample_id in sorted(missing_requested):
                print(
                    f"[WARN] Requested sample not found: {sample_id}"
                )

        else:
            selected_sample_ids = available_sample_ids[
                : args.n_qualitative
            ]

        print()
        print(
            f"[INFO] Generating qualitative panels for "
            f"{len(selected_sample_ids)} samples."
        )

        for sample_id in selected_sample_ids:

            safe_id = sanitize_filename(sample_id)

            comparative_path = (
                directories["qualitative"]
                / f"sample_{safe_id}_comparative.png"
            )

            success = plot_comparative_sample(
                sample_id,
                sample_metrics,
                heatmap_mapping,
                comparative_path,
                metadata,
                args.dpi,
                args.show,
            )

            if success:
                generated_files.append(comparative_path)

            difference_path = (
                directories["qualitative"]
                / f"sample_{safe_id}_standard_minus_inpx.png"
            )

            success = plot_difference_map(
                sample_id,
                heatmap_mapping,
                difference_path,
                metadata,
                args.dpi,
                args.show,
            )

            if success:
                generated_files.append(difference_path)

        # -----------------------------------------------------------
        # Sample-level reconstruction-error plots
        # -----------------------------------------------------------

        paired_error_path = (
            directories["distributions"]
            / "paired_mean_reconstruction_error.png"
        )

        plot_paired_sample_error_change(
            sample_metrics,
            directories["distributions"],
            metadata,
            args.dpi,
            args.show,
        )

        if paired_error_path.exists():
            generated_files.append(paired_error_path)

        for metric, label in [
            ("image_score_mean_error", "Mean Reconstruction Error"),
            ("heatmap_std", "Heatmap Standard Deviation"),
            ("heatmap_max", "Heatmap Maximum"),
        ]:
            if metric not in sample_metrics.columns:
                continue

            output_path = (
                directories["distributions"]
                / f"{metric}_distribution.png"
            )

            plot_condition_boxplot(
                sample_metrics,
                metric,
                label,
                output_path,
                experiment_title(metadata)
                + f"\n{label} by Condition",
                args.dpi,
                args.show,
            )

            if output_path.exists():
                generated_files.append(output_path)

    # ---------------------------------------------------------------
    # Spatial evidence
    # ---------------------------------------------------------------

    if evidence_metrics is not None and not evidence_metrics.empty:

        regional_error_path = (
            directories["evidence"]
            / "regional_mean_reconstruction_error.png"
        )

        plot_regional_errors(
            evidence_metrics,
            regional_error_path,
            metadata,
            args.dpi,
            args.show,
        )

        if regional_error_path.exists():
            generated_files.append(regional_error_path)

        regional_energy_path = (
            directories["evidence"]
            / "regional_energy_fraction.png"
        )

        plot_regional_energy(
            evidence_metrics,
            regional_energy_path,
            metadata,
            args.dpi,
            args.show,
        )

        if regional_energy_path.exists():
            generated_files.append(regional_energy_path)

        contrast_path = (
            directories["evidence"]
            / "inside_background_contrast.png"
        )

        plot_inside_background_contrast(
            evidence_metrics,
            contrast_path,
            metadata,
            args.dpi,
            args.show,
        )

        if contrast_path.exists():
            generated_files.append(contrast_path)

    # ---------------------------------------------------------------
    # Localization
    # ---------------------------------------------------------------

    if localization_metrics is not None:
        plot_localization_metrics(
            localization_metrics,
            directories["localization"],
            metadata,
            args.dpi,
            args.show,
        )

        plot_paired_standard_inpx(
            localization_metrics,
            directories["localization"],
            metadata,
            args.dpi,
            args.show,
        )

        generated_files.extend(
            directories["localization"].glob("*.png")
        )

    # ---------------------------------------------------------------
    # Detection
    # ---------------------------------------------------------------

    if detection_metrics is not None:
        plot_detection_metrics(
            detection_metrics,
            directories["detection"],
            metadata,
            args.dpi,
            args.show,
        )

        generated_files.extend(
            directories["detection"].glob("*.png")
        )

    # ---------------------------------------------------------------
    # Retention / degradation
    # ---------------------------------------------------------------

    if retention_metrics is not None:
        plot_retention_degradation(
            retention_metrics,
            directories["retention"],
            metadata,
            args.dpi,
            args.show,
        )

        generated_files.extend(
            directories["retention"].glob("*.png")
        )

    # ---------------------------------------------------------------
    # Summary figure
    # ---------------------------------------------------------------

    summary_path = (
        directories["summary"]
        / "smoke_test_summary.png"
    )

    create_summary_figure(
        sample_metrics,
        localization_metrics,
        retention_metrics,
        summary_path,
        metadata,
        args.dpi,
        args.show,
    )

    if summary_path.exists():
        generated_files.append(summary_path)

    # ---------------------------------------------------------------
    # Deduplicate generated paths
    # ---------------------------------------------------------------

    generated_files = list(
        dict.fromkeys(
            path.resolve()
            for path in generated_files
            if path.exists()
        )
    )

    write_visualization_summary(
        output_dir,
        metadata,
        generated_files,
        sample_metrics,
    )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("Visualization complete")
    print("=" * 72)
    print(f"Figures generated: {len(generated_files)}")
    print(f"Output directory : {output_dir}")
    print()

    for path in generated_files:
        print(f"  {path.relative_to(output_dir)}")


if __name__ == "__main__":
    main()