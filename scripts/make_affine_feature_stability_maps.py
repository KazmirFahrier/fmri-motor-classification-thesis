#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from run_detrended_pair_feature_selection import (
    PAIR_CLASSES,
    load_checkpoints,
    outer_splits,
    rank_pair_features,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence
from run_spatial_scale_feature_sweep import transform_scale


MAP_NAMES = ("coarse", "leg", "arm")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def feature_grid_affine(
    reference_affine: np.ndarray,
    reference_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
) -> np.ndarray:
    # scipy.ndimage.zoom with grid_mode=False maps the first and last voxel
    # centers onto each other. Preserve that same coordinate convention here.
    reference_extent = np.asarray(reference_shape, dtype=np.float64) - 1.0
    feature_extent = np.asarray(feature_shape, dtype=np.float64) - 1.0
    if np.any(feature_extent <= 0):
        raise ValueError(f"Feature shape must be at least two voxels per axis: {feature_shape}")
    scale = reference_extent / feature_extent
    result = np.asarray(reference_affine, dtype=np.float64).copy()
    result[:3, :3] = reference_affine[:3, :3] @ np.diag(scale)
    return result


def top_locations(
    values: np.ndarray,
    affine: np.ndarray,
    limit: int = 25,
) -> list[dict]:
    order = np.argsort(-values.reshape(-1), kind="stable")
    rows = []
    for index in order[:limit]:
        coordinate = np.asarray(np.unravel_index(int(index), values.shape))
        world = nib.affines.apply_affine(affine, coordinate)
        rows.append(
            {
                "feature_index": int(index),
                "grid_coordinate": coordinate.astype(int).tolist(),
                "reference_world_coordinate_mm": world.tolist(),
                "selection_frequency": float(values.reshape(-1)[index]),
            }
        )
    return rows


def selection_maps(
    checkpoint_dir: Path,
    baseline: dict,
    sequence_key: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict]:
    features, y, records = load_checkpoints(checkpoint_dir, [sequence_key])
    sequence, detrend_rows = preprocess_sequence(features.pop(sequence_key), records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    feature_shape = tuple(int(value) for value in baseline["native_feature_shape"])
    pair_x, pair_shape = transform_scale(
        mean_x, feature_shape, baseline["pair_transform"], batch_size
    )
    hyperparameters = {row["split"]: row for row in baseline["hyperparameters"]}
    splits = outer_splits(
        records,
        "subject",
        6,
        [int(value) for value in baseline["subject_seeds"]],
    )
    counts = {
        "coarse": np.zeros(mean_x.shape[1], dtype=np.int32),
        "leg": np.zeros(pair_x.shape[1], dtype=np.int32),
        "arm": np.zeros(pair_x.shape[1], dtype=np.int32),
    }
    for split in splits:
        hyper = hyperparameters[split["split"]]
        coarse_y = (y >= 2).astype(np.int64)
        coarse_ranking = rank_pair_features(
            mean_x, coarse_y, split["train_idx"], (0, 1)
        )
        coarse_count = int(hyper["selected_coarse_feature_count"])
        counts["coarse"][coarse_ranking[:coarse_count]] += 1
        for pair_name, classes in PAIR_CLASSES.items():
            ranking = rank_pair_features(pair_x, y, split["train_idx"], classes)
            feature_count = int(
                hyper["selected_pair_configurations"][pair_name]["feature_count"]
            )
            counts[pair_name][ranking[:feature_count]] += 1
    split_count = len(splits)
    maps = {
        name: values.reshape(feature_shape if name == "coarse" else pair_shape)
        .astype(np.float32)
        / split_count
        for name, values in counts.items()
    }
    metadata = {
        "model_scope": (
            "Training-fold ranking stability for the frozen conservative mean-window "
            "hierarchy. Temporal-contrast candidates and the repetition-similarity "
            "assignment term are not represented in these spatial maps."
        ),
        "split_count": split_count,
        "subject_seeds": baseline["subject_seeds"],
        "feature_shape": list(feature_shape),
        "pair_shape": list(pair_shape),
        "pair_transform": baseline["pair_transform"],
        "mean_detrended_variance_fraction": float(
            np.mean([row["mean_temporal_variance_fraction"] for row in detrend_rows])
        ),
        "selected_voxel_summary": {
            name: {
                "mean_selection_frequency": float(values.mean()),
                "maximum_selection_frequency": float(values.max()),
                "selected_in_at_least_half_of_folds": int(np.sum(values >= 0.5)),
                "selected_in_at_least_80_percent_of_folds": int(np.sum(values >= 0.8)),
            }
            for name, values in maps.items()
        },
    }
    return maps, metadata


def plot_maps(reference: np.ndarray, maps: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    titles = ("Sagittal max projection", "Coronal max projection", "Axial max projection")
    for row_index, name in enumerate(MAP_NAMES):
        data = maps[name]
        for column_index, axis in enumerate((0, 1, 2)):
            background = np.max(reference, axis=axis).T
            overlay = np.max(data, axis=axis).T
            axes[row_index, column_index].imshow(
                background, cmap="gray", origin="lower", interpolation="nearest"
            )
            image = axes[row_index, column_index].imshow(
                np.ma.masked_less_equal(overlay, 0),
                cmap="inferno",
                origin="lower",
                interpolation="nearest",
                alpha=0.78,
                vmin=0,
                vmax=1,
            )
            axes[row_index, column_index].set_title(
                f"{name.title()} - {titles[column_index]}"
            )
            axes[row_index, column_index].axis("off")
        figure.colorbar(image, ax=axes[row_index, :], fraction=0.018, label="Fold frequency")
    figure.suptitle(
        "Training-fold feature-ranking stability in representative T1w geometry\n"
        "Interpret as model selection frequency, not anatomical causality",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export affine-aware training-fold feature-ranking stability maps."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    baseline_path = Path(args.baseline_json)
    reference_path = Path(args.reference_image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(baseline_path.read_text())
    maps, metadata = selection_maps(
        checkpoint_dir, baseline, args.sequence_key, args.batch_size
    )

    reference_image = nib.load(reference_path)
    reference_shape = tuple(int(value) for value in reference_image.shape[:3])
    feature_shape = tuple(int(value) for value in baseline["native_feature_shape"])
    affine = feature_grid_affine(reference_image.affine, reference_shape, feature_shape)
    reference_volume = np.array(
        reference_image.dataobj[..., 0], dtype=np.float32, copy=True
    )
    reference_volume -= np.nanmin(reference_volume)
    reference_volume /= max(float(np.nanmax(reference_volume)), 1e-8)

    resampled_maps = {}
    outputs = []
    for name, values in maps.items():
        feature_image = nib.Nifti1Image(values, affine)
        feature_path = out_dir / f"{name}_selection_frequency_feature_grid.nii.gz"
        nib.save(feature_image, feature_path)
        resampled = resample_from_to(
            feature_image,
            (reference_shape, reference_image.affine),
            order=1,
        )
        reference_map_path = out_dir / f"{name}_selection_frequency_reference_grid.nii.gz"
        nib.save(resampled, reference_map_path)
        resampled_maps[name] = np.asarray(resampled.dataobj, dtype=np.float32)
        outputs.extend([feature_path, reference_map_path])

    figure_path = out_dir / "feature_selection_stability_reference_geometry.png"
    plot_maps(reference_volume, resampled_maps, figure_path)
    outputs.append(figure_path)
    metadata.update(
        {
            "status": "complete",
            "reference_image": reference_path.name,
            "reference_image_sha256": sha256_file(reference_path),
            "reference_shape": list(reference_shape),
            "reference_affine": reference_image.affine.tolist(),
            "feature_grid_affine": affine.tolist(),
            "top_locations": {
                name: top_locations(values, affine) for name, values in maps.items()
            },
            "outputs": [
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in outputs
            ],
            "interpretation_limit": (
                "The original model resized each subject's T1w-space BOLD grid by array index. "
                "These maps place aggregate feature-ranking frequencies into one representative "
                "subject's T1w affine solely to preserve orientation and physical extent. They "
                "are not group-normalized atlas maps, signed model weights, saliency maps, or "
                "evidence that a location causally generates a class prediction."
            ),
        }
    )
    metadata_path = out_dir / "feature_selection_stability.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                "status": "complete",
                "out_dir": str(out_dir),
                "split_count": metadata["split_count"],
                "selected_voxel_summary": metadata["selected_voxel_summary"],
                "interpretation_limit": metadata["interpretation_limit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
