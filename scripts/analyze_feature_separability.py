#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]

FILENAME_PATTERNS = [
    re.compile(r"^(?P<subject_id>sub-\d+)_run-(?P<run_id>\d+)_vol-(?P<vol_id>\d+)\.nii(?:\.gz)?$"),
    re.compile(r"^volume_(?P<subject_prefix>sub)_(?P<subject_num>\d+)_run_(?P<run_id>\d+)_(?P<vol_id>\d+)\.nii(?:\.gz)?$"),
]


def parse_file(path: Path) -> Tuple[str, int, int]:
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(path.name)
        if not match:
            continue
        data = match.groupdict()
        if data.get("subject_id") is not None:
            subject_id = data["subject_id"]
        else:
            subject_id = f"{data['subject_prefix']}-{int(data['subject_num']):02d}"
        return subject_id, int(data["run_id"]), int(data["vol_id"])
    raise ValueError(f"Cannot parse NIfTI filename: {path.name}")


def find_dataset_root(slug: str) -> Path | None:
    direct = Path("/kaggle/input") / slug
    if direct.exists():
        return direct
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        matches = sorted(p for p in datasets_root.rglob(slug) if p.is_dir())
        if matches:
            return matches[0]
    matches = sorted(p for p in Path("/kaggle/input").rglob(slug) if p.is_dir())
    return matches[0] if matches else None


def iter_class_files(batch_roots: Iterable[Path]) -> Iterable[Tuple[str, Path]]:
    for root in batch_roots:
        for class_name in CLASS_NAMES:
            candidates = [root / class_name]
            candidates.extend(p for p in root.rglob(class_name) if p.is_dir())
            class_dirs = sorted({p.resolve() for p in candidates if p.exists()})
            for class_dir in class_dirs[:1]:
                for path in sorted(class_dir.glob("*.nii.gz")) + sorted(class_dir.glob("*.nii")):
                    yield class_name, path


def load_index(batch_slugs: List[str]) -> Dict[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]]:
    roots: List[Path] = []
    missing: List[str] = []
    for slug in batch_slugs:
        root = find_dataset_root(slug)
        if root is None:
            missing.append(slug)
        else:
            roots.append(root)
    if missing:
        raise FileNotFoundError(f"Missing mounted batch datasets: {missing}")

    index: Dict[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]] = defaultdict(lambda: defaultdict(list))
    for class_name, path in iter_class_files(roots):
        subject_id, run_id, vol_id = parse_file(path)
        index[(subject_id, run_id)][class_name].append((vol_id, path))
    return index


def select_blocks(
    index: Dict[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]],
    subject_id: str | None,
    run_id: int | None,
    max_blocks: int,
    volumes_per_class: int,
) -> List[Tuple[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]]]:
    keys = sorted(index.keys())
    if subject_id is not None:
        keys = [key for key in keys if key[0] == subject_id]
    if run_id is not None:
        keys = [key for key in keys if key[1] == run_id]

    selected: List[Tuple[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]]] = []
    for key in keys:
        block = index[key]
        if all(len(block.get(class_name, [])) >= volumes_per_class for class_name in CLASS_NAMES):
            selected.append(
                (key, {class_name: sorted(block[class_name])[:volumes_per_class] for class_name in CLASS_NAMES})
            )
        if len(selected) >= max_blocks:
            break
    if not selected:
        raise ValueError(f"No complete blocks found with {volumes_per_class} volumes per class")
    return selected


def raw_volume(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape}: {path}")
    return data.astype(np.float32, copy=False)


def normalize_and_resize(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(data.shape) != target_shape:
        factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, factors, order=1)
    mean = float(data.mean())
    std = float(data.std())
    return ((data - mean) / max(std, 1e-6)).astype(np.float32, copy=False)


def load_samples(
    blocks: List[Tuple[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]]],
    target_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    records: List[Dict[str, object]] = []
    for block_id, ((subject_id, run_id), block) in enumerate(blocks):
        for class_id, class_name in enumerate(CLASS_NAMES):
            for vol_id, path in block[class_name]:
                raw = raw_volume(path)
                xs.append(normalize_and_resize(raw, target_shape).reshape(-1))
                ys.append(class_id)
                records.append(
                    {
                        "block_id": block_id,
                        "subject_id": subject_id,
                        "run_id": int(run_id),
                        "class_id": class_id,
                        "class_name": class_name,
                        "vol_id": int(vol_id),
                        "path": str(path),
                        "raw_mean": float(raw.mean()),
                        "raw_std": float(raw.std()),
                        "raw_min": float(raw.min()),
                        "raw_max": float(raw.max()),
                    }
                )
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64), records


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def nearest_centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    centroids = np.stack([train_x[train_y == class_id].mean(axis=0) for class_id in range(len(CLASS_NAMES))], axis=0)
    train_centroids = unit_rows(centroids)
    test_unit = unit_rows(test_x)
    return (test_unit @ train_centroids.T).argmax(axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def leave_one_out_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    preds: List[int] = []
    for idx in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        preds.append(int(nearest_centroid_predict(x[mask], y[mask], x[idx : idx + 1])[0]))
    return accuracy(y, np.asarray(preds, dtype=np.int64))


def within_block_template_results(x: np.ndarray, y: np.ndarray, records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    block_ids = sorted({int(record["block_id"]) for record in records})
    for block_id in block_ids:
        idx = np.asarray([int(record["block_id"]) == block_id for record in records], dtype=bool)
        block_records = [record for record in records if int(record["block_id"]) == block_id]
        block_y = y[idx]
        block_x = x[idx]
        rows.append(
            {
                "block_id": block_id,
                "subject_id": block_records[0]["subject_id"],
                "run_id": int(block_records[0]["run_id"]),
                "num_samples": int(block_y.size),
                "within_block_loo_template_accuracy": leave_one_out_accuracy(block_x, block_y),
            }
        )
    return rows


def leave_one_block_out_accuracy(x: np.ndarray, y: np.ndarray, records: List[Dict[str, object]]) -> float:
    preds: List[int] = []
    truths: List[int] = []
    block_ids = sorted({int(record["block_id"]) for record in records})
    for block_id in block_ids:
        test_idx = np.asarray([int(record["block_id"]) == block_id for record in records], dtype=bool)
        train_idx = ~test_idx
        if len(set(y[train_idx].tolist())) < len(CLASS_NAMES):
            continue
        preds.extend(nearest_centroid_predict(x[train_idx], y[train_idx], x[test_idx]).tolist())
        truths.extend(y[test_idx].tolist())
    return accuracy(np.asarray(truths, dtype=np.int64), np.asarray(preds, dtype=np.int64))


def raw_global_feature_accuracy(records: List[Dict[str, object]], y: np.ndarray) -> float:
    raw = np.asarray([[record["raw_mean"], record["raw_std"]] for record in records], dtype=np.float32)
    raw = (raw - raw.mean(axis=0, keepdims=True)) / np.maximum(raw.std(axis=0, keepdims=True), 1e-8)
    return leave_one_out_accuracy(raw, y)


def class_feature_summary(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_records = [record for record in records if int(record["class_id"]) == class_id]
        for feature_name in ["raw_mean", "raw_std", "raw_min", "raw_max"]:
            values = np.asarray([float(record[feature_name]) for record in class_records], dtype=np.float64)
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": feature_name,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return rows


def pairwise_template_distances(x: np.ndarray, y: np.ndarray) -> List[Dict[str, object]]:
    centroids = np.stack([x[y == class_id].mean(axis=0) for class_id in range(len(CLASS_NAMES))], axis=0)
    centroids = unit_rows(centroids)
    rows: List[Dict[str, object]] = []
    for left in range(len(CLASS_NAMES)):
        for right in range(left + 1, len(CLASS_NAMES)):
            cosine_similarity = float(centroids[left] @ centroids[right])
            rows.append(
                {
                    "class_a": CLASS_NAMES[left],
                    "class_b": CLASS_NAMES[right],
                    "cosine_similarity": cosine_similarity,
                    "cosine_distance": float(1.0 - cosine_similarity),
                }
            )
    return rows


def top_anova_voxels(
    x: np.ndarray,
    y: np.ndarray,
    target_shape: Tuple[int, int, int],
    top_n: int,
) -> List[Dict[str, object]]:
    class_masks = [y == class_id for class_id in range(len(CLASS_NAMES))]
    class_counts = np.asarray([mask.sum() for mask in class_masks], dtype=np.float32)
    class_means = np.stack([x[mask].mean(axis=0) for mask in class_masks], axis=0)
    overall_mean = x.mean(axis=0)

    between = ((class_means - overall_mean[None, :]) ** 2 * class_counts[:, None]).sum(axis=0) / (len(CLASS_NAMES) - 1)
    within = np.zeros_like(between)
    for class_id, mask in enumerate(class_masks):
        within += ((x[mask] - class_means[class_id][None, :]) ** 2).sum(axis=0)
    within /= max(int(y.size) - len(CLASS_NAMES), 1)
    f_scores = between / np.maximum(within, 1e-8)

    top_indices = np.argsort(f_scores)[-top_n:][::-1]
    rows: List[Dict[str, object]] = []
    for rank, flat_idx in enumerate(top_indices, start=1):
        coord = np.unravel_index(int(flat_idx), target_shape)
        means = class_means[:, flat_idx]
        rows.append(
            {
                "rank": rank,
                "x": int(coord[0]),
                "y": int(coord[1]),
                "z": int(coord[2]),
                "f_score": float(f_scores[flat_idx]),
                "max_mean_class": CLASS_NAMES[int(means.argmax())],
                "min_mean_class": CLASS_NAMES[int(means.argmin())],
                "mean_range": float(means.max() - means.min()),
                **{f"mean_{class_name}": float(means[class_id]) for class_id, class_name in enumerate(CLASS_NAMES)},
            }
        )
    return rows


def confusion_from_leave_one_out(x: np.ndarray, y: np.ndarray) -> List[List[int]]:
    preds: List[int] = []
    for idx in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        preds.append(int(nearest_centroid_predict(x[mask], y[mask], x[idx : idx + 1])[0]))
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for truth, pred in zip(y.tolist(), preds):
        matrix[int(truth), int(pred)] += 1
    return matrix.tolist()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Dict[str, object]) -> None:
    text = f"""# Feature Separability Diagnostic

## Dataset Slice

- Batch slugs: `{', '.join(summary['batch_slugs'])}`
- Blocks analyzed: `{summary['num_blocks']}`
- Samples analyzed: `{summary['num_samples']}`
- Target shape: `{summary['target_shape']}`
- Volumes per class per block: `{summary['volumes_per_class']}`

## Main Readout

- Raw global mean/std nearest-centroid accuracy: `{summary['raw_global_feature_accuracy']:.4f}`
- Within-block spatial-template accuracy: `{summary['within_block_template_accuracy_mean']:.4f}` mean
- Leave-one-block-out spatial-template accuracy: `{summary['leave_one_block_out_template_accuracy']:.4f}`

## Interpretation

{summary['interpretation']}
"""
    path.write_text(text)


def interpret(raw_acc: float, within_acc: float, cross_block_acc: float) -> str:
    notes: List[str] = []
    if raw_acc >= 0.50:
        notes.append("Raw global intensity features are suspiciously predictive, so scanner/run artifacts may be contributing.")
    else:
        notes.append("Raw global intensity features are not enough by themselves, which is good: the signal is not just mean/std intensity.")

    if within_acc >= 0.75:
        notes.append("Spatial class templates are strong within the same subject-run.")
    elif within_acc >= 0.45:
        notes.append("Spatial class templates are present but modest within the same subject-run.")
    else:
        notes.append("Spatial class templates are weak even within a run.")

    if cross_block_acc >= 0.50:
        notes.append("Some class template structure transfers across blocks, so a corrected baseline may be worth trying.")
    elif within_acc >= 0.75:
        notes.append("The signal appears mostly run-specific; focus next on temporal/event alignment, run normalization, and subject-aware evaluation.")
    else:
        notes.append("Cross-block transfer is weak; audit event windows and preprocessing before running larger models.")
    return " ".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze class-separating features in extracted fMRI volumes.")
    parser.add_argument("--batch-slugs", nargs="+", default=["thesis-batch-01"])
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=12)
    parser.add_argument("--volumes-per-class", type=int, default=16)
    parser.add_argument("--target-shape", nargs=3, type=int, default=[16, 16, 16])
    parser.add_argument("--top-voxels", type=int, default=100)
    parser.add_argument("--out-dir", default="/kaggle/working/feature_separability")
    args = parser.parse_args()

    target_shape = tuple(args.target_shape)
    index = load_index(args.batch_slugs)
    blocks = select_blocks(index, args.subject_id, args.run_id, args.max_blocks, args.volumes_per_class)
    print(f"selected_blocks={[(key[0], key[1]) for key, _ in blocks]}", flush=True)

    x, y, records = load_samples(blocks, target_shape)
    class_counts = Counter(int(value) for value in y.tolist())
    print(f"loaded shape={x.shape} class_counts={class_counts}", flush=True)

    raw_acc = raw_global_feature_accuracy(records, y)
    within_rows = within_block_template_results(x, y, records)
    within_acc = float(np.mean([row["within_block_loo_template_accuracy"] for row in within_rows]))
    cross_block_acc = leave_one_block_out_accuracy(x, y, records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sample_records.csv", records)
    write_csv(out_dir / "class_global_feature_summary.csv", class_feature_summary(records))
    write_csv(out_dir / "within_block_template_results.csv", within_rows)
    write_csv(out_dir / "pairwise_template_distances.csv", pairwise_template_distances(x, y))
    write_csv(out_dir / "top_anova_voxels.csv", top_anova_voxels(x, y, target_shape, args.top_voxels))

    summary = {
        "batch_slugs": args.batch_slugs,
        "selected_blocks": [{"subject_id": key[0], "run_id": int(key[1])} for key, _ in blocks],
        "num_blocks": len(blocks),
        "num_samples": int(y.size),
        "target_shape": list(target_shape),
        "volumes_per_class": int(args.volumes_per_class),
        "class_counts": {CLASS_NAMES[class_id]: int(class_counts[class_id]) for class_id in range(len(CLASS_NAMES))},
        "raw_global_feature_accuracy": raw_acc,
        "within_block_template_accuracy_mean": within_acc,
        "within_block_template_accuracy_min": float(min(row["within_block_loo_template_accuracy"] for row in within_rows)),
        "within_block_template_accuracy_max": float(max(row["within_block_loo_template_accuracy"] for row in within_rows)),
        "leave_one_block_out_template_accuracy": cross_block_acc,
        "within_block_loo_confusion": confusion_from_leave_one_out(x, y),
        "interpretation": interpret(raw_acc, within_acc, cross_block_acc),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(out_dir / "feature_separability_report.md", summary)
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
