"""Selection aware within run permutation null for the final preprocessing search.

The smoke version uses two permutations. After its observed result reconstructs the
corresponding six folds of the joint nested confirmation, only PERMUTATIONS changes to
200. Every draw repeats inner candidate selection and outer model fitting.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.svm import SVC


protocol_paths = list(Path("/kaggle/input").rglob("protocol.py"))
if len(protocol_paths) != 1:
    raise RuntimeError(f"Expected one mounted protocol.py, found {protocol_paths}")
sys.path.insert(0, str(protocol_paths[0].parent))

from protocol import (  # noqa: E402
    balanced_accuracy,
    inner_splits,
    load_and_preprocess,
    outer_splits,
    run_keys,
    standardize,
)


PERMUTATIONS = 2
PERMUTATION_SEED = 20260824
C_VALUE = 1e-4

SOURCE_FAMILIES = [
    {
        "grid": 24,
        "entries": [
            (sigma, f"sigma{str(sigma).replace('.', 'p')}_offset_3_length_8_sequence")
            for sigma in [0.0, 0.7, 1.1, 1.4]
        ],
        "inputs": [f"/kaggle/input/thesis-nativesmooth-shard{s}" for s in range(3)],
        "path_filter": "nativesmooth",
    },
    {
        "grid": 32,
        "entries": [
            (sigma, f"sigma{str(sigma).replace('.', 'p')}_offset_3_length_8_sequence")
            for sigma in [0.0, 1.4, 1.8, 2.2]
        ],
        "inputs": [f"/kaggle/input/thesis-ns32-shard{s}" for s in range(3)],
        "path_filter": "ns32",
    },
    {
        "grid": 48,
        "entries": [(0.0, "grid48_offset_3_length_8_sequence")],
        "inputs": [f"/kaggle/input/thesis-grid48-shard{s}" for s in range(3)],
        "path_filter": "grid48",
    },
]


def candidate_name(grid: int, sigma: float) -> str:
    return f"grid{grid}_sigma{sigma}"


def permuted_labels(
    labels: np.ndarray,
    records: list[dict],
    permutations: int,
    seed: int,
) -> list[np.ndarray]:
    groups = []
    keys = run_keys(records)
    for key in sorted(set(keys.tolist())):
        groups.append(np.flatnonzero(keys == key))
    rng = np.random.default_rng(seed)
    draws = [labels.copy()]
    for _ in range(permutations):
        shuffled = labels.copy()
        for indices in groups:
            shuffled[indices] = rng.permutation(labels[indices])
        draws.append(shuffled)
    return draws


def standardized_kernels(
    block: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = standardize(block, train_idx)
    train = ((block[train_idx] - mean) / scale).astype(np.float64)
    evaluate = ((block[eval_idx] - mean) / scale).astype(np.float64)
    train_kernel = train @ train.T
    eval_kernel = evaluate @ train.T
    return train_kernel, eval_kernel


def predict(
    train_kernel: np.ndarray,
    eval_kernel: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    model = SVC(
        C=C_VALUE,
        kernel="precomputed",
        decision_function_shape="ovr",
        random_state=0,
        cache_size=6000,
    )
    model.fit(train_kernel, train_labels)
    scores = model.decision_function(eval_kernel)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return scores.argmax(axis=1).astype(np.int64)


blocks: dict[str, np.ndarray] = {}
candidates: list[str] = []
y = None
records = None

for family in SOURCE_FAMILIES:
    grid = family["grid"]
    for sigma, key in family["entries"]:
        name = candidate_name(grid, sigma)
        block, labels, loaded_records = load_and_preprocess(
            family["inputs"],
            key,
            expect_features=grid ** 3,
            path_filter=family["path_filter"],
        )
        if y is None:
            y = labels
            records = loaded_records
        else:
            if not np.array_equal(labels, y) or loaded_records != records:
                raise RuntimeError(f"{name}: labels or records differ")
        blocks[name] = block
        candidates.append(name)
        print(f"{name}: {block.shape}", flush=True)

assert y is not None and records is not None
class_count = int(y.max()) + 1
label_draws = permuted_labels(y, records, PERMUTATIONS, PERMUTATION_SEED)

# A single prespecified six fold partition keeps the exact null computationally
# tractable. The main point estimate remains the separate 30 fold confirmation.
splits = outer_splits(records, seeds=(11,))
outer_scores = np.full((len(label_draws), len(splits)), np.nan, dtype=np.float64)
selections: list[list[str]] = [[""] * len(splits) for _ in label_draws]

started = time.time()
for fold_position, split in enumerate(splits):
    print(f"\n{split['split']}", flush=True)
    inner = inner_splits(records, split["train_idx"])
    inner_scores = np.zeros((len(label_draws), len(candidates)), dtype=np.float64)

    for candidate_position, name in enumerate(candidates):
        candidate_started = time.time()
        for inner_split in inner:
            train_kernel, eval_kernel = standardized_kernels(
                blocks[name], inner_split["train_idx"], inner_split["val_idx"]
            )
            for draw_position, draw_labels in enumerate(label_draws):
                prediction = predict(
                    train_kernel,
                    eval_kernel,
                    draw_labels[inner_split["train_idx"]],
                )
                inner_scores[draw_position, candidate_position] += balanced_accuracy(
                    draw_labels[inner_split["val_idx"]], prediction, class_count
                ) / len(inner)
            del train_kernel, eval_kernel
        print(
            f"  inner {name}: {time.time() - candidate_started:.1f}s",
            flush=True,
        )

    selected_positions = inner_scores.argmax(axis=1)
    for draw_position, candidate_position in enumerate(selected_positions):
        selections[draw_position][fold_position] = candidates[int(candidate_position)]

    for candidate_position, name in enumerate(candidates):
        draw_positions = np.flatnonzero(selected_positions == candidate_position)
        if not len(draw_positions):
            continue
        train_kernel, eval_kernel = standardized_kernels(
            blocks[name], split["train_idx"], split["val_idx"]
        )
        for draw_position in draw_positions:
            draw_labels = label_draws[int(draw_position)]
            prediction = predict(
                train_kernel,
                eval_kernel,
                draw_labels[split["train_idx"]],
            )
            outer_scores[draw_position, fold_position] = balanced_accuracy(
                draw_labels[split["val_idx"]], prediction, class_count
            )
        del train_kernel, eval_kernel

    if not np.isfinite(outer_scores[:, fold_position]).all():
        raise RuntimeError(f"Missing outer scores in {split['split']}")
    print(
        f"  observed selected {selections[0][fold_position]} "
        f"score {outer_scores[0, fold_position]:.4f}",
        flush=True,
    )

observed = float(outer_scores[0].mean())
null_scores = outer_scores[1:].mean(axis=1)
p_value = float((1 + np.sum(null_scores >= observed)) / (PERMUTATIONS + 1))

payload = {
    "status": "smoke" if PERMUTATIONS < 200 else "complete",
    "protocol": {
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "shuffle_unit": "within subject run",
        "outer_folds": len(splits),
        "outer_seed": 11,
        "inner_subject_folds": 4,
        "selection_metric": "inner independent balanced accuracy",
        "candidates": candidates,
        "classifier": "linear SVC with precomputed exact linear kernel",
        "c_value": C_VALUE,
    },
    "observed": {
        "balanced_accuracy": observed,
        "fold_scores": outer_scores[0].tolist(),
        "selected_candidates": selections[0],
        "selected_counts": dict(Counter(selections[0])),
    },
    "null": {
        "scores": null_scores.tolist(),
        "mean": float(null_scores.mean()),
        "standard_deviation": float(null_scores.std()),
        "maximum": float(null_scores.max()),
        "p_value": p_value,
        "selected_candidates_by_draw": selections[1:],
    },
    "runtime_seconds": float(time.time() - started),
}

output_path = Path("/kaggle/working/exact_pipeline_null.json")
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
