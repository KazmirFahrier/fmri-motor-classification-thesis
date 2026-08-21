# --- frozen protocol, inlined for Kaggle -------------------------------------------
# Kaggle kernels cannot import from this repository, so the pieces of the frozen
# protocol that every analysis depends on are reproduced here verbatim in behaviour:
# checkpoint ordering, subject-wise fold construction, subject-run centering, per-lag
# detrending, train-only standardization, and the design-constrained assignment rule.
#
# Ordering matters. load_checkpoints sorts by filename, and the fold assignment is a
# seeded shuffle of the sorted subject list, so any change in ordering silently changes
# every fold. With sharded extraction the subjects arrive split across several mounted
# directories, so they are gathered and re-sorted by subject id rather than by path.
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # noqa: BLE001
    linear_sum_assignment = None


def load_sharded(input_dirs, keys):
    """Load subject checkpoints spread across mounted kernel outputs."""
    paths = _gather(input_dirs)
    features = {k: [] for k in keys}
    labels, records = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for k in keys:
                features[k].append(data[k].astype(np.float32))
            labels.append(data["labels"].astype(np.int64))
            records.extend(json.loads(str(data["records_json"])))
    print(f"loaded {len(paths)} subjects", flush=True)
    return ({k: np.concatenate(v, axis=0) for k, v in features.items()},
            np.concatenate(labels, axis=0), records)


def _gather(input_dirs, path_filter=None):
    """Find subject checkpoints under mounted kernel outputs.

    Kaggle does not necessarily mount a chained kernel's output at
    /kaggle/input/<slug>/, so rather than assume a layout this searches the given
    directories and then falls back to scanning /kaggle/input entirely. On failure it
    prints what is actually mounted, because a wrong path assumption is otherwise
    indistinguishable from an empty upstream.
    """
    paths = []
    for directory in input_dirs:
        if Path(directory).exists():
            paths.extend(Path(directory).rglob("sub-*.npz"))
    if not paths:
        root = Path("/kaggle/input")
        if root.exists():
            paths = list(root.rglob("sub-*.npz"))
    if path_filter is not None:
        # Several extractions can be mounted at once, and each holds the same subject
        # ids under different keys. Without a filter a subject would be loaded twice and
        # the record list would no longer line up with the feature rows.
        paths = [p for p in paths if path_filter in str(p)]
    if not paths:
        root = Path("/kaggle/input")
        listing = sorted(str(p) for p in root.rglob("*"))[:40] if root.exists() else []
        raise ValueError(
            f"No sub-*.npz found. Tried {input_dirs} (filter={path_filter}) then a full "
            f"scan of /kaggle/input. Mounted contents (first 40): {listing}")
    return sorted(paths, key=lambda p: p.name)


def load_and_preprocess(input_dirs, key, expect_features=None, path_filter=None):
    """Load, preprocess and reduce to the lag-mean block one subject at a time.

    Both preprocessing steps group by subject-run, so they are entirely within-subject
    and can be applied as each file is read. Only the lag-mean survives, which is what
    every decoder here consumes.

    That matters at large grids. Holding the full event x lag x feature array for a
    48^3 grid would need about 10 GB before any copy; reducing per subject holds one
    subject's lags at a time and the final block is a single lag-mean.
    """
    paths = _gather(input_dirs, path_filter)
    blocks, labels, records = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            sequence = data[key].astype(np.float32)
            subject_records = json.loads(str(data["records_json"]))
            labels.append(data["labels"].astype(np.int64))
        if expect_features is not None and sequence.shape[2] != expect_features:
            raise ValueError(
                f"{path.name}: expected {expect_features} features, "
                f"got {sequence.shape[2]}")
        blocks.append(
            preprocess_sequence(sequence, subject_records).mean(axis=1, dtype=np.float32))
        records.extend(subject_records)
        del sequence
    print(f"loaded and preprocessed {len(paths)} subjects", flush=True)
    return (np.concatenate(blocks, axis=0), np.concatenate(labels, axis=0), records)


def outer_splits(records, fold_count=6, seeds=(11, 23, 37, 51, 71)):
    all_idx = np.arange(len(records), dtype=np.int64)
    subjects = np.asarray([str(r["subject_id"]) for r in records])
    subject_list = np.asarray(sorted(set(subjects.tolist())))
    splits = []
    for seed in seeds:
        shuffled = subject_list.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        for fold_idx in range(fold_count):
            held_out = shuffled[fold_idx::fold_count].tolist()
            val_mask = np.isin(subjects, held_out)
            splits.append({"split": f"subject_seed_{seed}_fold_{fold_idx}",
                           "train_idx": all_idx[~val_mask], "val_idx": all_idx[val_mask]})
    return splits


def inner_splits(records, train_idx, fold_count=4):
    subjects = np.asarray([str(records[int(i)]["subject_id"]) for i in train_idx])
    subject_list = np.asarray(sorted(set(subjects.tolist())))
    out = []
    for fold_idx in range(fold_count):
        held_out = subject_list[fold_idx::fold_count].tolist()
        mask = np.isin(subjects, held_out)
        out.append({"train_idx": train_idx[~mask], "val_idx": train_idx[mask]})
    return out


def run_keys(records):
    return np.asarray([f'{r["subject_id"]}|run-{int(r["run_id"])}' for r in records])


def center_by_subject_run(x, records):
    keys = run_keys(records)
    out = x.copy()
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        out[mask] -= out[mask].mean(axis=0)
    return out


def temporal_detrend_by_subject_run(x_centered, records, degree=1):
    if degree < 1:
        return x_centered.copy()
    keys = run_keys(records)
    out = x_centered.astype(np.float64)
    for key in sorted(set(keys.tolist())):
        indices = np.flatnonzero(keys == key)
        times = np.asarray([float(records[i]["event_start"]) for i in indices],
                           dtype=np.float64)
        times -= times.mean()
        times /= max(float(np.std(times)), 1e-8)
        design = np.stack([times ** p for p in range(1, degree + 1)], axis=1)
        design -= design.mean(axis=0, keepdims=True)
        q, _ = np.linalg.qr(design)
        group_x = out[indices]
        out[indices] = group_x - q @ (q.T @ group_x)
    return out.astype(np.float32)


def preprocess_sequence(sequence, records):
    for lag in range(sequence.shape[1]):
        sequence[:, lag] = temporal_detrend_by_subject_run(
            center_by_subject_run(sequence[:, lag], records), records, 1)
    return sequence


def standardize(x, train_idx):
    mean = x[train_idx].mean(axis=0, dtype=np.float64)
    scale = x[train_idx].std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def mean_smooth(x, shape, kernel_size, batch_size=256):
    if kernel_size <= 1:
        return x
    radius = kernel_size // 2
    result = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        volumes = x[start:stop].reshape((-1, *shape))
        padded = np.pad(volumes, ((0, 0), (radius, radius), (radius, radius),
                                  (radius, radius)), mode="reflect")
        smoothed = np.zeros_like(volumes, dtype=np.float32)
        for dx in range(kernel_size):
            for dy in range(kernel_size):
                for dz in range(kernel_size):
                    smoothed += padded[:, dx:dx + shape[0], dy:dy + shape[1],
                                       dz:dz + shape[2]]
        result[start:stop] = (smoothed / float(kernel_size ** 3)).reshape(stop - start, -1)
    return result


def balanced_assign_group(scores):
    group_size, class_count = scores.shape
    per_class = group_size // class_count
    if group_size % class_count != 0 or linear_sum_assignment is None:
        return scores.argmax(axis=1).astype(np.int64)
    expanded = np.repeat(scores, per_class, axis=1)
    row_idx, col_idx = linear_sum_assignment(-expanded)
    assignment = np.empty(group_size, dtype=np.int64)
    assignment[row_idx] = col_idx // per_class
    return assignment


def apply_balanced_assignment(scores, val_idx, records):
    pred = scores.argmax(axis=1).astype(np.int64)
    grouped = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        r = records[int(record_idx)]
        grouped[f'{r["subject_id"]}|run-{int(r["run_id"])}'].append(local_pos)
    for positions in grouped.values():
        positions = sorted(positions,
                           key=lambda p: records[int(val_idx[p])]["event_start"])
        pred[positions] = balanced_assign_group(scores[positions])
    return pred


def balanced_accuracy(y_true, y_pred, class_count):
    recalls = []
    for c in range(class_count):
        mask = y_true == c
        if int(mask.sum()):
            recalls.append(float((y_pred[mask] == c).sum()) / int(mask.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def svm_scores(block, y, train_idx, eval_idx, c_value=1e-4):
    """Linear SVM via a precomputed kernel.

    For a linear SVM the precomputed kernel is exactly the inner-product matrix, so no
    dual basis or eigendecomposition is needed: kernel_train = X X^T directly.
    """
    from sklearn.svm import SVC
    mean, scale = standardize(block, train_idx)
    x_train = ((block[train_idx] - mean) / scale).astype(np.float64)
    x_eval = ((block[eval_idx] - mean) / scale).astype(np.float64)
    model = SVC(C=c_value, kernel="precomputed", decision_function_shape="ovr",
                random_state=0)
    model.fit(x_train @ x_train.T, y[train_idx])
    return model.decision_function(x_eval @ x_train.T).astype(np.float64)


def evaluate(block, y, records, splits, class_count, c_value=1e-4, label=""):
    out = {"independent": [], "balanced": []}
    for split in splits:
        scores = svm_scores(block, y, split["train_idx"], split["val_idx"], c_value)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        for rule, prediction in (
            ("independent", scores.argmax(axis=1).astype(np.int64)),
            ("balanced", apply_balanced_assignment(scores, split["val_idx"], records)),
        ):
            out[rule].append(balanced_accuracy(y[split["val_idx"]], prediction, class_count))
    if label:
        print(f"  {label}: indep {np.mean(out['independent']):.4f} "
              f"balanced {np.mean(out['balanced']):.4f}", flush=True)
    return out
# --- end protocol ------------------------------------------------------------------

# ====================================================================================
"""Nest the smoothing-sigma choice, and store per-subject rows for later paired tests.

Native-resolution smoothing reached `0.8648` against `0.8098` unsmoothed, but the sigma
was chosen with all 30 folds visible. That is exactly the cohort-visible design search
this project has spent considerable effort measuring, and leaving it unnested would
repeat the error the manuscript already discloses elsewhere.

The four measured points say the cost of nesting tracks how stable the choice is across
folds — 0% surviving for the temporal window whose selections split five ways, 100% for
the box kernel where all 30 folds agreed. Accuracy was still rising at the largest sigma
tested, so the optimum is not bracketed and the selection could be unstable. This finds
out.

Per-subject accuracies are written out as well. The frozen hierarchy's own per-subject
rows were stored outside the repository and have been deleted, so no paired test against
it is currently possible; storing these means that when those rows are regenerated the
comparison can be made without re-running anything here.
"""
import json
import numpy as np
from collections import defaultdict

INPUTS = [f"/kaggle/input/thesis-nativesmooth-shard{s}" for s in range(3)]
SIGMAS = [0.0, 0.7, 1.1, 1.4]
KEY = "sigma{}_offset_3_length_8_sequence"
EXPECTED_SIGMA0 = 0.8098
TOLERANCE = 0.004

blocks, y, records = {}, None, None
for sg in SIGMAS:
    key = KEY.format(str(sg).replace(".", "p"))
    block, labels, recs = load_and_preprocess(INPUTS, key, path_filter="nativesmooth")
    if y is None:
        y, records = labels, recs
    else:
        assert np.array_equal(labels, y), f"sigma {sg}: labels differ"
    blocks[sg] = block
    print(f"sigma {sg}: {block.shape}", flush=True)

class_count = int(y.max()) + 1
splits = outer_splits(records)

fixed = {}
for sg in SIGMAS:
    fixed[sg] = evaluate(blocks[sg], y, records, splits, class_count, label=f"sigma {sg}")
observed = float(np.mean(fixed[0.0]["independent"]))
if abs(observed - EXPECTED_SIGMA0) > TOLERANCE:
    raise SystemExit(f"VALIDATION FAILED: sigma-0 {observed:.4f} != {EXPECTED_SIGMA0}")
print(f"VALIDATION PASSED: sigma-0 {observed:.4f}", flush=True)

nested = {"independent": [], "balanced": []}
selected = []
per_subject = {"independent": defaultdict(list), "balanced": defaultdict(list)}

for split in splits:
    inner = inner_splits(records, split["train_idx"])
    means = {}
    for sg in SIGMAS:
        vals = []
        for isp in inner:
            s = svm_scores(blocks[sg], y, isp["train_idx"], isp["val_idx"])
            vals.append(balanced_accuracy(y[isp["val_idx"]],
                                          s.argmax(axis=1).astype(np.int64), class_count))
        means[sg] = float(np.mean(vals))
    best = max(SIGMAS, key=lambda g: means[g])
    selected.append(best)

    val_idx = split["val_idx"]
    s = svm_scores(blocks[best], y, split["train_idx"], val_idx)
    grouped = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        grouped[str(records[int(record_idx)]["subject_id"])].append(local_pos)
    for rule, prediction in (
        ("independent", s.argmax(axis=1).astype(np.int64)),
        ("balanced", apply_balanced_assignment(s, val_idx, records)),
    ):
        nested[rule].append(balanced_accuracy(y[val_idx], prediction, class_count))
        for subject, positions in grouped.items():
            per_subject[rule][subject].append(balanced_accuracy(
                y[val_idx][positions], prediction[positions], class_count))
    print(f"{split['split']} selected sigma {best} -> "
          f"{nested['independent'][-1]:.4f}", flush=True)

summary = {str(sg): {r: float(np.mean(v)) for r, v in fixed[sg].items()} for sg in SIGMAS}
summary["nested"] = {r: float(np.mean(v)) for r, v in nested.items()}
counts = {str(sg): selected.count(sg) for sg in SIGMAS}
json.dump({
    "sigmas": SIGMAS, "summary": summary, "selected_counts": counts,
    "nested_folds": nested,
    "per_subject_means": {r: {s: float(np.mean(v)) for s, v in d.items()}
                          for r, d in per_subject.items()},
}, open("/kaggle/working/nativesmooth_nested.json", "w"), indent=2)

print("\n  sigma     independent   balanced")
for sg in SIGMAS:
    print(f"  {sg:<9} {summary[str(sg)]['independent']:.4f}        "
          f"{summary[str(sg)]['balanced']:.4f}")
print(f"  nested    {summary['nested']['independent']:.4f}        "
      f"{summary['nested']['balanced']:.4f}")
print(f"selected: {counts}")
best_fixed = max(SIGMAS, key=lambda g: summary[str(g)]["independent"])
print(f"\noracle sigma {best_fixed}: {summary[str(best_fixed)]['independent']:.4f}")
print(f"nested:            {summary['nested']['independent']:.4f}")
print(f"nesting cost:      "
      f"{summary[str(best_fixed)]['independent'] - summary['nested']['independent']:+.4f}")
