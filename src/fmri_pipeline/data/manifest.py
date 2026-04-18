from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from fmri_pipeline.utils.parsing import parse_sample_path


MANIFEST_COLUMNS = [
    "sample_id",
    "filepath",
    "class_name",
    "class_id",
    "subject_id",
    "run_id",
    "vol_id",
    "batch_id",
    "exists",
]


def _expand_roots(data_roots: Sequence[str | Path]) -> List[Path]:
    roots: List[Path] = []
    for r in data_roots:
        p = Path(r).expanduser().resolve()
        if not p.exists():
            continue
        roots.append(p)
    return roots


def _resolve_class_dir(root: Path, class_name: str) -> Path | None:
    direct = root / class_name
    if direct.exists():
        return direct

    candidates = [
        p
        for p in root.rglob(class_name)
        if p.is_dir()
    ]
    if not candidates:
        return None

    # Prefer the shallowest class directory when dataset roots include an extra wrapper folder.
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def _iter_nifti_files(class_dir: Path) -> Iterable[Path]:
    files = list(class_dir.glob("*.nii")) + list(class_dir.glob("*.nii.gz"))
    return sorted(files)


def build_manifest(
    data_roots: Sequence[str | Path],
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    roots = _expand_roots(data_roots)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    rows = []
    parse_failures: List[str] = []
    missing_class_dirs: List[str] = []

    for root in roots:
        for class_name in class_names:
            class_dir = _resolve_class_dir(root, class_name)
            if class_dir is None:
                missing_class_dirs.append(str(root / class_name))
                continue

            for nifti_path in _iter_nifti_files(class_dir):
                try:
                    parsed = parse_sample_path(nifti_path, class_name=class_name)
                except ValueError:
                    parse_failures.append(str(nifti_path))
                    continue

                rows.append(
                    {
                        "filepath": parsed.filepath,
                        "class_name": parsed.class_name,
                        "class_id": class_to_id[parsed.class_name],
                        "subject_id": parsed.subject_id,
                        "run_id": parsed.run_id,
                        "vol_id": parsed.vol_id,
                        "batch_id": parsed.batch_id,
                        "exists": Path(parsed.filepath).exists(),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        qc = {
            "num_samples": 0,
            "num_subjects": 0,
            "num_runs": 0,
            "class_counts": {},
            "roots": [str(r) for r in roots],
            "missing_class_dirs": missing_class_dirs,
            "parse_failures": parse_failures,
            "parse_failure_count": len(parse_failures),
        }
        return df, qc

    df = df.sort_values(["subject_id", "run_id", "vol_id", "class_id"]).reset_index(drop=True)
    df.insert(0, "sample_id", df.index.astype(int))

    qc = {
        "num_samples": int(len(df)),
        "num_subjects": int(df["subject_id"].nunique()),
        "num_runs": int(df[["subject_id", "run_id"]].drop_duplicates().shape[0]),
        "class_counts": {
            str(k): int(v)
            for k, v in df["class_name"].value_counts().sort_index().to_dict().items()
        },
        "roots": [str(r) for r in roots],
        "missing_class_dirs": missing_class_dirs,
        "parse_failures": parse_failures,
        "parse_failure_count": len(parse_failures),
    }

    return df[MANIFEST_COLUMNS], qc


def save_manifest_parquet(df: pd.DataFrame, out_index: str | Path) -> None:
    p = Path(out_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_manifest(index_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(index_path)
    missing_cols = set(MANIFEST_COLUMNS).difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Manifest missing expected columns: {sorted(missing_cols)}"
        )
    return df
