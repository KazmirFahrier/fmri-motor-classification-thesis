from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


NIFTI_PATTERNS = [
    re.compile(
        r"^(?P<subject_id>sub-\d+)_run-(?P<run_id>\d+)_vol-(?P<vol_id>\d+)\.nii(?:\.gz)?$"
    ),
    re.compile(
        r"^volume_(?P<subject_prefix>sub)_(?P<subject_num>\d+)_run_(?P<run_id>\d+)_(?P<vol_id>\d+)\.nii(?:\.gz)?$"
    ),
]

THESIS_BATCH_PATTERN = re.compile(r"^thesis-batch-(?P<batch_num>\d+)$")
BATCH_PATTERN = re.compile(r"^batch[_-](?P<batch_num>\d+)$")


@dataclass(frozen=True)
class ParsedSample:
    filepath: str
    class_name: str
    subject_id: str
    run_id: int
    vol_id: int
    batch_id: str


def find_batch_id(path: Path) -> str:
    for part in path.parts:
        match = THESIS_BATCH_PATTERN.match(part)
        if match:
            return f"batch_{int(match.group('batch_num')):02d}"

        match = BATCH_PATTERN.match(part)
        if match:
            return f"batch_{int(match.group('batch_num')):02d}"

    return "unknown_batch"


def parse_sample_path(path: str | Path, class_name: Optional[str] = None) -> ParsedSample:
    p = Path(path)
    inferred_class = class_name if class_name else p.parent.name

    subject_id = None
    run_id = None
    vol_id = None

    for pattern in NIFTI_PATTERNS:
        m = pattern.match(p.name)
        if not m:
            continue

        if "subject_id" in m.groupdict():
            subject_id = m.group("subject_id")
        else:
            subject_id = f"{m.group('subject_prefix')}-{int(m.group('subject_num')):02d}"

        run_id = int(m.group("run_id"))
        vol_id = int(m.group("vol_id"))
        break

    if subject_id is None or run_id is None or vol_id is None:
        raise ValueError(f"Filename does not match expected pattern: {p.name}")

    return ParsedSample(
        filepath=str(p),
        class_name=inferred_class,
        subject_id=subject_id,
        run_id=run_id,
        vol_id=vol_id,
        batch_id=find_batch_id(p),
    )
