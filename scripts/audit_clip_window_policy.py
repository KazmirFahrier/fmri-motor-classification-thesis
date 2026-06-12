#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def load_volume_ids(batch_slugs: List[str]) -> Dict[Tuple[str, int, str], List[int]]:
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

    groups: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    for class_name, path in iter_class_files(roots):
        subject_id, run_id, vol_id = parse_file(path)
        groups[(subject_id, run_id, class_name)].append(vol_id)
    return {key: sorted(values) for key, values in groups.items()}


def contiguous_segments(values: List[int]) -> List[List[int]]:
    if not values:
        return []
    segments: List[List[int]] = [[values[0]]]
    for value in values[1:]:
        if value == segments[-1][-1] + 1:
            segments[-1].append(value)
        else:
            segments.append([value])
    return segments


def count_clips(
    values: List[int],
    clip_length: int,
    clip_stride: int,
    clip_window_stride: int,
    hrf_shift: int,
) -> int:
    vol_set = set(values)
    total = 0
    for start_idx in range(0, len(values), clip_window_stride):
        shifted_start = values[start_idx] + hrf_shift
        target = [shifted_start + step * clip_stride for step in range(clip_length)]
        if all(value in vol_set for value in target):
            total += 1
    return total


def summarize_policy(
    groups: Dict[Tuple[str, int, str], List[int]],
    clip_length: int,
    clip_stride: int,
    clip_window_stride: int,
    hrf_shifts: List[int],
    example_limit: int,
) -> Dict[str, object]:
    segment_lengths: Counter[int] = Counter()
    group_lengths: Counter[int] = Counter()
    examples: List[Dict[str, object]] = []
    clip_counts = {str(shift): 0 for shift in hrf_shifts}
    groups_with_clips = {str(shift): 0 for shift in hrf_shifts}

    for key, values in sorted(groups.items()):
        subject_id, run_id, class_name = key
        group_lengths[len(values)] += 1
        segments = contiguous_segments(values)
        for segment in segments:
            segment_lengths[len(segment)] += 1
        per_shift = {}
        for shift in hrf_shifts:
            clips = count_clips(values, clip_length, clip_stride, clip_window_stride, shift)
            per_shift[str(shift)] = clips
            clip_counts[str(shift)] += clips
            if clips > 0:
                groups_with_clips[str(shift)] += 1
        if len(examples) < example_limit:
            examples.append(
                {
                    "subject_id": subject_id,
                    "run_id": int(run_id),
                    "class_name": class_name,
                    "volume_ids": values,
                    "contiguous_segments": segments,
                    "clip_counts_by_hrf_shift": per_shift,
                }
            )

    return {
        "num_subject_run_class_groups": len(groups),
        "group_length_counts": dict(sorted(group_lengths.items())),
        "contiguous_segment_length_counts": dict(sorted(segment_lengths.items())),
        "clip_length": int(clip_length),
        "clip_stride": int(clip_stride),
        "clip_window_stride": int(clip_window_stride),
        "hrf_shifts": hrf_shifts,
        "clip_counts_by_hrf_shift": clip_counts,
        "groups_with_clips_by_hrf_shift": groups_with_clips,
        "examples": examples,
        "interpretation": interpret(segment_lengths, clip_counts, hrf_shifts, clip_length),
    }


def interpret(
    segment_lengths: Counter[int],
    clip_counts: Dict[str, int],
    hrf_shifts: List[int],
    clip_length: int,
) -> str:
    common_segment = segment_lengths.most_common(1)[0][0] if segment_lengths else None
    zero_count = clip_counts.get("0")
    positive_shift_counts = [clip_counts[str(shift)] for shift in hrf_shifts if shift > 0]
    if common_segment == 8 and zero_count is not None and positive_shift_counts:
        if max(positive_shift_counts) < zero_count:
            return (
                "Extracted class folders appear to contain 8-volume event segments. "
                "Applying a positive HRF shift inside these folders reduces clips and cannot include post-event "
                "volumes from the original continuous run. Use hrf_shift=0 for this pre-extracted dataset, or "
                "rebuild extraction from raw 4D runs with the desired HRF shift before class-folder export."
            )
    return (
        "Review clip counts and contiguous segments. If folders already contain cropped event windows, later HRF "
        "shifts should be treated as within-window cropping, not true HRF-aligned extraction."
    )


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Clip Window Policy Audit",
        "",
        f"- Subject-run-class groups: `{summary['num_subject_run_class_groups']}`",
        f"- Group length counts: `{summary['group_length_counts']}`",
        f"- Contiguous segment length counts: `{summary['contiguous_segment_length_counts']}`",
        f"- Clip length: `{summary['clip_length']}`",
        f"- Clip stride: `{summary['clip_stride']}`",
        f"- Clip window stride: `{summary['clip_window_stride']}`",
        f"- Clip counts by HRF shift: `{summary['clip_counts_by_hrf_shift']}`",
        f"- Groups with clips by HRF shift: `{summary['groups_with_clips_by_hrf_shift']}`",
        "",
        "## Interpretation",
        "",
        str(summary["interpretation"]),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit clip construction on pre-extracted class-folder volumes.")
    parser.add_argument("--batch-slugs", nargs="+", default=[f"thesis-batch-{i:02d}" for i in range(1, 8)])
    parser.add_argument("--clip-length", type=int, default=6)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--clip-window-stride", type=int, default=1)
    parser.add_argument("--hrf-shifts", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--example-limit", type=int, default=8)
    parser.add_argument("--out-dir", default="/kaggle/working/clip_window_policy")
    args = parser.parse_args()

    groups = load_volume_ids(args.batch_slugs)
    summary = summarize_policy(
        groups,
        args.clip_length,
        args.clip_stride,
        args.clip_window_stride,
        args.hrf_shifts,
        args.example_limit,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(out_dir / "clip_window_policy_report.md", summary)
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
