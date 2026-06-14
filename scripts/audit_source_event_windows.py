#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import requests


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def raw_github_url(dataset_id: str, version: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/{version}/{path}"


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def load_repetition_time(dataset_id: str, version: str) -> float:
    url = raw_github_url(dataset_id, version, "task-motor_bold.json")
    metadata = json.loads(fetch_text(url))
    return float(metadata["RepetitionTime"])


def extracted_event_starts(records: list[dict]) -> dict[tuple[str, int, str], list[int]]:
    grouped: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for record in records:
        subject = str(record["subject_id"])
        run_id = int(record["run_id"])
        class_name = CLASS_NAMES[int(record["class_id"])]
        vol_start = min(int(vol_id) for vol_id in record["vol_ids"])
        # The source events are 16 s = 8 volume windows. Dense clips start at base/base+1/base+2.
        grouped[(subject, run_id, class_name)].add(vol_start - (vol_start % 8))
    return {key: sorted(value) for key, value in grouped.items()}


def source_event_starts(
    dataset_id: str,
    version: str,
    subject: str,
    run_id: int,
    repetition_time: float,
) -> tuple[dict[str, list[int]], list[dict]]:
    path = f"{subject}/ses-1/func/{subject}_ses-1_task-motor_run-{run_id:02d}_events.tsv"
    text = fetch_text(raw_github_url(dataset_id, version, path))
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    starts: dict[str, list[int]] = defaultdict(list)
    timing_rows = []
    for row in rows:
        trial_type = row["trial_type"]
        onset = float(row["onset"])
        duration = float(row["duration"])
        start_float = onset / repetition_time
        duration_float = duration / repetition_time
        timing_rows.append(
            {
                "trial_type": trial_type,
                "onset": onset,
                "duration": duration,
                "start_volume_float": start_float,
                "duration_volumes_float": duration_float,
            }
        )
        if trial_type in CLASS_NAMES:
            starts[trial_type].append(int(round(start_float)))
    return {class_name: sorted(starts[class_name]) for class_name in CLASS_NAMES}, timing_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare extracted class-folder event windows against source OpenNeuro BIDS events.tsv files. "
            "Only lightweight TSV/JSON files are downloaded from the OpenNeuro GitHub mirror."
        )
    )
    parser.add_argument("--records-json", required=True, help="Saved feature records.json with subject/run/class/vol_ids.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--dataset-id", default="ds004044")
    parser.add_argument("--version", default="2.0.3")
    args = parser.parse_args()

    records = json.loads(Path(args.records_json).read_text())
    repetition_time = load_repetition_time(args.dataset_id, args.version)
    extracted = extracted_event_starts(records)
    subjects = sorted(set(str(record["subject_id"]) for record in records))
    subject_runs = sorted(set((str(record["subject_id"]), int(record["run_id"])) for record in records))

    anomalies = []
    run_rows = []
    for subject, run_id in subject_runs:
        source_starts, timing_rows = source_event_starts(
            args.dataset_id,
            args.version,
            subject,
            run_id,
            repetition_time,
        )
        run_anomalies = []
        for class_name in CLASS_NAMES:
            expected = source_starts[class_name]
            observed = extracted.get((subject, run_id, class_name), [])
            if expected != observed:
                anomaly = {
                    "subject": subject,
                    "run_id": run_id,
                    "class_name": class_name,
                    "source_event_starts": expected,
                    "extracted_event_starts": observed,
                }
                anomalies.append(anomaly)
                run_anomalies.append(anomaly)
        non_integer_timing = [
            row
            for row in timing_rows
            if abs(row["start_volume_float"] - round(row["start_volume_float"])) > 1e-6
            or abs(row["duration_volumes_float"] - round(row["duration_volumes_float"])) > 1e-6
        ]
        if non_integer_timing:
            anomalies.append(
                {
                    "subject": subject,
                    "run_id": run_id,
                    "type": "non_integer_timing",
                    "rows": non_integer_timing,
                }
            )
            run_anomalies.extend(non_integer_timing)

        run_rows.append(
            {
                "subject": subject,
                "run_id": run_id,
                "source_target_starts": source_starts,
                "anomaly_count": len(run_anomalies),
            }
        )

    subject_anomaly_counts = {
        subject: sum(1 for anomaly in anomalies if anomaly.get("subject") == subject)
        for subject in subjects
    }
    report = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "records_json": args.records_json,
        "repetition_time": repetition_time,
        "subject_count": len(subjects),
        "run_count": len(subject_runs),
        "class_names": CLASS_NAMES,
        "anomaly_count": len(anomalies),
        "subject_anomaly_counts": subject_anomaly_counts,
        "anomalies": anomalies,
        "runs": run_rows,
        "note": (
            "An empty anomalies list means extracted event-window volume starts match source BIDS "
            "events.tsv onset/TR starts for the four target motor classes."
        ),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "repetition_time": repetition_time,
                "subject_count": len(subjects),
                "run_count": len(subject_runs),
                "anomaly_count": len(anomalies),
                "subjects_with_anomalies": [
                    subject for subject, count in subject_anomaly_counts.items() if count
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
