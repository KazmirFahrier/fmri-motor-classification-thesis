#!/usr/bin/env python3
"""Write deterministic hashes for the lightweight publication package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_FILES = [
    "experiments/confirmation/investigation_closeout.results.json",
    "findings_2026-08-18_interpretation/experiments/frozen_vs_nested_native_smoothing.results.json",
    "findings_2026-08-18_interpretation/experiments/q1_confirmation.results.json",
    "findings_2026-08-18_interpretation/scripts/compare_frozen_vs_nested_preprocessing.py",
    "docs/HCP_EXTERNAL_REPLICATION_PROTOCOL.md",
    "docs/JOURNAL_STRATEGY.md",
    "manuscript/Q1_MANUSCRIPT.md",
    "manuscript/tables/main_benchmark.csv",
    "manuscript/tables/main_benchmark.md",
    "manuscript/figures/protocol_separated_performance.pdf",
    "manuscript/figures/protocol_separated_performance.png",
    "scripts/build_q1_figures.py",
    "scripts/build_q1_publication_tables.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/artifact_manifest.json"),
    )
    args = parser.parse_args()

    entries = []
    for relative_name in DEFAULT_FILES:
        path = args.root / relative_name
        content = path.read_bytes()
        entries.append(
            {
                "path": relative_name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    payload = {
        "manifest_version": 1,
        "scope": "lightweight Q1 confirmation and publication package",
        "files": entries,
    }
    output = args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} hashes to {output}")


if __name__ == "__main__":
    main()
