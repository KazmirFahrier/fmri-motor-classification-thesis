from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from finalize_investigation_closeout import validate_checkpoints
from make_affine_feature_stability_maps import feature_grid_affine
from run_nested_repetition_consistency_assignment import ASSIGNMENTS


def write_checkpoint(path: Path, subject: str) -> None:
    records = []
    labels = []
    for run_id in (1, 2):
        for event_index, class_id in enumerate((0, 0, 1, 1, 2, 2, 3, 3)):
            labels.append(class_id)
            records.append(
                {
                    "subject_id": subject,
                    "run_id": run_id,
                    "event_start": event_index * 10,
                    "class_id": class_id,
                }
            )
    np.savez_compressed(
        path,
        offset_3_length_8_sequence=np.ones((16, 2, 3), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        records_json=np.asarray(json.dumps(records)),
    )


def test_checkpoint_validator_checks_balanced_run_composition(tmp_path: Path) -> None:
    write_checkpoint(tmp_path / "sub-01.npz", "sub-01")
    write_checkpoint(tmp_path / "sub-02.npz", "sub-02")
    protocol = {
        "dataset": {
            "subject_count": 2,
            "run_count": 4,
            "runs_per_subject": 2,
            "events_per_run": 8,
            "event_count": 32,
            "classes": ["a", "b", "c", "d"],
            "events_per_class_per_run": 2,
        },
        "representation": {
            "sequence_key": "offset_3_length_8_sequence",
            "sequence_shape_per_subject": [16, 2, 3],
        },
    }
    summary, manifest = validate_checkpoints(tmp_path, protocol)
    assert summary["status"] == "pass"
    assert summary["event_count"] == 32
    assert summary["class_counts"] == {"a": 8, "b": 8, "c": 8, "d": 8}
    assert len(manifest) == 2


def test_balanced_assignment_enumeration_is_complete_and_legal() -> None:
    assert ASSIGNMENTS.shape == (2520, 8)
    assert len({tuple(row.tolist()) for row in ASSIGNMENTS}) == 2520
    for assignment in ASSIGNMENTS:
        assert np.array_equal(np.bincount(assignment, minlength=4), [2, 2, 2, 2])


def test_feature_grid_affine_preserves_reference_center() -> None:
    reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference_affine[:3, 3] = [-72.0, -82.0, -70.0]
    reference_shape = (75, 81, 70)
    feature_shape = (24, 24, 24)
    result = feature_grid_affine(reference_affine, reference_shape, feature_shape)
    reference_center = nib.affines.apply_affine(
        reference_affine, (np.asarray(reference_shape) - 1) / 2
    )
    feature_center = nib.affines.apply_affine(
        result, (np.asarray(feature_shape) - 1) / 2
    )
    assert np.allclose(feature_center, reference_center, atol=1e-6)
