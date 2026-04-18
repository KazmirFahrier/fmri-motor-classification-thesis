from __future__ import annotations

import pandas as pd

from fmri_pipeline.data.splits import assert_no_subject_leakage, create_subjectwise_splits


def _dummy_manifest(num_subjects: int = 12) -> pd.DataFrame:
    rows = []
    sample_id = 0
    for s in range(1, num_subjects + 1):
        subject = f"sub-{s:02d}"
        for run_id in [1, 2]:
            for class_id, class_name in enumerate(
                [
                    "Left leg movements",
                    "Right leg movements",
                    "Forearm movements",
                    "Upper arm movements",
                ]
            ):
                for vol_id in [10, 11, 12]:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "filepath": f"/x/batch_01/{class_name}/{subject}_run-{run_id}_vol-{vol_id}.nii.gz",
                            "class_name": class_name,
                            "class_id": class_id,
                            "subject_id": subject,
                            "run_id": run_id,
                            "vol_id": vol_id,
                            "batch_id": "batch_01",
                            "exists": True,
                        }
                    )
                    sample_id += 1
    return pd.DataFrame(rows)


def test_subjectwise_split_no_leakage():
    df = _dummy_manifest()
    split_data = create_subjectwise_splits(df, seed=42, holdout_subject_count=2, cv_folds=5)
    assert_no_subject_leakage(split_data)

    holdout = set(split_data["holdout_subjects"])
    for fold in split_data["folds"]:
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])
        assert not train.intersection(val)
        assert not train.intersection(holdout)
        assert not val.intersection(holdout)
