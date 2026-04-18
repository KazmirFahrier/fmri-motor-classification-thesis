from __future__ import annotations

from fmri_pipeline.utils.parsing import parse_sample_path


def test_parse_sample_path_ok():
    parsed = parse_sample_path(
        "/tmp/dataset/batch_01/Forearm movements/sub-23_run-2_vol-104.nii.gz"
    )
    assert parsed.subject_id == "sub-23"
    assert parsed.run_id == 2
    assert parsed.vol_id == 104
    assert parsed.class_name == "Forearm movements"
    assert parsed.batch_id == "batch_01"
