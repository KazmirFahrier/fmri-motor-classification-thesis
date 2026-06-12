# Data Processing Specification

## Source Assumption
Data is organized as class folders under batch roots, with filenames:

`sub-XX_run-R_vol-VVV.nii.gz`

## Manifest Fields
- `filepath`
- `class_name`
- `class_id`
- `subject_id`
- `run_id`
- `vol_id`
- `batch_id`
- `exists`

## Split Policy
- Holdout subjects selected first (fixed count).
- Remaining subjects used for K-fold subject-wise CV.
- Strict no-overlap checks are enforced.

## Temporal Clip Construction
- Group by `(subject_id, run_id, class_id)`.
- Build clips from contiguous volume IDs with parameters:
  - `hrf_shift` (default `0` for the pre-extracted class-folder dataset)
  - `clip_length` (default one of 4/6/8)
  - `clip_stride`
  - `clip_window_stride`

Important: the current class-folder datasets already contain cropped event-window volumes. A positive
`hrf_shift` applied at `ClipDataset` time shifts only within those cropped class windows; it cannot
include post-event volumes from the original continuous 4D run. True HRF-shifted extraction must be
performed when rebuilding class folders from the raw BIDS runs.

## Transform Policy
- Resize to `target_shape`
- Optional train-time random crop and flips
- Z-score normalization

## QC Rules
- Fail on missing expected classes.
- Fail on filename parse errors in strict mode.
- Detect missing files and corrupted NIfTI files.
