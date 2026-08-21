# Storage backup and recovery map

This document records the storage cleanup completed on 2026-08-20. The GitHub repository is the canonical copy of source code, configuration, documentation, figures, and lightweight results. Private Kaggle datasets and completed Kaggle notebook outputs preserve large derived arrays, NIfTI content, checkpoints, and historical run artifacts.

## New private Kaggle archives

| Purpose | Kaggle handle | Kaggle size | Verification |
| --- | --- | ---: | --- |
| Targeted raw QC | `kazmirfahrierniloy/thesis-targeted-raw-qc-20260624` | 2,890,023,397 bytes | Remote checksum manifest SHA256 `82ef4fdb95b9092601ab2631380d396b547d64c33133673054ccba7d2411412c` |
| Targeted weak run QC | `kazmirfahrierniloy/thesis-targeted-weak-run-qc-20260713` | 1,310,395,399 bytes | Remote checksum manifest SHA256 `0385184da97ff96ec95f125fd540a2ada8a060a46287a158423a54ea02e0951c` |
| Twelve class artifacts | `kazmirfahrierniloy/thesis-twelve-class-artifacts-20260817` | 3,767,216,364 bytes | Remote checksum manifest SHA256 `3c3a4c7f2d69f6c24eca2f7bf3b9d85b476ad1fab3613f8d87785f1dd02314f4` |
| Surface projection artifacts | `kazmirfahrierniloy/thesis-surface-projection-20260813` | 3,026,243,720 bytes | Remote checksum manifest SHA256 `2efc38a41830f297ddcf4ab5f33fb4702874fe9a12c1c8dee5403e1e68fde64a` |
| Historical lightweight status archive | `kazmirfahrierniloy/thesis-lightweight-workspace-archive-20260820` | 94,321,347 bytes | Downloaded archive SHA256 `0092bed8d972f5df61999cd986b07ffb28086428021bfc05fc47268b063833e2`; 6,979 archived files |

Kaggle expands recognized archives during dataset processing. Consequently, uploaded `.nii.gz` files appear as `.nii` files in the dataset. Their decompressed scientific content was independently compared. The targeted raw QC sample produced SHA256 `f7d9b277a74d4bc0bba1f55febcbd15bebb30d38065e43eefb609f80513c9c54`, and the targeted weak run sample produced SHA256 `7409de6d751e7b670a85a311a61deaed0609c123d7a70193b44267d6bdd89e43` on both sides.

Exact NPZ recovery was also checked. The twelve class `seq/sub-01.npz` hash was `941d3152c3a9c52368ad29897b1b194e79b46bf180346a32657275fb2fe50f45`. The surface `surfseq/sub-01.npz` hash was `5fbafca2738e9444f29261b4a7ca73f9dd5e66bc21de6421e634996595f33c8e`.

The lightweight archive contains resident JSON, CSV, text, logs, figures, metadata, and copied source snapshots from historical `status_*` folders. Its `ICLOUD_PLACEHOLDERS.txt` lists 1,223 nonresident placeholders that were not reread into the archive. These placeholders are duplicated kernel or repository snapshots whose canonical source is GitHub or an existing Kaggle run.

## Existing large artifact coverage

The cleanup also relies on these previously verified remote assets:

1. Source batches `kazmirfahrier/thesis-batch-01` through `kazmirfahrier/thesis-batch-07`.
2. `kazmirfahrier/thesis-legacy-ablation-artifacts`.
3. `kazmirfahrier/thesis-legacy-full-artifacts`.
4. `kazmirfahrier/thesis-7batch-artifacts`.
5. `kazmirfahrier/thesis-wholebrain-artifacts`.
6. `kazmirfahrier/thesis-roi-temporal-artifacts`.
7. `b6uejhvvnmiwb/thesis-legacy-no-dropconnect-artifacts`.
8. `kazmirfahrierniloy/thesis-legacy-no-attention-artifacts`.
9. `kazmirfahrierlover/thesis-legacy-no-transformer-artifacts`.
10. `b6uejhvvnmiwb/thesis-legacy-full-artifacts-epoch25-final`.
11. Completed notebook output `b6uejhvvnmiwb/thesis-event-sequence-full-cohort`.
12. Completed notebook output `b6uejhvvnmiwb/thesis-temporal-basis-full-cohort`.
13. Completed twelve class offset zero shards `kazmirfahrierniloy/thesis-t12-offset0-shard0` through `shard2`.
14. Completed interpretation grid shards `kazmirfahrierniloy/thesis-gridsweep-shard0` through `shard2`.
15. Completed offset zero shards `kazmirfahrierniloy/thesis-offset0-shard0` through `shard2`.

## Recovery

Download any dataset with:

```bash
kaggle datasets download OWNER/DATASET -p RECOVERY_DIRECTORY --unzip
```

Restore the lightweight historical archive with:

```bash
tar -xzf thesis_lightweight_workspace_20260820.tarbackup
```

The datasets are private. Recovery therefore requires an authenticated Kaggle account with access to the owning account.

## Local cleanup scope

The cleanup removes only generated thesis storage:

1. The abandoned top level `.git` object store, which had no commit, branch, tracked file, or remote.
2. Top level `status_*` and `status_latest_*` result folders.
3. Local `kaggle_*_artifacts_dataset*` extraction and staging folders.
4. `tmp_kaggle_download*` duplicate downloads.
5. Local Kaggle status mirrors and the temporary lightweight archive staging directory.

The cleanup does not remove the canonical `/Users/USER/Documents/Thesis` repository, Kaggle kernel source directories, credentials, or unrelated projects.
