# Inter-Subject Normalization

Assessed: 2026-08-12.

## The objection

The frozen representation is produced by `reproduce_thesis_transform` in
`scripts/sweep_continuous_bold_windows.py`:

```python
extracted = zscore(resize(volume, extraction_shape))   # (100, 100, 100)
return zscore(resize(extracted, feature_shape))        # (24, 24, 24)
```

`resize` is `scipy.ndimage.zoom(..., order=1)`. Applied to `space-T1w` denoised
BOLD, this rescales the bounding box of each subject's native-space anatomy onto
a common grid. It is **not** registration: no template, no warp, no anatomical
correspondence. Two subjects' voxel `(12, 12, 12)` are the same fraction of the
way across their respective bounding boxes, not the same anatomical location.

This is a genuine methodological gap, and the objection has teeth: an unknown
share of the cross-subject difficulty characterised in this project may be
alignment that was never performed rather than an intrinsic property of the task.
A reviewer at a neuroimaging venue will stop reading at this point.

## What the release actually contains

The assumption that `derivatives/ciftify/` offers a drop-in grayordinate
substitute does **not** survive inspection of the bucket. Listing
`s3://openneuro.org/ds004044/` gives:

| Path | Contents | Usable as aligned time series? |
| --- | --- | --- |
| `derivatives/fmriprep/sub-XX/` | Exactly 12 files per subject, all `space-T1w` BOLD (`desc-preproc` and `desc-preproc_denoised`). No `anat/`, no `*_xfm.h5`, no MNI outputs. | No. This is the space already in use. |
| `derivatives/ciftify/sub-XX/results/` | One `ses-1_task-motor_hp200_s4_level2.feat/` directory. The `.dtseries.nii` files inside are **level-2 GLM outputs** — `cope`, `pe`, `tstat`, `varcope`, `res4d`, `weights`. | No. Session-level GLM statistics, not per-run event time series. |
| `derivatives/ciftify/sub-XX/native/` | 47 files including `sub-XX.{L,R}.midthickness.native.surf.gii`, `.white.`, `.pial.`, and `sub-XX.{L,R}.sphere.MSMSulc.native.surf.gii` plus the `MSMSulc/` registration outputs. | **Not directly, but this is the enabling asset.** |

Two corrections to earlier assumptions are worth recording:

- There is no preprocessed grayordinate BOLD (`*_Atlas.dtseries.nii`) anywhere in
  the release. "Re-run the decoder on grayordinates" is not a swap of input paths.
- fmriprep shipped **no** MNI-space volumes and **no** anatomical transforms, so
  the cheap volumetric route is also unavailable off the shelf.

All 62 cohort subjects do have ciftify derivatives, so coverage is complete and
the all-62 primary estimate is not threatened by this route.

## The viable path

The expensive part of surface normalization — cortical reconstruction and
inter-subject surface registration — has **already been computed and shipped**.
`native/` contains both the ribbon surfaces and the MSMSulc registration spheres.
What remains is projection and resampling:

1. `wb_command -volume-to-surface-mapping` with `-ribbon-constrained`, using
   `midthickness`, `white`, and `pial`, to sample each `space-T1w` run onto the
   subject's native surface. The BOLD is already in `T1w` space, which is the
   space these surfaces live in, so no additional registration is required.
2. `wb_command -metric-resample` with `ADAP_BARY_AREA`, using
   `sphere.MSMSulc.native` and the fsLR-32k sphere, to move every subject onto a
   common vertex correspondence.
3. Assemble left and right hemispheres into fsLR-32k grayordinates and re-run the
   frozen decoder unchanged.

This yields genuine anatomy-based inter-subject correspondence using only
released data, with no registration recomputed.

### Implementation status

`findings_2026-08-12/scripts/project_bold_to_surface.py` implements both steps directly on nibabel, so
Connectome Workbench is **not** required:

- Ribbon sampling averages trilinear samples at several depths between the white
  and pial surfaces, approximating `-ribbon-constrained` mapping.
- Spherical resampling uses `sphere.MSMSulc.native`, which is a perfect sphere of
  radius `100.000` (verified: standard deviation `0.000` across 141838 vertices),
  so all subjects share one spherical frame. Each subject is resampled onto a
  self-generated icosphere, which supplies vertex correspondence without needing
  the fsLR template.

Verified on `sub-01`: surfaces parse to 141838 vertices with matching topology
across white, pial, midthickness, and sphere; one run of `space-T1w` denoised BOLD
is `(77, 82, 73, 232)` at 2 mm.

### A suspected frame offset, investigated and withdrawn

An earlier pass on this document reported a blocking `c_ras` offset between the
ciftify surfaces and the fmriprep BOLD, estimated at `(0, +24, +6)` mm because that
translation raised in-FOV vertex coverage from `88.7%` to `98.3%` and mean sampled
intensity from `672` to `770`.

**That diagnosis does not survive scrutiny and is withdrawn.** The objective was
invalid: a translation that pushes the surface toward the centre of the field of
view raises both coverage and mean intensity *by construction*, whether or not it
improves anatomical alignment. The optimiser was rewarded for moving the mesh into
the brightest, best-covered part of the volume. Two supporting observations were
also wrong: the centroid comparisons put a **left hemisphere** surface against a
**whole-brain** mask, which manufactures a spurious ~31 mm `x` difference, and
compared a whole-head T1w against a brain-only EPI field of view, which
manufactures a `z` difference.

The evidence now points the other way. Checked against each subject's own raw T1w:

| Check | sub-01 | sub-07 | sub-23 |
| --- | ---: | ---: | ---: |
| White vertices in positive tissue | 100% | 100% | 100% |
| WM (2 mm inside white) | 292.6 | 302.9 | 315.5 |
| GM (mid-ribbon) | 285.6 | 290.8 | 302.3 |
| CSF (2 mm outside pial) | 281.4 | 277.9 | 292.4 |

The tissue ordering `WM > GM > CSF` holds in **every** subject, mean cortical
thickness is `2.95` mm — an anatomically correct value — and vertex normals agree
with the white-to-pial direction for `95.8%` of vertices, so the meshes are sound.
Sampling the BOLD at white vertices gives `602.8` against a BOLD median of `202`,
roughly three times the median, so the vertices land in brain rather than
background. The `88.7%` coverage figure is simply **normal EPI field-of-view
coverage** of the cortical surface, not evidence of misalignment.

### Why a hand-rolled registration cannot settle it either

A boundary-based cost was then implemented properly — area-weighted vertex normals,
trilinear sampling, normalised WM-versus-CSF contrast — and swept along each axis:

```
y:  -16:+0.0518  -8:+0.0457  0:+0.0259  +8:+0.0601  +16:+0.0574
z:  -16:+0.0334  -8:+0.0218  0:+0.0259  +8:+0.0410  +16:+0.0519
```

The cost is **U-shaped**: lowest near identity and rising toward *both* extremes. A
valid registration cost peaks at the correct alignment. One that improves in every
direction is measuring the brain-versus-background edge, where a bright inner sample
meets an empty outer sample, rather than the white-matter boundary. Consistently,
free optimisation gave `z` estimates spanning `40` mm across three subjects, which is
noise rather than a recovered quantity.

The root cause is the image. The raw T1w is **not bias-corrected and not intensity
normalised**: its brain-voxel histogram is broad and unimodal with no separable
grey and white peaks, and the WM/GM ratio at correctly placed surfaces is only
`1.03`–`1.04` where a usable T1w gives `1.5`–`2.0`. A hand-rolled boundary cost has
nothing to lock onto.

### Current position

There is **no established frame offset**, and the burden of proof has moved: the
positive checks above are consistent with the surfaces already being in the correct
frame, which is what one would expect from a ciftify pipeline that applies `c_ras`
itself.

What is *not* established is that alignment is correct to within a millimetre or
two, because no tool available here can demonstrate that on an uncorrected image.
Settling it requires proper tooling rather than more hand-rolled objectives:

1. **N4 bias correction on the T1w, then a real registration** (ANTs, or FSL
   `bbregister` which is purpose-built for surface-to-EPI). With a bias-corrected
   image the boundary cost becomes well posed.
2. **A visual check** — render the white and pial contours over the T1w and the
   mean EPI for several subjects. Cheap, and for a question this binary it is worth
   more than another summary statistic.

### The visual check, and the resolution

The visual check was run: white and pial contours rendered over both the T1w and
the mean EPI at the surface centroid slice for three subjects
(`figures/surface_alignment_check.png`).

It is unambiguous. **On the T1w the contours trace the cortical folding precisely**
— individual gyri and sulci are followed by the white contour with the pial contour
just outside it, across all three subjects. There is no frame offset. The ciftify
pipeline applied `c_ras` itself, as it is supposed to.

On the mean EPI the contours follow the brain, but visibly **extend past the
posterior and inferior edge of the image**. The EPI field of view is smaller than
the T1w and clips the lower cortex. That is the whole explanation for the ~11% of
vertices that fall outside: it is coverage, not registration.

**Resolution: there is no offset to correct, and the projection is unblocked.**
Two things follow for the implementation:

- Vertices outside the EPI field of view must be **explicitly marked invalid** and
  excluded from downstream features, not silently sampled as zero. Roughly 11% of
  cortical vertices are affected, concentrated in inferior and posterior regions,
  and the affected set will differ between subjects.
- Residual EPI-specific geometric distortion has **not** been assessed. Susceptibility
  distortion along the phase-encode direction is a normal property of EPI and would
  not be visible at this inspection level. The dataset ships no fieldmaps in the
  derivative trees examined, so the manuscript should state that surface sampling
  assumes the fmriprep EPI-to-T1w alignment and does not add a distortion correction
  of its own.

### End-to-end projection now runs

With the offset question closed, the full path was run on `sub-01` run 1:

| Quantity | Value |
| --- | ---: |
| Native vertices (left hemisphere) | 141838 |
| Target shared-sphere vertices | 10242 |
| Timepoints | 232 |
| Native vertices inside the EPI field of view | 0.8830 |
| Target vertices valid after resampling | 0.9188 |
| Mean temporal SD over valid vertices | 11.24 |
| Non-finite values | 0 |

Validity is propagated through the same barycentric resampling as the data, and a
target vertex counts as valid only when at least half its source weight comes from
in-FOV vertices. Invalid vertices are zeroed **and flagged** in the output, so
downstream code can exclude them rather than mistake a zero for a measurement.

### Cohort extraction

`build_surface_event_sequences.py` produces a **drop-in replacement** for the frozen
volumetric checkpoints: shape `(48, 8, 20484)` with the same `labels` and
`records_json` keys, so every existing decoder runs against it unchanged and the
only thing that differs is the spatial representation. That is what makes the
comparison clean — same events, same temporal window, same folds, volumetric
bounding-box rescale versus genuine surface correspondence.

Design points that matter:

- Volumes are selected exactly as the frozen extraction selects them,
  `event_start + offset + step`, so only the spatial axis changes.
- The resampling operator depends only on the two spheres, so it is built **once per
  subject and hemisphere** and applied to every volume. Rebuilding it per volume
  would dominate runtime.
- Each run's NIfTI is streamed, used, and deleted immediately; peak disk stays at
  about one run.
- Work is checkpointed per subject, so an interruption resumes by skipping completed
  subjects.
Two traps, both of which cost a relaunch:

- **Run index padding.** Raw BIDS pads the run index (`run-01`) while the fmriprep
  derivatives do not (`run-1`). Both forms are required, and getting it wrong yields
  a silent 404 on the events file.
- **The cohort is not contiguously numbered.** Subject ids skip `15`, `19`, `28`,
  `40`, `41` and `64`, and run up to `sub-68`. Generating `sub-01 … sub-62` therefore
  fails on five subjects that do not exist *and* silently omits five that do,
  yielding a 57-subject cohort that no longer matches the volumetric one. The
  extraction takes its subject list from the frozen checkpoint directory via
  `--subjects-from`, which both fixes the numbering and guarantees the two cohorts
  are identical — a precondition for any paired comparison.

Validated on `sub-01`: `(16, 8, 20484)` across two runs, `0.932` valid vertices,
93 s including the one-off surface download. Full cohort is 62 subjects × 6 runs,
roughly 138 GB streamed and several hours.

### How the comparison must be read

Once extraction finishes the swap is mechanical — point the existing decoders at the
surface checkpoint directory — but the reading is not, and three things have to be
handled before any difference is attributed to alignment.

**1. Normalisation must be matched.** The extraction stores raw sampled BOLD, on the
order of several hundred intensity units, while the volumetric path z-scores every
volume across the grid. `normalize_surface_checkpoints.py` applies the same
per-volume z-score, computed **over valid vertices only** so that a subject's
field-of-view coverage does not leak into their feature scale. Skipping this would
confound representation with normalisation.

**2. Raw feature count is a weaker confound than it looks.** The surface has `20484`
vertices against `13824` volumetric features, but both far exceed the roughly `2480`
training events, and the decoders are fitted in a dual basis whose rank is bounded by
the sample count. Effective model capacity is therefore **identical** between the two,
and an earlier concern that the comparison is dimensionality-confounded was
overstated. A rank-matched control is still cheap and worth running, but it is a
robustness check rather than a precondition.

**3. The genuine asymmetry is anatomical coverage, and it cuts against the surface.**
The volumetric bounding box contains everything — cortex, white matter, subcortical
structures, cerebellum, and non-brain. The surface representation contains **cortex
only**. For a motor task that is a real loss: cerebellar and subcortical motor
territory carries limb-related signal that the surface path discards entirely. On top
of that, roughly 7% of cortical vertices fall outside the EPI field of view.

So the surface representation trades information for correspondence. That makes the
possible outcomes asymmetric in a useful way:

- **Surface wins** — alignment matters enough to overcome losing subcortex and
  cerebellum. A strong result, and the headline the learning curve predicted.
- **Surface ties** — alignment is worth roughly what the discarded structures were
  worth. Genuinely informative, and it should be reported as such rather than as a
  null.
- **Surface loses** — either alignment adds little, or the discarded structures
  matter more. These are **not** distinguishable from this comparison alone, and the
  manuscript must not claim the first when it has only shown the disjunction. The
  clean follow-up is a volumetric decoder restricted to a cortical ribbon mask, which
  isolates coverage from correspondence.

## Result: surface alignment does not improve cross-subject decoding

All 62 subjects extracted and decoded under the identical 30-fold protocol.

| Model | Rule | Volumetric | Surface (per-subject valid) | Surface (intersected) |
| --- | --- | ---: | ---: | ---: |
| `linear_svm` | independent | **0.8051** | 0.7460 | 0.6003 |
| `logistic_l2` | independent | **0.8028** | 0.7382 | 0.5936 |
| `correlation_centroid` | independent | **0.6963** | 0.6722 | 0.4904 |
| `linear_svm` | balanced | **0.8441** | 0.7916 | 0.6256 |

The surface loses by roughly `0.06`. Restricting to the 12834 vertices valid in every
subject (`0.6265` of the sphere) is much worse still, which is unsurprising: it
discards real measurements from every subject to accommodate the worst-covered one.

### The dissociation that makes this interesting

A surface loss on its own would be ambiguous — this document said so before the run,
and the ambiguity was between "alignment adds little" and "the discarded subcortex
and cerebellum mattered". Within-subject leave-one-run-out decoding resolves part of
it, because it removes inter-subject alignment from the problem entirely:

| | Volumetric | Surface | Difference |
| --- | ---: | ---: | ---: |
| **Within-subject** (`linear_svm`, independent) | 0.6912 | **0.7332** | **+0.0420** |
| **Cross-subject** (`linear_svm`, independent) | **0.8051** | 0.7460 | −0.0591 |

The surface representation carries **more** class information than the volumetric one,
and transfers **worse**. The projection did not degrade the data — the opposite. This
is a transfer failure, not an information loss, and it rules out the explanation that
ribbon sampling and barycentric resampling blurred the signal away.

Measuring correspondence directly agrees. Inter-subject correlation of per-class
response maps is `0.0393` on the surface against `0.0465` volumetric, and a
leave-one-subject-out group-map matching test gives `0.798` against `0.823`.

### The pipeline was verified before this was believed

"Anatomical surface registration does not beat a bounding-box rescale" is a strong
claim and the obvious first objection is that the registration was never actually
applied. It was. Resampling each subject's midthickness coordinates onto the shared
sphere and correlating across subjects gives:

| Sphere used | Inter-subject anatomical-position correlation |
| --- | ---: |
| `sphere.MSMSulc.native` (registered) | **0.9887** |
| `sphere.native` (unregistered) | 0.9786 |

MSMSulc is being used, and it does improve anatomical correspondence over the
unregistered sphere. The pipeline is sound; the result is about the method, not a bug.

### What is and is not established

**Established.** Surface projection preserves and slightly improves the information
content of the representation, while transferring worse across subjects than a
volumetric bounding-box rescale. Anatomical alignment of cortical folding is
therefore *not* sufficient to improve cross-subject motor decoding on this cohort.

**Not established.** Whether the cross-subject deficit is because surface alignment
adds nothing over bounding-box rescaling for *functional* correspondence, or because
the discarded subcortex and cerebellum carry signal that is unusually consistent
across subjects. The within-subject result weakens the second explanation — losing
those structures did not cost information — but does not eliminate it, since a
structure could aid transfer specifically without adding within-subject
discriminability. **The ribbon-masked volumetric control separates these and should
be run before the manuscript commits to either.**

## Connectivity hyperalignment recovers over half the transfer deficit

Per-fold connectivity hyperalignment, template rebuilt from each fold's training
subjects, 324 parcels, orthogonal Procrustes on vertex-to-parcel connectivity
profiles:

| Model | Rule | Volumetric | Surface | Hyperaligned | Gain from alignment |
| --- | --- | ---: | ---: | ---: | ---: |
| `linear_svm` | independent | **0.8051** | 0.7460 | 0.7789 | **+0.0329** |
| `logistic_l2` | independent | **0.8028** | 0.7382 | 0.7777 | **+0.0396** |
| `linear_svm` | balanced | **0.8441** | 0.7916 | 0.8037 | +0.0121 |
| `logistic_l2` | balanced | **0.8456** | 0.7907 | 0.8081 | +0.0174 |
| `correlation_centroid` | independent | 0.6963 | 0.6722 | 0.6074 | **-0.0648** |

Paired across the 30 folds for `linear_svm` independent: mean difference `+0.0329`,
CI95 `[+0.0238, +0.0429]`, improving **27 of 30 folds**. This is a reliable effect,
not fold noise.

### Reading it

**Functional alignment does what anatomical alignment could not.** Surface projection
alone *lost* `0.059` relative to volumetric; hyperalignment gives back `0.033` of
that, a little over half. The deficit was genuinely a correspondence problem and it
is partly fixable without any new data, using only label-free connectivity structure.

**It does not overtake the volumetric baseline.** At `0.7789` the hyperaligned
surface remains `0.026` behind `0.8051`. The pipeline that discards subcortex,
cerebellum, and a tenth of cortex does not catch up with the crude bounding-box
rescale that keeps them, even after functional alignment.

**It harms the correlation classifier badly** (`-0.065`). Per-parcel orthogonal maps
are fitted to connectivity profiles estimated from 384 task-locked samples, so they
carry estimation noise. A discriminative model can learn to discount that; a
nearest-centroid classifier comparing raw normalised patterns cannot, and it inherits
the noise directly. This is worth reporting rather than hiding — it shows the
alignment is not a free improvement to the representation but a trade the decoder has
to be able to exploit.

### The residual gap now has a specific candidate

The arc across three experiments is consistent:

| Quantity | Value |
| --- | ---: |
| Surface advantage **within** subject | +0.042 |
| Surface deficit **across** subjects | −0.059 |
| Recovered by functional alignment | +0.033 |
| **Residual deficit** | **−0.026** |

The representation is better, the correspondence was worse, and most of the
correspondence problem yields to hyperalignment. What remains is roughly `0.026`, and
the obvious candidate is the one structural difference left: the volumetric bounding
box contains subcortex and cerebellum and the surface does not.

**This makes the ribbon-masked volumetric control decisive rather than optional.**
Restricting the volumetric decoder to a cortical ribbon removes the coverage
advantage while keeping everything else. If volumetric accuracy falls to roughly the
hyperaligned surface level, the residual is coverage and the alignment story is
complete. If it does not, something about the surface pipeline is still costing
accuracy and needs to be found.

## The surface contributes no information the volumetric grid lacks

The ribbon-masked volumetric control is the direct test of the coverage hypothesis,
but labelling the `24^3` bounding-box grid anatomically requires re-downloading every
subject's surfaces *and* BOLD affines — several gigabytes for one control.
Concatenating the two representations bounds the same question far more cheaply, and
asks something independently useful: does surface alignment add anything to the
existing pipeline?

Regularisation has to be swept, because the joint space has `34308` features and the
`C` selected for the volumetric decoder alone is not necessarily right for it:

| C | Combined `linear_svm` independent |
| --- | ---: |
| 1e-5 | 0.7716 |
| **1e-4** | **0.8051** |
| 1e-3 | 0.8042 |
| 1e-2 | 0.8042 |

At its best the combined decoder reaches `0.805095` against volumetric alone at
`0.805088`. Paired across the 30 folds:

| Quantity | Value |
| --- | ---: |
| Mean difference | **+0.000006** |
| CI95 | `[-0.0081, +0.0084]` |
| Folds better / worse | 14 / 16 |
| Folds identical | **0 / 30** |
| Per-fold SD of the difference | 0.0233 |

No fold is unchanged, and individual folds move by up to `0.058` in either direction,
so the surface features are demonstrably being used. They simply contribute **exactly
nothing on net**. This is not a null from an inert feature block; it is a null from
features that shift predictions without improving them.

### What that does and does not settle

It **strengthens the coverage explanation** considerably. If the surface carried
information the volumetric grid lacked — better functional correspondence, higher
within-subject signal — concatenation would capture it, since the decoder is free to
use both. It does not, which is what one expects if the volumetric representation
already contains the cortical signal in coarser form *plus* the subcortical and
cerebellar structure the surface discards.

It does **not** prove the residual gap is coverage. The surface information could be
a strict subset of the volumetric information for reasons unrelated to which
structures are included — for instance if the projection, however faithful, cannot
represent anything the `24^3` grid does not already resolve. The ribbon-masked
control remains the decisive test and the manuscript should not close this question
without it.

It also settles a practical point in the negative: **surface alignment offers no
route to a better headline number for this project.** The `0.8314` frozen result is
not improved by adding a functionally aligned cortical surface representation.

## A radial-partition probe, and how its premise was wrong

A cheap proxy for the ribbon mask: partition the `24^3` grid by radial distance from
its centre and decode from each shell separately. The stated premise was that cortex
lies near the outer surface of the bounding box and subcortex near its centre.

| Partition | Voxels | Independent | Balanced |
| --- | ---: | ---: | ---: |
| `r < 0.40` | 2176 | **0.7411** | 0.8025 |
| `r 0.40–0.60` | 5032 | **0.7861** | 0.8314 |
| `r 0.60–0.80` | 5488 | 0.3943 | 0.3991 |
| `r 0.80–1.00` | 1128 | 0.3524 | 0.3479 |
| all | 13824 | 0.8098 | 0.8485 |

**The premise was wrong.** The signal is concentrated centrally, not peripherally.
The outer 40% of the radius — `6616` voxels, `48%` of the grid — decodes near chance.
That is not because cortex is uninformative; it is because those shells are largely
**outside the brain**. `reproduce_thesis_transform` rescales the bounding box of the
whole imaged volume, so the outer shells contain skull, scalp, and air, and the brain
occupies roughly the central `60%` of the radius.

So the partition separates brain from non-brain, not cortex from subcortex, and it
**does not answer the coverage question it was built for**. Recording that plainly
matters more than salvaging an interpretation: a radial shell in a bounding-box
rescale has no anatomical meaning, and I should have predicted that before running it.

### What it does establish

Two things survive, both worth keeping.

**Nearly half the volumetric feature space is dead weight.** The outer `48%` of
voxels contribute almost nothing, which is consistent with the earlier finding that
the discriminative signal sits in many low-variance directions rather than in the
leading principal components — a large share of the grid is simply not brain.

**The central region alone supports `0.7411`.** From `2176` voxels, `16%` of the
grid, the decoder reaches within `0.04` of the hyperaligned surface's `0.7789` and
within `0.065` of the full volumetric `0.8098`. Whatever those central voxels are —
deep white matter, subcortical structures, or ventricle-adjacent tissue — they carry
substantial class information that a cortical surface representation cannot access.

That is **consistent with** the coverage explanation for the residual gap, but it is
weaker evidence than the ribbon-masked control, because "central in a bounding box"
still is not an anatomical label. The honest summary is that the coverage explanation
has survived two cheap probes without being confirmed by either, and the ribbon mask
remains the test that would settle it.

## The ribbon-masked control refutes the coverage explanation

Every previous probe pointed at coverage — the volumetric bounding box retains
subcortex, cerebellum, and white matter that the surface discards — without ever
confirming it. The control was deferred because it appeared to need ~12 GB of
re-downloaded data.

It did not. Labelling the grid needs each subject's BOLD **geometry**, not their BOLD
data, and a NIfTI header is 348 bytes. An HTTP range request for the first 64 KB of
each gzipped volume yields the shape and affine, verified against `sub-01` to match
the values read from the full 190 MB file. Cost fell to about 870 MB of surfaces plus
4 MB of headers. **This should have been checked before running two proxies that could
not answer the question.**

The ribbon frequency map — the fraction of subjects for whom each grid voxel lies
between the white and pial surfaces — was thresholded and the decoder rerun on cortex
alone:

| Condition | Voxels | `linear_svm` independent | Balanced |
| --- | ---: | ---: | ---: |
| Ribbon, frequency ≥ 0.25 | 6662 | **0.8177** | 0.8557 |
| Ribbon, frequency ≥ 0.50 | 5348 | 0.8097 | 0.8496 |
| Ribbon, frequency ≥ 0.75 | 3882 | 0.7894 | 0.8336 |
| Full volumetric grid | 13824 | 0.8051 | 0.8441 |
| Hyperaligned surface | 20484 | 0.7789 | 0.8037 |

**Restricting the volumetric decoder to cortex does not cost accuracy — it slightly
improves it.** At the `0.25` threshold cortex-only reaches `0.8177`, above the full
grid's `0.8051`, consistent with the ANOVA result that discarding uninformative voxels
helps. Even the strictest mask, keeping only 3882 voxels that are ribbon in three
quarters of subjects, still reaches `0.7894` — above the hyperaligned surface.

### What this settles, and what it opens

**The coverage explanation is refuted.** The volumetric advantage does not come from
subcortex or cerebellum. A cortex-only volumetric decoder beats a cortex-only surface
decoder by roughly the same margin as before, so the residual `0.026` is *not* about
which structures each representation contains.

That was the leading hypothesis through three separate probes and it is wrong. The
combination of results now forces a harder conclusion:

- The surface representation is **more reliable** (`0.6665` versus `0.5983`) and
  **more informative within subject** (`0.7332` versus `0.6912`).
- Its cross-subject deficit is **not** coverage.
- Functional alignment recovers only about half of it.

So the remaining explanation is **correspondence itself**: for this task, MSMSulc
folding-based surface registration provides no cross-subject advantage over a crude
volumetric bounding-box rescale, and connectivity hyperalignment closes only part of
the difference. Cortical folding and somatotopic functional location evidently do not
correspond tightly enough across subjects for surface alignment to pay for the
resampling it costs.

That is a stronger and more surprising claim than the coverage story it replaces, and
it is supported rather than merely suggested. It should be stated as a finding: **on
this cohort, anatomy-based surface normalisation does not improve cross-subject motor
decoding, and a bounding-box rescale is a competitive alternative.**

### Why this makes functional alignment more interesting, not less

The dissociation is close to the ideal setup for hyperalignment. The surface
representation has *more* signal and *worse* correspondence, so the deficit is
precisely the kind that a functional alignment is designed to repair. If connectivity
hyperalignment can close a `0.06` transfer gap on a representation that already
carries `+0.04` more information, the aligned surface would end up ahead of the
volumetric baseline. That is now the sharpest open question in the programme.

### Cost and constraints

- Requires Connectome Workbench, which is not currently a project dependency.
- Volumes are streamed per run from S3 and deleted immediately, as the existing
  extraction already does, so the 30 GB of local free space is not a blocker.
- 62 subjects x 6 runs = 372 projections. This is the two-week item, not the
  one-day item.

### A cheaper partial answer

If Workbench is unavailable, an affine T1w-to-MNI registration estimated from the
raw `sub-XX/ses-1/anat/` T1w and applied to the BOLD would still be a real
improvement over a bounding-box rescale, and would let the manuscript say that
the effect survives *some* anatomical normalization. It is weaker than surface
alignment and should be described as such.

## Why either outcome is publishable

- If accuracy rises substantially, inter-subject alignment was a real limiting
  factor, which is itself a useful finding about why this task has looked hard.
- If accuracy is unchanged, the effect survives proper normalization and the
  central claims become considerably harder to contest.

The current status is that the gap is **documented and scoped**, not closed. Until
it is closed, Limitation 2 in the [Publication Plan](../../docs/PUBLICATION_PLAN.md) must
stand.
