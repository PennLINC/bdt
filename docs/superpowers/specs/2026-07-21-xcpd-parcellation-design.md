# Use XCP-D's parcellation code for NIfTI and CIFTI

Date: 2026-07-21
Status: approved (design)

## Problem

BDT's volumetric parcellation is hand-rolled. Specs 2 and 3 added
`src/bdt/interfaces/parcellate.py` (`ParcellateVolumetric`,
`ParcellateVolumetricTimeseries`) and `utils/cifti.py:tsv_correlation` with
bespoke numpy, *while XCP-D's equivalents were already vendored* in
`src/bdt/interfaces/connectivity.py` (`NiftiParcellate`, `TSVConnect`) and never
wired up. The result is two parallel implementations with different coverage
semantics, different output formats, and no shared provenance with XCP-D.

The CIFTI path is not affected: `_init_parcellate_cifti_wf` is already a faithful
port of XCP-D's `init_parcellate_cifti_wf` over the vendored
`CiftiParcellateWorkbench` / `CiftiVertexMask` / `CiftiMask`. It stays as-is.

Goal: every parcellation path delegates to a real nilearn/Workbench masker, with
XCP-D's coverage definition and output format, and the hand-rolled numpy is
deleted.

## Routing

Dispatch happens in the parcellate factory. `parcellate_scalar` and
`parcellate_timeseries` share it; a scalar is just the `n_timepoints == 1` case.

| atlas form | detected by | implementation |
| --- | --- | --- |
| CIFTI | `context.role_is_cifti(node, role)` | `_init_parcellate_cifti_wf` — unchanged |
| 3D dseg | atlas `ndim == 3` | XCP-D `NiftiParcellate` (`NiftiLabelsMasker`) |
| 4D, thresholded | `ndim == 4` and `role_suffix == 'dseg'` | binarize each volume, `NiftiLabelsMasker` per volume |
| 4D, unthresholded | `ndim == 4` and `role_suffix == 'probseg'` | `NiftiMasker` + explicit weighted mean |

The 4D sub-case key needs no new plumbing: `tractogram_to_pseg`'s
`dynamic_suffix` already yields `dseg` when a `threshold` parameter is set and
`probseg` otherwise, and Spec 2 made that propagate through
`node_output_entities`, so `context.role_suffix(node, 'atlas')` is authoritative
at build time.

### Determining `ndim` at build time

The factory dispatches while the graph is being built, so `ndim` cannot simply be
read off the atlas file in every case. It is resolved by provenance:

- **Selection atlas** — the matched path exists on disk during graph
  construction, so the factory reads the header (`nb.load(path).ndim`). This is a
  header read, not a data read.
- **Processing atlas** — the file does not exist yet, but is 4D by construction:
  `tractogram_to_pseg` is the only action that produces an atlas, and it stacks
  bundles via `ConcatenateNiftis`. So a processing-node atlas is always 4D, and
  the thresholded/unthresholded sub-case comes from `role_suffix`.

Warping preserves `ndim`, so a build-time reading taken from the original
selection remains valid for the warped atlas that actually reaches the masker.
If the header cannot be read, that is a hard error naming the path rather than a
guessed default.

`NiftiLabelsMasker` requires a 3D integer label image, so it cannot consume a 4D
stack directly — hence the per-volume application in the thresholded branch,
which also preserves overlapping bundles.

## Interfaces

`NiftiParcellate` and `TSVConnect` are used as vendored, with one dtype fix
(below). The 4D branches go in a new `ProbSegParcellate` interface, which
handles both sub-cases behind a `binarize` flag — they share all coverage,
labels, and output bookkeeping and differ only in the per-parcel reduction.

`src/bdt/interfaces/parcellate.py` is deleted, along with
`utils/cifti.py:tsv_correlation` and their tests.

### Unthresholded 4D: weighted mean

```python
masker = NiftiMasker(mask_img=brain_mask, standardize=None)
W = masker.fit_transform(atlas)                 # (n_parcels, n_voxels_in_mask)
D = masker.transform(data)                      # (n_timepoints, n_voxels_in_mask)
values = (W @ D.T) / W.sum(axis=1)[:, None]     # (n_parcels, n_timepoints)
```

One vectorized masker call, no per-volume splitting, and the average is computed
strictly within the brain mask. Verified against a brute-force
`Σ(w·d)/Σ(w)` on a synthetic case: agreement to 6 decimal places.

`NiftiMapsMasker` is deliberately *not* used here. Its extraction ends in
`linalg.lstsq(maps_data[maps_mask, :], data[maps_mask, :])`, i.e. a least-squares
unmixing. Even applied one map at a time it returns `Σ(w·d)/Σ(w²)`, a scaling
coefficient rather than a mean — inflated relative to the weighted mean whenever
the map is non-binary (measured: 7.5008 vs 5.2585 on a random probability map, a
43% difference). nilearn offers no weighted-average mode, so the two-line
explicit computation above is the definition itself and is what we use.

`standardize=None` is passed at every `NiftiMasker` call site: `standardize=False`
raises a `FutureWarning` on nilearn 0.14.0 (already visible in the current test
output from `utils/cifti.py`).

## Coverage

One definition everywhere, XCP-D's:

```
coverage = |parcel ∩ brain_mask| / |parcel|
```

XCP-D computes this by passing *the binarized atlas* — never the data — through a
pair of `NiftiLabelsMasker(strategy='sum')` instances, one with `mask_img` and one
without, and dividing. The data file is structurally uninvolved.

This supersedes the hand-rolled `valid = isfinite(data) & (data != 0)`, which
measured a different, data-dependent quantity. The governing assumption is that
the brain mask already excludes NaN and zero-variance voxels.

For psegs the weight-domain analogue is `Σ(w·mask)/Σ(w)`, which is likewise
data-independent. In the thresholded branch the weights are binary, so it reduces
to XCP-D's voxel-count formula exactly.

Parcels below `min_coverage` become NaN. Because the labels table supplies the
*full expected* parcel list, parcels lost entirely when the atlas is warped (e.g.
the MNI152NLin2009cAsym → MNI152NLin6Asym hop in `nifti_parcellate.yml`) are
NaN-filled in place rather than silently shifting the remaining columns — a case
the hand-rolled code could not represent.

### Defect in the vendored code (must fix)

`NiftiLabelsMasker(strategy='sum')` returns `0` for `uint8` input on nilearn
0.14.0, while returning the correct value for float:

```
uint8  -> [0 0]            float32 -> [256. 256.]      (true count: 256)
```

`connectivity.py:104` builds the binarized atlas with
`(atlas_img.get_fdata() > 0).astype(np.uint8)`, so the denominator is zero and
coverage evaluates to `inf`:

```
uint8   (as vendored) : [inf inf]
float32 (fixed)       : [0.75 0.25]     <- correct for the fixture
```

Since `inf < min_coverage` is `False`, coverage thresholding silently becomes a
no-op and the coverage TSV ships full of `inf`. This has never surfaced because
`connectivity.py` is vendored but currently unused.

Fix: build the binarized atlas as `float32`. It lands as an explicitly commented
divergence from upstream, not a silent edit, with a test asserting a
known-coverage fixture returns `0.75`.

## Mandatory inputs

### Brain mask

Auto-discovered, space- and session-matched (`suffix=mask, desc=brain`), using the
existing `find_reference` pattern in `factories.py`. `nifti_parcellate.yml` needs
no edit: ds008325 ships
`sub-125511_ses-1_..._space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz`.

If no mask matches, this is a **hard build-time error naming the failed query**.
There is deliberately no data-derived fallback: synthesizing a mask from the
data's finite/non-zero support would reintroduce data dependence into coverage
through the back door, contradicting the definition above. A tract scalar in a
space with no matching mask is a spec/YAML problem to fix, not something to guess
at.

### Atlas labels

`atlas_labels` is mandatory in `NiftiParcellate`, and it is not cosmetic — the
labels table is what supplies the full expected parcel list used for NaN-filling
warped-away parcels. Two provenances:

- **Selection atlas** (AtlasPack): resolve the sibling `_dseg.tsv` from the
  `Match` path at build time. It must come from the *original selection*, not from
  the warp node's output — the warped atlas in the node cwd has no sidecar
  beside it.
- **Processing atlas** (`tractogram_to_pseg`): extend role wiring in
  `workflow.py` to carry a secondary `tsv` edge alongside `outputnode.out`. The
  node already exposes `outputnode.tsv` via `EntitiesToSegTSV`; only the compiler
  drops it today.

The compiler change closes the known limitation recorded in Spec 2 ("region names
are index-based; atlas dseg.tsv sidecar not carried by role wiring"). It is the
riskiest part of this work because role wiring is shared by every action, so it
must be strictly additive: wire `tsv` only when the upstream exposes it *and* the
downstream declares it wants it.

## Outputs

XCP-D format throughout:

- `timeseries.tsv` — wide, real region **names** as columns (a scalar is a 1-row
  table), `na_rep='n/a'`.
- `coverage.tsv` — `Node` index column plus `coverage`.
- FC via `TSVConnect` — relmat with `Node` row labels, replacing
  `tsv_correlation`. This also resolves the missing-row-labels finding from the
  Spec 3 final review.

`ExtraProduct.volumetric_extension` (added in Spec 3) already routes the coverage
product to `.tsv` for volumetric nodes, so the sink plan needs no change.

This changes `tract_parcellate.yml`'s outputs: the tidy
`index/name/mean/coverage` long format becomes wide + a separate coverage TSV.
That pipeline is the only current consumer, and the change is accepted.

`TSVConnect.temporal_mask` is optional and stays unwired; BDT has no censoring
concept today.

## Testing

- Per-branch unit tests against synthetic atlases with analytically known answers,
  pinning the equivalences verified during design: binarized single-map
  `NiftiLabelsMasker` == plain mean within mask; `(W @ D.T) / W.sum(axis=1)` ==
  brute-force `Σ(w·d)/Σ(w)`.
- A coverage test on a fixture with known geometry (`0.75` / `0.25`) that fails
  against the `uint8` denominator bug.
- A regression test that a parcel absent from the warped atlas returns NaN in
  place rather than shifting columns.
- Node-level round-trips wherever a nipype node is involved, since direct function
  calls miss node-only failures (as the `correlate_tsv` `NameError` showed).
- Acceptance: `nifti_parcellate.yml` and `tract_parcellate.yml` both compile and
  route to the expected masker.

## Out of scope

- `ConnectPlot` and other reportlets.
- Censoring / `temporal_mask` wiring.
- The same-space/different-grid crash from the Spec 3 review (`weight[..., None] *
  data0` broadcast mismatch) — the maskers' `resampling_target=None` contract
  makes grid agreement an explicit precondition; enforcing it is separate work.
