# Design: volumetric `parcellate_scalar` (Strategy A)

**Date:** 2026-07-20
**Status:** Approved (design)

**Scope:** Spec 2 of 2. Spec 1 (the transform query + chaining subsystem —
`src/bdt/transforms/` + `bdt.interfaces.transforms.ResolveApplyTransforms`) is
complete. This spec adds the volumetric (NIfTI) parcellation path to the
existing `parcellate_scalar` action, which currently supports CIFTI inputs only.
It *consumes* Spec 1's `ResolveApplyTransforms` to warp a volumetric atlas into
the scalar's space when they differ.

## Problem

`init_parcellate_scalar_wf` (`src/bdt/engine/factories.py`) is CIFTI-only: it
routes every node through `_init_parcellate_cifti_wf` (dscalar × dlabel →
pscalar, coverage-aware). The real pipeline (`scripts/tract_parcellate.yml`)
has two volumetric nodes that build but cannot run:

- **`fa_roi`** — scalar = QSIRecon FA (`ACPC`), atlas = `bundle_rois` (a 4D dseg
  of per-bundle masks, `ACPC`). Same space → parcellate directly.
- **`cbf_roi`** — scalar = ASLPrep CBF (`MNI152NLin6Asym`), atlas = `bundle_rois`
  (`ACPC`). Different spaces → the atlas must be warped `ACPC → MNI152NLin6Asym`
  (chain: computed `ACPC→T1w` bridge + ASLPrep's stored `T1w→MNI152NLin6Asym`)
  before parcellation.

The atlas (`bundle_rois`, from `tractogram_to_pseg`) is a **4D dseg/pseg**: one
volume per bundle, with a BIDS label sidecar `dseg.tsv` (columns `index`, `name`)
on the node's `outputnode.tsv`. This is a different shape than the existing
`bdt.utils.cifti.nifti_parcellate_to_tsv`, which assumes a 3D integer-label atlas
(`nilearn.NiftiLabelsMasker`).

## Atlas forms supported

Three, per the approved scope:

1. **4D dseg** (per-region binary masks) — *treated like a pseg*.
2. **4D pseg** (per-region probabilistic weight maps).
3. **3D dseg** (single integer-label image).

The 4D forms unify under one **voxel-value-weighted mean** per volume; a binary
4D dseg reduces to the plain in-mask mean. The 3D dseg is the label-wise mean.

## Components

### 1. Routing — `init_parcellate_scalar_wf`

`init_parcellate_scalar_wf` branches on the resolved `scalar`'s file type,
determined at compile time from the context (CIFTI `.dscalar.nii` vs volumetric
`.nii`/`.nii.gz`):

- **CIFTI** → the existing `_init_parcellate_cifti_wf` path, **unchanged**
  (byte-for-byte identical results for the current CIFTI stories).
- **volumetric NIfTI** → the new Strategy-A subworkflow below.

### 2. Cross-space atlas warp (consumes Spec 1)

Mirrors the cross-space handling in `init_map_scalar_to_surface_wf`, but uses
Spec 1's full-graph image resample rather than a single hop:

- `atlas_space = context.role_space(node, 'atlas')`,
  `scalar_space = context.role_space(node, 'scalar')`;
  `cross_space = atlas_space is not None and scalar_space is not None and atlas_space != scalar_space`.
- When `cross_space`, insert a `ResolveApplyTransforms` node
  (`bdt.interfaces.transforms`):
  - `moving` = the atlas image, `reference` = the scalar (defines the output
    grid), `source` = `atlas_space`, `target` = `scalar_space`.
  - `local_transforms` = subject transform files discovered across the input
    derivative datasets via `context.provider` (e.g. ASLPrep's
    `from-T1w_to-MNI152NLin6Asym` xfm). Discovery is at compile time; the graph
    build / path find / apply run at runtime inside the node.
  - `bridges` = the computed `from-ACPC_to-T1w` `.mat` emitted by
    `_register_acpc_to_t1w` (ANTsPy rigid, brain-masked) — wired **only when
    `ACPC` is one of the two endpoints**, consistent with the locked decision
    that BDT computes its own `ACPC↔T1w` bridge and does *not* use QSIPrep's
    stored `ACPC↔anat` transforms. The references for the registration are
    resolved exactly as in `init_map_scalar_to_surface_wf` (fixed = the T1w
    anatomical + brain mask; moving = the `space-ACPC` anatomical + brain mask).
  - `interpolation` = `nearest` for a **dseg** atlas (labels stay crisp),
    `linear` for a **pseg** atlas (fractional weights preserved). The atlas's
    dseg-vs-pseg nature is known at compile time from its suffix.
  - The full multi-hop chain (e.g. `ACPC → T1w → MNI152NLin6Asym`) is resolved
    by Spec 1's `chain_for_image_resample` over the local + bridge + TemplateFlow
    edges — that generality is why Spec 1 exists.
- When **not** `cross_space` (e.g. `fa_roi`), the atlas is parcellated directly
  (no warp, no bridge, no registration references needed).

The 4D atlas is warped by applying the resolved (spatial) transform to each
volume onto the scalar grid, yielding a 4D atlas on the scalar's grid.

### 3. Volumetric parcellation interface

A new coverage-aware `SimpleInterface` (in `bdt.interfaces`, e.g.
`interfaces/parcellate.py`) consuming the (possibly warped) atlas image, the
scalar image, its label sidecar `dseg.tsv` (when present), and `min_coverage`.
For each region *r*:

- weight `w_r(v)` = the atlas value of region *r* at voxel *v* (4D dseg/pseg →
  the value in volume *r*; 3D dseg → 1 where `label == r`, else 0).
- validity `V(v)` = the scalar voxel is finite **and** nonzero (matches the CIFTI
  path's "has data" definition).
- **weighted mean:** `mean_r = Σ_{v∈V} w_r(v)·scalar(v) / Σ_{v∈V} w_r(v)`.
- **coverage fraction:** `coverage_r = Σ_{v∈V} w_r(v) / Σ_{all v} w_r(v)`.
- if `coverage_r < min_coverage` → `mean_r = NaN` (mirrors the CIFTI
  NaN-masking of low-coverage parcels).

`min_coverage` default `0.5`, matching `_init_parcellate_cifti_wf`.

### 4. Output

One tidy long TSV, one row per region, columns:

- `index` — the region's integer id (4D: the volume index in input order; 3D: the
  label value).
- `name` — the region label from the atlas's `dseg.tsv` sidecar; falls back to
  the string of `index` for a bare 3D dseg with no sidecar.
- `mean` — the (NaN-masked) weighted mean.
- `coverage` — the per-region coverage fraction.

This matches the tidy style of `parcellate_scalar_as_tract_profile` and keeps
the `parcellate_scalar` action's single `.tsv` output (`out=_o('map', '.tsv',
… preserve_source=True)`); the action spec is unchanged.

## Error handling

- Cross-space with no resolvable transform chain → `NoTransformPathError` (Spec 1)
  propagates from the node, naming source, target, and reachable spaces.
- Cross-space needing the `ACPC↔T1w` bridge but with no `context.provider` to
  resolve the registration references → a clear `ValueError` naming the node
  (mirrors `init_map_scalar_to_surface_wf`).
- A scalar whose file type is neither CIFTI dscalar nor volumetric NIfTI → a
  clear error rather than silent misrouting.

## Testing

Small synthetic NIfTIs, hermetic (no network, no ANTs):

- **Same-space 4D dseg:** a known marker within one bundle mask → exact weighted
  (in-mask) mean; other regions independent.
- **4D pseg:** probabilistic weights → weighted mean differs from the plain mean
  by the expected amount.
- **3D dseg:** integer-label path → per-label mean, names from a sidecar and the
  index-fallback with none.
- **Coverage:** a region with data over less than `min_coverage` of its weight →
  `mean` NaN, `coverage` reported.
- **Cross-space seam:** a synthetic affine xfm passed as `local_transforms`
  warps the atlas onto the scalar grid, then parcellation lands the expected
  region means — exercising the `ResolveApplyTransforms → parcellate` wiring
  without network/ANTs (the ACPC↔T1w bridge registration node is not exercised
  in unit tests; its inputs are stubbed / the cross-space test uses a non-ACPC
  pairing so no bridge is needed).
- **CIFTI unchanged:** a CIFTI scalar still routes through
  `_init_parcellate_cifti_wf`.

## Out of scope (this spec)

- Changes to the CIFTI parcellation path or the `parcellate_scalar` action spec.
- Producing the `ACPC↔T1w` bridge node — it already exists
  (`_register_acpc_to_t1w`); this spec only wires it as a `bridges` input.
- Bridges other than `ACPC↔T1w` (any other computed registration) — the same
  `NotImplementedError` policy as `init_map_scalar_to_surface_wf` applies to the
  *computed* bridge, though the resolved chain may extend arbitrarily far via
  discovered local + TemplateFlow edges.
- `parcellate_scalar_as_tract_profile` (the along-tract path) — already
  implemented separately.
