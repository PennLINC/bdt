# Design: volumetric `parcellate_timeseries` + `functional_connectivity`

**Date:** 2026-07-20
**Status:** Approved (design)

**Scope:** The sibling of the volumetric `parcellate_scalar` path (2026-07-20
volumetric-parcellate-scalar). That work made only `parcellate_scalar`
volumetric; `parcellate_timeseries` and `functional_connectivity` are still
CIFTI-only, so a NIfTI bold (`nifti_parcellate.yml`) is wrongly routed into the
CIFTI subworkflow (`CiftiCreateDenseFromTemplate`/`CiftiVertexMask`) and fails.
This spec adds the volumetric path for both actions so a NIfTI bold + NIfTI
atlas parcellate and correlate end to end. It reuses Spec 1
(`ResolveApplyTransforms`) and the volumetric-scalar machinery.

## Problem

`scripts/nifti_parcellate.yml`: a fMRIPrep bold in `MNI152NLin6Asym` (NIfTI)
parcellated by an AtlasPack `4S456Parcels` dseg in `MNI152NLin2009cAsym` (NIfTI),
then correlated. Both `init_parcellate_timeseries_wf` and
`init_functional_connectivity_wf` are hard-wired to the CIFTI path
(`_init_parcellate_cifti_wf` / `CiftiCorrelation`), which cannot consume NIfTI
(and needs `wb_command`, not installed). The two template spaces differ, so the
atlas must be warped into the bold's space first — the easy cross-space case
(both standard templates → Spec 1's TemplateFlow `.h5` edges bridge them; no
ACPC bridge, no `wb_command`).

## Key decisions

1. **Route on CIFTI-ness, not extension.** A role fed by a *processing* node has
   no `extension` entity (`node_output_entities` drops it), so the extension
   check `parcellate_scalar` uses cannot route `functional_connectivity` (whose
   input is the parcellate node). Use `_produces_cifti(spec, resolved)`
   (`bdt.outputs.plan`), which propagates CIFTI-ness through processing nodes.
2. **Reuse the volumetric-scalar cross-space warp.** The atlas is warped into the
   data's space with `ResolveApplyTransforms` exactly as in
   `_init_parcellate_volumetric_wf`; here the chain is standard→standard (TF
   edges), nearest for the dseg atlas.
3. **Wide (timepoints × regions) TSV** for the parcellated timeseries — the
   orientation `functional_connectivity` needs (`tsv_correlation` correlates
   columns). Region columns are keyed by index (same sidecar-naming limitation
   as `parcellate_scalar`).
4. **Coverage: NaN-mask + a separate coverage TSV** (user-selected). Low-coverage
   region columns are NaN-masked in the wide TSV (so FC yields NaN for them,
   mirroring the CIFTI path's NaN-masked parcels), and a per-region coverage TSV
   is emitted as the action's `coverage` `ExtraProduct`.

## Components

### A. Routing signal — `FactoryContext.role_is_cifti(node, role)`

New method backed by `_produces_cifti(self.spec, self.resolved)`, memoized like
`_entities_by_node`. Returns whether the file feeding `role` is CIFTI; when no
`spec` is available (build-stub path), falls back to the resolved match's
`is_cifti(path)` (or `False`). Used by the three factories below.
`init_parcellate_scalar_wf` is switched from its `role_extension` check to
`role_is_cifti` for consistency and to correctly handle a processing-fed scalar.

### B. `init_parcellate_timeseries_wf` — routing + volumetric subworkflow

Branch on `role_is_cifti(node, 'timeseries')`:
- CIFTI → `_init_parcellate_cifti_wf(node, name, 'timeseries', 'parcellated.ptseries.nii')` (unchanged).
- volumetric → a new subworkflow that mirrors `_init_parcellate_volumetric_wf`:
  `inputnode(timeseries, atlas)`; when the atlas and data spaces differ, insert
  `ResolveApplyTransforms` (nearest for a dseg / linear for a pseg, reference =
  the bold's spatial grid, `local_transforms` from `discover_transforms`, ACPC
  bridge only if ACPC is an endpoint — not the case here); then
  `ParcellateVolumetricTimeseries`. `outputnode` exposes `out` (wide TSV) and
  `coverage` (per-region coverage TSV).

To avoid duplication, the shared cross-space-warp scaffolding of
`_init_parcellate_volumetric_wf` is factored into a helper both the scalar and
timeseries subworkflows call (returns the atlas field — warped or direct — given
the `in_role`).

### C. `ParcellateVolumetricTimeseries` interface

New `SimpleInterface` (in `bdt.interfaces.parcellate`): inputs `timeseries`
(4D NIfTI), `atlas` (3D dseg or 4D dseg/pseg), `min_coverage` (default 0.5),
`out_file`, `coverage_file`. A voxel is **valid** (has data) when its across-time
mean is finite and nonzero — one static validity mask, matching the scalar path's
`finite & nonzero`. Per region *r*: `weight_r`; `coverage_r = Σ_valid w_r / Σ_all
w_r`; and a weighted mean **per timepoint** `t`, `mean_r(t) = Σ_valid w_r·data(t)
/ Σ_valid w_r`, reusing the scalar path's region-weight computation. Outputs:
- `out_file` — wide TSV, rows = timepoints, columns = region indices; a region's
  whole column is NaN when `coverage_r < min_coverage`.
- `coverage_file` — per-region TSV (`index`, `coverage`).

A pure `_parcellate_volumetric_timeseries(...)` function holds the logic (like
`_parcellate_volumetric`), reusing the region-weight computation.

### D. Coverage `ExtraProduct` made format-aware

`ExtraProduct` gains a `volumetric_extension` field (default `None`). In
`build_sink_plan`, an `ExtraProduct` is emitted for a volumetric node when it is
**not** `cifti_only` **or** has a `volumetric_extension`; the product's extension
is `ep.extension` for a CIFTI node and `ep.volumetric_extension` for a volumetric
node. The `parcellate_timeseries` coverage product becomes
`ExtraProduct('coverage', 'boldmap', '.pscalar.nii', volumetric_extension='.tsv',
cifti_only=False, stat='coverage')`. The volumetric subworkflow's
`outputnode.coverage` feeds it. (CIFTI behavior is unchanged: it still emits the
`.pscalar.nii` coverage map.)

### E. `init_functional_connectivity_wf` — routing

Branch on `role_is_cifti(node, 'timeseries')`:
- CIFTI ptseries → existing `CiftiCorrelation` → `.pconn.nii` (unchanged).
- volumetric wide TSV → a node running the existing
  `bdt.utils.cifti.tsv_correlation` → a region×region correlation relmat TSV on
  `outputnode.out`.

The `functional_connectivity` action's primary output is already `.tsv` with a
`cifti_extension='.pconn.nii'` override, so the volumetric product is a `.tsv`
with no action-spec change.

## Error handling

- Cross-space with no resolvable chain → `NoTransformPathError` (Spec 1)
  propagates.
- A CIFTI-vs-volumetric mismatch between the parcellated series and FC is
  impossible: both route on the same `_produces_cifti` signal for the same data
  lineage.

## Testing

Hermetic synthetic NIfTIs, no network / no `wb_command`:

- **Timeseries parcellation:** a small 4D bold + 4D/3D atlas → wide TSV with the
  expected per-timepoint weighted means and shape `(T, n_regions)`.
- **Coverage:** a region covered below `min_coverage` → its whole column NaN in
  the wide TSV, and the coverage TSV reports the fraction.
- **Sink plan:** a volumetric `parcellate_timeseries` node emits a `.tsv` primary
  **and** a `.tsv` coverage product; a CIFTI node still emits `.ptseries.nii` +
  `.pscalar.nii` coverage (format-aware `ExtraProduct` regression).
- **Routing:** `role_is_cifti` true for a CIFTI-lineage input, false for a NIfTI
  one, propagated through the parcellate processing node to FC; CIFTI
  `parcellate_timeseries`/`functional_connectivity` still build the CIFTI nodes.
- **FC volumetric:** `tsv_correlation` on a known wide TSV → the expected
  correlation relmat.
- **Cross-space build:** atlas `MNI152NLin2009cAsym` vs bold `MNI152NLin6Asym`
  builds `warp_atlas` and no `register_acpc`.
- **Acceptance:** `scripts/nifti_parcellate.yml` compiles end to end (bold →
  volumetric parcellate → volumetric FC).

## Out of scope (this spec)

- Region names remain index-based (the atlas `dseg.tsv` sidecar is still not
  carried by role wiring; the `atlas_labels` seam exists but is unwired).
- Changes to the CIFTI `parcellate_timeseries`/`functional_connectivity` paths
  beyond the routing branch.
- Retrofitting `parcellate_scalar`'s output/coverage — only its routing signal
  changes (to `role_is_cifti`).
