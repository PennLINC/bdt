# Per-parcel statistics for parcellate_scalar

Date: 2026-07-21
Status: approved (design)

## Problem

A parcellated scalar map reports one number per parcel — the mean — and the format
does not say so. The tract profile, by contrast, is tidy and names its statistics
(`bundle`, `node`, `mean`, `std`). Users want the same for parcellated maps, and
want to choose which statistics are computed.

## Parameter

`parcellate_scalar` accepts a `statistics` parameter:

```yaml
- name: cbf_roi
  action: parcellate_scalar
  inputs: {scalar: load_cbf, atlas: bundle_rois}
  parameters:
    statistics: [mean, standard_deviation]
```

Default `['mean']`. Only `mean` and `standard_deviation` are implemented; any
other value raises at build time, listing the supported set. The names are chosen
to match nilearn's `strategy` vocabulary so the volumetric path can pass them
straight through.

`parcellate_timeseries` is **not** changed. A per-parcel statistic there would be
across *time* rather than across *voxels* — a different quantity that needs its
own definition.

## Statistic semantics

`standard_deviation` is the **population** SD (ddof=0), across the voxels or
vertices of a parcel. Verified during design that this is what
`NiftiLabelsMasker(strategy='standard_deviation')` returns (matches
`numpy.std()`, not `std(ddof=1)`), and Workbench's `-method STDEV` is likewise
population (`SAMPSTDEV` is the sample variant). The two modalities therefore
agree without correction.

## Output format

**One tidy TSV per node**, regardless of modality — a row per parcel, a column
per requested statistic:

```
node        mean     standard_deviation
LH_Vis_1    0.4213   0.0821
LH_Vis_2    0.5104   0.1132
```

Column names are the parameter values verbatim (`standard_deviation`, with the
underscore), matching the tract profile's readable style. The `node` column holds
the parcel name from the atlas labels table.

The coverage TSV is unchanged (`Node`, `coverage`).

**CIFTI additionally emits one `.pscalar.nii` per statistic**, because a CIFTI
scalar map holds exactly one value per parcel and cannot carry several statistics
in one file. The volumetric path has no native per-statistic file — the tidy TSV
*is* its output.

## Entity naming

A statistic contributes a `stat-` entity. Two rules:

1. **Normalize** the value to BIDS-legal characters: strip everything
   non-alphanumeric except `+`. So `standard_deviation` → `standarddeviation`.
2. **Concatenate with `+`** when the source already carries a statistic, source
   first. `+` is legal in BIDS entity values (BDT's own entity patterns are
   `[a-zA-Z0-9+]+`).

```
CBF   (no source stat)   ->  stat-mean            stat-standarddeviation
ALFF  (stat-alff)        ->  stat-alff+mean       stat-alff+standarddeviation
```

This applies **only to the per-statistic CIFTI files**. The tidy TSV holds every
statistic at once, so it keeps the source's `stat-` untouched (an ALFF map's
table stays `stat-alff`), preserving the current
`test_parcellate_scalar_preserves_source_naming` behavior.

## Implementation

### Volumetric

Both volumetric interfaces gain a `statistics` input and emit the tidy table.

- **3D dseg** — `NiftiLabelsMasker(strategy=<statistic>)` once per statistic,
  reusing the existing coverage machinery unchanged. `mean` and
  `standard_deviation` are both native nilearn strategies.
- **4D pseg** — `ProbSegParcellate` computes the weighted mean it already does,
  plus, when asked, the weighted population SD
  `sqrt(Σw(d − μ)² / Σw)` over the mask-restricted voxels.

Parcels below `min_coverage` are NaN in every statistic column, as today.

### CIFTI

`_init_parcellate_cifti_wf` builds one `CiftiParcellateWorkbench` per statistic
(`cor_method='MEAN'` / `'STDEV'`), each followed by the existing coverage-mask
step, and exposes them on `outputnode` as `out_<statistic>`. A new interface
merges the resulting pscalars into the single tidy TSV, so both modalities
produce the same table.

The `cifti_to_tsv` derive stays as-is for `parcellate_timeseries`; only
`parcellate_scalar` uses the new N→1 merge.

### Sink plan

`build_sink_plan` multiplies the native CIFTI product across the requested
statistics, giving each its composed `stat-` entity and pointing each at the
matching `outputnode` field. The tidy TSV is a single product in both modalities.

## Testing

- Numeric: `standard_deviation` matches `numpy.std()` (ddof=0) per parcel, for
  both the 3D dseg and 4D pseg paths; the weighted SD matches a brute-force
  computation over the mask.
- Format: the tidy table has a `node` column plus one column per requested
  statistic, in the order requested; defaults to `mean` alone.
- Low-coverage parcels are NaN in every statistic column.
- Entity naming: `standard_deviation` normalizes to `standarddeviation`;
  composition yields `stat-alff+mean` for a source that has one and `stat-mean`
  for one that does not; the tidy TSV keeps the source's `stat-`.
- Plan: a CIFTI scalar node with two statistics plans two pscalars and exactly
  one TSV; a volumetric one plans exactly one TSV and no pscalar.
- An unsupported statistic raises at build time naming the supported set.
- Acceptance: `tract_parcellate.yml` with `statistics: [mean,
  standard_deviation]` compiles and its outputs carry the expected names.

## Out of scope

- `median`, `sum`, `minimum`, `maximum`, `variance` — nilearn and Workbench both
  support them, and the validation list is the only thing to extend, but they are
  not requested now.
- Per-parcel statistics for `parcellate_timeseries`.
- The `map` suffix on the coverage product (not a BEP-016 suffix) — a separate
  pre-existing issue.
