# Design: `init_tractogram_to_pseg_wf`

Date: 2026-07-17
Status: Approved for planning
Scope: End-to-end — factory + interfaces + action spec + output plan

## Problem

`init_tractogram_to_pseg_wf` (`bdt/src/bdt/engine/factories.py`) is currently
non-working pseudocode. It must turn a set of bundle-wise tractograms
(`.tck.gz`, ACPC space) into a 4D probabilistic/discrete segmentation NIfTI —
one volume per bundle — plus a BIDS label TSV mapping each volume to its bundle
name. Downstream (`scripts/tract_parcellate.yml`), the result is consumed as an
`atlas` by `parcellate_scalar_as_roi`.

The pseudocode has several concrete defects to fix:

- It reads the `threshold` **parameter** via `context.role_space(node, 'threshold')`
  (a role/entity lookup) — wrong; it must read `node.parameters.get('threshold')`.
- Its `outputnode` uses `seg`/`tsv` fields, but the compiler
  (`workflow.py:123`) wires downstream consumers and passthrough sinks from
  `outputnode.out`. The primary segmentation must live on `out`.
- It references four interfaces that do not exist (`Tckmap`, `ConcatenateNiftis`,
  `ThresholdNifti`, `EntitiesToSegTSV`).
- It has a `seg_buffer` node constructed with a misplaced `name=` kwarg (inside
  `IdentityInterface(...)` instead of on `pe.Node`).
- The docstring's suffix rule is self-contradictory (both branches say
  "threshold is a number").

## Decisions

| Question | Decision |
|---|---|
| Scope | **End-to-end**: factory + interfaces + `actions.py` OutputSpec + `outputs/plan.py`. |
| Reference grid for `tckmap` | **Resolve via `FactoryContext.find_reference`** (a `space-ACPC` DWI reference, session-matched), like `map_scalar_to_surface`. No new role. |
| Interfaces | **Reuse** `nipype.algorithms.misc.Gunzip` + `nipype.interfaces.mrtrix3.ComputeTDI`; **custom** `ConcatenateNiftis`, `ThresholdNifti`, `EntitiesToSegTSV`. |
| probseg values / threshold | **Normalize each bundle map to [0,1]** (peak/unit-max); `threshold` compares against the normalized values. |
| Suffix naming | `threshold` unset → **`probseg`** (4D, [0,1]); `threshold` set → **`dseg`** (4D, binarized `value > threshold`). |
| Grouped-list input | **Already delivered** by the existing pipeline (see below); verify, do not re-implement. |

### Note on the grouped-list input (correction)

An earlier framing assumed the compiler could not hand `inputnode.tractograms` a
list. That is incorrect. `tractograms` is a grouped role (`fan_out=False`), so:

- `_classify_selections` (`pipeline.py:105`) marks the feeding `select_data`
  node **grouped**;
- `_combinations` (`pipeline.py:144`) passes its **entire match list**;
- `sel_paths` (`pipeline.py:233`) yields a **list of paths**;
- the compiler's source `IdentityInterface.out` forwards that list unchanged into
  `inputnode.tractograms`.

`init_resample_surface_scalar_wf` already relies on this mechanism for its
grouped L/R `surfaces` (covered by `test_compile_resample_surface_scalar`).
Therefore **no compiler change is required**; the plan only *verifies* the
list reaches the new factory's `MapNode`.

## Factory design

`inputnode` fields: `tractograms` (a **list** of `.tck.gz` paths).
`outputnode` fields: `out` (the seg — probseg or dseg) and `tsv` (label table).

The ACPC reference is resolved **at build time** from `context`
(`find_reference`, session-matched to `tractograms` via
`context.role_session(node, 'tractograms')`); it raises if no `FactoryContext`
provider is supplied. `threshold = node.parameters.get('threshold')` decides
whether the binarize node is inserted.

```
inputnode.tractograms (list)
  ├─► MapNode Gunzip (iterfield=in_file)              # .tck.gz -> .tck   [reuse nipype.algorithms.misc.Gunzip]
  │     └─► MapNode ComputeTDI (iterfield=in_file)    # per-bundle raw TDI on the reference grid [reuse nipype mrtrix3]
  │           reference = <find_reference result>     #   (fixed, non-iterated input)
  │           └─► ConcatenateNiftis(normalize=True)   # peak-normalize each vol -> [0,1], stack -> 4D probseg
  │                 └─► [threshold set] ThresholdNifti(threshold, binarize=True)   # value > threshold -> 4D dseg
  │                       └─► outputnode.out          # the seg
  └─► EntitiesToSegTSV(entity='bundle')               # filenames -> index/name TSV (input order preserved)
        └─► outputnode.tsv
```

When `threshold` is unset, `ConcatenateNiftis` connects straight to
`outputnode.out`. Both `ConcatenateNiftis` and `EntitiesToSegTSV` consume the
same `inputnode.tractograms` list, so the 4th-dim volume order and the TSV row
order are identical by construction.

`ComputeTDI`'s `reference` (template image) fixes the output grid; `in_file` is
the per-bundle `.tck`; default contrast (`tdi`, streamline count) is used.

## Interfaces (new module `bdt/interfaces/tractography.py`)

All new interfaces are `nipype` `SimpleInterface`s built on `nibabel`/`numpy`
(no new heavy dependencies). Reused as-is: `nipype.algorithms.misc.Gunzip`,
`nipype.interfaces.mrtrix3.ComputeTDI`.

- **`ConcatenateNiftis`** — inputs `in_files: list[File]`, `normalize: Bool=True`,
  `out_file`. When `normalize`, each 3D input is divided by its own positive max
  (empty maps stay all-zero) to land in `[0, 1]`; inputs are then stacked along a
  new 4th axis into one 4D NIfTI (affine/header from the first input; shape
  consistency asserted). Output `out_file` (4D probseg).
- **`ThresholdNifti`** — inputs `in_file: File`, `threshold: Float`,
  `binarize: Bool=True`, `out_file`. Emits `(data > threshold)` as the segmentation
  (uint8 when `binarize`), preserving 4D shape and header. Only instantiated when
  `threshold is not None`.
- **`EntitiesToSegTSV`** — inputs `in_files: list[File]`, `entity: Str='bundle'`,
  `out_file`. Parses the entity value from each filename (BIDS `key-value`,
  order-preserving) and writes a TSV with columns `index` (1-based volume order)
  and `name` (the entity value). Raises if the entity is missing from any file.

## Action spec + output plan changes

The suffix must switch on `threshold`, and the label TSV must be sinked — both
kept registry-driven (no `if/elif` ladder), via two small generic extensions.

1. **`OutputSpec.dynamic_suffix`** — optional `Callable[[dict], str]` taking
   `node.parameters` and returning the primary suffix. `build_sink_plan`
   (`outputs/plan.py`) applies it (when present) in place of the static
   `out.suffix`. Extension is unchanged (`.nii.gz`).

2. **`ExtraProduct` label-TSV support** — add `match_primary_suffix: bool=False`;
   when set, the extra product's suffix follows the resolved primary suffix. The
   extra-product loop (`outputs/plan.py:307`) already honors `cifti_only=False`,
   so a volumetric node can carry an extra product.

`tractogram_to_pseg`'s `OutputSpec` becomes:

```python
out=_o(
    'probseg', '.nii.gz', 'dwi',
    primary_role='tractograms',
    dynamic_suffix=lambda p: 'dseg' if p.get('threshold') is not None else 'probseg',
    extra=(ExtraProduct('tsv', 'probseg', '.tsv',
                        cifti_only=False, match_primary_suffix=True),),
)
```

(`_o` gains a `dynamic_suffix` passthrough; `ExtraProduct` gains
`match_primary_suffix`. Both default to the current behavior, so no other action
changes.)

## Tests

Mirror the `map_scalar_to_surface` pattern in `test/engine/test_nipype_workflow.py`
(tool-free build-time assembly with a `DictDataProvider` stub + `FactoryContext`):

- **Probseg path** (no `threshold`): factory builds; graph contains no
  `ThresholdNifti` node; `ConcatenateNiftis` connects to `outputnode.out`.
- **Dseg path** (`threshold: 0.0`): a `ThresholdNifti` node is inserted between
  concat and `outputnode.out`.
- **Reference resolution**: `find_reference` resolves the ACPC reference from the
  stub provider; a missing/absent provider raises `ValueError`.
- **Outputnode contract**: `outputnode` exposes both `out` and `tsv`.
- **Grouped list reaches the factory** (verification of the "no compiler change"
  claim): compile `scripts/tract_parcellate.yml`'s `bundle_rois` node via
  `init_bdt_wf` with a multi-match stub selection and assert `inputnode.tractograms`
  is the full list feeding the `MapNode`s.

Output-plan tests in `test/engine/test_output_plan.py`:

- `threshold` unset → primary product suffix `probseg`; `threshold` set →
  `dseg` (exercises `dynamic_suffix`).
- The label-TSV extra product is planned (`.tsv`, `source_field='tsv'`, suffix
  matching the primary).

Interface unit tests (run-time, tiny synthetic NIfTIs / fake filenames):

- `ConcatenateNiftis`: normalization to `[0,1]`, correct 4D stacking/order,
  all-zero input stays zero.
- `ThresholdNifti`: `value > threshold` semantics, `binarize` dtype, 4D preserved.
- `EntitiesToSegTSV`: correct `index`/`name` rows in input order; missing entity
  raises.

## Out of scope / follow-ups

- General fan-out over multi-match selections and list-valued *roles* (multiple
  upstream nodes into one role) — the compiler's noted follow-up; unrelated to the
  grouped single-selection path this feature uses.
- BIDS strictness of a 4D `dseg`/`probseg` label file (the label TSV uses
  `index`/`name`; a 4D discrete `dseg` is non-standard BIDS but is the project's
  intended representation here).
- Whether `parcellate_scalar_as_roi` needs the label TSV alongside the atlas NIfTI
  when consuming `bundle_rois` — a downstream concern handled by existing atlas
  machinery.
```

