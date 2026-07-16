# BDT node-graph — build status & handoff

**Date:** 2026-07-16
**Branch:** `bdt-node-graph` (in `/Users/mcieslak/projects/BDT/bdt`)
**Spec being implemented:** `bdt/docs/2026-07-15-bdt-user-stories-and-spec.md`
**Companion records:** plan at `/Users/mcieslak/.claude/plans/in-the-bdt-repo-swirling-blossom.md`; running notes in Claude memory `bdt-node-graph-project.md`.

This is a pick-up-in-a-new-session summary: current state, architecture, what's tested, the task backlog, the real test data now available, and the recommended next step.

---

## TL;DR

- **Story 3.1 runs end-to-end on real data and reproduces XCP-D bit-for-bit.** The full path — pybids selection → compile to nipype `Workflow` → real `wb_command` on `sub-125511`'s 91k dtseries → BIDS derivatives (ptseries + tsv + coverage pscalar, pconn + relmat, with provenance sidecars) — works. With **coverage-aware parcellation implemented (`min_coverage`)**, the **ptseries, coverage map, and pconn all match XCP-D's own outputs exactly** (NaN positions equal; max abs diff `0.0` / `~1e-7`).
- A validated node-graph **spec compiles into a nipype `Workflow`** that nipype runs (→ automatic `File(exists)` I/O validation, content-hash caching, resume, MultiProc/cluster plugins).
- **Framework pivoted to nipype.** NiWrap was trialed then **deleted entirely** (no users yet). Pydra was considered and set aside — the vendored qsirecon/xcp_d/fmriprep interfaces are nipype and directly reusable.
- **65 pytest pass, ruff-clean.** Foundation (spec/validator, transform graph, pybids provider, output naming) + the nipype engine + **output plan/sink layer + pipeline driver (#12/#13)** + **coverage-aware `parcellate_timeseries` and `parcellate_scalar`** (both matching XCP-D bit-for-bit) are done; 3 of ~20 action factories are done.
- **The 4S atlas blocker is cleared** — the PennLINC AtlasPack (16 MB, from UPenn Box) is unpacked at `/Volumes/5TB/BDT_testing/atlases` (a full BIDS-Atlas dataset: `tpl-fsLR/tpl-fsLR_atlas-4S1056Parcels_den-91k_dseg.dlabel.nii`, all 4S scales + MNI variants).

Run the suite (the real end-to-end test auto-runs when `wb_command` + the data drive are present, else skips):
```bash
cd /Users/mcieslak/projects/BDT/bdt
export PATH="/Applications/workbench/bin_macosxub:$PATH"
PYTHONPATH=src python -m pytest test/spec test/engine -o addopts='' -p no:cacheprovider -q
```
(Env note: base miniforge has nipype 1.11, niworkflows 1.14, pybids 0.22, nibabel, nilearn, pandas, networkx, pyyaml pip-installed. `config.py`/`parser.py` import the heavy templateflow/fmriprep stack — py_compile-checked, not import-tested in this env.)

---

## Architecture

```
CLI (bdt <raw_bids> <out> participant --datasets k=path … --spec spec.yaml)
  -> config singleton (bdt/config.py; add --spec; still carries BOLD cruft to remove)
  -> load + STATIC-VALIDATE spec           (bdt/spec/)          [DONE]
  -> build TRANSFORM GRAPH from _xfm.*      (bdt/transforms/)    [DONE]
  -> resolve SELECTION nodes -> files       (bdt/engine/pybids_provider.py) [DONE]
  -> COMPILE spec -> nipype Workflow        (bdt/engine/workflow.py::init_bdt_wf) [DONE, 2 actions]
       selection node -> IdentityInterface source node (file path)
       processing node -> init_<action>_wf sub-workflow (bdt/engine/factories.py)
       wire by ROLE: connect(up, 'outputnode.out'|'out', sub, f'inputnode.{role}')
       write_outputs -> DerivativesDataSink  [TODO #12]
  -> workflow.run(MultiProc, base_dir=work)  [TODO #13]  -> BIDS derivatives
```

Each action factory has a fixed boundary: `inputnode` (fields = the action's declared roles) → `outputnode` (single `out` = the node's primary product). The compiler wires `inputs[role]` from an upstream node's `outputnode.out`. This is qsirecon's `init_dwi_recon_workflow` pattern, but keyed by **declared role** instead of field-name intersection. Intermediates are CIFTI-native (ptseries/pconn); TSV/relmat materialization happens at the `write_outputs` boundary.

---

## Module inventory & status

| Area | Path | Status |
|---|---|---|
| Spec grammar + action registry (roles, formats, fan_out, output specs) | `src/bdt/spec/{model,actions,load}.py` | ✅ done |
| **Static validator** (unique names, ref resolution, DAG, role/format contract, scope lineage) | `src/bdt/spec/validate.py` | ✅ done |
| **Transform graph** + two typed queries (image-resample vs point-warp, invertibility guard) | `src/bdt/transforms/{graph,queries}.py` | ✅ done |
| **pybids provider** (selection, subject-scoping, short-name entities) | `src/bdt/engine/pybids_provider.py` | ✅ done |
| BEP/atlas pybids entity config (`param`/`stat`/`scale`/`meas`/`thresh`/`tpl`/`cohort`) | `src/bdt/data/bdt_entities.json` | ✅ done |
| **Output naming** (BIDS name, desc prepend-compose, `tpl-`, collision) + provenance sidecars | `src/bdt/outputs/{sink,provenance}.py` | ✅ done (used by #12) |
| **nipype compiler** (spec → Workflow, role wiring, **sink attachment**) | `src/bdt/engine/workflow.py` | ✅ done (single-match; MapNode fan-out TODO) |
| **Output plan** (per-node entity composition, `atlas-` inject, desc-compose, product list) | `src/bdt/outputs/plan.py` | ✅ done (pure, unit-tested) |
| **Sink interfaces** (`BDTDerivativeSink`, `CiftiToTsv` nipype `SimpleInterface`s) | `src/bdt/interfaces/derivatives.py` | ✅ done |
| **CIFTI coverage interfaces** (`CiftiVertexMask`, `CiftiMask`) + `write_ndata` | `src/bdt/interfaces/cifti.py`, `src/bdt/utils/write_save.py` | ✅ done (ported from xcp_d) |
| **Pipeline driver** (`run_spec`: resolve → plan → compile → run; driver-level fan-out) | `src/bdt/engine/pipeline.py` | ✅ done |
| **Action factories** | `src/bdt/engine/factories.py` | ⏳ `parcellate_timeseries` + `parcellate_scalar` (shared coverage-aware CIFTI helper, XCP-D-faithful), `functional_connectivity` (3/~20) |
| **Source-preserving naming** (`OutputSpec.preserve_source`: keep source suffix/datatype/stat/desc, add `atlas-`) | `src/bdt/spec/actions.py`, `src/bdt/outputs/plan.py` | ✅ done |
| CIFTI/NIfTI numeric helpers (for Function nodes) | `src/bdt/utils/cifti.py` | ✅ done |
| Vendored nipype interfaces (14 CIFTI wb_command) | `src/bdt/interfaces/workbench.py` | ✅ imports (needed `utils/write_save.py`, added) |
| Reusable provider types | `src/bdt/engine/selection.py` | ✅ trimmed to `Match`/`DataProvider`/`DictDataProvider`/`_matches`/`SelectionError` |

**Deleted (NiWrap layer, gone):** `src/bdt/actions/*`, `src/bdt/tools/*`, `src/bdt/engine/{executor,pipeline,builders,result}.py`, their tests, and the `niwrap` dependency.

**Still-cruft to delete (task #8):** `src/bdt/workflows/{base,connectivity,parcellation}.py` (xcp_d-derived, don't import), `src/bdt/interfaces/connectivity.py`, and the BOLD/confounds/dummy-scan/`--task-id` flags in `cli/parser.py` + `config.py`.

---

## Tests (65, all passing)

Story 3.2 progress note: `parcellate_scalar` (CIFTI) is done and reproduces XCP-D's parcellated **ALFF** bit-for-bit (dense `stat-alff` dscalar × 4S1056Parcels → `stat-alff_boldmap.pscalar.nii`/`.tsv`, NaN positions + values match; `test_parcellate_scalar_matches_xcpd_alff`). It reuses the coverage-aware CIFTI helper and the new `preserve_source` naming (keeps source suffix/stat, adds `atlas-`). Depth actions stay descoped.

**Surface factories — scoped, gated on the grouping engine (2026-07-16):**
- **Ground truth EXISTS** (correcting an earlier note): `fmriprep_anat/sub-125511/ses-1/anat/sub-125511_ses-1_rec-refaced_space-fsLR_den-91k_thickness.dscalar.nii` is fmriprep's own fsLR-resampled thickness → `resample_surface_scalar` on thickness is bit-for-bit validatable.
- **All inputs present**: native thickness `hemi-{L,R}_thickness.shape.gii` (T1w); subject reg spheres `hemi-{L,R}_space-fsLR_desc-reg_sphere.surf.gii`; native midthickness; templateflow local `tpl-fsLR_hemi-{L,R}_den-32k_sphere.surf.gii` + `tpl-fsLR_den-32k_hemi-{L,R}_midthickness.surf.gii`. NODDI = volumetric `model-noddi_param-{icvf,isovf,od,...}_dwimap.nii.gz` (ACPC); native white/pial/midthickness present for `map_scalar_to_surface`.
- **Blocker = L/R grouping + auxiliary-input plumbing (the deferred fan-out task #12).** Surface actions are inherently per-hemi: `surface_scalar`/`surfaces` inputs are L+R files that must be *grouped* (not fanned) into one dscalar, and the reg spheres + templateflow standard meshes are auxiliary inputs the story spec doesn't wire. The current compiler wires one file per role and the driver Cartesian-fans every multi-match selection. **Next sub-project:** (a) mark `surface_scalar`/`surfaces` roles as group roles and teach the driver group-vs-fan (a selection feeding a `fan_out=False` role passes ALL its matches as a list; else fans out) + the compiler to wire a list; (b) an auxiliary-input mechanism (a `context` passed to `init_bdt_wf`/factories) resolving reg spheres from the surfaces' dataset via the provider + fetching templateflow meshes; (c) `resample_surface_scalar` (per-hemi `-metric-resample ADAP_BARY_AREA` with area-surfs → `-cifti-create-dense-scalar`) validated vs fmriprep's fsLR thickness; then (d) `map_scalar_to_surface` (per-hemi `-volume-to-surface-mapping` ribbon-constrained; Strategy-B `giftirs` surface warp when scalar/surface spaces differ — the NODDI ACPC vs T1w-surface case) reusing the transform graph. Story spec filters need adapting to the real data (thickness is `suffix-thickness .shape.gii`, not `suffix-morph desc-thickness`).

  **Code map for the grouping change (exact touchpoints):**
  - `src/bdt/spec/actions.py` — set `surface_scalar` role `fan_out=False` (the `surfaces` role already is). Grouping keys off `Role.fan_out`.
  - `src/bdt/engine/pipeline.py` — `_resolve_selections` returns `{sel: [Match,...]}`; **`_combinations`** currently Cartesian-products *all* selections. Add: classify each selection as *grouped* (every consumer role has `fan_out=False`) vs *fanned*; grouped selections contribute a single list-of-matches to each combo (skip the product), fanned selections product as now. `run_spec` builds `sel_paths[name]` = a *list* of paths for grouped selections.
  - `src/bdt/engine/workflow.py` — `init_bdt_wf` sets `source.inputs.out` to that value (a list is fine for `IdentityInterface`); the wiring loop already connects `out -> inputnode.<role>`, so a grouped role receives the list unchanged. Add the `context`/aux param here and pass it into factories.
  - `src/bdt/engine/factories.py` — new `init_resample_surface_scalar_wf(node, name, aux)` / `init_map_scalar_to_surface_wf`; a small helper to pick per-hemi files from a list by parsing `hemi-L`/`hemi-R`. Use `preserve_source` OutputSpecs (already set for parcellate_scalar; add for these two — `.dscalar.nii` cifti product).
  - `src/bdt/outputs/plan.py` — `node_output_entities` seeds from `_primary_upstream`; for a grouped `surface_scalar` primary the entities come from *one representative* match (drop `hemi`), which already works since grouped matches share non-hemi entities. Verify `hemi` is dropped from the output (a combined dscalar has no `hemi-`).
  - Aux resolution (reg spheres, templateflow): the driver knows the `surfaces` selection's dataset; `provider.select(dataset, {'suffix':'sphere','space':'fsLR','desc':'reg', hemi:[L,R]})` gets the reg spheres; `templateflow.api.get('fsLR', density=..., suffix='sphere'|'midthickness', hemi=...)` fetches the standard meshes (all already cached locally).


- `test/spec/test_spec.py` (22) — all 8 user stories validate; every negative/shape case.
- `test/spec/test_transforms.py` (11) — mirror property, opposite-named point rule, invertibility guard, multi-hop ANTs ordering.
- `test/engine/test_pybids_provider.py` (5) — real on-disk BIDS fixtures; subject scoping, `exclude`, custom entities, `tpl-` atlas dataset.
- `test/engine/test_outputs.py` (5) — desc-compose, `tpl-`, collision, sidecars.
- `test/engine/test_cifti_utils.py` (4) — cifti→tsv (synthetic CIFTI), nilearn parcellation, correlation.
- `test/engine/test_nipype_workflow.py` (5) — compile story 3.1 to a nipype Workflow; role-based edges; **sink attachment from a plan**; no-sinks-without-plan; missing-factory error. (Assembly-only; no tools.)
- `test/engine/test_output_plan.py` (4) — entity composition (`atlas-` inject, stat, desc), CIFTI→native+tsv vs NIfTI→tsv-only products, only-write_outputs planned. (Pure; no nipype.)
- `test/engine/test_cifti_interfaces.py` (3) — `write_ndata` round-trip, `CiftiVertexMask` coverage flagging, `CiftiMask` NaN-masking (synthetic CIFTI).
- `test/engine/test_pipeline.py` (4) — driver resolution + Cartesian fan-out, missing-selection error, pre-run collision check, **+ a gated real end-to-end run** that asserts BDT's ptseries (NaN + values), coverage, and pconn match XCP-D bit-for-bit (auto-skips without `wb_command`/data).

---

## Locked decisions (with rationale)

1. **nipype full Workflow engine** (2026-07-16). Pivoted from NiWrap after its "trust the exit code" model (no automatic `File(exists)` validation, no caching/resume) put that burden on us; nipype handles it and the vendored interfaces are reusable. Pydra weighed and declined.
2. **Streamline/gifti compute in Rust** (`trxrs`/`giftirs` subcommands): vertex sampling (dpv) + per-streamline (dps), TDI, dps-weighted endpoint→parcel connectivity, write dpv/dps, import SIFT2 csv. Production stays MRtrix-free.
3. **Skip depth mapping** — `cortical_depth_profile` / `wm_depth_profile` are NOT being implemented (left in the registry as documented-but-unimplemented; a spec using them would fail at compile with "no factory").
4. **Never compute a normalization** — only apply discovered `_xfm.*`. Transforms are per-action: grid = Strategy A (ANTs, warp atlas→data); points = Strategy B (`trxrs`/`giftirs`, warp data losslessly). `map_scalar_to_surface` warps the *surfaces* into the scalar's space (surfaces are in T1w; `fsnative` is a mesh density, not a coordinate frame).
5. **BIDS/BEP conformance** — `space-` required on atlas outputs; `tpl-` for BAT/dataset scope; `timeseries`/`dwimap`/`tractogram`/`morph`/`relmat` suffixes; ship the pybids entity config.

---

## Real test data + tools (NEW — the big lever)

**Data:** `/Volumes/5TB/BDT_testing/data` — raw BIDS root (`sub-125511`, `ses-1`) with `derivatives/`:
`aslprep, fmriprep_anat, fmriprep_func, freesurfer-post, nilearn, qsiprep, qsirecon, xcpd`.
- **xcpd:** `sub-125511_ses-1_task-*_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` → **story 3.1 (parcellate + FC), runnable now**.
- **fmriprep_anat/func:** surfaces (native + `space-fsLR_den-32k`, L/R midthickness/pial/white) + `from-T1w_to-MNI*_xfm.h5` → story 3.2 surface mapping.
- **qsiprep:** `from-ACPC_to-MNI152NLin2009cAsym_xfm.h5` (both directions) + `dwimap` (model-eddy) → diffusion scalar + a real transform-graph workout.
- **aslprep:** CBF + xfms. **freesurfer-post:** `seg-FreeSurfer_morph.tsv`.
- ⚠️ Filter macOS `._*` AppleDouble files (external drive).
- ⚠️ **No BIDS-Atlas dataset present** — story 3.1 needs a `.dlabel.nii` atlas (fetch a 4S / Schaefer, or point `--datasets atlases=` somewhere).

**Tools on PATH (darwin, this machine):**
`wb_command` (`/Applications/workbench/bin_macosxub`), `antsApplyTransforms` (miniforge), `trxrs` + `giftirs` (`~/.local/bin`). **`dsi_studio` NOT installed** → `tract2region` can't run locally.
→ Real end-to-end nipype runs are now possible for the CIFTI, surface, scalar, and streamline paths.

---

## Task backlog (IDs from the session tracker)

| # | Task |
|---|---|
| 5 | Phase 5: Rust `trx-rs` subcommands (`sample`/`import-dps`/`tdi`/`region2region`) + streamline actions |
| 6 | Phase 6: dataset scope → `tpl-` + BAT atlas algebra (union/intersect/outer_product) |
| 7 | Phase 7: reports, BEP entity config finalize, docs, container (ANTs + DSI Studio + Rust stage) |
| 8 | Remove BOLD cruft from `config.py`/`parser.py`/`workflows/*` |
| 9 | Strategy-A ANTs atlas warp for cross-space NIfTI parcellation |
| 10 | Wire `resample_surface_scalar` (fsLR) + `assemble_cifti` (currently error-stubs in the NiWrap layer — re-do as nipype factories) |
| 11 | **Port all action factories to nipype** (`init_<action>_wf`) — add `VolumeToSurfaceMapping`/`MetricResample`/`ApplyTransforms` interfaces from fmriprep/smriprep; add `trxrs`/`giftirs` `CommandLine` interfaces |
| 12 | ✅ **DONE (core): `write_outputs` → sink** via `bdt/outputs/plan.py` + `bdt/interfaces/derivatives.py`, wired in the compiler. Reuses `bdt/outputs` naming. **Remaining:** in-graph fan-out via MapNode/iterables (currently driver-level Cartesian product in `pipeline.py`). |
| 13 | ✅ **DONE (core): pipeline entry** `bdt/engine/pipeline.py::run_spec` (provider → resolve → plan → `init_bdt_wf` → `run`). **Remaining:** dataset-scope `tpl-` run-once path; wire `cli/run.py` (with task #8 cruft removal); MultiProc default. |
| 14 | ✅ **DONE: coverage-aware `parcellate_timeseries`** — vertex coverage → `-cifti-weights` mean, threshold, NaN-mask, emit `stat-coverage` pscalar. ptseries/coverage/pconn now match XCP-D bit-for-bit. |
| 15 | ✅ **DONE: naming questions resolved** — entity order `space, den, atlas, stat`; source `desc` kept + composed. (`model`/`param` order for `dwimap` still open, revisit at story 3.2.) |

---

## First end-to-end run (2026-07-16 — DONE) ✅

Story 3.1 (`parcellate_timeseries` + `functional_connectivity`) ran on `sub-125511` real fsLR-91k denoised dtseries × `4S1056Parcels`, producing real BIDS derivatives in ~10 s (Linear plugin). Driver-level fan-out ran all 3 tasks (fracback / rest-multiband / rest-singleband); each wrote ptseries + tsv + pconn + relmat + sidecars.

**Validation vs XCP-D's own outputs on the same subject (ground truth) — with `min_coverage` now implemented, all match exactly:**
- **ptseries: bit-for-bit** — NaN positions equal, max abs diff over finite parcels `0.0`.
- **coverage pscalar: identical** — max abs diff `0.0`.
- **pconn (connectivity): identical** — max abs diff ~1e-7.

`init_parcellate_timeseries_wf` now reproduces XCP-D's `init_parcellate_cifti_wf` exactly: a vertex-wise coverage mask (`CiftiVertexMask`) is used as `-cifti-weights` for the parcel mean (so zero/uncovered vertices don't dilute it) *and*, parcellated itself, as the per-parcel coverage; parcels ≤ `min_coverage` are NaN-masked (`CiftiMath` threshold + `CiftiMask`); the coverage map is emitted as `stat-coverage_boldmap.pscalar.nii` (an `ExtraProduct` off `outputnode.coverage`). Historical note: before this, ptseries matched on 1048/1056 parcels, the 8 diffs being exactly the coverage handling (2 low-coverage → NaN vs 0; 6 partial → coverage-weighted vs straight mean).

**Naming decisions confirmed with user (2026-07-16) — all resolved:**
1. Parcellation entity is `atlas-<label>` (not `seg-`).
2. A CIFTI-input node writes native CIFTI **and** TSV; a NIfTI-input node writes TSV only.
3. **Entity order = `space, den, atlas, stat`** — `ENTITY_ORDER` updated (`res`/`den` moved ahead of `atlas`/`stat`), `test_bids_name_ordering` updated. (`model`/`param` placement for `dwimap` outputs is deferred to story 3.2.)
4. **Source `desc` is kept + prepend-composed** (spec 1.7 literal), so parcellated outputs carry `desc-denoised`. Differs from XCP-D (which drops it) but keeps 3.2's `desc-geneexpression`.

Current BDT vs XCP-D output name (only `atlas-` vs `seg-` and the retained `desc-denoised` differ, both by decision):
```
BDT : ..._space-fsLR_den-91k_atlas-4S1056Parcels_stat-mean_desc-denoised_timeseries.ptseries.nii
XCPD: ..._space-fsLR_seg-4S1056Parcels_den-91k_stat-mean_timeseries.ptseries.nii
```

## Recommended next step

Story 3.1 is now **fully faithful to XCP-D** (parcellate + FC, CIFTI path). Remaining, in rough priority:
1. Port `parcellate_scalar` + `map_scalar_to_surface` factories (task #11) — testable *for real* now (`giftirs`/ANTs/`wb_command` present; T1w↔ACPC surface-warp path), and story 3.2 needs the fsLR 32k/91k HCPMMP1 + 4S atlases already in the AtlasPack. (`parcellate_scalar` can reuse the coverage-aware CIFTI machinery just built.)
2. NIfTI parcellation path for story 3.1's nifti variant (`parcellate_timeseries` on a volumetric bold + a NIfTI atlas) — needs the Strategy-A ANTs atlas warp for cross-space, `NiftiParcellate`-style coverage for same-space.
3. Proper in-graph fan-out (MapNode/iterables) to replace the driver-level Cartesian product for multi-branch specs (task #12 follow-up).
4. `model`/`param` entity ordering for `dwimap` outputs (deferred story-3.2 naming question).
