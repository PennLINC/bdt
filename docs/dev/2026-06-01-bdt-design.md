# BDT Design Spec
**Date:** 2026-06-01 (rev. 2026-06-02)
**Status:** Approved; transform system revised per plan review
**Participants:** Taylor Salo, Matt Cieslak

> **Rev 2026-06-02:** §7 reworked into a geometry-polymorphic transform model (grid / point / diffusion-model) per `2026-06-01-bdt-plan-review.md`. Decisions adopted: streamlines warp tracts→atlas (Strategy B) via `trxrs`; surface cross-mesh handled with `wb_command -label-resample`; `odx-rs` is in scope for v1 (`parcellate_fixel`/`parcellate_odf`); endpoint→parcel connectivity lives in `bdt` on trxrs read-only bindings. The three Rust repos (`trx-rs`, `gifti-rs`, `odx-rs`) are now actually invoked, via `CommandLine` interfaces, and require a container build stage.
>
> **Rev 2026-06-02b (verified against the Rust repos):** Binary/package names and CLI surfaces were checked against the cloned `trx-rs`, `gifti-rs`, `odx-rs` sources. Confirmed: binaries are `trxrs`, `giftirs`, `odx`; the trx-rs Python package is **`trxrs`** (`trxrs.load(p).positions()/.offsets()`), not `trx-python`. Corrected here: (1) `odx` operates on an **`.odx` container** — SH/ODF/fixel inputs must be ingested (`odx convert` / `from_mrtrix` / `from_pyafq_aodf`) before `odx transform`; (2) `odx transform` follows the **image/same-named** h5 convention (opposite of `trxrs`/`giftirs`) and has `--mode mrtrix` (default) vs `--mode ants` (fixel-cardinality-preserving, paired h5s); (3) `odx transform` is **CLI-only** — the PyO3 bindings expose reading/peaks/conversion but no transform function. Repo provenance: `trx-rs` = `tee-ar-ex/trx-rs`; `gifti-rs`, `odx-rs`, `itk-transforms-rs` = `PennLINC`.

---

## 1. Overview

`bdt` is a Python package implementing two BIDS Apps as nipype workflows:

- **BDT** (`bdt` CLI) — BIDS Derivatives Transformer: applies atlases to BIDS derivative datasets and writes out parcellated data and connectivity matrices.
- **BAT** (`bat` CLI) — BIDS Atlas Transformer: manipulates BIDS Atlas datasets via atlas algebra (intersection, union, outer product).

Both tools live in the same package (`src/bdt/`) with two entry points, share infrastructure (config, BIDS I/O, transform engine, interfaces), and follow the nipost BIDS App pattern established by the fMRIPost family.

---

## 2. Package Layout

```
src/bdt/
├── cli/
│   ├── bdt_run.py       ← bdt entry point
│   ├── bdt_parser.py    ← bdt CLI parser
│   ├── bdt_workflow.py  ← bdt workflow builder
│   ├── bdt_version.py   ← bdt version checks
│   ├── bat_run.py       ← bat entry point
│   ├── bat_parser.py    ← bat CLI parser
│   ├── bat_workflow.py  ← bat workflow builder
│   └── bat_version.py   ← bat version checks
├── config.py            ← shared singleton (bdt_workflow + bat_workflow sections)
├── workflows/
│   ├── bdt/
│   │   ├── base.py          ← init_bdt_wf, init_single_subject_wf
│   │   ├── timeseries.py    ← init_timeseries_run_wf (BOLD, ASL 4D; any format)
│   │   ├── scalar.py        ← init_scalar_run_wf (CBF, FA/MD, thickness, etc.)
│   │   ├── streamlines.py   ← init_streamlines_run_wf (TRX/TCK; warps tracts→
│   │   │                        atlas space via trxrs, then endpoint connectivity)
│   │   └── diffusion.py     ← init_diffusion_model_run_wf (SH/ODF/fixel via odx)
│   ├── bat/
│   │   ├── base.py          ← init_bat_wf, init_bat_dataset_wf
│   │   └── algebra.py       ← init_intersect_wf, init_union_wf, init_outer_product_wf
│   ├── parcellation.py      ← shared: init_load_atlases_wf (warps atlases into
│   │                            grid-data space; defined here, not just referenced),
│   │                            init_parcellate_{nifti,gifti,cifti,fixel,odf}_wf,
│   │                            init_surface_volume_parcellate_wf,
│   │                            init_label_resample_wf (wb_command -label-resample)
│   └── connectivity.py      ← init_functional_connectivity_{nifti,gifti,cifti}_wf,
│                                init_streamline_connectivity_wf
├── interfaces/
│   ├── bids.py              ← DerivativesDataSink (existing)
│   ├── connectivity.py      ← NiftiParcellate, TSVConnect, ConnectPlot, etc. (existing)
│   ├── workbench.py         ← WBCommand wrappers (existing)
│   ├── reportlets.py        ← AboutSummary, SubjectSummary (existing)
│   ├── ants.py              ← ApplyTransforms wrapper (missing, add)
│   ├── nilearn.py           ← IndexImage, NiftiLabelsMasker (missing, add)
│   ├── censoring.py         ← Censor (missing, add)
│   ├── plotting.py          ← PlotCiftiParcellation (missing, add)
│   ├── rust.py              ← CommandLine wrappers: TrxTransform, TrxConvert,
│   │                            GiftiTransform, OdxTransform (new; shell out to
│   │                            trxrs / giftirs / odx binaries)
│   ├── gifti.py             ← SurfaceVolumeParcellate (vertex-sample a volume
│   │                            atlas after giftirs transform); surface↔surface
│   │                            parcellation routes through workbench.py (new)
│   ├── tractography.py      ← StreamlineConnectivity (endpoint→parcel via trxrs
│   │                            read-only bindings), BundleStats (new)
│   ├── odx.py               ← FixelParcellate, OdfParcellate (new)
│   └── atlas.py             ← AtlasIntersect, AtlasUnion, AtlasOuterProduct (new)
└── utils/
    ├── bids.py              ← collect_derivatives, collect_atlases (extend for ASL, dMRI, GIFTI)
    ├── transforms.py        ← build_transform_graph, chain_for_image_resample,
    │                            chain_for_point_warp (new)
    ├── atlas.py             ← atlas collection/selection (extracted from bids.py)
    ├── utils.py             ← (existing)
    └── filemanip.py         ← (existing)
```

`pyproject.toml` entry points:
```toml
[project.scripts]
bdt = "bdt.cli.bdt_run:main"
bat = "bdt.cli.bat_run:main"
```

---

## 3. CLI

### BDT

Standard BIDS App pattern (participant level). Atlas datasets are passed via `--datasets` alongside derivative datasets. Atlas selection and operation configuration live entirely in the spec file.

```bash
bdt <bids_dir> <output_dir> participant \
  --datasets fmriprep=/path/to/fmriprep \
             aslprep=/path/to/aslprep \
             qsirecon=/path/to/qsirecon \
             atlases=/path/to/bids-atlas-dataset \
  --spec /path/to/bdt_spec.yaml \
  --participant-label sub-01
```

### BAT

Dataset level (atlases are not per-subject). `bids_dir` is the input BIDS Atlas dataset; no `--datasets` is needed for the single-dataset case.

```bash
bat <bids_dir> <output_dir> dataset \
  --spec /path/to/bat_spec.yaml
```

If atlases from multiple input datasets need to be combined, `--datasets` can be used to name additional atlas datasets.

---

## 4. Spec Grammar

### `bdt_spec.yaml`

A list of `sources`. Each source binds a derivative type (identified by BIDS entities) to a set of atlases and operations. Atlases are specified as lists of entity-filter dicts, passed directly to pybids `.get()`. File format (NIfTI/GIFTI/CIFTI) is auto-detected from the matched file's extension at workflow-build time.

```yaml
sources:
  - suffix: bold
    datasets: [fmriprep]
    operations: [parcellate_timeseries, functional_connectivity]
    atlases:
      - atlas: HCPMMP1
      - atlas: Schaefer400
        desc: 400Parcels17Networks

  - suffix: cbf
    datasets: [aslprep]
    operations: [parcellate_scalar]
    atlases:
      - atlas: HCPMMP1
        res: 2

  - suffix: asl
    datasets: [aslprep]
    operations: [parcellate_timeseries]
    atlases:
      - atlas: HCPMMP1

  - suffix: tractogram   # BEP 046: suffix is `tractogram` (not `tractography`)
    datasets: [qsirecon]
    operations: [streamline_connectivity, bundle_stats]
    atlases:
      - atlas: Schaefer2018
        seg: 17networks
        scale: 400         # BEP 017/atlas: scale- = region count (was desc-400Parcels17Networks)

  - suffix: dwimap       # BEP 016: diffusion model is `dwimap` + model-/param- (not suffix `fod`)
    datasets: [qsirecon]
    model: csd
    param: wm
    operations: [parcellate_odf]
    atlases:
      - atlas: HCPMMP1
```

Supported operations: `parcellate_timeseries`, `parcellate_scalar`, `functional_connectivity`, `streamline_connectivity`, `bundle_stats`, `parcellate_fixel`, `parcellate_odf`.

> **BIDS/BEP entity & suffix note.** Sources are selected with BIDS entities, so the spec must understand the entities the relevant BEPs define: `model-`/`param-` for diffusion models ([BEP 016](#13-additional-considerations)), `tract-`/`track-` for tractograms ([BEP 046](#13-additional-considerations)), and `seg-`/`scale-` for atlas realizations (merged BIDS *Templates and atlases* + [BEP 017](#13-additional-considerations); `scale-` is the BIDS-blessed encoding of Schaefer's region count, replacing the `desc-400Parcels17Networks` idiom). See §14 for the full conformance map.

**Tractography I/O.** Per [BEP 046](#13-additional-considerations), the **only BIDS-compliant tractography format is TRX (`.trx`)**; `.tck` and `.trk` are explicitly *not* BIDS-valid and are accepted only as ingest conveniences, converted to `.trx` with `trxrs convert` (trx-rs reads `.trk`/`.tck` but writes only TRX). The tractogram suffix is `tractogram`, optionally carrying `tract-<name>` (anatomical structure) and `track-<method>` (controlled vocab: `act`/`eudx`/`fact`/`ifod`/`sdstream`/`tensor`/`pft`/`ptt`/`set`). `streamline_connectivity` warps the tractogram into the atlas's space (Strategy B; see §7.4) and assigns each streamline's two endpoints to parcels to build an N×N matrix (emitted per [BEP 017](#13-additional-considerations) as `relmat`; see §8); `bundle_stats` reports per-tractogram count / mean length / density (a whole-tractogram summary, *not* a connectivity matrix — see §8).

### `bat_spec.yaml`

A list of `operations`. Each operation names an atlas algebra step, its inputs (entity-filter dicts), and the BIDS entities for the output atlas. Entity values must be alphanumeric (BIDS requirement; no hyphens or underscores).

```yaml
operations:
  - name: corticalSubcortical
    operation: union
    inputs:
      - atlas: HCPMMP1
      - atlas: Tian
        seg: S2
    output_entities:
      atlas: HCPMMPTian

  - name: networkParcels
    operation: intersect
    inputs:
      - atlas: Schaefer400
        desc: 400Parcels17Networks
      - atlas: RSN
        desc: networks
    output_entities:
      atlas: Schaefer400RSN
      desc: 400Parcels17NetworksRSN

  - name: parcelBundles
    operation: outer_product
    inputs:
      - atlas: Schaefer400
      - atlas: AFQ
        seg: bundles
    output_entities:
      atlas: Schaefer400AFQ
```

Supported operations: `union`, `intersect`, `outer_product`.

---

## 5. Config Singleton

Single `config.py` following the nipost singleton pattern (ToML serialisation for cross-process use). Sections:

| Section | Shared? | Key fields |
|---|---|---|
| `environment` | yes | version, nipype_version, exec_env |
| `execution` | yes | bids_dir, output_dir, datasets, spec, participant_label |
| `workflow` | yes | spaces, file_format (`auto`/`nifti`/`gifti`/`cifti`) |
| `bdt_workflow` | bdt only | min_coverage, dummy_scans, correlation_lengths, output_correlations |
| `bat_workflow` | bat only | output_space, interpolation |
| `nipype` | yes | nprocs, omp_nthreads, plugin |

`file_format = auto` (default): format is detected from each collected file's extension at workflow-build time and used to route to the appropriate format-specific parcellation subworkflow.

---

## 6. Workflow Hierarchy

### BDT

`init_bdt_wf` iterates over subjects. `init_single_subject_wf` reads the spec, calls `collect_derivatives()` for each source entry, and dispatches each matched file to a data-type run workflow based on the entry's `operations`.

```
init_bdt_wf()
  └── init_single_subject_wf(subject_id)
        │  reads spec; collects files per source entry; raises on ambiguity
        │
        ├── init_timeseries_run_wf(source_file, atlas_filters, operations)
        │     ├── init_load_atlases_wf        ← warp atlases to data space
        │     ├── init_parcellate_*_wf        ← nifti / gifti / cifti dispatch
        │     └── init_functional_connectivity_*_wf   ← if requested
        │
        ├── init_scalar_run_wf(source_file, atlas_filters, operations)
        │     ├── init_load_atlases_wf
        │     └── init_parcellate_*_wf        ← parcel-mean only, no FC
        │
        ├── init_streamlines_run_wf(source_file, atlas_filters, operations)
        │     ├── TrxConvert                  ← .trk → .trx on ingest (if needed)
        │     ├── TrxTransform                ← warp tracts → atlas space
        │     │                                  (chain_for_point_warp)
        │     └── init_streamline_connectivity_wf  ← endpoint→parcel N×N matrix
        │
        └── init_diffusion_model_run_wf(source_file, atlas_filters, operations)
              ├── OdxTransform                ← resample + SH-reorient to atlas/data space
              └── init_parcellate_{fixel,odf}_wf
```

### BAT

Dataset-level (no subject loop):

```
init_bat_wf()
  └── init_bat_dataset_wf()
        │  reads bat spec; collects input atlases via collect_atlases()
        │
        ├── init_intersect_wf(inputs, output_entities)
        ├── init_union_wf(inputs, output_entities)
        └── init_outer_product_wf(inputs, output_entities)
```

### Format dispatch

Format is detected from the source file extension at workflow-build time:

| Extension | Format | Parcellation tool |
|---|---|---|
| `.nii.gz` | NIfTI volumetric | ANTs `ApplyTransforms` (atlas→grid) + nilearn `NiftiLabelsMasker` |
| `.func.gii`, `.shape.gii` | GIFTI surface | Connectome Workbench (surface atlas, `-label-resample` if cross-mesh) **or** `giftirs transform` + vertex-sample (volume atlas) |
| `.dtseries.nii`, `.dscalar.nii` | CIFTI dense | Connectome Workbench |
| `.trx`, `.tck` (`.trk`→convert) | Streamlines | `trxrs transform` (tracts→atlas) + endpoint connectivity |
| SH/ODF/fixel, ingested to `.odx` (from `.nii.gz` FOD, `.mif`, DSI Studio, pyAFQ aodf) | Diffusion model | `odx convert`/`from_*` → `.odx`, then `odx transform` (resample + SH reorient, image-direction h5) + `OdfParcellate`/`FixelParcellate` |

---

## 7. Transform System

### 7.1 Principle: transforms are geometry-polymorphic

There is no single "right direction" to move data and atlases into a shared space — it depends on the **geometry** of the data being parcellated:

- **Grid data** (volumetric NIfTI, CIFTI dense): warping the data re-interpolates signal on every step, so we **warp the static label atlas into the data's grid** once, with `GenericLabel` interpolation. This is the classic xcp_d/qsirecon pattern.
- **Point data** (GIFTI vertices, TCK/TRX streamlines): a surface or tractogram is a set of *coordinates*. Warping coordinates is **lossless** — there is no signal to resample — whereas warping a coarse label volume into a low-res diffusion grid with nearest-neighbor *loses small parcels*. So for point data we **warp the points** (or the atlas vertices), not the label volume.
- **Diffusion-model grid** (SH/ODF/fixels): resampled like grid data, but additionally requires **SH reorientation** (and, for fixels, cardinality handling). Handled by `odx transform`.

This yields **three transform modalities**, not one:

| Data geometry | Operation | Tool | h5 direction needed | Interp. concern |
|---|---|---|---|---|
| **Grid** (NIfTI vol, CIFTI dense) | resample atlas → data grid | ANTs `ApplyTransforms` | **image/pull**: `from-{atlas}_to-{data}` | yes — `GenericLabel` |
| **Points** (GIFTI verts, TCK/TRX) | warp data coords (or atlas verts) | `giftirs transform`, `trxrs transform` | **opposite-named**: `from-{target}_to-{source}` | none — lossless |
| **Diffusion-model grid** (SH/ODF/fixels) | resample + reorient | `odx transform` | image/pull, with SH reorientation | yes, + fixel cardinality |

### 7.2 Graph engine with two typed queries

`utils/transforms.py` implements a single graph-based transform engine, but exposes **two typed queries** so callers cannot accidentally apply a transform backwards:

- **Nodes:** spaces (e.g., `MNI152NLin6Asym`, `T1w`, `boldref`, `fsLR`, `ACPC`/native-DWI)
- **Edges:** transforms discovered from `_xfm.*` files in the provided derivative datasets, supplemented by TemplateFlow standard→standard transforms. Each edge records the file, the transform type (`affine` vs `displacement`/`warp`), and whether it is invertible.
- **`build_transform_graph(datasets)`** — scans datasets at workflow-build time and returns a `networkx.DiGraph`. The filename regex indexes `.h5`, `.txt`, `.mat`/`*GenericAffine.mat`, and warp files **with or without** the `mode-image` token.
- **`chain_for_image_resample(graph, src, tgt)`** — returns `(files, tool='ApplyTransforms', invert_flags)` for **grid/pull** semantics (atlas → data grid).
- **`chain_for_point_warp(graph, src, tgt)`** — returns `(files, tool, invert_flags)` for **point** semantics. Per the trx-rs/gifti-rs convention, *to warp points from space A to space B you pass `from-B_to-A_xfm`* — so this query traverses the graph in the **opposite** direction from the image query and hands the result to `trxrs`/`giftirs`.

Both queries raise an explicit error if no path exists, listing available nodes.

### 7.3 The direction-convention landmine (highest correctness risk)

ANTs `ApplyTransforms` uses **image/pull** semantics; the Rust point-warpers use the **opposite** convention. Reusing the same chain for both silently produces mirrored/garbage output. Two invariants the engine must enforce:

1. **Modality-aware selection.** Never share a chain between an image resample and a point warp. Callers must use the typed query that matches the data geometry; the two queries traverse the graph in opposite directions.
2. **Warp invertibility.** `itk-transforms-rs` **cannot invert a displacement field** (only affine chains), and `trxrs --invert` only flips affines. Therefore a point warp **requires the correctly-named warp file to physically exist** — it cannot be synthesized by inversion. The graph tracks which directions are backed by a real warp; `chain_for_point_warp` raises an explicit error when only the wrong-direction warp is present. fMRIPrep/QSIPrep emit both directions, so this is satisfiable — but the engine must check rather than assume.

### 7.4 Per-data-type workflow sketches

- **Grid (BOLD/CBF/FA NIfTI, CIFTI dense):** `init_load_atlases_wf` calls `chain_for_image_resample(atlas_space → data_space)`, chains transforms via `ApplyTransforms` (`GenericLabel`), then parcellates in the data's grid.
- **Surface (GIFTI):** two sub-cases — see §7.5.
- **Streamlines (TRX/TCK):** **Strategy B (warp tracts → atlas).** `init_streamlines_run_wf` calls `chain_for_point_warp(streamline_space → atlas_space)` and warps the tractogram into the atlas's space with `trxrs transform`. Once tracts are in (e.g.) MNI, any MNI atlas applies without re-warping, and surface/bundle atlases compose uniformly. Endpoint→parcel connectivity then runs in atlas space (§3 of the plan; via trxrs read-only bindings).
- **Diffusion model (SH/ODF/fixel):** the input is first ingested into an `.odx` container (`odx convert` / `from_mrtrix` / `from_pyafq_aodf`), then `odx transform` resamples + SH-reorients it into the target space before `parcellate_odf`/`parcellate_fixel`. Critically, `odx transform` uses the **image/same-named** h5 convention (`from-{source}_to-{target}` to move source→target) — the *same* direction as ANTs and the **opposite** of `trxrs`/`giftirs`. Fixel transport has two modes: `--mode mrtrix` (default; single `--transform` h5, fixels may be duplicated/dropped at non-uniform warps) and `--mode ants` (paired `--transform`/`--transform-inverse` h5s, preserving fixel cardinality). Cardinality-preserving fixel parcellation therefore needs *both* transform directions present in the graph; the v1 `parcellate_odf` path (SH mean per parcel) only needs the single forward h5.

### 7.5 Surface (GIFTI) transform & parcellation

`gifti-rs`'s role is narrow: it warps **`*.surf.gii` geometry only** (it fails on input without a POINTSET and never touches `.func/.shape/.label.gii` data) and does **not** parcellate or cross-mesh resample. So:

- **Surface data with a surface atlas on the same mesh:** parcellate via Connectome Workbench (`wb_command -cifti-parcellate`), reusing the existing CIFTI parcellation path — **not** a hand-rolled nibabel interface.
- **Surface data with a surface atlas on a *different* mesh** (e.g. atlas on `fsaverage`, data on `fsLR-32k`): add a `wb_command -label-resample` step (registration spheres) to bring the atlas onto the data's mesh before parcellation. *(Decision: BDT adds the resample step rather than requiring pre-resampled atlases.)*
- **Surface data with a *volumetric* atlas:** warp the subject's `*.surf.gii` (e.g. midthickness in T1w) into the volume atlas's space with `giftirs transform`, then sample the label volume at each vertex — a lossless surface parcellation by a volume atlas, no `wb_command` needed. This same vertex-warp enables streamline→surface endpoint mapping (warp surfaces into streamline space, nearest-vertex assign endpoints), addressing former open question 4.5.

### 7.6 Rust binary dependencies

Because `giftirs transform` has **no Python bindings**, `trxrs transform`/`convert` ship only **read-only** Python bindings (the `trxrs` package: `load().positions()/.offsets()`, plus a `convert()` helper), and `odx transform` is likewise **CLI-only** (the `odx` PyO3 package exposes reading, peak-finding, and format conversion — `load`, `sh`, `peaks_from_sh`, `from_mrtrix`, `from_pyafq_aodf`, … — but **no `transform` function**), all three transform integrations are nipype `CommandLine` interfaces shelling out to the Rust binaries — exactly how ANTs, MRtrix, and `wb_command` are already wrapped. The diffusion-model *parcellation* (not transform) does read `.odx` via the `odx` PyO3 bindings.

The container therefore needs a **Rust build stage** (or pre-built release binaries) so the verified binaries `trxrs`, `giftirs`, and `odx` are on `PATH` (repos: `tee-ar-ex/trx-rs`, `PennLINC/gifti-rs`, `PennLINC/odx-rs`; shared engine `PennLINC/itk-transforms-rs`). See the implementation plan for the Docker stage and the I/O standardization on TRX/TCK (`.trk` is converted on ingest with `trxrs convert`, since trx-rs reads but cannot write `.trk`).

---

## 8. Outputs

All BDT outputs are BIDS derivatives. `DerivativesDataSink` copies all BIDS entities from the source file and injects atlas entities in canonical BIDS order — `space- → atlas- → seg- → scale- → res- → den- → desc-` (merged BIDS *Derivatives from atlases*). Two consequences enforced throughout this section: **(a) `space-` is REQUIRED** on every atlas-derived output (it disambiguates the coordinate system the parcellation ran in); **(b)** measure/model semantics live in dedicated entities (`meas-`, `model-`, `param-`, `stat-`) rather than overloaded `desc-`. Suffixes follow the relevant BEPs rather than the raw source suffix (see §14 for the full map and BEP pointers).

### Time series (`parcellate_timeseries`, `functional_connectivity`) — BEP 012, BEP 017

Parcellated time series use the **`timeseries`** suffix ([BEP 012](#13-additional-considerations)), not `bold`. Functional connectivity is a relationship matrix: **`meas-<label>_relmat.dense.tsv`** with a REQUIRED `_relmat.json` sidecar and a node-index sidecar ([BEP 017](#13-additional-considerations)).

```
sub-{label}/func/
  # parcellated time series (BEP 012)
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-][_desc-]_timeseries.tsv
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-][_desc-]_timeseries.json
  # functional connectivity matrix (BEP 017) — meas- identifies the measure
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-{label}_relmat.dense.tsv
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-{label}_relmat.json   ← REQUIRED sidecar
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-{label}_nodeindices.tsv ← REQUIRED node map
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-{label}_nodeindices.json
  # OPTIONAL Connectome-Workbench-native artifacts (not BEP-canonical; emitted only for CIFTI inputs):
  #   …_timeseries.ptseries.nii , …_meas-{label}_relmat.pconn.nii
sub-{label}/perf/
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-][_desc-]_timeseries.tsv   ← ASL 4D parcellated
```

> **`_relmat.json` REQUIRED fields** ([BEP 017](#13-additional-considerations)): `NodeFiles`, `RelationshipMeasure`, `Weighted`, `Directed`, `ValidDiagonal`, `StorageFormat`, `Software`. **`meas-` examples:** `pearsoncorrelation`, `partialcorrelation`. The `_nodeindices.tsv` maps each matrix index → source-atlas ROI (for a single-atlas square matrix it is an adapted copy of the atlas `dseg.tsv`). `.dense.` is mandatory between `_relmat` and the extension for dense matrices (`.sparse.` for sparse).

### Scalar maps (`parcellate_scalar`) — BEP 011, BEP 012, BEP 016

The parcellated suffix mirrors the source-derivative's BEP, and measure/model semantics live in `stat-`/`model-`/`param-` rather than the suffix:

```
sub-{label}/func/
  # ALFF/ReHo/etc. are stat- values on a derivative map, NOT an `alff` suffix (BEP 012)
  sub-{label}[_ses-][_task-]_space-{space}_atlas-{label}[_seg-][_scale-]_stat-alff[_desc-]_timeseries.tsv
sub-{label}/perf/
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-][_desc-]_cbf.tsv          ← ASL CBF (perf)
sub-{label}/dwi/
  # FA/MD etc. carry model-/param- per BEP 016 (e.g. model-tensor, param-fa)
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-]_model-{label}_param-{label}_dwimap.tsv
sub-{label}/anat/
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-][_desc-]_morph.tsv         ← anatomical (BEP 011)
```

> `morph` (tabular morphometrics) is the [BEP 011](#13-additional-considerations) suffix and is correct as-is; its columns should follow the BEP 011 `Morphometrics` table. `stat-` measures (`alff`/`falff`/`reho`/…) and the `<source>map` convention are [BEP 012](#13-additional-considerations); `model-`/`param-` on `dwimap` are [BEP 016](#13-additional-considerations). CBF parcellation follows the ASL/perf derivative conventions.

### Streamlines (`streamline_connectivity`, `bundle_stats`) — BEP 017

Endpoint connectivity is a relationship matrix per [BEP 017](#13-additional-considerations): the measure is a `meas-<label>` entity (`count`/`length`/`density`/`denlen`), the suffix is `relmat` with the mandatory `.dense.` qualifier, and the `_relmat.json` + `_nodeindices.*` sidecars are REQUIRED (same fields as the FC case above).

```
sub-{label}/dwi/
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-count_relmat.dense.tsv   ← N×N endpoint connectivity
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-count_relmat.json         ← REQUIRED sidecar
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-count_nodeindices.tsv      ← REQUIRED node map
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-]_meas-count_nodeindices.json
  # bundle_stats is a per-tractogram summary (count/mean length/density), NOT a matrix → plain TSV, not relmat:
  sub-{label}[_ses-]_space-{space}[_tract-][_track-]_desc-bundlestats_tractogram.tsv
```

> **`bundle_stats` is not connectivity.** [BEP 017](#13-additional-considerations) reserves `relmat` for pairwise matrices; a whole-tractogram count/length/density summary has no `relmat` home, so it is written as a descriptive TSV keyed to the `tractogram` ([BEP 046](#13-additional-considerations)). Streamline *count/length/density between ROI pairs*, by contrast, **are** `meas-` variants of `relmat` and belong in the matrix above.

### Diffusion model (`parcellate_fixel`, `parcellate_odf`) — BEP 016

```
sub-{label}/dwi/
  sub-{label}[_ses-]_space-{space}_atlas-{label}[_seg-][_scale-]_model-{label}_param-{label}_dwimap.tsv
```

Per [BEP 016](#13-additional-considerations), the model (`model-csd`, `model-noddi`, …) and the parameter (`param-wm`/`param-gm`/`param-csf` for CSD SH; `param-fa` etc.) are REQUIRED entities, and the BIDS-native diffusion-model image is a **plain NIfTI** (e.g. SH as `I×J×K×45`) — `.odx`/`.mif` are tool-internal only and never appear in a BIDS tree. Fixel parcellation is **not** a simple parcel-mean (fixel→parcel must respect fixel cardinality/direction); ODF parcellation summarizes per-parcel ODF/SH coefficients. Both consume `odx`-resampled (SH-reoriented) inputs that were ingested from the BIDS `dwimap` NIfTI.

### BAT outputs

A new BIDS Atlas dataset at `output_dir/`, conforming to the merged-BIDS *Templates and atlases* derivatives spec (`src/derivatives/atlas.md`). Atlas files live under a `tpl-<label>/` directory named for the spatial reference of the input atlases. BAT produces discrete segmentation atlases only (`dseg`); probabilistic atlases and tractography are not in scope.

```
output_dir/
  dataset_description.json                       ← DatasetType: derivative
  atlas-{label}_description.json                 ← REQUIRED (validator error if absent; see below)
  tpl-{tpl}/
    [<datatype>/]
      tpl-{tpl}_atlas-{label}[_seg-{label}][_scale-{label}][_res-{label}][_desc-{label}]_dseg.nii.gz
      tpl-{tpl}_atlas-{label}[_seg-{label}][_scale-{label}][_res-{label}][_desc-{label}]_dseg.tsv
      tpl-{tpl}_atlas-{label}[_seg-{label}][_scale-{label}][_res-{label}][_desc-{label}]_dseg.json
  # OPTIONAL provenance for combined atlases (union / outer_product): which source parcel each output label came from
  atlas-{label}_space-{tpl}_nodelabels.tsv       ← BEP 017 §4.3
  atlas-{label}_space-{tpl}_nodelabels.json
```

Two BIDS-conformance requirements (both merged into BIDS today):

1. **Canonical entity order is `atlas → seg → scale → res → desc`** (`schema/rules/entities.yaml`). The earlier draft wrote `desc` before `seg`, which is invalid; `seg-`/`scale-` precede `desc-`. `scale-` is the spec-blessed encoding of region count (e.g. Schaefer's 400), preferred over folding it into `desc-`.
2. **`atlas-<label>_description.json` is REQUIRED** — the validator raises `ATLAS_DESCRIPTION_REQUIRED` (level: *error*) for any file carrying both `tpl-` and `atlas-` with no sibling description file. BAT MUST write it, with at least the REQUIRED metadata fields **`AtlasName`** and **`License`** (note the schema key is `AtlasName`, not `Name`); `Description`/`Authors` are recommended. Provenance (the source atlases and the operation) SHOULD go in `Description`/`DerivedFrom`.

The `dseg.tsv` MUST have `index` + `name` columns (`SegmentationLookup` rule). For **union** and **outer_product**, BAT SHOULD also emit a `nodelabels.tsv`/`.json` ([BEP 017](#13-additional-considerations) §4.3) mapping each output label back to its `SourceAtlasName` + `SourceAtlasIndex` + `SourceAtlasLabel`, since the offset-relabeling otherwise loses that provenance.

The `atlas-` entity and any additional entities (`seg`, `scale`, `desc`) come from `output_entities` in the bat spec. The `tpl-` entity is inherited from the spatial reference of the input atlases (all inputs to a given BAT operation must share the same template space; BAT raises an error if they do not). If the inherited `tpl-` is not a standard TemplateFlow identifier, `SpatialReference` metadata is REQUIRED on the output files.

**Scope clarification (resolves the prior contradiction).** BAT operates on **discrete label images (`dseg`) only** and emits `dseg`. "Tractography is not in scope" means BAT does not process streamline *data* (`.trx`/`.tck`) — it does **not** mean bundle atlases are excluded. A bundle atlas already represented as a `dseg` label image (e.g. AFQ bundles rasterized to a template) is a valid input, so the `parcelBundles` `outer_product` example (Schaefer400 × AFQ bundles → sub-parcels) is in scope. Probabilistic (`probseg`) atlases remain out of scope.

---

## 9. Missing Interfaces

Four interfaces are referenced in the existing codebase but not yet implemented:

| Interface | File | Note |
|---|---|---|
| `ApplyTransforms` | `interfaces/ants.py` | Wrapper around `nipype.interfaces.ants.ApplyTransforms` with bdt defaults |
| `IndexImage` | `interfaces/nilearn.py` | Extracts a single volume from a 4D image |
| `Censor` | `interfaces/censoring.py` | Removes volumes from a time series based on a temporal mask TSV |
| `PlotCiftiParcellation` | `interfaces/plotting.py` | Visualises parcellated CIFTI scalars via Connectome Workbench |

New interfaces needed for v1:

| Interface | File | Purpose |
|---|---|---|
| `TrxTransform` | `interfaces/rust.py` | CommandLine wrapper for `trxrs transform` (warp streamline coords) |
| `TrxConvert` | `interfaces/rust.py` | CommandLine wrapper for `trxrs convert` (`.trk` → `.trx`/`.tck` on ingest) |
| `GiftiTransform` | `interfaces/rust.py` | CommandLine wrapper for `giftirs transform` (warp `*.surf.gii` vertices) |
| `OdxTransform` | `interfaces/rust.py` | CommandLine wrapper for `odx transform` (resample + SH reorient an `.odx`; positional in/out, `--transform`, `--mode mrtrix\|ants`, optional `--transform-inverse`) |
| `OdxConvert` | `interfaces/rust.py` | CommandLine wrapper for `odx convert` (ingest `.nii.gz` FOD / `.mif` / DSI Studio / pyAFQ aodf → `.odx`) before transform |
| `SurfaceVolumeParcellate` | `interfaces/gifti.py` | Sample a volume label atlas at warped surface vertices (lossless surface-by-volume parcellation) |
| `StreamlineConnectivity` | `interfaces/tractography.py` | Endpoint→parcel assignment → N×N matrix, built on trxrs read-only bindings (`positions()`/`offsets()` + nibabel label lookup, small radial search) |
| `BundleStats` | `interfaces/tractography.py` | Per-tractogram statistics: count, mean length, density |
| `FixelParcellate` | `interfaces/odx.py` | Per-parcel fixel summary (respects fixel cardinality, not a plain mean); reads the `.odx` via the `odx` PyO3 bindings |
| `OdfParcellate` | `interfaces/odx.py` | Per-parcel ODF/SH coefficient summary; reads the `.odx` via `odx.load(...).sh(...)` (PyO3), not `nibabel` |
| `AtlasIntersect` | `interfaces/atlas.py` | Voxel-wise label **restriction**: keep voxels labeled in both atlases, relabel to the atlas-1 parcel (distinct from outer product) |
| `AtlasUnion` | `interfaces/atlas.py` | Combine two atlas label images; overlap resolved by a **precedence rule** (not a hard error) |
| `AtlasOuterProduct` | `interfaces/atlas.py` | Cartesian sub-parcels: combined label `(d1-1)*n2 + d2` on the overlap (distinct from intersect) |

> **Surface-with-surface-atlas parcellation is *not* a new interface** — it routes through the existing `wb_command` path (`interfaces/workbench.py`), adding only `init_label_resample_wf` for the cross-mesh case. `gifti-rs` does not parcellate, so it is not used for this step.

---

## 10. Error Handling

The "explicit over heuristics" principle applies at three points:

1. **Atlas collection** (`collect_atlases`): if an entity-filter dict matches more than one file, raise an error listing all matches and instructing the user to add more entities to disambiguate.

2. **Transform graph** (`chain_for_image_resample` / `chain_for_point_warp`): if no path exists between atlas space and data space, raise an error naming both spaces and listing all nodes currently in the graph.

3. **Source file collection** (`collect_derivatives`): if a spec entry matches zero files, raise an error. If it matches more than one file per run (e.g., two CBF maps with different `desc`), raise an error asking the user to add a `desc` filter to the spec entry.

4. **Point-warp invertibility** (`chain_for_point_warp`): a streamline/surface warp requires the correctly-named warp file to physically exist (displacement fields cannot be inverted; `trxrs --invert` only flips affines). If only the wrong-direction warp is present, raise an explicit error rather than silently producing mirrored output.

5. **BAT template mismatch** (atlas algebra): all inputs to a BAT operation must share the same `tpl-` space; raise an error listing the mismatched inputs.

---

## 11. Testing Strategy

### Unit tests (no data required, run by default)

- `test_spec.py` — spec YAML parsing, entity dict validation, error on unknown operations
- `test_transforms.py` — `build_transform_graph` plus both typed queries (`chain_for_image_resample` pull direction, `chain_for_point_warp` opposite direction) with synthetic graph fixtures; widened-regex (`.mat`/no-`mode`) indexing; no-path and point-warp-not-invertible error paths
- `test_atlas.py` — `collect_atlases` with a mocked pybids layout; ambiguity error path
- `test_interfaces.py` — `Censor`, `IndexImage`, `AtlasIntersect`, and other lightweight interfaces

### Workflow build tests (no execution, uses `mock_config()`)

- `test_bdt_base.py` — `init_bdt_wf` and `init_single_subject_wf` build without error
- `test_bdt_workflows.py` — `init_load_atlases_wf`, `init_timeseries_run_wf`, `init_scalar_run_wf`, `init_streamlines_run_wf` (Strategy B: convert→warp→connectivity), `init_diffusion_model_run_wf`, and the surface workflows (`init_surface_volume_parcellate_wf`, `init_label_resample_wf`) build for each format/geometry
- `test_bat_workflows.py` — `init_bat_wf` and each algebra workflow build without error

### Integration tests (`@pytest.mark.integration`, real datasets required)

- BOLD fMRI: fMRIPrep derivatives + small BIDS Atlas dataset
- ASL: ASLPrep derivatives
- dMRI streamlines: QSIRecon tractography (ACPC/native) + MNI parcellation — exercises `trxrs convert`/`transform` and endpoint connectivity (requires the Rust binaries on `PATH`)
- dMRI model: SH/ODF map + `odx transform` + `parcellate_odf`
- Surface: `.func.gii` + cross-mesh `.label.gii` (exercises `wb_command -label-resample`); and `.surf.gii` + volume atlas (exercises `giftirs transform` + `SurfaceVolumeParcellate`)
- BAT: intersect two small atlases, verify output is a valid BIDS Atlas dataset; union with overlap (precedence); intersect ≠ outer_product regression
- **BIDS-validation (§14):** run `bids-validator` (or the schema validator) over BDT and BAT outputs and assert no errors — specifically that BAT emits `atlas-<label>_description.json` (no `ATLAS_DESCRIPTION_REQUIRED`), entity order is canonical (`seg` before `desc`), every atlas-derived file carries `space-`, and connectivity outputs ship the `_relmat.json` + `_nodeindices.*` sidecars
- Container: build with the Rust stage and assert `trxrs`/`giftirs`/`odx` resolve on `PATH`

---

## 12. Open Questions (deferred)

- **4.1** Transform selection heuristics: should there ever be automatic selection vs strict errors for ambiguous transforms?
- **4.3** Subject-specific atlases (e.g., individual bundle segmentations) — BAT may need a subject-level workflow mode.
- Subject-specific atlases in BDT (e.g., individual parcellations from fMRIPrep): collection strategy not yet defined.

### Resolved in this revision

- **Streamline space (was a contradiction):** **Strategy B** — warp tracts → atlas space via `trxrs transform` (§7.4). The lossless-points argument overrides the §7 grid principle for point data.
- **Surface cross-mesh resampling:** BDT adds a `wb_command -label-resample` step (§7.5) rather than requiring pre-resampled atlases.
- **odx-rs scope:** **in scope for v1** — `parcellate_fixel`/`parcellate_odf` and `odx transform` (§7.4, §8).
- **Endpoint→parcel assignment:** lives in `bdt` as `StreamlineConnectivity`, built on trxrs read-only bindings (§9).
- **4.5 (streamline–surface mapping):** the `giftirs` vertex-warp (§7.5) makes nearest-vertex endpoint assignment tractable; the prior `density_map` per-parcel rasterization (O(parcels × streamlines)) is dropped in favor of endpoint lookup.

---

## 13. Additional Considerations

While reviewing this plan, cross-reference against the BIDS specification (@bids-specification) and the following BIDS Extension proposals (BEPs), which describe plans to add relevant elements to BIDS. These BEPs are not merged into BIDS, so they are both subject to change and the bdt developers may choose to ignore or modify certain elements.
BEP 011 Structural Derivatives (a GitHub pull request onto the BIDS repository): https://github.com/bids-standard/bids-specification/pull/518
BEP 012 Functional Preprocessing Derivatives (a GitHub pull request onto the BIDS repository): https://github.com/bids-standard/bids-specification/pull/519
BEP 016 Diffusion Weighted Imaging Derivatives (a GitHub pull request onto the BIDS repository): https://github.com/bids-standard/bids-specification/pull/2211
BEP 017 Connectivity Schema (a Google Doc describing the BEP): https://docs.google.com/document/d/1ugBdUF6dhElXdj3u9vw0iWjE6f_Bibsro3ah7sRV0GA/
BEP 039 Dimensionality Reduction-Based Networks (a Google Doc describing the BEP): https://docs.google.com/document/d/1GTWsj0MFQedXjOaNk6H0or6IDVFyMAysrJ9I4Zmpz2E/
BEP 041 Statistical Model Derivatives (a Google Doc describing the BEP): https://docs.google.com/document/d/1KHzp-yk8KXvkUIhtN71WU0m4P4kKT9C1yvI-i9_kNeY/
BEP 046 Tractography (a Google Doc describing the BEP): https://docs.google.com/document/d/1ubDQ2RhgjnfGqoeukzEkPV9YEHhfYMERrj7-3b0c2HI/edit

> **Local copies (reviewed 2026-06-02).** The four Google-Doc BEPs have been exported to Markdown in the project root (`BIDS Extension Proposal 17 (BEP017)…md`, `BEP39 …md`, `BIDS Extension Proposal 41 (BEP041)…md`, `BEP046 …md`) and the GitHub-PR BEPs (011/012/016) were read via `gh`. §14 records the conformance review against all of them plus merged BIDS.

Also, we are strongly considering using NiWrap instead of Nipype for bdt and bat. Consider that in any reviews or implementation plans.

---

## 14. BIDS / BEP Conformance Map

Reviewed 2026-06-02 against merged BIDS (the cloned `bids-specification`, `master`) and the seven BEPs in §13. "Merged" = enforceable by the validator today; "BEP-pending" = stable enough to adopt but subject to change. Outputs in §8 already reflect these.

### 14.1 Merged BIDS (validator-enforceable now)

| Item | Requirement | Where |
|---|---|---|
| `atlas-`, `seg-`, `scale-` entities | All exist on `master`; `scale-` = region count (Schaefer) | §4, §8 |
| Atlas-derivative entity order | `space → atlas → seg → scale → res → den → desc` | §8 (all BDT outputs) |
| `space-` on atlas derivatives | **REQUIRED** to disambiguate coordinate system | §8 |
| BAT entity order | `atlas → seg → scale → res → desc` (earlier `desc`-before-`seg` was invalid) | §8 BAT |
| `atlas-<label>_description.json` | **REQUIRED** (`ATLAS_DESCRIPTION_REQUIRED`, error); `AtlasName`+`License` required | §8 BAT |
| `dseg.tsv` columns | `index` + `name` required | §8 BAT |
| Non-standard `tpl-` | `SpatialReference` metadata REQUIRED | §8 BAT |

### 14.2 BEP-pending (adopt, but track upstream)

| BDT/BAT element | Conformant form | BEP |
|---|---|---|
| Parcellated time series | `…_timeseries.tsv` (not `_bold.tsv`) | **BEP 012** |
| ALFF/ReHo/etc. | `stat-<label>` on a `…map`/`timeseries` (not an `alff` suffix) | **BEP 012** |
| Anatomical morphometrics | `…_morph.tsv` (columns per BEP 011 `Morphometrics`) | **BEP 011** |
| Diffusion-model source & scalar | `suffix: dwimap` + REQUIRED `model-`/`param-`; SH = plain NIfTI `I×J×K×N` | **BEP 016** |
| Functional & streamline connectivity | `…_meas-<label>_relmat.dense.tsv` + REQUIRED `_relmat.json` + `_nodeindices.{tsv,json}` | **BEP 017** |
| Connectivity measure | `meas-` entity (`pearsoncorrelation`, `count`, `length`, `density`, …) | **BEP 017** |
| BAT combined-atlas provenance | `nodelabels.tsv`/`.json` (source atlas + index per output label) | **BEP 017 §4.3** |
| Tractogram source | `suffix: tractogram` (not `tractography`); `tract-`/`track-` entities; **TRX-only** (`.tck`/`.trk` non-BIDS, ingest-convert) | **BEP 046** |

### 14.3 Out of scope (confirmed orthogonal)

- **BEP 041 (Statistical Model Derivatives):** GLM/stat-model outputs; connectivity explicitly delegated to BEP 017. BDT/BAT apply atlases, they do not fit models — no overlap.
- **BEP 039 (Dimensionality-Reduction Networks):** ICA/PCA/gradient *components* are continuous `…map` files with `model-`/`param-`/`item-`, **not** discrete `dseg`. BDT/BAT's discrete-atlas scope keeps these out; this BEP is the reference *if* continuous/overlapping "soft atlases" are ever added.

### 14.4 Cross-BEP entity pattern

The `model-`/`param-` pair recurs across BEP 012/016/039; `meas-` is the BEP 017 connectivity idiom; `seg-`/`scale-` are the atlas idiom (merged + BEP 017). BDT's spec grammar (§4) and `DerivativesDataSink` (§8) thread these dedicated entities through rather than overloading `desc-`. The `bdt_spec.yaml` source/atlas filters are plain pybids entity dicts, so they already accept `model`/`param`/`tract`/`track`/`seg`/`scale`/`meas` once pybids' config knows them; BDT ships a derivatives entity config covering the BEP entities it consumes.
