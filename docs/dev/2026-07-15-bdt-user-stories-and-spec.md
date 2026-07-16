# BDT / BAT — User Stories & Node-Spec Design

**Date:** 2026-07-15
**Status:** Draft for discussion
**Supersedes:** `2026-07-15-bdt-node-spec-mockups.md` (consolidates it with the subsequent design decisions).
**Builds on:** `2026-06-01-bdt-design.md`, `2026-06-01-bdt-implementation-plan.md`, `2026-06-01-bdt-plan-review.md`, `HackathonNotes.md`.

This document is the single current reference for (1) the proposed BDT/BAT **spec grammar** and (2) the **user stories** it must express, each written in that grammar.

---

## 0. What changed since the 2026-06-01 design

The approved design used a **declarative `sources`** model: a flat list binding an atlas set to a derivative type, applying one of a fixed 4-verb menu, each source processed independently. The hackathon showed that several real analyses are **ordered, multi-step pipelines** where one operation's output is the next one's input (map a scalar to a surface → resample → parcellate; convert bundles → ROI atlas → sample a scalar). That can't be expressed by independent sources.

So the config becomes a **QSIRecon-inspired node graph with steps**. The deltas:

| Area | 2026-06-01 design | This design |
|---|---|---|
| Config model | declarative `sources` (bind atlas→derivative, fixed 4 ops) | **node graph**: named nodes, data flows node→node via `inputs:` |
| Atlas selection | implicit / CLI `--atlases` | first-class **`select_atlases` node**; `--atlases` flag removed |
| Analysis levels | participant only | **participant (`nodes:`) + dataset (`dataset:`, run once)** |
| Transform principle | global "warp atlas, not data" | **per-action**: grid=warp atlas (Strategy A); points=warp data losslessly (Strategy B) |
| New capabilities | — | surface mapping + depth profiles, subcortical CIFTI assembly, streamline scalar mapping (dpv/dps) + dps-weighted connectivity, inline atlas construction |
| Outputs | copy source entities | **`desc` prepend-compose** + **provenance sidecars**; dataset-level → `tpl-` |

Unchanged: nipype BIDS App, niworkflows, the `explicit-over-heuristics` principle, and BAT as atlas algebra (now expressed in the same node grammar).

---

## 1. Spec grammar

A spec is a YAML document with up to two node lists — `dataset:` (run once) and `nodes:` (run per participant):

```yaml
dataset:   # optional: subject-independent resources, computed once
- <node>
nodes:     # per-subject pipeline
- <node>
```

### 1.1 Node anatomy

Every node has a unique `name` and an `action`. There are two kinds:

```yaml
# SELECTION node — reads files from disk
- name: <unique id>
  action: select_data | select_atlases
  dataset: <name>                 # a --datasets key
  filters: {<entity>: <value>}    # pybids entity filters; value may be a list
  exclude: [{<entity>: <value>}]  # optional negative filters

# PROCESSING node — consumes upstream nodes
- name: <unique id>
  action: <verb>
  inputs:                         # role -> upstream node (or list of nodes)
    <role>: <node>
    <role>: [<node>, <node>]
  parameters: {<k>: <v>}          # action-specific
  desc: <label>                   # optional output disambiguator (see 1.7)
  write_outputs: true             # materialize as derivatives (default: false)
```

**No input shorthand.** Every wired input is a **named role** under `inputs:` — even single-input nodes (`inputs: {timeseries: parcellate_bold}`). `inputs:` means exactly one thing: wires from upstream nodes. Selection nodes never take `inputs:`; processing nodes never take `dataset:`. Because `dataset:` and `inputs:` are distinct keys, a dataset reference can never be confused with a node reference.

### 1.2 Two analysis levels (scope)

**Scope is a property of a node's lineage**, like format and space:

- **participant** (`nodes:`) — some input in its lineage varies by subject (a subject's bold, NODDI, surfaces, aseg). Runs per subject.
- **dataset** (`dataset:`) — its *entire* lineage is subject-independent (a neuromaps map, a standard-space atlas, atlas algebra). Runs **once**; outputs go to `tpl-<space>/`.

Rules: participant nodes **may** reference dataset nodes (a per-subject step consuming a once-computed reference or a shared atlas); a dataset node may **not** reference a participant node. The framework **validates** that a node placed in `dataset:` has a fully subject-independent lineage and errors if a subject-varying input leaks in — placement is explicit, not auto-hoisted.

> "Dataset-level (runs once)" is *not* "group-level aggregation across subjects" (e.g. averaging parcellated maps over a cohort). That reduction is downstream of BDT and out of scope here.

### 1.3 Selection & fan-out

- `select_data` matches **one or more** files; the downstream chain **fans out** over each match (one branch per file), outputs disambiguated by their own entities. "All NODDI params" or "every non-NODDI scalar" is just a loose filter; narrow with `filters:`/`exclude:` (Q1).
- `select_atlases` matches one or more atlases; a parcellation node applies each and tags outputs with `atlas-<label>`.
- A role may take a **list** of nodes (`atlas: [a, b]`) → fan out over them.

### 1.4 Atlases: selected or constructed

Anything feeding an `atlas` role is either **selected** (`select_atlases` from an atlas dataset) or **constructed** from data by an action (e.g. `tractogram_to_dseg` builds a bundle-ROI dseg). Parcellation doesn't care which produced the atlas. (BDT/BAT boundary for inline construction: open question Q3.)

### 1.5 Transforms are per-action

The old global "warp the atlas, never the data" principle becomes a per-action property, because the two data geometries need opposite handling (plan-review §1, §5):

- **Grid / image-resample actions (Strategy A — ANTs `ApplyTransforms`).** `parcellate_timeseries` and `parcellate_scalar` warp the **atlas/label volume into the data's space** with `GenericLabel` interpolation. `resample_subcortical` is the same family but the other direction: it warps the **continuous scalar data onto the fixed standard MNI152NLin6Asym 2mm grayordinate grid** with continuous (linear/spline) interpolation — *not* `GenericLabel` (that would quantize a continuous map). So "Strategy A" means "grid resample via `ApplyTransforms`," in **either** direction — not specifically "atlas into data space."
- **Point actions (Strategy B).** `map_scalar_to_surface`, `resample_surface_scalar`, `map_scalar_to_streamlines`, `region2region` move the **data's points losslessly** (surface spheres / streamline coordinate warps).
- **Wrappers inherit their primitive's strategy.** `parcellate_scalar_as_roi` is Strategy A (it parcellates a dseg — warps the ROI dseg into the scalar's space); `parcellate_scalar_as_tract_profile` is Strategy B (it wraps `map_scalar_to_streamlines`).

The transform engine must therefore expose **two typed queries** over one graph — an image-resample chain and a point-warp chain — because they traverse the graph in opposite directions and a point warp requires the correctly-named transform to physically exist (it can't be synthesized by inverting a displacement field). See Q5.

### 1.6 CLI (atlases are in-spec now)

```bash
bdt /data/raw_bids /data/derivatives/bdt-<name> participant \
  --datasets \
    xcpd=/data/derivatives/xcp_d \
    smriprep=/data/derivatives/smriprep \
    qsirecon=/data/derivatives/qsirecon/derivatives/qsirecon-<flavor> \
    aslprep=/data/derivatives/aslprep \
    atlases=/data/atlases/bids-atlas-dataset \
  --spec /data/configs/<name>.yaml \
  --participant-label 01 \
  --work-dir /work/bdt-<name>
```

### 1.7 Outputs — naming, disambiguation, provenance

- **Entity composition.** A written output's entities are composed from its source file(s) + `atlas-<label>` + any action-specific entity (`stat-`, `seg-`, …) + suffix.
- **Disambiguation via `desc` (prepend-compose, never overwrite).** When a node sets `desc:`, it **compounds** with any upstream desc as one camelCase alphanumeric token — `output_desc = existing_desc + Capitalize(node_desc)` (else just `node_desc`). E.g. source `desc-geneexpression` + node `desc: strict` → `desc-geneexpressionStrict`. This is BIDS-legal (one `desc-<alnum>` token), unlike `+`-delimited encodings. Compounding is in processing order; only set `desc` where you actually need to disambiguate, so it stays bounded.
- **Provenance → JSON sidecar, always.** The filename `desc` only has to be *unique and recognizable*; the authoritative record lives in the sidecar via BIDS `GeneratedBy` + `Sources`:
  ```json
  {
    "GeneratedBy": [{"Name": "bdt", "Node": "parcellate_strict",
                     "Action": "parcellate_timeseries",
                     "Parameters": {"min_coverage": 0.7}}],
    "Sources": ["bids:xcpd:sub-01/func/..._desc-denoised_bold.dtseries.nii",
                "bids:atlases:tpl-fsLR_atlas-4S456Parcels_dseg.dlabel.nii"]
  }
  ```
- **`write_outputs`** marks which nodes materialize (default false; intermediates stay in the work dir). Per-node independent — a node feeding a downstream step can still be written.
- **Dataset-level outputs → `tpl-<space>/`** (not duplicated across subjects).
- **Collision is an error, not an overwrite.** If two `write_outputs` nodes resolve to the same path, the validator errors and asks for a disambiguating `desc`.

### 1.8 Static validation (before nipype build)

Reject bad specs early rather than at build time with a generic "unknown node": unique names; every `inputs:` value resolves to a prior node; every `dataset:` value resolves to a `--datasets` key; the graph is acyclic; each node's roles match the action's declared contract (required/optional roles + accepted input formats), e.g. `functional_connectivity` must be fed a *parcellated* series; dataset-node lineage is subject-independent; no two written outputs collide. (The per-action contract is Q2.)

---

## 2. Action catalog

| action | kind | inputs (roles) / source | produces | notes |
|---|---|---|---|---|
| `select_data` | selection | `dataset:` + `filters:` | data file(s) | fans out over matches |
| `select_atlases` | selection | `dataset:` + `filters:` | atlas(es) | ≥1 match ok |
| `parcellate_timeseries` | processing | `timeseries`, `atlas` | parcel×time (ptseries/tsv) | warps atlas→data (Strategy A) |
| `parcellate_scalar` | processing | `scalar`, `atlas` | parcel means (tsv/pscalar) | anat→`morph` suffix rule |
| `functional_connectivity` | processing | `timeseries` (parcellated) | relmat (pconn/tsv) | consumes a *parcellated* series |
| `map_scalar_to_surface` | processing | `scalar`, `surfaces` | surface scalar (func.gii/dscalar) | cortex only; volume→surface (Strategy B) |
| `resample_surface_scalar` | processing | `surface_scalar`, `surfaces` | surface scalar on target mesh | fsnative→fsLR via registration spheres; fsLR surface is **32k/164k** (91k = dense-CIFTI grayordinates, not a surface density) |
| `cortical_depth_profile` | processing | `scalar`, `surfaces` | per-depth surface scalars | ribbon sampling, N depths |
| `wm_depth_profile` | processing | `scalar`, `surfaces` | per-depth maps | fixed mm distances into WM |
| `resample_subcortical` | processing | `scalar`, `structures` | subcortical grayordinate volume | Strategy A image-resample: warp the continuous scalar onto the standard MNI152NLin6Asym 2mm grayordinate grid, continuous interpolation (not `GenericLabel`) |
| `assemble_cifti` | processing | `surface`, `volume` | dense CIFTI (dscalar/dtseries) | staples cortex surface + subcortex volume |
| `tractogram_to_dseg` | processing | `tractograms` | dseg atlas | TDI→binarize→label (inline atlas construction) |
| `map_scalar_to_streamlines` | processing | `scalar`, `streamlines` | annotated TRX (dpv + dps) | sibling of `map_scalar_to_surface`; `per_vertex`→**dpv**, `per_streamline: <stat>`→**dps** |
| `parcellate_scalar_as_roi` | processing | `scalar`, `atlas` | per-bundle mean (tsv) | wrapper over `parcellate_scalar` on the bundle dseg (Strategy A: warp ROI dseg → scalar space, per-ROI mean) |
| `parcellate_scalar_as_tract_profile` | processing | `scalar`, `bundles` | per-node profile (tsv) | wrapper: `map_scalar_to_streamlines` (dpv) → n_nodes |
| `tract2region` | processing | `bundles`, `atlas` | fiber counts per ROI | DSI Studio Tract2Region; optional dps weighting |
| `region2region` | processing | `streamlines`, `atlas` | region×region relmat(s) | edges weighted by named **dps** (count/sift2/fa/cbf); one node → many matrices |
| `atlas_union` / `atlas_intersect` / `atlas_outer_product` | processing | `a`, `b` | new atlas (dseg) | BAT algebra as nodes (dataset level) |

Geometry symmetry: `map_scalar_to_surface` (per-vertex GIFTI metric), `map_scalar_to_streamlines` (dpv/dps TRX), and `parcellate_*` (per-parcel) are all "sample a volume onto a geometry, attach values to its elements."

---

## 3. User stories

### 3.1 Parcellate XCP-D bold with a new atlas (4S456Parcels)

Re-parcellate existing XCP-D denoised BOLD with an atlas you didn't run XCP-D with.

```yaml
nodes:
- name: load_bold
  action: select_data
  dataset: xcpd
  filters: {space: fsLR, desc: denoised, suffix: bold, extension: .dtseries.nii}
- name: atlas_4s456
  action: select_atlases
  dataset: atlases
  filters: {atlas: 4S456Parcels, extension: .dlabel.nii}
- name: parcellate_bold
  action: parcellate_timeseries
  inputs: {timeseries: load_bold, atlas: atlas_4s456}
  parameters: {min_coverage: 0.5}
  write_outputs: true
- name: fc_bold
  action: functional_connectivity
  inputs: {timeseries: parcellate_bold}
  parameters: {xdf_covariance: true}
  write_outputs: true
```

Outputs (`func/`): `..._atlas-4S456Parcels_stat-mean_timeseries.tsv` and `..._atlas-4S456Parcels_stat-pearsoncorrelation_relmat.tsv` (+ `.json`). A CIFTI dlabel atlas parcellates cortex *and* subcortex in one `wb_command -cifti-parcellate` call.

### 3.2 GM-NODDI → fsLR, compared to thickness / ALFF / gene expression

Map a volumetric GM-NODDI scalar to the surface and to fsLR, parcellate and depth-profile it, and compare against a surface morphometric (thickness), a functional map (ALFF), and a subject-independent reference (neuromaps gene expression). Note the **gene-expression parcellation is `dataset`-level** — identical for every subject, computed once.

```yaml
# --- runs ONCE for the dataset ---
dataset:
- name: atlas_hcpmmp
  action: select_atlases
  dataset: atlases
  filters: {atlas: HCPMMP1, den: 32k}
- name: atlas_4s456
  action: select_atlases
  dataset: atlases
  filters: {atlas: 4S456Parcels, den: 32k}
- name: load_geneexpr
  action: select_data
  dataset: neuromaps
  filters: {suffix: map, desc: geneexpression, space: fsLR, den: 32k}
- name: geneexpr_parc
  action: parcellate_scalar
  inputs: {scalar: load_geneexpr, atlas: atlas_hcpmmp}
  write_outputs: true            # -> tpl-fsLR/..._atlas-HCPMMP1_desc-geneexpression_map.tsv

# --- runs PER SUBJECT (may reference dataset nodes) ---
nodes:
- name: surfaces
  action: select_data
  dataset: smriprep
  filters: {hemi: [L, R], suffix: [pial, white, midthickness], extension: .surf.gii}

# GM-NODDI: volume -> surface -> fsLR -> parcellate + ribbon profile
- name: load_noddi
  action: select_data
  dataset: qsirecon
  filters: {suffix: dwimap, model: noddi, space: ACPC}   # no `param` -> all params, fans out
- name: noddi_on_surface
  action: map_scalar_to_surface
  inputs: {scalar: load_noddi, surfaces: surfaces}
  parameters: {source_space: fsnative}
- name: noddi_fslr
  action: resample_surface_scalar
  inputs: {surface_scalar: noddi_on_surface, surfaces: surfaces}
  parameters: {target_space: fsLR, target_density: 32k}
  write_outputs: true
- name: noddi_parc
  action: parcellate_scalar
  inputs: {scalar: noddi_fslr, atlas: [atlas_hcpmmp, atlas_4s456]}
  write_outputs: true
- name: noddi_depth
  action: cortical_depth_profile
  inputs: {scalar: load_noddi, surfaces: surfaces}
  parameters: {n_surfaces: 14, include_pial: true, include_white: true}
  write_outputs: true

# cortical thickness (already a surface scalar)
- name: load_thickness
  action: select_data
  dataset: smriprep
  filters: {suffix: morph, desc: thickness, space: fsnative}
- name: thickness_fslr
  action: resample_surface_scalar
  inputs: {surface_scalar: load_thickness, surfaces: surfaces}
  parameters: {target_space: fsLR, target_density: 32k}
- name: thickness_parc
  action: parcellate_scalar
  inputs: {scalar: thickness_fslr, atlas: atlas_hcpmmp}
  write_outputs: true

# ALFF from XCP-D (already fsLR)
- name: load_alff
  action: select_data
  dataset: xcpd
  filters: {suffix: boldmap, stat: alff, space: fsLR, den: 32k}
- name: alff_parc
  action: parcellate_scalar
  inputs: {scalar: load_alff, atlas: atlas_hcpmmp}
  write_outputs: true
```

Shows: two-level scope (gene-expr once → `tpl-fsLR/`), multi-input nodes (`scalar`+`surfaces`), one `surfaces` node fanned to many consumers, list-valued `atlas`, participant nodes referencing dataset-level atlases.

**Whole-brain CIFTI variant (cortex + subcortex).** A dense CIFTI is two geometries: cortex = surface, subcortex/cerebellum = volume voxels in named structures. Subcortical signal is **volumetric** and must not be projected to the surface. To build a full-brain dense NODDI CIFTI:

```yaml
- name: subcort_structures
  action: select_atlases
  dataset: atlases
  filters: {atlas: HCPSubcortical, space: MNI152NLin6Asym, res: 2}
- name: noddi_subcort
  action: resample_subcortical
  inputs: {scalar: load_noddi, structures: subcort_structures}
  parameters: {target_space: MNI152NLin6Asym, resolution: 2}   # needs a T1w->MNI xfm (see below)
- name: noddi_cifti
  action: assemble_cifti
  inputs: {surface: noddi_fslr, volume: noddi_subcort}   # 32k cortex + MNI 2mm subcortex = den-91k dense CIFTI
  write_outputs: true
```

Note the densities: the surface scalar (`noddi_fslr`) is on the fsLR **32k** mesh, and only the *assembled* dense CIFTI is **den-91k** (32k cortex + MNI152NLin6Asym-2mm subcortex ≈ 91,282 grayordinates). Parcellating that dense CIFTI would use a den-91k `.dlabel.nii`; the cortex-only surface parcellations above use den-32k atlases.

The **transform asymmetry** is the key design point: cortex reaches fsLR for free via sMRIPrep's precomputed surface spheres, but the subcortex reaches *standard* grayordinates only through a volumetric `T1w → MNI152NLin6Asym` warp. BDT will not *compute* a normalization (per the hackathon DECISIONS), but it can **apply** the one preprocessing already produced (found as an edge in the transform graph). If no such warp exists, `resample_subcortical` errors, or you build a subject-space CIFTI with T1w-space subcortical structures. See Q6.

### 3.3 WM-depth analysis of the non-NODDI scalars

Sample QSIRecon scalars at fixed depths into white matter, excluding the GM-NODDI maps.

```yaml
nodes:
- name: atlas_hcpmmp
  action: select_atlases
  dataset: atlases
  filters: {atlas: HCPMMP1, den: 32k}
- name: white_surface
  action: select_data
  dataset: smriprep
  filters: {hemi: [L, R], suffix: white, extension: .surf.gii}
- name: load_scalars
  action: select_data
  dataset: qsirecon
  filters: {suffix: dwimap, space: T1w}
  exclude: [{model: noddi}]              # skip GM-NODDI; fans out over the rest
- name: wm_depth
  action: wm_depth_profile
  inputs: {scalar: load_scalars, surfaces: white_surface}
  parameters:
    origin: white
    direction: inward
    distances_mm: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  write_outputs: true
- name: wm_depth_parc
  action: parcellate_scalar
  inputs: {scalar: wm_depth, atlas: atlas_hcpmmp}
  write_outputs: true
```

Note the depth sweep is a **dimension inside the output**, not N filenames differing by a `desc` token (Q4).

### 3.4 Tract-wise means (CBF + FA), tract-as-ROI and tract-profile

Summarize scalars from other pipelines along bundles. `load_bundles` fans to two roles: `tractograms` for atlas construction, `bundles`/`streamlines` for scalar mapping.

```yaml
nodes:
- name: load_bundles
  action: select_data
  dataset: qsirecon
  filters: {model: gqi, suffix: streamlines, extension: .trx, space: ACPC}
- name: bundle_rois
  action: tractogram_to_dseg          # TDI -> binarize -> dseg (inline atlas construction)
  inputs: {tractograms: load_bundles}
  parameters: {threshold: 0.0}
  write_outputs: true

# CBF (ASLPrep)
- name: load_cbf
  action: select_data
  dataset: aslprep
  filters: {suffix: cbf, desc: basil}
- name: cbf_roi
  action: parcellate_scalar_as_roi
  inputs: {scalar: load_cbf, atlas: bundle_rois}       # warps ROI dseg -> CBF space
  write_outputs: true
- name: cbf_profile
  action: parcellate_scalar_as_tract_profile
  inputs: {scalar: load_cbf, bundles: load_bundles}
  parameters: {n_nodes: 100}
  write_outputs: true

# FA (QSIRecon tensor)
- name: load_fa
  action: select_data
  dataset: qsirecon
  filters: {suffix: dwimap, model: tensor, param: fa, space: ACPC}
- name: fa_roi
  action: parcellate_scalar_as_roi
  inputs: {scalar: load_fa, atlas: bundle_rois}
  write_outputs: true
- name: fa_profile
  action: parcellate_scalar_as_tract_profile
  inputs: {scalar: load_fa, bundles: load_bundles}
  parameters: {n_nodes: 100}
  write_outputs: true
```

`parcellate_scalar_as_tract_profile` is a wrapper over `map_scalar_to_streamlines` (dpv → n_nodes, Strategy B). `parcellate_scalar_as_roi` is *not*: it parcellates the bundle **dseg** built by `tractogram_to_dseg` (Strategy A — warp the ROI dseg into the scalar's space, per-ROI mean), which is why its `atlas` role takes `bundle_rois`. See §3.6 and Q7.

### 3.5 tract2region — count fibers intersecting atlas ROIs

```yaml
nodes:
- name: load_bundles
  action: select_data
  dataset: qsirecon
  filters: {model: gqi, suffix: streamlines, extension: .trx, space: ACPC}
- name: atlas_schaefer200
  action: select_atlases
  dataset: atlases
  filters: {atlas: 4S200Parcels, extension: .dlabel.nii}
- name: tract2region
  action: tract2region
  inputs: {bundles: load_bundles, atlas: atlas_schaefer200}
  parameters: {connectivity_type: [pass, end], connectivity_value: count}
  write_outputs: true
```

### 3.6 Region×region connectivity, weighted by count / SIFT2 / FA / CBF

Edge weights are per-streamline **dps** arrays carried by the tractogram: SIFT2, FA, and CBF are genuine stored dps; `count` is the unweighted default (every streamline contributes 1 — unit weights, not a stored array). SIFT2 rides in the qsirecon TRX; cross-pipeline scalars are sampled onto the streamlines with `map_scalar_to_streamlines` (Q8). One `region2region` node emits one matrix per weight — the old separate `weights:` role is gone.

```yaml
nodes:
- name: load_streamlines
  action: select_data
  dataset: qsirecon
  filters: {model: msmtcsd, suffix: streamlines, extension: .trx, space: ACPC}  # carries dps: sift2
- name: load_fa
  action: select_data
  dataset: qsirecon
  filters: {suffix: dwimap, model: tensor, param: fa, space: ACPC}
- name: load_cbf
  action: select_data
  dataset: aslprep
  filters: {suffix: cbf, desc: basil}

# attach scalars as named dps (chain to accumulate onto one tractogram)
- name: annotate_fa
  action: map_scalar_to_streamlines
  inputs: {scalar: load_fa, streamlines: load_streamlines}
  parameters: {name: fa, per_streamline: mean, per_vertex: true}
- name: annotate_cbf
  action: map_scalar_to_streamlines
  inputs: {scalar: load_cbf, streamlines: annotate_fa}     # warps points into CBF space to sample
  parameters: {name: cbf, per_streamline: mean}
  write_outputs: true          # .trx now carries dps: sift2, fa, cbf

- name: atlas_schaefer400
  action: select_atlases
  dataset: atlases
  filters: {atlas: Schaefer400, desc: 400Parcels17Networks}

- name: conn
  action: region2region
  inputs: {streamlines: annotate_cbf, atlas: atlas_schaefer400}
  parameters:
    search_radius: 2
    edges:
      - {weight: count}                    # raw streamline count
      - {weight: sift2, stat_edge: sum}    # SIFT2-weighted (dps from qsirecon)
      - {weight: fa,    stat_edge: mean}   # FA-weighted
      - {weight: cbf,   stat_edge: mean}   # CBF-weighted
  write_outputs: true
```

Notes: dps-as-universal-weight matches MRtrix `tck2connectome -scale_file/-tck_weights_in` and DSI Studio scalar `connectivity_value`s. `map_scalar_to_streamlines` warps the *streamline points* into each scalar's space to sample (Strategy B), attaches values, leaves geometry in place. The "official" per-subject SIFT2 weighting (Rob Smith) needs qsiprep + qsirecon and is a prerequisite outside BDT's scope; BDT consumes the weights as a dps. Output matrices are distinguished by a `stat-`/`desc-` entity per weight.

### 3.7 Extract time series from a subject-space FreeSurfer segmentation

```yaml
nodes:
- name: load_bold
  action: select_data
  dataset: xcpd
  filters: {suffix: bold, desc: denoised, space: T1w, extension: .nii.gz}
- name: atlas_aseg
  action: select_atlases
  dataset: freesurfer
  filters: {suffix: dseg, desc: aseg}
- name: parcellate_bold
  action: parcellate_timeseries
  inputs: {timeseries: load_bold, atlas: atlas_aseg}
  write_outputs: true
```

Two carried-over concerns: FreeSurfer derivatives aren't BIDS, so `select_atlases dataset: freesurfer` needs BIDS-ified FS or a dedicated loader (Q6); and the seg and bold must share a space (both `T1w` here) or the seg must be warped.

### 3.8 [BAT] Atlas algebra (dataset level, same grammar)

BAT is atlas algebra expressed as `dataset`-level nodes — `select_atlases` → an algebra action → write.

```yaml
dataset:
- name: cortical
  action: select_atlases
  dataset: atlases
  filters: {atlas: HCPMMP1}
- name: subcortical
  action: select_atlases
  dataset: atlases
  filters: {atlas: Tian, seg: S2}
- name: cortical_subcortical
  action: atlas_union
  inputs: {a: cortical, b: subcortical}
  parameters: {output_atlas: HCPMMPTian}
  write_outputs: true          # -> a new BIDS Atlas dataset under tpl-<space>/
```

---

## 4. Design decisions (settled)

- **Node graph, not declarative sources.** Flat `nodes:`/`dataset:` lists; data flows node→node via `inputs:`.
- **No input sugar.** Selection nodes use `dataset:` + `filters:`; processing nodes use `inputs:` (named roles) only. Eliminates the dataset-vs-node namespace ambiguity.
- **Atlases are nodes** (`select_atlases`), selected in-spec; `--atlases` CLI flag removed. Atlases may also be *constructed* by an action.
- **Two levels:** `dataset:` (subject-independent, run once → `tpl-`) and `nodes:` (per subject); participant may reference dataset, not vice versa; scope is validated.
- **`desc` prepend-composes** (camelCase, BIDS-legal), never overwrites; **provenance lives in the JSON sidecar** (`GeneratedBy` + `Sources`). Output collisions are errors.
- **`write_outputs`** marks materialized nodes (default false).
- **Transforms are per-action** (grid=Strategy A / points=Strategy B), needing two typed queries over one transform graph.
- **Subcortical CIFTI is volumetric**: `resample_subcortical` + `assemble_cifti`; never projected to the surface.
- **All streamline edge weights are dps** (count/SIFT2/FA/CBF); `map_scalar_to_streamlines` (dpv/dps) is the shared primitive under tract profiles, tract-ROI means, and weighted connectivity.

## 5. Open questions

- **Q1 — `select_data` multiplicity.** Confirm selection fans out over multiple matches (needed for "all NODDI params") and that >1 match is never an error, only narrowed by filters/`exclude`.
- **Q2 — per-action role/format contract.** Declare each action's required/optional roles and accepted input formats so the static validator (§1.8) can reject bad wiring. Precondition for everything else.
- **Q3 — BDT/BAT boundary.** `tractogram_to_dseg` lets BDT construct atlases mid-run; §3.8 shows BAT algebra in the same grammar. Decide whether inline construction lives in BDT (lean: yes, consume inline-built atlases) while reusable cross-subject atlas datasets stay BAT (dataset level).
- **Q4 — encoding continuous sweeps.** Depths/thresholds should be a *dimension inside one file* (dtseries "depth as time", or a column), not N filenames differing by a `desc` token. `desc` prepend closes the *variant-label* half of naming; this is the other half.
- **Q5 — transform two-query engine.** Implement image-resample vs point-warp queries over one graph, with the direction-convention and warp-invertibility checks from plan-review §5; error when only the wrong-direction warp exists.
- **Q6 — subject-space / non-BIDS atlases & subcortical normalization.** Loader for FreeSurfer (non-BIDS) atlases; subject/session scoping for per-subject atlases; and the `T1w→MNI152NLin6Asym` gating for standard subcortical CIFTI (apply preprocessing's warp, or build subject-space CIFTI, else error).
- **Q7 — wrappers vs recipes.** Decide whether the bundle-summary actions stay as named sugar or become documented recipes over their primitives: `parcellate_scalar_as_tract_profile` = `map_scalar_to_streamlines` (dpv) → n_nodes (Strategy B); `parcellate_scalar_as_roi` = `parcellate_scalar` on a bundle dseg (Strategy A). They wrap *different* primitives, so "collapse them" isn't a single decision.
- **Q8 — `trx-rs` features to add (we own these).** (i) write dpv/dps into TRX, (ii) import an external per-streamline vector (SIFT2 `.csv`) as a named dps, (iii) dps-weighted endpoint→parcel connectivity from a label volume — which also closes the plan-review "endpoint→parcel has no home" gap. Build in `trx-rs` (Rust, reusable), with MRtrix `tcksample`/`tck2connectome` as the validation oracle. Also: the Rust binaries (`trxrs`/`giftirs`/`odx`) need a container build stage on `PATH`.
