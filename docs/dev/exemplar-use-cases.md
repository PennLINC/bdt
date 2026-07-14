# Exemplar BDT and BAT use cases

**Status:** Proposed acceptance scenarios for planned behavior, not a claim that
every path is implemented in the current pre-alpha release.

These examples translate the user-facing [usage](../usage.rst), [spec
grammar](../spec.rst), [workflow](../workflows.rst), and [output
conventions](../outputs.rst) into concrete research scenarios. Their feature
expectations come from the [design specification](2026-06-01-bdt-design.md),
[implementation plan](2026-06-01-bdt-implementation-plan.md), [plan
review](2026-06-01-bdt-plan-review.md), and [meeting notes](bdt_meeting_notes.md).
Entity selectors are illustrative and must be refined to match the files in a
specific derivative or atlas dataset. In particular, BDT should error rather
than guess when a selector is ambiguous or a required transform path is absent,
and all inputs to one BAT operation must be discrete `dseg` atlases in the same
`tpl-` space.

## BDT: applying atlases to derivatives

### 1. Resting-state functional connectomes from dense fMRI

A developmental-neuroscience team wants Schaefer-400 regional time series and
functional-connectivity matrices from fMRIPrep CIFTI outputs. This exercises
CIFTI format dispatch, time-series parcellation, connectivity calculation, and
BEP-style `timeseries` and `meas-*_relmat.dense.tsv` outputs with node metadata.

```yaml
sources:
  - suffix: bold
    task: rest
    space: fsLR
    den: 91k
    datasets: [fmriprep]
    operations: [parcellate_timeseries, functional_connectivity]
    atlases:
      - atlas: Schaefer2018
        seg: 17networks
        scale: 400
```

### 2. Regional perfusion level and dynamics after stroke

A cerebrovascular study wants both mean CBF per HCP-MMP1 parcel and parcel-wise
ASL time series for later hemodynamic modeling. The two source entries exercise
4D and scalar NIfTI dispatch, explicit selection among multiple ASLPrep
derivatives, label-safe atlas-to-grid resampling, and `perf/` TSV outputs.

```yaml
sources:
  - suffix: cbf
    desc: basil
    datasets: [aslprep]
    operations: [parcellate_scalar]
    atlases: &perfusion_atlases
      - atlas: HCPMMP1
        res: 2
  - suffix: asl
    desc: preproc
    datasets: [aslprep]
    operations: [parcellate_timeseries]
    atlases: *perfusion_atlases
```

### 3. Cortical-thickness summaries with an atlas on another mesh

An aging study has sMRIPrep cortical-thickness GIFTI files on `fsaverage5`, but
its chosen Destrieux labels are distributed on the full `fsaverage` mesh. BDT
should use registration spheres and Workbench label resampling before computing
parcel means, exercising GIFTI scalar dispatch and the planned cross-mesh
surface path. The right hemisphere is represented by a second, analogous source
entry.

```yaml
sources:
  - suffix: thickness
    space: fsaverage5
    hemi: L
    datasets: [smriprep]
    operations: [parcellate_scalar]
    atlases:
      - atlas: Destrieux
        den: 164k
        hemi: L
```

### 4. Structural connectomes and tractogram quality summaries

A connectomics group wants Schaefer-400 endpoint-connectivity matrices from
QSIRecon iFOD tractograms, plus streamline count and mean length for cohort QC.
This exercises BEP-046 entity filtering, TRX ingest, point-coordinate warping
from native diffusion space to atlas space, endpoint-to-parcel assignment, and
the distinction between a `relmat` connectome and a non-matrix bundle-statistics
TSV.

```yaml
sources:
  - suffix: tractogram
    track: ifod
    datasets: [qsirecon]
    operations: [streamline_connectivity, bundle_stats]
    atlases:
      - atlas: Schaefer2018
        seg: 17networks
        scale: 400
```

### 5. Regional crossing-fiber organization in multiple sclerosis

A diffusion study wants parcel-level white-matter ODF coefficients and fixel
summaries from a CSD `dwimap`, using a white-matter atlas. This exercises
BEP-016 `model-`/`param-` selection, ingestion into the internal ODX container,
spatial resampling with SH reorientation, and the distinct ODF and
cardinality-aware fixel reducers. The implementation plan identifies the fixel
reducer as follow-up work, making this an explicit acceptance target for the
full planned workflow.

```yaml
sources:
  - suffix: dwimap
    model: csd
    param: wm
    datasets: [qsirecon]
    operations: [parcellate_odf, parcellate_fixel]
    atlases:
      - atlas: JHU
        seg: whitematter
```

## BAT: creating derived atlases

### 1. A cortical-subcortical connectome atlas

A whole-brain connectomics study needs HCP-MMP1 cortex and Tian S2 subcortex in
one label image. An overlap-tolerant union should keep the first atlas at shared
boundary voxels, offset labels from the second atlas, and record source-label
provenance in `nodelabels.tsv`.

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
      desc: FirstPrecedence
```

### 2. Preserve detailed subcortical nuclei in a whole-brain atlas

A movement-disorders team wants CIT168 nuclei embedded in a coarser
Schaefer-Tian atlas. Listing CIT168 first makes its detailed labels win in
overlap while the second atlas fills the rest of the brain. This exercises the
same union primitive with scientifically meaningful precedence and atlases
collected from multiple input atlas datasets.

```yaml
operations:
  - name: detailedSubcortex
    operation: union
    inputs:
      - atlas: CIT168
      - atlas: SchaeferTian
        scale: 400
    output_entities:
      atlas: CIT168SchaeferTian
      scale: 400
```

### 3. Restrict cortical parcels to the default-mode network

A cognitive study needs the portions of Schaefer parcels lying inside a
discrete default-mode-network mask. `intersect` should retain the first atlas's
parcel identities only where the second atlas is nonzero; it must not create
Cartesian parcel-by-network labels.

```yaml
operations:
  - name: defaultModeParcels
    operation: intersect
    inputs:
      - atlas: Schaefer2018
        seg: 17networks
        scale: 400
      - atlas: Yeo2011
        seg: DMN
    output_entities:
      atlas: SchaeferDMN
      scale: 400
```

### 4. Split anatomical parcels by vascular territory

A stroke study wants regions that preserve anatomical parcel identity while
also distinguishing arterial territories. The outer product of an anatomical
atlas and a discrete vascular-territory atlas should assign a unique label to
each observed parcel-territory pair, enabling territory-aware regional CBF
analysis and retaining pair provenance in `nodelabels.tsv`.

```yaml
operations:
  - name: parcelTerritories
    operation: outer_product
    inputs:
      - atlas: HarvardOxford
      - atlas: ArterialTerritories
        seg: vascular
    output_entities:
      atlas: HarvardOxfordVascular
```

### 5. Parcel-by-bundle regions for gray-white matter coupling

A multimodal lab wants to study BOLD signal near the cortical terminations of
major tracts. It supplies an AFQ bundle atlas already rasterized as a discrete
`dseg` (not streamline data) and crosses it with Schaefer-400. The outer product
creates one label for each overlapping parcel-bundle pair, exercising the
parcel-by-bundle scenario from the design discussions while remaining within
BAT's discrete-atlas scope.

```yaml
operations:
  - name: parcelBundles
    operation: outer_product
    inputs:
      - atlas: Schaefer2018
        seg: 17networks
        scale: 400
      - atlas: AFQ
        seg: bundles
    output_entities:
      atlas: SchaeferAFQ
      scale: 400
```

Every BAT scenario is expected to emit a valid BIDS Atlas derivative: a `dseg`
image and `dseg.tsv`, the required root-level
`atlas-<label>_description.json`, canonical entity order, and inherited
`tpl-` space. Union and outer-product scenarios additionally require source
label provenance; a template mismatch or an ambiguous atlas selector is an
error rather than an invitation to choose heuristically.
