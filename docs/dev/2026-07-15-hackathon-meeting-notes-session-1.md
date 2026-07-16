## Key Outcomes

The team conducted a full-day hackathon to define the scope, user stories, config file structure, and output organization for **BDT (BIDS Derivatives Transformer)**, a new post-processing BIDS app. Core purpose: enable cross-modality derivative mapping and atlas parcellation without rerunning upstream pipelines. Key decisions reached include no spatial normalization within BDT, a preference for QSI Recon-style stepwise config files over source-with-operations format, and shared workflow code between BDT and existing apps (XCPD, QSI Recon). 
## Decisions Made

- **No spatial normalization in BDT**: Users must compute registrations in upstream software (fMRI Prep, QSI Prep); BDT will only support rigid T1W↔ACPC alignment as the sole exception. 
- **Atlases are transformed, not target data**: Target data stays in its native space; atlases/streamlines are warped to match. 
- **Config format**: Adopt **QSI Recon-style stepwise spec** (explicit ordered steps with named inputs/outputs) rather than flat sources-with-operations structure. 
- **Shared workflow code**: Parcellation logic shared between XCPD and BDT to prevent implementation drift. 
- **No gradient nonlinearity distortion correction** in first pass; error out if detected. 
- **Rigid registration only** for T1W↔ACPC; no affine, no MSM Sulc reproduction within BDT. 
- **Region-to-region connectivity matrices** deferred to a separate user story from track-to-region. 
- **Depth sampling / white matter depth analysis** deferred until stakeholders clarify expected outputs. 
- **Gene expression / group-level maps** stored at TPL root level (not duplicated per subject); symlinks or dual-download approach for AWS use cases. 

## Primary Use Cases Defined

- **Forgot an atlas**: Re-parcellate XCPD or QSI Recon outputs with a new atlas (e.g., 4S456) without full rerun. 
- **Cross-modality mapping**: Map QSI Recon scalars (e.g., NODI/ICVF) onto FreeSurfer surfaces → resample to FSLR 91K → parcellate. 
- **Tract-wise scalar means**: Warp bundle streamlines into CBF/ASL map space → compute bundle-wise mean values. 
- **Track profiles (AFQ-style)**: Use DIPY to compute per-node scalar values along bundle centroids instead of simple ROI averages. 
- **Track-to-region intersection**: Use DSI Studio (bleeding-edge version in BDT container) to compute fiber counts, mean FA/MD, geometry measures per ROI. 
- **SIFT-weighted connectivity**: Apply single-subject response functions + SIFT weights from QSI Recon; no new FOD estimation. 

## Open Questions

- Whether scalar GIFTIs (func.gii / shape.gii) should be written to output alongside SIFTIs. 
- Handling gradient nonlinearity correction detection — no standard BIDS metadata field exists yet. 
- Whether BDT should support extended functional connectivity measures (bootstrapped exact timing, covariance estimates) beyond what XCPD provides. 
- BIDS entity/suffix naming consistency across modalities (e.g., `bold-map`, `dwi-map`, `cbf-map`). 
- First positional argument convention: raw BIDS vs. primary derivative dataset. 

## Action Items

- **Parker**: Create single-subject Grumpy dataset (raw BIDS + all derivatives in one folder) and document available files and dataset size. 
- **Howard**: Draft track profile user story with expected output file structure; pull example CSV from PyAFQ. 
- **Adam**: Locate Rob Smith paper on single-subject SIFT weighting; confirm whether mu multiplication is required. 
- **Team**: Generate QSI Recon-style YAML config equivalents for each user story using ChatGPT/Claude as starting point. 
- **Team**: Review expected outputs for any non-obvious file types and draft mini example tables (3-row format). 
- **Team**: Add scope decisions (no spatial normalization, rigid only) to a dedicated policy/scope section in the design doc. 

## Next Session

Reconvene at **1:30 PM Eastern** to split tasks and begin drafting specs. 
