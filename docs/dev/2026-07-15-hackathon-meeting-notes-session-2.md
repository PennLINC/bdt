## Key Outcomes

The team reviewed and significantly extended the BDT (BIDS Derivatives Tool) pipeline specification, reconciling new ideas from the day's session with prior planning documents. Key outcomes include a finalized YAML-based config schema inspired by QSI Recon, a cleaner atlas selection approach using `load_data`/`select_data` nodes instead of CLI flags, and a working action catalog covering parcellation, connectivity, surface mapping, tractometry, and dataset-level operations. 
## Decisions Made

- **YAML config over CLI flags** for atlas selection: use `select_data` nodes with BIDS filters rather than colon-separated CLI syntax, enabling multi-entity filtering (e.g., `atlas`, `seg`, `scale`, `desc`) 
- **`inputs`** **(plural) as a dictionary** field on all nodes, replacing inconsistent `input`/`inputs` usage; named inputs must match named outputs of upstream nodes 
- **`desc`** **entity** (not a QSI Recon-style suffix directory) used to differentiate output variants (e.g., `desc-XDF` for covariance-estimated connectivity) 
- **`parcelate_scalar`** **replaces** **`parcelate_scalar_as_roi`**; bundles converted to PSEG (not DSEG) via `tractogram_to_pseg`, with threshold as a parameter 
- **Dataset-level section** added to spec for operations that run once (e.g., atlas resampling), avoiding per-subject redundancy and race conditions 
- **TRX library** used for streamline transforms now; ANTS TRX support deferred until ITK 6 official release 
- **`template`** **entity** (not `space`) required for population-level/template-flow files (e.g., `tpl-FSLR`) 
- Surface fallback parameters needed for inconsistent fMRI Prep naming (`white` vs. `smooth_wm`) 

## Action Catalog Confirmed

- `select_data` / `select_atlases` — BIDS-filter-based file selection 
- `parcelate_time_series` — parcellate BOLD; outputs time series + coverage map 
- `functional_connectivity` — takes parcellated time series; optional XDF covariance 
- `parcelate_scalar` — generic scalar parcellation with `summary_stats` parameter (default: mean) 
- `resample_surface_scalar` — maps scalars from FS Native → FSLR 91K 
- `tractogram_to_pseg` — converts bundles to PSEG atlas with threshold parameter 
- `track_to_region` — streamline-to-ROI mapping; supports scalar weighting (FA, CBF) via TRX data-per-vertex 
- `annotate_streamlines` — chains scalar annotations onto TRX file for reuse across matrices 

## Open Questions

- Whether `load_surfaces` should return named hemisphere outputs (`LH_pial`, `RH_white`, etc.) or require separate per-surface nodes 
- How to handle atlas space mismatches across file formats (NIfTI vs. CIFTI vs. GIFTI) — proposed: auto-detect and warp if transforms available, else error 
- BABS integration for dataset-level nodes: run dataset portion first, save as external dataset, load via CLI 
- Group-level/template map parcellation (e.g., histology maps) — BDT may handle surface-to-surface warping given complexity of workbench transforms 

## Pending Confirmation

- Whether `select_data` can dispatch to correct file format internally via `detect_format`, or requires separate `select_surfaces` action 
- Common response function ingress path for QSI Recon (currently mounted folder + filenames in spec; needs cleanup) 
- SIFT weighting node completeness — group mean B0 and response function scaling not yet fully addressed 

## Action Items

- **Taylor:** Update existing user stories — remove `parcelate_scalar_as_roi`, update `track_to_region` example to include FA/CBF scalar input 
- **Taylor:** Add `load_data`/`select_data` node example to YAML config in HackMD 
- **Matthew:** Paste finalized action catalog and full YAML examples into HackMD; push to `doc_dev` folder in BDT repo 
- **Matthew:** Clone the correct BDT GitHub repo locally 
- **Team:** Clean up user story examples, then align with GRUMPY dataset outputs to identify any missing QSI Recon derivatives 
