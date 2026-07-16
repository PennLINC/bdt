## Key Outcomes

The team reviewed an AI-generated (Opus) BDT pipeline implementation using NIRAP and decided to pivot back to **NiPype** as the workflow foundation due to familiarity, available existing code, and NIRAP's lack of content hashing/resumability. The Opus-generated code contained significant issues including residual NiPype imports in NIRAP-structured workflows. A ground-up, unit-test-driven development strategy was adopted, with test data being uploaded to OpenNeuro for shared use. 
## Decisions Made

- **Pivot to NiPype** over NIRAP/Pydra for initial implementation; NIRAP/Pydra migration deferred to a later stage 
- **No NIRAP engine** in BDT workflows; sequential per-subject execution considered acceptable given limited intra-subject parallelism 
- **Ground-up development strategy** adopted: build individual interfaces/workflows with tests before top-level orchestration 
- **TRX single-file input** preferred over individual per-bundle `.tck.gz` files for tracked profile workflows 
- **NiPype Apache license headers** retained pending deeper licensing review, given heavy derivation from fMRIPrep/NiWorkflows 
- **Track-to-region** to be reimplemented from scratch in BDT, then imported into QSI Recon, avoiding circular dependency risk 
- **XCPD parcellation outputs**: CIFTI + TSV for surface data; TSV only for NIfTI inputs — no implicit NIfTI-to-CIFTI conversion 

## Completed

- Opus generated a ~5,000-line BDT codebase including interfaces, workflows, engine, tests, and BIDS output naming 
- YAML workflow spec, transform graph validator, typed queries, PyBIDS data collection, and BIDS output naming confirmed functional 
- ANTs run-to-run reproducibility root cause identified; per-thread derivative handling solution implemented via Codex 
- TRXRS TT-to-TCK conversion tested; profiles align closely with native DSI Studio outputs despite minor coordinate offset 

## Blockers

- **Opus-generated workflows still import NiPype** (`pe.Node`, `pe.Workflow`) despite NIRAP target — needs correction with NIRAP example repos as reference 
- **ANTs AVX instruction issue** on Apple Silicon (Rosetta/compatibility mode) may affect local Docker testing 
- **ASL prep test data** processed with v25.1.0; whole-FOV CBF masking fix requires v26.0.3 
- **OpenNeuro upload** of 13 GB test dataset in progress; per-file transfer slow 

## Pending Confirmation

- Whether NIRAP wrappers are already first-class in Pydra 2.0 (follow up with Satra) 
- Migas dashboard credentials — need to confirm with Matthias before full adoption 
- BIDS config comprehensiveness for all expected BDT outputs needs validation 
- Licensing strategy for files synthesizing NiPype, fMRIPrep, and NiWorkflows code 

## Open Questions

- Whether DSI Studio TRK coordinate offset is a half-voxel bug or floating-point/discretization error from TT.GZ integer encoding 
- How to handle missing bundles (< 10 streamlines) in tracked profile workflows 
- NiPype "silent crash" behavior (memory errors continuing past failed nodes) — whether NIRAP/Pydra handles this better 

## Action Items

- **Matthew:** Clone NIRAP repo and new CPAC version to workspace as Opus reference for next code generation round 
- **Matthew:** Remove NIRAP from current implementation; test user workflows against grumpy dataset using local NiPype 
- **Parker:** Complete OpenNeuro upload of 13 GB test subject dataset 
- **Taylor:** Finalize V2 HackMD user stories document; remove unimplemented white matter depth story 
- **Adam:** Finalize tracked region user story with expected config, CLI, and outputs 
- **Howard:** Review tracked profile user story 
- **Tien:** Complete volumetric XCPD parcellation user story (CIFTI + NIfTI variants) 
- **Taylor:** Follow up with Satra re: NIRAP + Pydra 2.0 integration status (via Dorota) 
- **All:** Each team member drafts assigned user story with config YAML, CLI command, and expected outputs list 
