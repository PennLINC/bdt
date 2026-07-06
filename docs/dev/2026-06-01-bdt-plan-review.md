# BDT/BAT Plan Review — Design, Feasibility, Cleanliness

**Date:** 2026-06-01
**Reviewer:** Claude (Opus 4.8)
**Scope:** Feedback on `2026-06-01-bdt-design.md` and `2026-06-01-bdt-implementation-plan.md`, with focus on the missing transform story for GIFTI and streamline (TCK/TRX) data, grounded in the `odx-rs`, `trx-rs`, `gifti-rs`, and `itk-transforms-rs` repos.

---

## Top line

The volumetric half of this plan is solid and closely follows proven xcp_d/qsirecon patterns. But the part flagged as missing — gifti and streamline transforms — isn't just under-specified, it's built on a principle that *doesn't hold for non-volumetric data*, and as written the plan **never actually calls the three Rust repos that were added**. The good news: the repos are well-designed and cover the gap cleanly, once the transform model is reworked to be aware of data geometry. There's also one genuinely un-owned capability (streamline→parcel endpoint assignment) that no repo currently provides — that may need to be cloned or built.

> Note: the implementation plan references an existing `src/bdt/` scaffold (clearly mined from xcp_d) that isn't in this directory, so the plan was reviewed as a spec rather than against live code.

---

## 1. The core problem: transforms are geometry-polymorphic; the plan models only one geometry

The design's governing principle (§7) is:

> *transform atlases to data space, not data to atlas space (avoids repeated interpolation of the data)*

This is exactly right **for volumetric/grid data** (BOLD, CBF, FA NIfTI, CIFTI dense): warping the 4D data would re-interpolate signal on every step, so you warp the static label atlas once with `GenericLabel`. xcp_d and qsirecon both do precisely this.

But the rationale — "avoids repeated interpolation of the data" — **is meaningless for point data**. A GIFTI surface is a set of vertex coordinates; a tractogram is a set of streamline coordinates. Warping those *moves coordinates losslessly* — there is no signal resampling to avoid. In fact, for point data the principle inverts: warping the **points** is exact, while warping a **label volume** into a coarse diffusion grid with nearest-neighbor *loses small parcels*. This is the whole reason `trx-rs` and `gifti-rs` exist, and the plan's transform section (Task 6, the `networkx` + ANTs `ApplyTransforms` engine) has no concept of it.

So there are really **three transform modalities**, not one:

| Data geometry | Operation | Tool | h5 direction | Interp. concern |
|---|---|---|---|---|
| **Grid** (NIfTI vol, CIFTI dense) | resample atlas→data grid | ANTs `ApplyTransforms` | **same-named** (`from-atlas_to-data`) | yes — use `GenericLabel` |
| **Points** (GIFTI verts, TCK/TRX streamlines) | warp the *data's* coords (or atlas verts) | `giftirs transform`, `trxrs transform` | **opposite-named** | none — lossless |
| **Diffusion model grid** (SH/ODF/fixels) | resample + reorient | `odx transform` | same-named, but with SH reorientation | yes, + fixel cardinality |

The plan collapses all of this into modality #1. Everything below follows from that.

---

## 2. How the Rust repos actually slot in (and the integration reality)

Verified capabilities and, importantly, the integration surface of each:

| Repo | What it does for BDT | Integration path | Caveat |
|---|---|---|---|
| [itk-transforms-rs](itk-transforms-rs/README.md) | Shared RAS+ engine: reads ANTs `.h5`/`.txt`/`.mat`, `map_point`, `jacobian_at` | Rust lib only (consumed by the other three) | **Cannot invert a displacement field** — only affine chains (`itk-transforms-rs/src/chain.rs`) |
| [trx-rs](trx-rs/README.md) | `trxrs transform` (warp streamline coords); TRX/TCK IO; group-based connectivity; set ops | **`transform` is CLI-only**; Python bindings are **read-only** (positions/dps/dpv/groups) | Reads `.trk` but **cannot write it**; writes TRX/TCK |
| [gifti-rs](gifti-rs/README.md) | `giftirs transform` (warp `.surf.gii` vertices); GIFTI IO incl. label tables; auto FreeSurfer C_RAS | **CLI only — no Python bindings** | Warps **geometry only** — fails if input has no POINTSET; will not touch `.func/.shape/.label.gii` data |
| [odx-rs](odx-rs/README.md) | `odx transform` (resample SH/ODF/fixel grids); DSI↔MRtrix↔ODX conversion | Full PyO3 bindings (`odx` pkg) + CLI | No parcellation; **not used anywhere in the current plan** (see §6) |

**Feasibility consequence the plan must confront:** because `giftirs` has no Python bindings and `trxrs transform` is CLI-only, the integration **must** be nipype `CommandLine` interfaces shelling out to the Rust binaries — exactly how ANTs, MRtrix, and `wb_command` are already wrapped. That's clean and idiomatic, but it means:

- the Rust binaries (`trxrs`, `giftirs`, `odx`) have to be **built and on `PATH` in the container** — a new Docker build stage (Rust toolchain or pre-built release binaries). This belongs in the implementation plan and isn't mentioned.
- the plan's current gifti/streamline interfaces use **dipy and nibabel instead** (Task 11 `GiftiParcellate`, Task 12 `BundleAtlasIntersect`). As written, the repos that were added are dead weight. These tasks need rewriting around the CLIs.

---

## 3. Streamlines — the biggest gap, and a decision the plan hasn't made

Three separate problems here.

**(a) The "already in the same space" assumption is usually false.** Design §7 says *"tractography bundle atlases are already defined in the same space as the streamlines"* and so streamlines *"do not use the transform engine."* But qsirecon streamlines are typically in ACPC/T1w/native-DWI space, while bundle/parcellation atlases live in MNI. Moving between them is the core need — and it's precisely what `trxrs transform` was built for. This assumption deletes the very feature the repo provides.

**(b) No strategy has been chosen, and the two legitimate ones pull in opposite directions:**

- **Strategy A — warp the atlas into streamline space** (what qsirecon does today: `WarpConnectivityAtlases` → ANTs `MultiLabel` → MRtrix `tck2connectome`, in `qsirecon/qsirecon/interfaces/utils.py`). Consistent with the stated principle. Does **not** need `trx-rs`'s transform at all — only ANTs. Lossy on small parcels in low-res DWI grids.
- **Strategy B — warp the streamlines into atlas space** (what `trxrs transform` enables). Lossless geometry; once tracts are in MNI you can apply *any* MNI atlas without re-warping, and intersect with surface parcels and bundle atlases uniformly.

The design principle argues for A; the fact that `trx-rs` was added argues for B. **This contradiction needs an explicit decision in the doc.** Recommendation: B as the default for streamlines specifically, because the lossless-points argument (§1) overrides the principle here, and B composes far better with surface/bundle atlases.

**(c) Even after a space is chosen, there's a missing capability with no home.** Turning a tractogram + a label atlas into a region×region matrix requires **endpoint→parcel assignment** (assign each streamline's two ends to parcels, with a small radial search). Verified: `trx-rs`'s connectivity (`trx-rs/src/ops/connectivity.rs`) computes group→group connectivity **only from pre-existing TRX groups** — it does *not* look up endpoints in a NIfTI label volume. So nothing currently available does this step. Options:

1. Build it in Python (trxrs read-only bindings give you `positions()` + `offsets()` → endpoints; nibabel atlas → world-to-voxel → label lookup → N×N). ~50 lines, fully feasible.
2. Add it to `trx-rs` in Rust.
3. Keep delegating to MRtrix `tck2connectome` (then `trx-rs` is only IO/transform, not connectivity — but note `tck2connectome` wants `.tck`, and `trxrs` can write `.tck`, so this composes).

Meanwhile the plan's actual implementation (Task 12, dipy `density_map` per parcel) is the wrong primitive on three counts: it counts streamlines *passing through* parcels (not endpoint connectivity), it's `O(n_parcels × n_streamlines)` rasterization (catastrophic at 400 parcels × millions of streamlines — the exact scalability fear in open question §4.5), and it bypasses `trx-rs` entirely. qsirecon's endpoint approach (`tck2connectome` / DSI Studio, endpoint-based with radial search) is the established, correct reference.

Also: the spec examples use `.trk`, but `trx-rs` is write-only to TRX/TCK. Standardize the tractography I/O on TRX (or TCK) and convert `.trk` on ingest with `trxrs convert`.

---

## 4. GIFTI — "transform" and "parcellate" are being conflated

Two issues.

**(a) The parcellation interface reinvents — worse — what `wb_command` already does.** Task 11's `GiftiParcellate` is a hand-rolled nibabel `darray`-mean (with a dead `ndim==1` branch and fragile label-table handling). xcp_d already parcellates surface data with `wb_command -cifti-parcellate` (`xcp_d/xcp_d/interfaces/workbench.py`) including coverage thresholding (`min_coverage`) — and the BDT CIFTI path already calls `init_parcellate_cifti_wf`. Drop the nibabel interface; route GIFTI through the same `wb_command` path (or convert `.func.gii` + `.label.gii` into the CIFTI parcellation that already exists). `gifti-rs` does **not** parcellate, so it is not the tool for this step.

**(b) Cross-mesh resampling has no Rust home — and the plan needs to say what it requires.** If the atlas `.label.gii` is on a *different* mesh than the data (e.g. atlas on `fsaverage`, data on `fsLR-32k`), you must resample across surfaces with registration spheres (`wb_command -label-resample`). Confirmed: `gifti-rs` explicitly does **not** do this, and neither does xcp_d — xcp_d requires atlases pre-built on fsLR 32k/91k (`xcp_d/xcp_d/utils/atlas.py`) and only matches brainordinates with `-cifti-create-dense-from-template`. So BDT must either **(i)** require surface atlases pre-resampled to the data's mesh (simplest, matches xcp_d), or **(ii)** add a `wb_command -label-resample` step. Pick one and state it.

**(c) Where `gifti-rs` *actually* earns its place:** not in surface-with-surface-atlas parcellation, but in **surface-with-volumetric-atlas** sampling and in **surface↔streamline** mapping. Concretely: warp a subject's `*.surf.gii` (e.g. midthickness in T1w) into a volumetric atlas's space with `giftirs transform`, then sample the label volume at each vertex — a clean, lossless surface parcellation by a *volume* atlas, no `wb_command` needed. This same vertex-warp is the enabler for the streamline→surface mapping that §4.5 frets about (warp surfaces into streamline space, then nearest-vertex assign endpoints). That's the role to document for `gifti-rs` — it's narrower than "the gifti transform tool," and the plan should scope it that way. (`freesurfer-post` morphometry like thickness/curv as `.shape.gii` is the concrete consumer of the surface-parcellation path.)

---

## 5. The direction-convention landmine (highest correctness risk)

This is subtle and will silently produce mirrored/garbage output if missed. The `find_transform_chain` (Task 6) builds a graph from `from-X_to-Y` files and returns the chain for ANTs `ApplyTransforms` — i.e. **image/pull semantics**. The Rust point-warpers follow the **opposite** convention: per trx-rs's README, *"to warp tracts from space A to space B, pass `from-B_to-A_xfm.h5`."*

The trap: if the same graph/query is reused for a streamline warp `SS→AS` by calling `find_transform_chain(graph, SS, AS)`, it returns the `from-SS_to-...` files — but `trxrs` needs the reverse-named `from-AS_to-SS`. Same physical BIDS pair, applied backwards. Two things the engine must encode that it currently doesn't:

1. **Modality-aware selection.** A point-warp query must traverse the graph in the *opposite* direction from a grid-resample query (equivalently: call the chain for `target→source` and hand it to `trxrs`/`giftirs`; `odx` uses the image direction).
2. **Warp invertibility.** Since `itk-transforms-rs` can't invert displacement fields and `trxrs --invert` only flips affines, a point warp **requires the correctly-named file to physically exist** — it cannot be synthesized. The graph must track which direction is backed by a real warp and error explicitly when only the wrong-direction warp is present. fMRIPrep/QSIPrep emit both directions, so this is satisfiable — but the engine has to check.

Cleanest fix: keep one graph, expose two typed queries — `chain_for_image_resample(src→tgt)` and `chain_for_point_warp(src→tgt)` — each returning `(files, tool, invert_flags)`. Also widen the regex (Task 6 requires the literal `_mode-image_` and only matches `.h5|.txt`) to index `.mat` / `*GenericAffine.mat` and warp files that lack `mode-image`.

---

## 6. odx-rs — out of scope as written; intent needed

`odx-rs` is genuinely impressive (full Python bindings, DSI↔MRtrix↔ODX, SH-reorienting `odx transform`), but **the current plan never parcellates SH/ODF/fixel data** — `dwimap` scalars (FA/MD) are handled as plain volumetric NIfTI via `parcellate_scalar`. So `odx` has no role in the plan as written. If the intent is to **parcellate diffusion models** (fixel-based analysis, per-parcel ODF summaries) or to use `odx transform` to move diffusion-derived maps between spaces, that's a real scope expansion with its own design needs (fixel→parcel is not a simple mean; SH reorientation matters). Worth either deferring explicitly or designing a `parcellate_fixel`/`parcellate_odf` operation deliberately.

---

## 7. Cleanliness — concrete issues (independent of the above)

- **`AtlasIntersect` and `AtlasOuterProduct` are byte-for-byte identical** (Task 13, both compute `(d1-1)*n2 + d2` on the overlap). The design claims distinct semantics (intersect = restrict; outer product = Cartesian sub-parcels) but the code doesn't realize it. Real bug.
- **`init_load_atlases_wf` is never defined** — it's referenced in design §6 and Tasks 15–16 and is *the heart of volumetric atlas warping*, but no task creates it. The "known gaps" note catches `init_parcellate_nifti_wf` but misses this larger one. (xcp_d's `init_load_atlases_wf` in `xcp_d/xcp_d/workflows/parcellation.py` is the obvious template.)
- **`ApplyTransforms` wrapper** (Task 7): `if not self.inputs.float: self.inputs.float = True` forces `float=True` permanently — a user can never set it `False`.
- **BAT scope contradiction:** design §8 says BAT outputs `dseg` only and *"tractography is not in scope,"* yet the `bat_spec` example `parcelBundles` feeds `AFQ seg-bundles` into `outer_product`.
- **`AtlasUnion` hard-errors on any voxel overlap** — real cortical+subcortical atlases overlap at boundaries; a precedence rule is likely better than a crash.
- **density_map performance** (Task 12) as in §3(c).

---

## 8. What's missing / what to clone

No Rust *repo* is strictly missing for the transform mechanics — `itk-transforms-rs` + `trx-rs` + `gifti-rs` (+ `odx-rs` if scope expands) cover point-warping completely. But two capabilities have **no home**:

1. **Streamline endpoint→parcel assignment / connectivity-from-a-label-volume.** Nothing currently available does it (`trx-rs` connectivity is group-based only). Decide where it lives: new Python interface in `bdt`, a new subcommand in `trx-rs`, or delegate to MRtrix `tck2connectome`. If there's a Rust repo for this, it should be pulled in; otherwise build it in Python on the trxrs read-only bindings.
2. **Cross-surface mesh resampling** (`fsaverage↔fsLR` label/metric). No Rust tool covers it; it stays a `wb_command` dependency. Not a missing repo, but the plan must declare whether it requires pre-resampled surface atlases (recommended) or adds the `wb_command` step.

And one packaging gap: the Rust binaries need a **container build stage** so `trxrs`/`giftirs`/`odx` are on `PATH`.

---

## Open decisions before reworking the transform sections

1. **Streamlines: Strategy A (warp atlas→tracts, qsirecon-style) or B (warp tracts→atlas via `trxrs`)?** Recommend B for the lossless-points reason, but it contradicts the stated §7 principle — a deliberate call is needed.
2. **Surface atlases: require pre-resampled to the data mesh, or add `wb_command -label-resample`?**
3. **Is `odx-rs` in scope for v1** (parcellate diffusion models), or deferred?
4. **Where does endpoint→parcel assignment live** — and is there a repo for it to look at?

A natural follow-up deliverable: a replacement **§7 Transform System** (the modality table, the two-query engine API, and per-data-type workflow sketches showing exactly where `trxrs`/`giftirs`/`odx` get invoked and in which direction) plus a corrected set of gifti/streamline tasks.
