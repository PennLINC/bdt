# Design: transform query + chaining subsystem

**Date:** 2026-07-18
**Status:** Approved (design)

> **Reconciliation note (2026-07-19):** Component A below (the graph + chain
> queries) is **already implemented and tested** in `src/bdt/transforms/`
> (`graph.py` + `queries.py`, networkx-backed, with *two* chain queries —
> `chain_for_image_resample` and `chain_for_point_warp` — and an `extra_edges`
> injection hook). Per user decision, the implementation plan builds only the
> missing pieces on top of it: TemplateFlow edge enumeration (feeding
> `extra_edges`) and the runtime `nitransforms` resolve/apply node. It does
> **not** create `utils/transforms.py` or a single-BFS/no-networkx variant.

**Scope:** Spec 1 of 2. This spec covers a reusable transform query/chaining
subsystem. Spec 2 (the volumetric `parcellate_scalar` path that consumes it) is
a separate follow-up.

## Problem

BDT actions often need an image in a space it is not stored in — e.g. warping a
bundle-ROI atlas (in `ACPC`) into a scalar's space (`MNI152NLin6Asym`) before
volumetric parcellation. The transforms needed to bridge two spaces are spread
across:

- **subject-specific transforms** stored in the derivative datasets as BIDS
  `from-X_to-Y_mode-image_xfm.{h5,txt,mat}` files (fMRIPrep/ASLPrep/QSIPrep
  outputs — typically stored in *both* directions);
- a **computed bridge** BDT produces at runtime (the rigid `ACPC↔T1w`
  transform from `_register_acpc_to_t1w`, which is not a file on disk until the
  workflow runs);
- **standard-space transforms** on TemplateFlow (inter-template `xfm`s).

BDT needs to determine the *shortest chain* of these transforms between an
arbitrary source and target space, and apply it to resample an image.

Prior art: https://github.com/nipreps/fmripost-template/pull/13 (BFS shortest
path over `(from, to)` space edges + TemplateFlow enumeration). Its body note
about "downloading all transforms" is outdated — `tflow.ls` lists metadata only;
only the transforms on the chosen path are fetched.

## Key decisions

1. **Two coupled specs; this is Spec 1** (the chaining subsystem). Spec 2 is the
   volumetric `parcellate_scalar` path.
2. **Graph edges** = discovered local dataset transforms + computed bridges
   (ACPC↔T1w) + TemplateFlow inter-template edges (via `ls`) + inverses. One
   uniform BFS. Only the transforms on the chosen path are fetched from TF.
3. **Resolution site: runtime, in one nipype node.** The graph is built, the
   path found, and the chain applied inside `_run_interface` — because some
   edges (the ACPC↔T1w bridge) are produced by an upstream runtime node rather
   than existing on disk at build time.
4. **Inverses: affines only.** A reverse edge (`invert=True`) is synthesized only
   for *linear* transforms (`.txt`/`.mat`/`.lta`, incl. the computed bridge),
   which `nitransforms` inverts exactly. Nonlinear `.h5` warps are traversed
   only in their stored direction, relying on NiPreps storing both
   `from-X_to-Y` and `from-Y_to-X` (as the datasets do). A needed nonlinear
   reverse that isn't on disk is simply not an edge → BFS fails cleanly.
5. **Application engine: `nitransforms`** (pure Python, already a dependency, the
   NiPreps standard). The ANTs CLI is not installed here; `nitransforms`
   composes the ITK `.h5` + affine `.txt`/`.mat` chain and resamples natively.

## Components

### A. `src/bdt/utils/transforms.py` — pure graph functions (no nipype/network)

Small, dependency-light, unit-testable in isolation:

- `xfm_endpoints(path) -> (from_space, to_space, is_linear)`
  Parse `from-`/`to-` (and `template-`, treated as the `to` for a TF transform)
  from the filename; infer linearity from extension (`.txt`/`.mat`/`.lta` →
  linear, `.h5` → nonlinear).
- `build_transform_graph(entries) -> edges`
  `entries` is an iterable of `(path, from_space, to_space, is_linear)`. Emits a
  forward edge `from→to` with `invert=False` always, plus a reverse edge
  `to→from` with `invert=True` **only if `is_linear`**. An edge carries
  `(neighbor_space, path, invert)`.
- `find_transform_chain(edges, source, target) -> [(path, invert), ...]`
  BFS shortest path (fewest hops); returns the ordered chain of
  `(path, invert)`. Raises `ValueError` naming `source`, `target`, and the set
  of spaces reachable from `source` when no path exists.

### B. `src/bdt/interfaces/transforms.py` — `ResolveApplyTransforms` (runtime)

A `SimpleInterface` that resolves and applies in one node.

Inputs:
- `source`, `target` — space strings (e.g. `'ACPC'`, `'MNI152NLin6Asym'`).
- `moving` — image to warp (e.g. the atlas).
- `reference` — image defining the output grid (e.g. the scalar).
- `local_transforms` — list of discovered xfm files (see "Discovery" below).
- `bridges` — list of computed transform files (e.g. the ACPC↔T1w `.mat`);
  parsed for endpoints the same way as local transforms.
- `interpolation` — resampling interpolation, default `'linear'`
  (consumers pass `'nearest'` for label/dseg images).
- `out_file` — output filename.

`_run_interface`:
1. Parse endpoints of `local_transforms + bridges` via `xfm_endpoints`.
2. Enumerate TemplateFlow edges: `get_templates()` +
   `ls(tpl, suffix='xfm', extension='h5')` (metadata only).
3. `build_transform_graph(...)` over all entries.
4. `find_transform_chain(edges, source, target)`.
5. Fetch only the TF files on the path (`templateflow_get` helper).
6. Build a `nitransforms.manip.TransformChain` honoring `invert` flags (linear
   inverted via `~`), resample `moving` onto `reference` with `interpolation`,
   write `out_file`.

Outputs:
- `out_file` — the warped image.
- `out_transforms`, `out_inversions` — the resolved chain (for provenance/debug).

**Discovery (refinement approved):** `local_transforms` is a concrete **file
list**, discovered at *compile time* via the existing `BIDSDataProvider` (which
already parses `from`/`to`/`mode`) and passed into the node. The graph/path/apply
still run at runtime. This reuses BDT's provider (no re-globbing BIDS at
runtime) and makes the node trivially testable with a hand-built list.

### C. TemplateFlow handling

Adopt the PR's approach: `get_templates()` + `ls(...)` for candidate edges (no
download). A `templateflow_get`-style helper fetches only the path transforms.
The linear-inverse policy applies to TF edges too (TF `.h5` are nonlinear →
forward-only unless the reverse file is listed).

## Error handling

- No path between spaces → `ValueError` naming source, target, and reachable
  spaces.
- A nonlinear reverse that isn't on disk is never an edge, so BFS fails cleanly
  rather than selecting an inverse `nitransforms` can't realize.
- A path transform file that can't be loaded/fetched → error surfaced from the
  apply step.

## Testing

- **Graph functions** (pure, no I/O): synthetic `(from,to)` filename lists,
  including the **real ds008325 set** — computed `ACPC→T1w` bridge + aslprep
  `T1w→MNI152NLin6Asym` ⇒ a 2-hop chain `ACPC→T1w→MNI152NLin6Asym`. Assert:
  linear transforms yield reverse edges; nonlinear `.h5` do not; a missing link
  raises with the reachable-spaces message.
- **`ResolveApplyTransforms`**: tiny synthetic **affine** transforms + small
  label images applied via `nitransforms` (no ANTs, no network) — warp a label
  image through a known translation chain and assert voxels land where expected;
  assert `nearest` interpolation preserves integer labels.

## Out of scope (this spec)

- The volumetric `parcellate_scalar` path (Spec 2) — it consumes
  `ResolveApplyTransforms` to warp the atlas into the scalar's space, then
  parcellates.
- Producing the ACPC↔T1w bridge node — it already exists
  (`_register_acpc_to_t1w`); Spec 2 wires it as a `bridges` input where needed.
- Non-`image` transform modes (e.g. `mode-points`), and cohort/resolution
  disambiguation of TemplateFlow transforms beyond what `ls` returns.
