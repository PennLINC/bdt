# BIDS-derivative `dataset_description.json` for BDT output

**Date:** 2026-07-23
**Status:** Approved design, ready for implementation plan

## Problem

BDT writes derivative files with JSON provenance sidecars whose `Sources`
entries are `bids:<dataset_key>:<relpath>` URIs, where `<dataset_key>` is the
`--datasets KEY=PATH` key (see `bdt/outputs/provenance.py:bids_uri` and
`bdt/outputs/plan.py:_sources`). Per BIDS, those URIs only resolve if the output
dataset's `dataset_description.json` carries a `DatasetLinks` dict mapping each
key to its location. BDT currently writes **no** `dataset_description.json` at
the output root, so the emitted `Sources` URIs are dangling and the output tree
is not a valid BIDS-Derivative dataset.

A `write_derivative_description` function exists in `bdt/utils/bids.py` but is
dead, buggy stale code copied from xcp_d (wrong dataset `Name`, an
`enumerate(dataset_links)` bug that iterates dict *keys* as if they were paths,
and it is never called from the pipeline — only referenced in an old planning
doc).

## Goal

On every run, BDT writes a spec-compliant `dataset_description.json` at the
output root with:

- `DatasetType: "derivative"`, `BIDSVersion`, `Name`, `HowToAcknowledge`.
- A `GeneratedBy` chain: BDT's own record prepended to the aggregated
  `GeneratedBy` entries inherited from the input derivative datasets.
- A `DatasetLinks` dict mapping every input dataset key to an absolute path
  (with `templateflow` mapped to its canonical URL), so the `Sources` URIs in
  the per-file sidecars resolve.

## Design decisions (settled)

- **DatasetLinks value encoding:** absolute filesystem paths; `templateflow`
  (if present as a key) maps to `https://github.com/templateflow/templateflow`.
- **Which datasets are linked:** all `--datasets` entries, **plus** the raw
  positional `bids_dir` as a `"raw"` link.
- **GeneratedBy source:** aggregate across all input derivative datasets that
  have a `dataset_description.json` with a `GeneratedBy` list, dedup, then
  prepend BDT's own record.
- **Stale code:** remove `write_derivative_description` from `utils/bids.py`.
- **Defaults chosen without asking (single, easily-changed constants):**
  `BIDS_VERSION = "1.10.0"`, default `Name = "BDT derivatives"`; env-var names
  normalized to `BDT_DOCKER_TAG` / `BDT_SINGULARITY_URL`.

## Architecture

### 1. New module: `src/bdt/outputs/dataset_description.py`

Kept free of a pybids dependency, consistent with the rest of `bdt/outputs/`.

Module constant:

```python
BIDS_VERSION = '1.10.0'
```

**`dataset_generated_by() -> dict`**

The dataset-level BDT provenance record (distinct in shape from
`provenance.generated_by`, which is the per-*node* sidecar record):

```python
{
    'Name': 'BDT',
    'Version': __version__,
    'CodeURL': f'https://github.com/nipreps/bdt/archive/{__version__}.tar.gz',
}
```

If `BDT_DOCKER_TAG` is set in the environment, add
`'Container': {'Type': 'docker', 'Tag': f'nipreps/bdt:{tag}'}`; else if
`BDT_SINGULARITY_URL` is set, add `'Container': {'Type': 'singularity',
'URI': url}`.

**`write_dataset_description(output_dir, datasets, bids_dir=None, name='BDT derivatives') -> Path`**

- `output_dir: str | Path` — derivatives root (already created by the caller).
- `datasets: dict[str, str | Path]` — the `--datasets` `{key: root}` mapping.
- `bids_dir: str | Path | None` — the raw positional dataset, or `None`.
- `name: str` — dataset `Name`.

Steps:

1. **Aggregate GeneratedBy.** For each `root` in `datasets.values()` (skip the
   `templateflow` key), attempt to read `root/dataset_description.json`. If it
   parses and has a list-valued `GeneratedBy`, extend a collected list with its
   entries. Dedup order-preserving (compare full dict equality). Best-effort:
   a missing or malformed input description is silently skipped — never raises.
   Prepend `dataset_generated_by()` at the front of the collected list.
2. **DatasetLinks.** Start empty. For each `key, path` in `datasets`:
   `templateflow` -> the canonical URL; otherwise `os.path.abspath(path)`.
   If `bids_dir` is given, add `'raw' -> os.path.abspath(bids_dir)`, but only if
   `'raw'` is not already a `--datasets` key — if it is, skip the raw link and
   emit a warning (real `Sources` keys win).
3. **Assemble** the description dict:
   ```python
   {
       'Name': name,
       'BIDSVersion': BIDS_VERSION,
       'DatasetType': 'derivative',
       'HowToAcknowledge': 'Include the generated boilerplate in the methods section.',
       'GeneratedBy': generated_by,
       'DatasetLinks': dataset_links,
   }
   ```
4. **Write** `output_dir/dataset_description.json` with `json.dumps(..., indent=2)`,
   **overwriting** any existing file (deterministic — `run_spec` always knows the
   full input set on each run). Return the written `Path`.

Export both `dataset_generated_by` and `write_dataset_description` from
`bdt/outputs/__init__.py`.

### 2. Wire into `run_spec` (`src/bdt/engine/pipeline.py`)

- Add parameter `bids_dir: str | Path | None = None`.
- Immediately after `output_dir.mkdir(parents=True, exist_ok=True)` (~line 211),
  before the subject/combo loop, call:
  ```python
  from bdt.outputs.dataset_description import write_dataset_description
  write_dataset_description(output_dir, datasets, bids_dir=bids_dir)
  ```
  Writing once, up front, means the description exists even if a later workflow
  crashes. `bids_dir` defaults to `None`, so the existing direct `run_spec` test
  calls are unaffected (they simply get no `raw` link).

### 3. CLI (`src/bdt/cli/run.py`)

Pass the already-parsed positional through:
`run_spec(..., bids_dir=opts.bids_dir)`.

### 4. Remove stale code

Delete `write_derivative_description` from `src/bdt/utils/bids.py`. Confirmed no
importer in `src/` or `test/`; the only reference is in a historical planning
doc, which needs no change.

## Data flow

```
--datasets k=path ... + positional bids_dir
        │
        ▼
run_spec(datasets, output_dir, bids_dir=...)
        │  (once, before the per-subject loop)
        ▼
write_dataset_description
   ├─ dataset_generated_by()  ──┐
   ├─ read each input root's    │
   │   dataset_description.json ├─► GeneratedBy = [BDT, *inherited(deduped)]
   │   GeneratedBy (best effort)┘
   └─ DatasetLinks = {key: abspath(path), templateflow: URL, raw: abspath(bids_dir)}
        │
        ▼
output_dir/dataset_description.json   (resolves the bids:<key>:... Sources URIs)
```

## Testing (`test/engine/test_outputs.py`)

**Unit — `write_dataset_description`:**
- Build a temp tree: dataset `A` *with* a `dataset_description.json` containing a
  `GeneratedBy` list; dataset `B` *without* one; a `templateflow` key; and a
  separate `bids_dir`.
- Assert the written JSON has:
  - `DatasetType == 'derivative'`, `BIDSVersion == BIDS_VERSION`.
  - `GeneratedBy[0]['Name'] == 'BDT'` with the right `Version`/`CodeURL`; A's
    inherited entries following it; no duplicates.
  - `DatasetLinks`: `A`/`B` mapped to their abspaths, `templateflow` -> URL,
    `raw` -> abspath(bids_dir).
- Edge: a `--datasets` key literally named `raw` suppresses the bids_dir link
  (and no crash); a missing/malformed input `dataset_description.json` is
  skipped without raising.

**Integration — `run_spec`:**
- Extend an existing pipeline test to assert `output_dir/dataset_description.json`
  exists and parses as valid JSON with `DatasetType == 'derivative'`.

## Out of scope

- `.bidsignore`, `atlas-<label>_description.json`, and other dataset-level
  sidecars (tracked separately).
- Merge-on-rerun semantics — the file is regenerated wholesale each run.
- Threading `bids_dir` anywhere beyond `run_spec`/the CLI.
