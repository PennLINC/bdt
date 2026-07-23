# BDT `dataset_description.json` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every run, BDT writes a BIDS-Derivative-compliant `dataset_description.json` at the output root, with a `DatasetLinks` dict that makes the `bids:<key>:...` `Sources` URIs in per-file sidecars resolve.

**Architecture:** A new pybids-free module `bdt/outputs/dataset_description.py` builds the description — a BDT `GeneratedBy` record prepended to entries aggregated (deduped) from the input derivative datasets, plus `DatasetLinks` mapping every `--datasets` key to an absolute path (`templateflow`→URL) and the raw `bids_dir` to a `raw` link. `run_spec` calls it once, up front; the CLI threads the positional `bids_dir` through. The stale xcp_d-copied `write_derivative_description` in `utils/bids.py` is removed.

**Tech Stack:** Python 3.12, pytest, micromamba env `bdtenv`. Standard library only (`json`, `os`, `pathlib`, `warnings`).

## Global Constraints

- Run all Python/pytest via `micromamba run -n bdtenv …` (this subproject uses `bdtenv`, not `lincapps`).
- `BIDS_VERSION = '1.10.0'` and default `Name = 'BDT derivatives'` are single module constants.
- Env-var names for container provenance are `BDT_DOCKER_TAG` / `BDT_SINGULARITY_URL`.
- `DatasetLinks` values are absolute filesystem paths, except `templateflow` → `https://github.com/templateflow/templateflow`.
- The module must not depend on pybids (consistent with the rest of `bdt/outputs/`).
- `write_dataset_description` must never raise on a missing/malformed input `dataset_description.json` — inheritance is best-effort.
- Version control is user-managed: do **not** run `git commit`. Each task ends with a checkpoint that runs the tests; staging/committing is left to the user.

---

### Task 1: `dataset_description.py` module + unit tests

**Files:**
- Create: `src/bdt/outputs/dataset_description.py`
- Modify: `src/bdt/outputs/__init__.py`
- Test: `test/engine/test_outputs.py` (append)

**Interfaces:**
- Produces:
  - `dataset_generated_by() -> dict` — `{'Name': 'BDT', 'Version': <ver>, 'CodeURL': <url>[, 'Container': {...}]}`.
  - `write_dataset_description(output_dir: str | Path, datasets: dict[str, str | Path], bids_dir: str | Path | None = None, name: str = 'BDT derivatives') -> Path` — writes `output_dir/dataset_description.json`, returns its `Path`.
  - Module constant `BIDS_VERSION = '1.10.0'`.

- [ ] **Step 1: Write the failing unit tests**

Append to `test/engine/test_outputs.py`. Add `import os` beside the existing `import json` at the top of the file, then append these tests at the end:

```python
from bdt import __version__
from bdt.outputs import dataset_generated_by, write_dataset_description


def _write_input_desc(root, generated_by):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'dataset_description.json').write_text(
        json.dumps({'Name': root.name, 'GeneratedBy': generated_by})
    )


def test_dataset_generated_by_basic(monkeypatch):
    monkeypatch.delenv('BDT_DOCKER_TAG', raising=False)
    monkeypatch.delenv('BDT_SINGULARITY_URL', raising=False)
    rec = dataset_generated_by()
    assert rec['Name'] == 'BDT'
    assert rec['Version'] == __version__
    assert rec['CodeURL'].endswith(f'{__version__}.tar.gz')
    assert 'Container' not in rec


def test_dataset_generated_by_docker_container(monkeypatch):
    monkeypatch.setenv('BDT_DOCKER_TAG', '1.2.3')
    monkeypatch.delenv('BDT_SINGULARITY_URL', raising=False)
    rec = dataset_generated_by()
    assert rec['Container'] == {'Type': 'docker', 'Tag': 'nipreps/bdt:1.2.3'}


def test_dataset_generated_by_singularity_container(monkeypatch):
    monkeypatch.delenv('BDT_DOCKER_TAG', raising=False)
    monkeypatch.setenv('BDT_SINGULARITY_URL', 'docker://nipreps/bdt:1.2.3')
    rec = dataset_generated_by()
    assert rec['Container'] == {'Type': 'singularity', 'URI': 'docker://nipreps/bdt:1.2.3'}


def test_write_dataset_description_aggregates_and_links(tmp_path, monkeypatch):
    monkeypatch.delenv('BDT_DOCKER_TAG', raising=False)
    monkeypatch.delenv('BDT_SINGULARITY_URL', raising=False)
    a = tmp_path / 'A'
    _write_input_desc(a, [{'Name': 'smriprep', 'Version': '0.1'}])
    b = tmp_path / 'B'  # exists but has no dataset_description.json
    b.mkdir()
    tf = tmp_path / 'tf'
    tf.mkdir()
    bids = tmp_path / 'rawbids'
    bids.mkdir()
    out = tmp_path / 'out'
    out.mkdir()

    dest = write_dataset_description(
        out,
        {'A': str(a), 'B': str(b), 'templateflow': str(tf)},
        bids_dir=str(bids),
    )
    assert dest == out / 'dataset_description.json'
    desc = json.loads(dest.read_text())
    assert desc['DatasetType'] == 'derivative'
    assert desc['BIDSVersion'] == '1.10.0'
    assert desc['Name'] == 'BDT derivatives'
    # BDT record first; inherited smriprep entry present
    assert desc['GeneratedBy'][0]['Name'] == 'BDT'
    assert {'Name': 'smriprep', 'Version': '0.1'} in desc['GeneratedBy']
    # DatasetLinks: absolute paths, templateflow URL, raw link
    assert desc['DatasetLinks']['A'] == os.path.abspath(str(a))
    assert desc['DatasetLinks']['B'] == os.path.abspath(str(b))
    assert desc['DatasetLinks']['templateflow'] == (
        'https://github.com/templateflow/templateflow'
    )
    assert desc['DatasetLinks']['raw'] == os.path.abspath(str(bids))


def test_write_dataset_description_dedups_generated_by(tmp_path):
    entry = {'Name': 'smriprep', 'Version': '0.1'}
    a = tmp_path / 'A'
    _write_input_desc(a, [entry])
    b = tmp_path / 'B'
    _write_input_desc(b, [entry])  # same entry surfaced by two inputs
    out = tmp_path / 'out'
    out.mkdir()
    gb = json.loads(
        write_dataset_description(out, {'A': str(a), 'B': str(b)}).read_text()
    )['GeneratedBy']
    assert gb.count(entry) == 1
    assert gb[0]['Name'] == 'BDT'


def test_write_dataset_description_raw_key_collision(tmp_path):
    # a --datasets key literally named 'raw' wins; bids_dir link is skipped
    raw_ds = tmp_path / 'rawds'
    raw_ds.mkdir()
    bids = tmp_path / 'rawbids'
    bids.mkdir()
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.warns(UserWarning):
        dest = write_dataset_description(out, {'raw': str(raw_ds)}, bids_dir=str(bids))
    links = json.loads(dest.read_text())['DatasetLinks']
    assert links['raw'] == os.path.abspath(str(raw_ds))


def test_write_dataset_description_ignores_malformed_input(tmp_path):
    a = tmp_path / 'A'
    a.mkdir()
    (a / 'dataset_description.json').write_text('{ not valid json')
    out = tmp_path / 'out'
    out.mkdir()
    gb = json.loads(write_dataset_description(out, {'A': str(a)}).read_text())['GeneratedBy']
    assert len(gb) == 1
    assert gb[0]['Name'] == 'BDT'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_outputs.py -k dataset -v`
Expected: FAIL at import (`cannot import name 'dataset_generated_by' from 'bdt.outputs'`).

- [ ] **Step 3: Create the module**

Create `src/bdt/outputs/dataset_description.py` (prepend the standard NiPreps license header used by the sibling files in `src/bdt/outputs/`, then):

```python
"""Dataset-level BIDS-derivative description (``dataset_description.json``).

Writes the derivatives-root ``dataset_description.json`` with a ``DatasetLinks``
dict so the ``bids:<dataset_key>:...`` ``Sources`` URIs emitted in per-file
sidecars resolve, and a ``GeneratedBy`` chain aggregated from the input
derivative datasets with BDT's own record prepended.  Kept free of a pybids
dependency, like the rest of :mod:`bdt.outputs`.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

BIDS_VERSION = '1.10.0'
_TEMPLATEFLOW_URL = 'https://github.com/templateflow/templateflow'


def dataset_generated_by() -> dict:
    """The dataset-level BDT provenance record for ``GeneratedBy[0]``.

    Distinct in shape from :func:`bdt.outputs.provenance.generated_by`, which is
    the per-*node* sidecar record.  Adds a ``Container`` sub-dict from the
    ``BDT_DOCKER_TAG`` / ``BDT_SINGULARITY_URL`` environment variables when set.
    """
    from bdt import __version__

    record: dict = {
        'Name': 'BDT',
        'Version': __version__,
        'CodeURL': f'https://github.com/nipreps/bdt/archive/{__version__}.tar.gz',
    }
    docker_tag = os.environ.get('BDT_DOCKER_TAG')
    singularity_url = os.environ.get('BDT_SINGULARITY_URL')
    if docker_tag:
        record['Container'] = {'Type': 'docker', 'Tag': f'nipreps/bdt:{docker_tag}'}
    elif singularity_url:
        record['Container'] = {'Type': 'singularity', 'URI': singularity_url}
    return record


def _inherited_generated_by(datasets: dict[str, str | Path]) -> list[dict]:
    """``GeneratedBy`` entries aggregated from input derivatives (order-preserving dedup)."""
    collected: list[dict] = []
    for key, root in datasets.items():
        if key == 'templateflow':
            continue
        desc_path = Path(root) / 'dataset_description.json'
        try:
            desc = json.loads(desc_path.read_text())
        except (OSError, ValueError):
            continue  # missing or malformed input description -> skip (best effort)
        entries = desc.get('GeneratedBy')
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if entry not in collected:
                collected.append(entry)
    return collected


def write_dataset_description(
    output_dir: str | Path,
    datasets: dict[str, str | Path],
    bids_dir: str | Path | None = None,
    name: str = 'BDT derivatives',
) -> Path:
    """Write a BIDS-derivative ``dataset_description.json`` at ``output_dir``.

    ``datasets`` is the ``{key: root}`` mapping from ``--datasets``; each key
    becomes a ``DatasetLinks`` entry (absolute path, or the canonical URL for
    ``templateflow``).  When ``bids_dir`` is given it is linked as ``raw`` unless
    a ``--datasets`` key already claims that name.  Overwrites any existing file.
    """
    dataset_links: dict[str, str] = {}
    for key, root in datasets.items():
        if key == 'templateflow':
            dataset_links[key] = _TEMPLATEFLOW_URL
        else:
            dataset_links[key] = os.path.abspath(str(root))
    if bids_dir is not None:
        if 'raw' in dataset_links:
            warnings.warn(
                "A --datasets key named 'raw' shadows the raw bids_dir link; "
                'skipping the bids_dir link.',
                stacklevel=2,
            )
        else:
            dataset_links['raw'] = os.path.abspath(str(bids_dir))

    generated_by = [dataset_generated_by(), *_inherited_generated_by(datasets)]

    description = {
        'Name': name,
        'BIDSVersion': BIDS_VERSION,
        'DatasetType': 'derivative',
        'HowToAcknowledge': 'Include the generated boilerplate in the methods section.',
        'GeneratedBy': generated_by,
        'DatasetLinks': dataset_links,
    }

    dest = Path(output_dir) / 'dataset_description.json'
    dest.write_text(json.dumps(description, indent=2))
    return dest
```

- [ ] **Step 4: Export from the package**

Edit `src/bdt/outputs/__init__.py`. After the existing `from bdt.outputs.sink import ...` line, add:

```python
from bdt.outputs.dataset_description import dataset_generated_by, write_dataset_description
```

Then add both names to the `__all__` tuple (keep it alphabetized as it currently is):

```python
    'dataset_generated_by',
    'write_dataset_description',
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_outputs.py -k dataset -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Checkpoint**

Run the full outputs test module and lint the new file:
`micromamba run -n bdtenv python -m pytest test/engine/test_outputs.py -v`
`micromamba run -n bdtenv ruff check src/bdt/outputs/dataset_description.py`
Expected: all pass, no lint errors. (Version control is user-managed — stage/commit per your workflow, or skip.)

---

### Task 2: Wire into `run_spec` and the CLI + integration test

**Files:**
- Modify: `src/bdt/engine/pipeline.py` (add `bids_dir` param to `run_spec`; call `write_dataset_description`)
- Modify: `src/bdt/cli/run.py` (pass `bids_dir=opts.bids_dir`)
- Modify: `test/engine/test_cli_run.py` (update two `fake_run_spec` stub signatures)
- Test: `test/engine/test_pipeline.py` (append integration test)

**Interfaces:**
- Consumes: `write_dataset_description(output_dir, datasets, bids_dir=...)` from Task 1.
- Produces: `run_spec(..., bids_dir: str | Path | None = None)` — new trailing keyword parameter; writes `output_dir/dataset_description.json` before the per-subject loop.

- [ ] **Step 1: Write the failing integration test**

Append to `test/engine/test_pipeline.py`. It relies on `parse_spec`, `DictDataProvider`, `SelectionError`, `pytest`, `json`, and `Path`, all already imported at the top of that file except `json` — add `import json` to the top-of-file imports if it is not already present, then append:

```python
def test_run_spec_writes_dataset_description(tmp_path):
    """run_spec writes a derivative dataset_description.json before any workflow runs."""
    from bdt.engine.pipeline import run_spec

    ds = tmp_path / 'ds'
    ds.mkdir()
    bids = tmp_path / 'rawbids'
    bids.mkdir()
    out = tmp_path / 'out'

    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'sel',
                    'action': 'select_data',
                    'dataset': 'ds',
                    'filters': {'suffix': 'bold'},
                }
            ]
        }
    )
    provider = DictDataProvider({'ds': []})  # matches nothing -> SelectionError

    # The description is written up front; resolution then fails on the empty match.
    with pytest.raises(SelectionError):
        run_spec(
            spec,
            {'ds': str(ds)},
            out,
            tmp_path / 'work',
            subjects=['01'],
            provider=provider,
            validate=False,
            bids_dir=str(bids),
        )

    desc = json.loads((out / 'dataset_description.json').read_text())
    assert desc['DatasetType'] == 'derivative'
    assert desc['DatasetLinks']['ds'] == os.path.abspath(str(ds))
    assert desc['DatasetLinks']['raw'] == os.path.abspath(str(bids))
    assert desc['GeneratedBy'][0]['Name'] == 'BDT'
```

Add `import os` to the top-of-file imports of `test/engine/test_pipeline.py` if it is not already present.

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_pipeline.py::test_run_spec_writes_dataset_description -v`
Expected: FAIL — `run_spec` raises `TypeError` for the unexpected `bids_dir` keyword (param not added yet).

- [ ] **Step 3: Add the `bids_dir` parameter and write call in `run_spec`**

In `src/bdt/engine/pipeline.py`, add the new parameter to the `run_spec` signature. Change:

```python
    plugin_args: dict | None = None,
    validate: bool = True,
) -> list[RunResult]:
```

to:

```python
    plugin_args: dict | None = None,
    validate: bool = True,
    bids_dir: str | Path | None = None,
) -> list[RunResult]:
```

Then, immediately after these existing lines:

```python
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
```

insert:

```python

    from bdt.outputs.dataset_description import write_dataset_description

    # Written once, up front, so the derivatives root is a valid BIDS-Derivative
    # dataset (and the sidecar bids:<key>:... Sources URIs resolve) even if a
    # later workflow crashes.
    write_dataset_description(output_dir, datasets, bids_dir=bids_dir)
```

- [ ] **Step 4: Pass `bids_dir` from the CLI**

In `src/bdt/cli/run.py`, in the `run_spec(...)` call inside `main`, add the `bids_dir` argument. Change:

```python
        results = run_spec(
            opts.spec,
            datasets,
            opts.output_dir,
            work_dir,
            subjects=opts.participant_label or None,
            plugin=plugin,
            plugin_args=plugin_args,
        )
```

to:

```python
        results = run_spec(
            opts.spec,
            datasets,
            opts.output_dir,
            work_dir,
            subjects=opts.participant_label or None,
            plugin=plugin,
            plugin_args=plugin_args,
            bids_dir=opts.bids_dir,
        )
```

- [ ] **Step 5: Update the CLI test stubs to accept `bids_dir`**

In `test/engine/test_cli_run.py`, both `fake_run_spec` definitions (around lines 46 and 103) must accept the new keyword or they raise `TypeError`. Change each occurrence of:

```python
    def fake_run_spec(
        spec_arg, datasets, output_dir, work_dir, subjects=None, plugin='Linear', plugin_args=None
    ):
```

to:

```python
    def fake_run_spec(
        spec_arg,
        datasets,
        output_dir,
        work_dir,
        subjects=None,
        plugin='Linear',
        plugin_args=None,
        bids_dir=None,
    ):
```

In the first stub (the one that populates `captured` in `test_cli_maps_args_to_run_spec`), also add `bids_dir=str(bids_dir)` to the `captured.update(...)` call so the passthrough is asserted:

```python
        captured.update(
            spec=spec_arg,
            datasets=datasets,
            output_dir=str(output_dir),
            work_dir=str(work_dir),
            subjects=subjects,
            plugin=plugin,
            bids_dir=str(bids_dir),
        )
```

Then add this assertion after the existing `assert captured['plugin'] == 'Linear'` line in `test_cli_maps_args_to_run_spec`:

```python
    assert captured['bids_dir'] == str(bids)  # positional raw dataset threaded through
```

- [ ] **Step 6: Run the affected tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_pipeline.py::test_run_spec_writes_dataset_description test/engine/test_cli_run.py -v`
Expected: PASS (the new integration test and both CLI tests).

- [ ] **Step 7: Checkpoint**

Run the pipeline and CLI test modules together:
`micromamba run -n bdtenv python -m pytest test/engine/test_pipeline.py test/engine/test_cli_run.py -v`
Expected: all pass (real-data tests remain skipped). (Version control is user-managed — stage/commit per your workflow, or skip.)

---

### Task 3: Remove stale `write_derivative_description`

**Files:**
- Modify: `src/bdt/utils/bids.py` (delete `write_derivative_description`)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (pure deletion of dead code).

- [ ] **Step 1: Confirm it is dead code**

Run: `micromamba run -n bdtenv grep -rn 'write_derivative_description' src test`
Expected: exactly one hit — the `def write_derivative_description(...)` line in `src/bdt/utils/bids.py`. (A reference in `docs/dev/2026-06-01-bdt-implementation-plan.md` is a historical doc and needs no change.) If any `src/` or `test/` file *imports or calls* it, stop and reassess — this task assumes none do.

- [ ] **Step 2: Delete the function**

In `src/bdt/utils/bids.py`, remove the entire `write_derivative_description` function — from the `def write_derivative_description(input_dir, output_dir, dataset_links=None):` line through the end of its body (the line `out_desc.write_text(json.dumps(desc, indent=4))` inside the final `else:` block), i.e. the whole block ending just before `def validate_input_dir(...)`. Leave `validate_input_dir` and everything else intact.

- [ ] **Step 3: Verify nothing broke**

Run: `micromamba run -n bdtenv grep -rn 'write_derivative_description' src test`
Expected: no output (function gone, no references).

Run: `micromamba run -n bdtenv ruff check src/bdt/utils/bids.py`
Expected: no errors (if `packaging.version.Version` was imported only inside the deleted function, it was a local import there — confirm no now-unused module-level import remains; the function imported `json`/`os`/`Version` locally, so there is nothing to clean up at module scope).

- [ ] **Step 4: Checkpoint**

Run the full test suite:
`micromamba run -n bdtenv python -m pytest -q`
Expected: pass (real-data tests skipped). (Version control is user-managed — stage/commit per your workflow, or skip.)

---

## Self-Review

**Spec coverage:**
- New module `dataset_description.py` → Task 1. ✓
- `dataset_generated_by` + Container env vars → Task 1 (Steps 1, 3). ✓
- Aggregate + dedup + prepend GeneratedBy → Task 1 (`_inherited_generated_by`, tests). ✓
- DatasetLinks absolute paths + templateflow URL + `raw` link + `raw`-key collision → Task 1 (Steps 1, 3). ✓
- Best-effort on missing/malformed input description → Task 1 (`test_write_dataset_description_ignores_malformed_input`). ✓
- Body fields (`Name`/`BIDSVersion`/`DatasetType`/`HowToAcknowledge`) → Task 1. ✓
- Wire into `run_spec`, write once before loop, `bids_dir` default `None` → Task 2. ✓
- CLI passes `opts.bids_dir` → Task 2. ✓
- Remove stale `write_derivative_description` → Task 3. ✓
- Tests (unit in `test_outputs.py`, integration in `test_pipeline.py`) → Tasks 1 & 2. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `write_dataset_description(output_dir, datasets, bids_dir=None, name='BDT derivatives') -> Path` and `dataset_generated_by() -> dict` are used identically across Tasks 1 and 2; `run_spec`'s new `bids_dir` keyword matches the CLI call and both test stubs. ✓
