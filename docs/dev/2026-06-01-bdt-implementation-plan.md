# BDT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Rev 2026-06-02 (per `2026-06-01-bdt-plan-review.md`):** The transform engine is now **geometry-polymorphic** (see design §7). Concrete changes from the first draft: Task 6 exposes two typed queries and widens the xfm regex; Task 7 fixes the `float` trait bug; Tasks 11–13 are reworked (GIFTI routes through `wb_command`; streamlines use endpoint→parcel connectivity on trxrs bindings instead of `density_map`; `AtlasOuterProduct`/`AtlasIntersect`/`AtlasUnion` are made distinct and overlap-tolerant). New tasks added: **Task 6b** Rust `CommandLine` interfaces, **Task 13b** `init_load_atlases_wf`, **Task 17b** odx diffusion-model interfaces + workflow, and **Task 22** the Docker Rust build stage. Streamlines adopt **Strategy B** (warp tracts→atlas via `trxrs`).

> **Rev 2026-06-02b (verified against the Rust repos):** The Rust binary/CLI surfaces and Python packages were checked against the cloned `trx-rs`, `gifti-rs`, `odx-rs` sources, and the previously-"intended" wrapper shapes were corrected in place. Material fixes: Python dep `trx-python`→**`trxrs`** and `_load_endpoints` now uses `trxrs.load().positions()/.offsets()` (Tasks 1, 12); all Rust subcommands take **positional** `input output` with the verified flags (`--force`/`--overwrite`, `odx --mode/--transform-inverse`) and a **subcommand-pattern correctness fix** so the subcommand isn't dropped at runtime (Task 6b); `odx` operates on **`.odx` containers** with an ingest step and reads via the PyO3 `compact_to_ijk()`/`sh()` API, not nibabel (Task 17b); Dockerfile uses the real repo URLs + correct `cargo install` package/`--bin` names (Task 22). See the "Known gaps" note at the end for what remains unverified.

> **Rev 2026-06-02c (BIDS/BEP conformance):** Output naming and the spec parser were aligned to merged BIDS and the seven BEPs (design §14). Material changes folded in below: parcellated time series use the `timeseries` suffix not `bold` (BEP 012); diffusion sources/outputs use `suffix: dwimap` + `model-`/`param-` not `fod` (BEP 016); tractogram sources use `suffix: tractogram` + TRX-only (BEP 046); connectivity outputs are `meas-<label>_relmat.dense.tsv` with REQUIRED `_relmat.json` + `_nodeindices.*` sidecars (BEP 017); BAT emits the REQUIRED `atlas-<label>_description.json`, uses canonical `seg`-before-`desc` order, a `tpl-` (not `space-`) output entity, and an optional `nodelabels.tsv` provenance file (merged BIDS atlas spec + BEP 017 §4.3). Affected: Task 3 (spec entities/suffixes), Task 12 (relmat + sidecars), Tasks 19–21 (BAT outputs), Tasks 15/17 (output suffixes), plus the **BIDS/BEP conformance checklist** at the end.

**Goal:** Implement BDT (BIDS Derivatives Transformer) and BAT (BIDS Atlas Transformer) as two CLI entry points in the `bdt` package, each backed by a nipype workflow that applies atlases to BIDS derivative data.

**Architecture:** Both CLIs share `config.py`, `utils/`, and `interfaces/`; BDT workflows live under `workflows/bdt/` and BAT under `workflows/bat/`. Format/geometry dispatch (grid NIfTI / CIFTI dense / GIFTI surface / TRX-TCK streamlines / SH-ODF-fixel diffusion model) happens at workflow-build time by inspecting file extension. **Grid** atlases are warped to data space via a `networkx` transform graph built from `_xfm.*` files (ANTs `ApplyTransforms`, pull semantics). **Point** data (surfaces, streamlines) is warped via the Rust binaries `giftirs`/`trxrs` using the *opposite* graph direction (`chain_for_point_warp`). **Diffusion models** are resampled + SH-reoriented via `odx`.

**Tech Stack:** Python 3.12, nipype, niworkflows (LiterateWorkflow), pybids, nibabel, nilearn, networkx, PyYAML, ANTs (via nipype), Connectome Workbench (wb_command), MRtrix (optional, `tck2connectome` reference); Rust binaries on `PATH`: **`trxrs`** (streamline transform/convert + read-only Python bindings for endpoint lookup), **`giftirs`** (surface vertex transform, CLI-only), **`odx`** (SH/ODF/fixel transform + PyO3 bindings). `dipy` is retained only for `bundle_stats` length/count, **not** for connectivity.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/bdt/cli/bdt_run.py` | Rename from `run.py` | BDT entry point |
| `src/bdt/cli/bdt_parser.py` | Rename from `parser.py` | BDT CLI parser |
| `src/bdt/cli/bdt_workflow.py` | Rename from `workflow.py` | BDT workflow builder |
| `src/bdt/cli/bdt_version.py` | Rename from `version.py` | BDT version checks |
| `src/bdt/cli/bat_run.py` | Create | BAT entry point |
| `src/bdt/cli/bat_parser.py` | Create | BAT CLI parser |
| `src/bdt/cli/bat_workflow.py` | Create | BAT workflow builder |
| `src/bdt/config.py` | Modify | Add `bdt_workflow`, `bat_workflow` sections; `execution.spec` |
| `src/bdt/utils/spec.py` | Create | Spec YAML parser |
| `src/bdt/utils/atlas.py` | Create | `collect_atlases` with entity-filter dicts |
| `src/bdt/utils/transforms.py` | Create | `build_transform_graph`, `chain_for_image_resample`, `chain_for_point_warp` |
| `src/bdt/interfaces/ants.py` | Create | `ApplyTransforms` wrapper with bdt defaults |
| `src/bdt/interfaces/nilearn.py` | Create | `IndexImage`, `NiftiLabelsMasker` |
| `src/bdt/interfaces/censoring.py` | Create | `Censor` |
| `src/bdt/interfaces/plotting.py` | Create | `PlotCiftiParcellation` |
| `src/bdt/interfaces/rust.py` | Create | `TrxTransform`, `TrxConvert`, `GiftiTransform`, `OdxTransform` (CommandLine wrappers) |
| `src/bdt/interfaces/gifti.py` | Create | `SurfaceVolumeParcellate` (vertex-sample a volume atlas) |
| `src/bdt/interfaces/tractography.py` | Create | `StreamlineConnectivity` (endpoint→parcel via trxrs bindings), `BundleStats` |
| `src/bdt/interfaces/odx.py` | Create | `FixelParcellate`, `OdfParcellate` |
| `src/bdt/interfaces/atlas.py` | Create | `AtlasIntersect`, `AtlasUnion`, `AtlasOuterProduct` (distinct semantics) |
| `src/bdt/workflows/bdt/__init__.py` | Create | BDT workflows package |
| `src/bdt/workflows/bdt/base.py` | Create (move from `workflows/base.py`) | `init_bdt_wf`, `init_single_subject_wf` |
| `src/bdt/workflows/bdt/timeseries.py` | Create | `init_timeseries_run_wf` |
| `src/bdt/workflows/bdt/scalar.py` | Create | `init_scalar_run_wf` |
| `src/bdt/workflows/bdt/streamlines.py` | Create | `init_streamlines_run_wf` (Strategy B: convert→warp→endpoint connectivity) |
| `src/bdt/workflows/bdt/diffusion.py` | Create | `init_diffusion_model_run_wf` (odx transform + fixel/odf parcellation) |
| `src/bdt/workflows/parcellation.py` | Modify | Add `init_load_atlases_wf`, `init_label_resample_wf`, `init_surface_volume_parcellate_wf`, `init_parcellate_{fixel,odf}_wf` |
| `Dockerfile` | Modify | Add Rust build stage so `trxrs`/`giftirs`/`odx` are on `PATH` |
| `src/bdt/workflows/bat/__init__.py` | Create | BAT workflows package |
| `src/bdt/workflows/bat/base.py` | Create | `init_bat_wf`, `init_bat_dataset_wf` |
| `src/bdt/workflows/bat/algebra.py` | Create | `init_intersect_wf`, `init_union_wf`, `init_outer_product_wf` |
| `src/bdt/tests/__init__.py` | Create | Make `bdt.tests` importable |
| `src/bdt/tests/tests.py` | Create (copy from `test/tests.py`) | `mock_config` fixture |
| `pyproject.toml` | Modify | Add `bat` entry point; add `networkx`, `nibabel`, `dipy` deps |
| `test/test_spec.py` | Create | Unit tests for spec parser |
| `test/test_transforms.py` | Create | Unit tests for transform graph |
| `test/test_atlas.py` | Create | Unit tests for `collect_atlases` |
| `test/test_interfaces.py` | Create | Unit tests for lightweight interfaces |
| `test/test_bdt_base.py` | Modify | Workflow build tests |
| `test/test_bat_workflows.py` | Create | BAT workflow build tests |

---

## Phase 1 — Foundation

### Task 1: Rename CLI files and add `bat` entry point

**Files:**
- Rename: `src/bdt/cli/run.py` → `src/bdt/cli/bdt_run.py`
- Rename: `src/bdt/cli/parser.py` → `src/bdt/cli/bdt_parser.py`
- Rename: `src/bdt/cli/workflow.py` → `src/bdt/cli/bdt_workflow.py`
- Rename: `src/bdt/cli/version.py` → `src/bdt/cli/bdt_version.py`
- Modify: `pyproject.toml`
- Modify: `test/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_cli.py — replace existing imports at the top
from bdt.cli import bdt_run         # was: from bdt.cli import run
from bdt.cli.bdt_parser import parse_args
from bdt.cli.bdt_workflow import build_boilerplate, build_workflow
```

Run: `pytest test/test_cli.py -v`
Expected: ImportError — `bdt.cli.bdt_run` does not exist

- [ ] **Step 2: Rename the files**

```bash
cd src/bdt/cli
mv run.py bdt_run.py
mv parser.py bdt_parser.py
mv workflow.py bdt_workflow.py
mv version.py bdt_version.py
```

Inside `bdt_parser.py`, update the import:
```python
# was: from bdt.cli.version import check_latest, is_flagged
from bdt.cli.bdt_version import check_latest, is_flagged
```

Inside `bdt_workflow.py`, update the import:
```python
# was: from bdt.cli.parser import parse_args
from bdt.cli.bdt_parser import parse_args
```

Inside `bdt_run.py`, update the imports:
```python
# was:
from bdt.cli.parser import parse_args
from bdt.cli.workflow import build_workflow
# becomes:
from bdt.cli.bdt_parser import parse_args
from bdt.cli.bdt_workflow import build_workflow
```

- [ ] **Step 3: Update pyproject.toml entry points**

```toml
[project.scripts]
bdt = "bdt.cli.bdt_run:main"
bat = "bdt.cli.bat_run:main"
```

Also add missing dependencies:
```toml
dependencies = [
  "acres",
  "dipy",                 # bundle_stats only (length/count), not connectivity
  "fmriprep @ git+https://github.com/nipreps/fmriprep.git@master",
  "networkx",
  "nibabel",
  "nipype >= 1.8.5",
  "nireports @ git+https://github.com/nipreps/nireports.git@main",
  "nitransforms >= 24.0.2",
  "niworkflows @ git+https://github.com/nipreps/niworkflows.git@master",
  "odx",                  # odx-rs PyO3 bindings (diffusion-model parcellation reads .odx)
  "pybids >= 0.15.6",
  "pyyaml",
  "sdcflows @ git+https://github.com/nipreps/sdcflows.git@main",
  "smriprep @ git+https://github.com/nipreps/smriprep.git@master",
  "trxrs",                # trx-rs read-only bindings: load().positions()/.offsets() for endpoint lookup
  "typer",
]
```

> **Binary dependencies (not Python deps):** `trxrs`, `giftirs`, and `odx` CLI binaries must be on `PATH`. **Verified package/binary names** (against the cloned repos): the trx-rs Python distribution is **`trxrs`** (module `trxrs._core`; import `trxrs`), *not* `trx-python`; the odx Python distribution is **`odx`** (module `odx._odx`). The Python packages cover *read-only* reads (`trxrs.load`, `odx.load`/`sh`/`peaks_from_sh`/`from_*`), but `trxrs transform`/`convert`, `giftirs transform`, and `odx transform` are **CLI-only** — `odx` exposes no Python `transform` function. These binaries are installed by the Docker Rust build stage in **Task 22**; local/dev runs of the streamline, surface-by-volume, and diffusion-model paths require them too. (If `trxrs`/`odx` are not yet on PyPI, install from the repos: `tee-ar-ex/trx-rs` `python/`, `PennLINC/odx-rs` `python/`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cli.py -v`
Expected: PASS (imports resolve)

- [ ] **Step 5: Commit**

```bash
git add src/bdt/cli/ pyproject.toml test/test_cli.py
git commit -m "refactor: rename cli files to bdt_* prefix; add bat entry point"
```

---

### Task 2: Update config.py

Add `spec` to `execution`, split `workflow` into shared `workflow` + `bdt_workflow`, add `bat_workflow`.

**Files:**
- Modify: `src/bdt/config.py`
- Create: `test/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_config.py
def test_bdt_workflow_section():
    from bdt import config
    assert hasattr(config, 'bdt_workflow')
    assert hasattr(config.bdt_workflow, 'dummy_scans')
    assert hasattr(config.bdt_workflow, 'min_coverage')
    assert hasattr(config.bdt_workflow, 'correlation_lengths')
    assert hasattr(config.bdt_workflow, 'output_correlations')

def test_bat_workflow_section():
    from bdt import config
    assert hasattr(config, 'bat_workflow')
    assert hasattr(config.bat_workflow, 'output_space')
    assert hasattr(config.bat_workflow, 'interpolation')

def test_execution_spec():
    from bdt import config
    assert hasattr(config.execution, 'spec')
    assert config.execution.spec is None

def test_shared_workflow_no_dummy_scans():
    from bdt import config
    assert not hasattr(config.workflow, 'dummy_scans')
    assert not hasattr(config.workflow, 'cifti_output')
```

Run: `pytest test/test_config.py -v`
Expected: FAIL — `bdt_workflow` section does not exist

- [ ] **Step 2: Edit `src/bdt/config.py`**

In the `execution` class, add after `atlases = []`:
```python
spec = None
"""Path to the BDT or BAT spec YAML file."""
```

Add `'spec'` to `execution._paths`:
```python
_paths = (
    'bids_dir',
    'datasets',
    'bids_database_dir',
    'fs_license_file',
    'layout',
    'log_dir',
    'output_dir',
    'spec',
    'templateflow_home',
    'work_dir',
)
```

Replace the existing `workflow` class:
```python
class workflow(_Config):
    """Shared workflow settings (both BDT and BAT)."""

    ignore = None
    """Ignore particular steps."""
    spaces = None
    """Output spatial references."""
    file_format = 'auto'
    """Force a specific file format: 'auto', 'nifti', 'gifti', or 'cifti'."""


class bdt_workflow(_Config):
    """BDT-specific workflow settings."""

    dummy_scans = None
    """Number of initial scans to treat as non-steady-state."""
    min_coverage = 0.5
    """Minimum fraction of atlas parcel covered by the brain mask."""
    correlation_lengths = None
    """List of TR counts for windowed FC (None = whole-series only)."""
    output_correlations = True
    """Write parcel-parcel correlation TSV alongside parcellated time series."""


class bat_workflow(_Config):
    """BAT-specific workflow settings."""

    output_space = None
    """Template space for BAT output atlases (inherited from inputs if None)."""
    interpolation = 'GenericLabel'
    """ANTs interpolation mode used when resampling atlas label images."""
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_config.py -v`
Expected: PASS

- [ ] **Step 4: Grep for `config.workflow.dummy_scans` and `config.workflow.cifti_output` in the existing codebase and update those references**

```bash
grep -rn "config\.workflow\.dummy_scans\|config\.workflow\.cifti_output" src/bdt/
```

Update each hit to `config.bdt_workflow.dummy_scans` / `config.bdt_workflow.cifti_output`.

- [ ] **Step 5: Commit**

```bash
git add src/bdt/config.py test/test_config.py
git commit -m "feat(config): add bdt_workflow and bat_workflow sections; add execution.spec"
```

---

### Task 3: Spec parser

**Files:**
- Create: `src/bdt/utils/spec.py`
- Create: `test/test_spec.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_spec.py
import pytest

VALID_BDT_YAML = """
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
        res: '2'
"""

VALID_BAT_YAML = """
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
      - atlas: RSN
    output_entities:
      atlas: Schaefer400RSN
"""


def test_load_bdt_spec(tmp_path):
    from bdt.utils.spec import load_bdt_spec
    f = tmp_path / 'bdt_spec.yaml'
    f.write_text(VALID_BDT_YAML)
    spec = load_bdt_spec(f)
    assert len(spec.sources) == 2
    assert spec.sources[0].suffix == 'bold'
    assert spec.sources[0].datasets == ['fmriprep']
    assert spec.sources[0].atlases == [
        {'atlas': 'HCPMMP1'},
        {'atlas': 'Schaefer400', 'desc': '400Parcels17Networks'},
    ]
    assert 'parcellate_timeseries' in spec.sources[0].operations
    assert spec.sources[1].suffix == 'cbf'
    assert spec.sources[1].atlases == [{'atlas': 'HCPMMP1', 'res': '2'}]


def test_bdt_spec_unknown_operation(tmp_path):
    from bdt.utils.spec import load_bdt_spec
    bad = tmp_path / 'bad.yaml'
    bad.write_text(
        'sources:\n'
        '  - suffix: bold\n'
        '    datasets: [fmriprep]\n'
        '    operations: [do_magic]\n'
        '    atlases:\n'
        '      - atlas: HCPMMP1\n'
    )
    with pytest.raises(ValueError, match='Unknown BDT operations'):
        load_bdt_spec(bad)


def test_load_bat_spec(tmp_path):
    from bdt.utils.spec import load_bat_spec
    f = tmp_path / 'bat_spec.yaml'
    f.write_text(VALID_BAT_YAML)
    spec = load_bat_spec(f)
    assert len(spec.operations) == 2
    assert spec.operations[0].name == 'corticalSubcortical'
    assert spec.operations[0].operation == 'union'
    assert spec.operations[0].inputs == [{'atlas': 'HCPMMP1'}, {'atlas': 'Tian', 'seg': 'S2'}]
    assert spec.operations[0].output_entities == {'atlas': 'HCPMMPTian'}


def test_bat_spec_unknown_operation(tmp_path):
    from bdt.utils.spec import load_bat_spec
    bad = tmp_path / 'bad.yaml'
    bad.write_text(
        'operations:\n'
        '  - name: x\n'
        '    operation: subtract\n'
        '    inputs:\n'
        '      - atlas: A\n'
        '    output_entities:\n'
        '      atlas: B\n'
    )
    with pytest.raises(ValueError, match='Unknown BAT operation'):
        load_bat_spec(bad)
```

Run: `pytest test/test_spec.py -v`
Expected: ImportError — `bdt.utils.spec` does not exist

- [ ] **Step 2: Create `src/bdt/utils/spec.py`**

```python
"""Spec YAML parsers for BDT and BAT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_BDT_OPERATIONS = frozenset(
    [
        'parcellate_timeseries',
        'parcellate_scalar',
        'functional_connectivity',
        'streamline_connectivity',
        'bundle_stats',
        'parcellate_fixel',
        'parcellate_odf',
    ]
)
VALID_BAT_OPERATIONS = frozenset(['union', 'intersect', 'outer_product'])


# Reserved (non-entity) keys in a source entry; everything else is a BIDS
# entity filter passed to pybids .get() — including the BEP entities BDT
# consumes: model/param (BEP 016), tract/track (BEP 046), seg/scale (atlas).
_SOURCE_RESERVED = frozenset(['datasets', 'operations', 'atlases'])


@dataclass
class BDTSource:
    suffix: str
    datasets: list[str]
    operations: list[str]
    atlases: list[dict]
    filters: dict  # extra BIDS entity filters (model, param, tract, track, seg, scale, …)

    def __post_init__(self):
        unknown = set(self.operations) - VALID_BDT_OPERATIONS
        if unknown:
            raise ValueError(
                f'Unknown BDT operations: {sorted(unknown)}. '
                f'Valid: {sorted(VALID_BDT_OPERATIONS)}'
            )

    @property
    def entity_query(self) -> dict:
        """Full pybids .get() query for collecting this source's files."""
        return {'suffix': self.suffix, **self.filters}


@dataclass
class BDTSpec:
    sources: list[BDTSource]


def load_bdt_spec(path) -> BDTSpec:
    """Parse a bdt_spec.yaml file.

    A source entry's ``suffix`` plus any extra entity keys (e.g. ``model``,
    ``param``, ``tract``, ``track``, ``seg``, ``scale``) become the BIDS query
    for ``collect_derivatives``; ``datasets``/``operations``/``atlases`` are
    reserved. This lets sources select BEP-typed derivatives, e.g.
    ``{suffix: dwimap, model: csd, param: wm}`` (BEP 016) or
    ``{suffix: tractogram, track: ifod}`` (BEP 046).
    """
    data = yaml.safe_load(Path(path).read_text())
    sources = []
    for s in data['sources']:
        s = dict(s)
        suffix = s.pop('suffix')
        filters = {k: v for k, v in s.items() if k not in _SOURCE_RESERVED}
        known = {k: s[k] for k in _SOURCE_RESERVED if k in s}
        sources.append(BDTSource(suffix=suffix, filters=filters, **known))
    return BDTSpec(sources=sources)


@dataclass
class BATOperation:
    name: str
    operation: str
    inputs: list[dict]
    output_entities: dict

    def __post_init__(self):
        if self.operation not in VALID_BAT_OPERATIONS:
            raise ValueError(
                f'Unknown BAT operation: {self.operation!r}. '
                f'Valid: {sorted(VALID_BAT_OPERATIONS)}'
            )


@dataclass
class BATSpec:
    operations: list[BATOperation]


def load_bat_spec(path) -> BATSpec:
    """Parse a bat_spec.yaml file."""
    data = yaml.safe_load(Path(path).read_text())
    return BATSpec(operations=[BATOperation(**op) for op in data['operations']])
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_spec.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Commit**

```bash
git add src/bdt/utils/spec.py test/test_spec.py
git commit -m "feat(utils): add spec YAML parser for BDT and BAT"
```

---

### Task 4: Atlas utilities

The current `collect_atlases` in `utils/bids.py` uses named string atlas labels. Replace it with an entity-filter-dict approach in a new `utils/atlas.py`. The existing `connectivity.py` already imports `from bdt.utils.atlas import select_atlases` (a broken import); this task fixes that.

**Files:**
- Create: `src/bdt/utils/atlas.py`
- Create: `test/test_atlas.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_atlas.py
from unittest.mock import MagicMock
import pytest


def test_collect_atlases_single_match():
    from bdt.utils.atlas import collect_atlases
    layout = MagicMock()
    layout.get.return_value = ['/data/tpl-MNI_atlas-HCPMMP1_dseg.nii.gz']
    result = collect_atlases(layout, [{'atlas': 'HCPMMP1'}])
    assert result == ['/data/tpl-MNI_atlas-HCPMMP1_dseg.nii.gz']
    layout.get.assert_called_once_with(return_type='file', atlas='HCPMMP1')


def test_collect_atlases_multiple_filters():
    from bdt.utils.atlas import collect_atlases
    layout = MagicMock()
    layout.get.side_effect = [
        ['/data/tpl-MNI_atlas-HCPMMP1_dseg.nii.gz'],
        ['/data/tpl-MNI_atlas-Schaefer400_desc-400Parcels17Networks_dseg.nii.gz'],
    ]
    result = collect_atlases(
        layout,
        [{'atlas': 'HCPMMP1'}, {'atlas': 'Schaefer400', 'desc': '400Parcels17Networks'}],
    )
    assert len(result) == 2


def test_collect_atlases_ambiguous_raises():
    from bdt.utils.atlas import collect_atlases
    layout = MagicMock()
    layout.get.return_value = [
        '/data/tpl-MNI_atlas-HCPMMP1_res-1_dseg.nii.gz',
        '/data/tpl-MNI_atlas-HCPMMP1_res-2_dseg.nii.gz',
    ]
    with pytest.raises(ValueError, match='Multiple atlases found'):
        collect_atlases(layout, [{'atlas': 'HCPMMP1'}])


def test_collect_atlases_no_match_raises():
    from bdt.utils.atlas import collect_atlases
    layout = MagicMock()
    layout.get.return_value = []
    with pytest.raises(ValueError, match='No atlas found'):
        collect_atlases(layout, [{'atlas': 'NonExistent'}])
```

Run: `pytest test/test_atlas.py -v`
Expected: ImportError — `bdt.utils.atlas` does not exist

- [ ] **Step 2: Create `src/bdt/utils/atlas.py`**

```python
"""Atlas collection utilities."""
from __future__ import annotations

from bids import BIDSLayout


def collect_atlases(layout: BIDSLayout, atlas_filters: list[dict]) -> list[str]:
    """Return atlas file paths matching a list of entity-filter dicts.

    Each dict in ``atlas_filters`` is passed directly to ``layout.get()``.
    Raises ``ValueError`` if any filter matches zero or more than one file.
    """
    results = []
    for filt in atlas_filters:
        matches = layout.get(return_type='file', **filt)
        if not matches:
            raise ValueError(
                f'No atlas found for filter {filt!r}. '
                'Check that the atlas dataset is indexed and the entities are correct.'
            )
        if len(matches) > 1:
            bulleted = '\n'.join(f'  - {m}' for m in matches)
            raise ValueError(
                f'Multiple atlases found for filter {filt!r}:\n{bulleted}\n'
                'Add more entity filters (e.g., res, desc, seg) to disambiguate.'
            )
        results.append(matches[0])
    return results


def select_atlases(layout: BIDSLayout, atlas_filters: list[dict]) -> list[str]:
    """Alias for collect_atlases (keeps connectivity.py import working)."""
    return collect_atlases(layout, atlas_filters)
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_atlas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Commit**

```bash
git add src/bdt/utils/atlas.py test/test_atlas.py
git commit -m "feat(utils): add collect_atlases with entity-filter dict approach"
```

---

### Task 5: Create `bdt.tests` package and restructure `workflows/bdt/`

Existing conftest imports `from bdt.tests.tests import mock_config`, but `src/bdt/tests/` does not exist. Fix by creating the package. Also create the `workflows/bdt/` and `workflows/bat/` package shells so later tasks can populate them.

**Files:**
- Create: `src/bdt/tests/__init__.py`
- Create: `src/bdt/tests/tests.py` (copy of `test/tests.py`)
- Create: `src/bdt/workflows/bdt/__init__.py`
- Create: `src/bdt/workflows/bat/__init__.py`

- [ ] **Step 1: Create `src/bdt/tests/__init__.py`**

```python
"""BDT test utilities."""
```

- [ ] **Step 2: Copy `test/tests.py` → `src/bdt/tests/tests.py`**

```bash
cp test/tests.py src/bdt/tests/tests.py
```

The file is identical — no edits needed. The `test/tests.py` copy can stay as-is; it is used by the existing `run_local_tests.py`.

- [ ] **Step 3: Create `src/bdt/workflows/bdt/__init__.py`**

```python
"""BDT sub-workflows."""
```

- [ ] **Step 4: Create `src/bdt/workflows/bat/__init__.py`**

```python
"""BAT sub-workflows."""
```

- [ ] **Step 5: Verify conftest fixture resolves**

Run: `pytest test/conftest.py --collect-only -q`
Expected: no import errors

- [ ] **Step 6: Commit**

```bash
git add src/bdt/tests/ src/bdt/workflows/bdt/__init__.py src/bdt/workflows/bat/__init__.py
git commit -m "feat: add bdt.tests package; scaffold workflows/bdt and workflows/bat"
```

---

## Phase 2 — Transform Engine

### Task 6: Transform graph with two typed queries (`utils/transforms.py`)

> **Reworked (review §5).** The engine must be **modality-aware**. ANTs `ApplyTransforms` uses image/pull semantics; the Rust point-warpers (`trxrs`/`giftirs`) use the *opposite* convention. A single untyped `find_transform_chain` reused for both silently mirrors point data. So we expose two typed queries and track per-edge transform type + invertibility. The filename regex is widened to index `.mat`/`GenericAffine.mat` and warp files lacking the `mode-image` token.

**Files:**
- Create: `src/bdt/utils/transforms.py`
- Create: `test/test_transforms.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_transforms.py
import pytest
import networkx as nx


def test_build_transform_graph_empty(tmp_path):
    from bdt.utils.transforms import build_transform_graph
    g = build_transform_graph({'ds': tmp_path})
    assert isinstance(g, nx.DiGraph)
    assert len(g.nodes) == 0


def test_build_transform_graph_finds_xfm(tmp_path):
    from bdt.utils.transforms import build_transform_graph
    xfm = tmp_path / 'sub-01' / 'anat'
    xfm.mkdir(parents=True)
    (xfm / 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5').touch()
    (xfm / 'sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5').touch()
    g = build_transform_graph({'ds': tmp_path})
    assert g.has_edge('T1w', 'MNI152NLin6Asym')
    assert g.has_edge('MNI152NLin6Asym', 'T1w')


def test_build_transform_graph_indexes_mat_and_no_mode(tmp_path):
    """Regex must catch .mat / GenericAffine and files without mode-image."""
    from bdt.utils.transforms import build_transform_graph
    d = tmp_path / 'anat'
    d.mkdir()
    (d / 'sub-01_from-T1w_to-MNI152NLin6Asym_xfm.mat').touch()            # no mode-image
    (d / 'sub-01_from-fsnative_to-fsLR_xfm.txt').touch()
    g = build_transform_graph({'ds': tmp_path})
    assert g.has_edge('T1w', 'MNI152NLin6Asym')
    assert g.has_edge('fsnative', 'fsLR')


def test_image_resample_uses_pull_direction(tmp_path):
    """chain_for_image_resample(atlas→data) returns from-atlas_to-data files."""
    from bdt.utils.transforms import build_transform_graph, chain_for_image_resample
    d = tmp_path / 'anat'
    d.mkdir()
    f = d / 'sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5'
    f.touch()
    g = build_transform_graph({'ds': tmp_path})
    files, tool, invert = chain_for_image_resample(g, 'MNI152NLin6Asym', 'T1w')
    assert files == [str(f)]
    assert tool == 'ApplyTransforms'
    assert invert == [False]


def test_point_warp_uses_opposite_direction(tmp_path):
    """chain_for_point_warp(streamlines→atlas) needs from-atlas_to-streamlines file."""
    from bdt.utils.transforms import build_transform_graph, chain_for_point_warp
    d = tmp_path / 'anat'
    d.mkdir()
    # To warp points SS→AS, trxrs needs the from-AS_to-SS warp:
    f = d / 'sub-01_from-MNI152NLin6Asym_to-ACPC_mode-image_xfm.h5'
    f.touch()
    g = build_transform_graph({'ds': tmp_path})
    files, tool, invert = chain_for_point_warp(g, 'ACPC', 'MNI152NLin6Asym')
    assert files == [str(f)]
    assert tool in ('trxrs', 'giftirs')


def test_point_warp_missing_direction_raises(tmp_path):
    """A displacement warp cannot be inverted; wrong-direction-only must error."""
    from bdt.utils.transforms import build_transform_graph, chain_for_point_warp
    d = tmp_path / 'anat'
    d.mkdir()
    # Only the forward (image-direction) warp exists; point warp the other way is impossible.
    (d / 'sub-01_from-ACPC_to-MNI152NLin6Asym_mode-image_xfm.h5').touch()
    g = build_transform_graph({'ds': tmp_path})
    with pytest.raises(ValueError, match='no invertible|No transform path|cannot be inverted'):
        chain_for_point_warp(g, 'ACPC', 'MNI152NLin6Asym')


def test_image_resample_no_path_raises(tmp_path):
    from bdt.utils.transforms import build_transform_graph, chain_for_image_resample
    d = tmp_path / 'anat'
    d.mkdir()
    (d / 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5').touch()
    g = build_transform_graph({'ds': tmp_path})
    with pytest.raises(ValueError, match='No transform path'):
        chain_for_image_resample(g, 'MNI152NLin6Asym', 'T1w')
```

Run: `pytest test/test_transforms.py -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/utils/transforms.py`**

```python
"""Geometry-aware graph-based transform chain utilities.

Two typed queries prevent applying a transform backwards:
  * chain_for_image_resample — ANTs ApplyTransforms (image/pull semantics)
  * chain_for_point_warp     — trxrs/giftirs (opposite, point semantics)

Per the trx-rs/gifti-rs convention, to warp points from space A to space B you
pass the ``from-B_to-A`` transform. Displacement fields cannot be inverted
(itk-transforms-rs limitation), so a point warp requires the correctly-named
warp to physically exist.
"""
from __future__ import annotations

import re
from pathlib import Path

import networkx as nx

# Widened: optional mode-* token; .h5/.txt/.mat (incl. GenericAffine.mat); warp files.
_XFM_PATTERN = re.compile(
    r'from-(\w+)_to-(\w+)(?:_mode-[a-z]+)?_xfm\.(h5|txt|mat)$'
)
# Affine extensions are invertible by ANTs/trxrs; .h5 displacement fields are not.
_AFFINE_EXTS = {'txt', 'mat'}


def build_transform_graph(datasets: dict) -> nx.DiGraph:
    """Scan derivative datasets for *_xfm.* files and build a directed transform graph.

    Each edge carries:
      * 'file' — absolute path to the transform
      * 'xfm_type' — 'affine' or 'displacement'
      * 'invertible' — whether the transform can be inverted in place
    """
    graph = nx.DiGraph()
    for dataset_path in datasets.values():
        for xfm_file in Path(dataset_path).rglob('*_xfm.*'):
            m = _XFM_PATTERN.search(xfm_file.name)
            if not m:
                continue
            src, tgt, ext = m.group(1), m.group(2), m.group(3)
            is_affine = ext in _AFFINE_EXTS or 'GenericAffine' in xfm_file.name
            graph.add_edge(
                src, tgt,
                file=str(xfm_file.absolute()),
                xfm_type='affine' if is_affine else 'displacement',
                invertible=is_affine,
            )
    return graph


def _chain_files(graph: nx.DiGraph, src: str, tgt: str) -> list[str]:
    available = sorted(graph.nodes())
    for space in (src, tgt):
        if space not in graph:
            raise ValueError(
                f'Space "{space}" not in transform graph. Available spaces: {available}'
            )
    try:
        path = nx.shortest_path(graph, src, tgt)
    except nx.NetworkXNoPath:
        raise ValueError(
            f'No transform path from "{src}" to "{tgt}". Available spaces: {available}'
        )
    return [graph.edges[path[i], path[i + 1]]['file'] for i in range(len(path) - 1)]


def chain_for_image_resample(graph: nx.DiGraph, src: str, tgt: str):
    """Resample an image (e.g. atlas) from ``src`` space into ``tgt`` grid.

    ANTs ApplyTransforms pull semantics: the chain is the from-src_to-tgt path.

    Returns
    -------
    (files, tool, invert_flags) : (list[str], str, list[bool])
    """
    files = _chain_files(graph, src, tgt)
    return files, 'ApplyTransforms', [False] * len(files)


def chain_for_point_warp(graph: nx.DiGraph, src: str, tgt: str, tool: str = 'trxrs'):
    """Warp point data (streamlines/surface verts) from ``src`` space into ``tgt``.

    Per trx-rs/gifti-rs: to move points src→tgt, pass the ``from-tgt_to-src`` xfm.
    We therefore look up the *reverse* graph path (tgt→src). Because displacement
    fields cannot be inverted, every edge on that path must physically exist in the
    needed direction; if only the wrong-direction (non-invertible) warp is present,
    raise rather than synthesize.

    Returns
    -------
    (files, tool, invert_flags) : (list[str], str, list[bool])
    """
    available = sorted(graph.nodes())
    for space in (src, tgt):
        if space not in graph:
            raise ValueError(
                f'Space "{space}" not in transform graph. Available spaces: {available}'
            )
    # Reverse direction: trxrs/giftirs consume the from-tgt_to-src files.
    try:
        rev_path = nx.shortest_path(graph, tgt, src)
    except nx.NetworkXNoPath:
        # No real from-tgt_to-src warp exists. Could the inverse be used instead?
        if nx.has_path(graph, src, tgt):
            if all(
                graph.edges[u, v]['invertible']
                for u, v in zip(*(lambda p: (p[:-1], p[1:]))(nx.shortest_path(graph, src, tgt)))
            ):
                # All-affine inverse: trxrs --invert can flip these.
                fwd = _chain_files(graph, src, tgt)
                return fwd, tool, [True] * len(fwd)
            raise ValueError(
                f'Cannot warp points "{src}"→"{tgt}": only the wrong-direction warp '
                f'exists and displacement fields cannot be inverted. A from-{tgt}_to-{src} '
                f'warp must be provided.'
            )
        raise ValueError(
            f'No transform path to warp points from "{src}" to "{tgt}". '
            f'Available spaces: {available}'
        )
    files = [graph.edges[rev_path[i], rev_path[i + 1]]['file'] for i in range(len(rev_path) - 1)]
    return files, tool, [False] * len(files)
```

> **Note on directionality:** the point-warp inverse-affine branch is an optimization; if it proves fiddly during implementation, drop it and require the explicitly-named warp (the strict behavior the design mandates). Keep the `test_point_warp_missing_direction_raises` guarantee either way.

- [ ] **Step 3: Run tests**

Run: `pytest test/test_transforms.py -v`
Expected: PASS (7 tests)

- [ ] **Step 4: Commit**

```bash
git add src/bdt/utils/transforms.py test/test_transforms.py
git commit -m "feat(utils): add geometry-aware transform engine with image/point queries"
```

---

### Task 6b: Rust binary `CommandLine` interfaces (`interfaces/rust.py`)

> **New (review §2).** `giftirs` and `trxrs transform`/`convert` have no/read-only Python bindings, so they must be wrapped as nipype `CommandLine` interfaces shelling out to the binaries — exactly like ANTs/MRtrix/wb_command. `odx transform` also gets a CLI wrapper for uniformity (its PyO3 bindings are used directly in the parcellation interfaces, Task 17b).

**Files:**
- Create: `src/bdt/interfaces/rust.py`
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing test** (importable + correct `cmd`; no binary needed at import time)

```python
# add to test/test_interfaces.py
def test_rust_interfaces_cmdline():
    from bdt.interfaces.rust import (
        TrxTransform, TrxConvert, GiftiTransform, OdxTransform, OdxConvert,
    )
    assert TrxTransform().cmd == 'trxrs'
    assert TrxConvert().cmd == 'trxrs'
    assert GiftiTransform().cmd == 'giftirs'
    assert OdxTransform().cmd == 'odx'
    assert OdxConvert().cmd == 'odx'
```

Run: `pytest test/test_interfaces.py::test_rust_interfaces_cmdline -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/rust.py`**

Use `CommandLine` / `CommandLineInputSpec` with `argstr` traits. The argstrs below are **verified against the cloned `trx-rs` / `gifti-rs` / `odx-rs` CLI sources** (`src/bin/trxrs.rs`, `giftirs-cli/src/main.rs`, `src/bin/odx.rs`). Key facts baked in: every subcommand takes **positional `input` then `output`** (there is no `--output` flag); `trxrs`/`giftirs` use the **opposite-named** point-warp h5 while `odx` uses the **same-named** image h5; `giftirs` overwrite flag is `--overwrite`, `trxrs` is `--force`; `odx transform` adds `--mode {mrtrix,ants}` and `--transform-inverse` and operates on `.odx` containers (no `--reference`).

```python
"""nipype CommandLine wrappers for the Rust binaries (trxrs, giftirs, odx)."""
from __future__ import annotations

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


# --- trxrs transform --------------------------------------------------------
class _TrxTransformInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0,
                   desc='input streamlines (.trx/.tck)')
    out_file = File('warped.trx', usedefault=True, argstr='%s', position=1,
                    desc='output TRX (positional, immediately after input)')
    # trx-rs convention: to warp tracts from space A to space B, pass from-B_to-A_xfm.
    transform = File(exists=True, mandatory=True, argstr='--transform %s',
                     desc='reverse-named warp/affine (see chain_for_point_warp)')
    reference = File(exists=True, argstr='--reference %s',
                     desc='optional target-space NIfTI/TRX; sets the output header')
    invert = traits.Bool(False, usedefault=True, argstr='--invert',
                         desc='numerically invert an affine-only chain (warps cannot be inverted)')
    force = traits.Bool(True, usedefault=True, argstr='--force',
                        desc='overwrite the output if it exists')


class _TrxTransformOutputSpec(TraitedSpec):
    out_file = File(desc='warped streamlines')


class TrxTransform(CommandLine):
    """Warp streamline coordinates with `trxrs transform`."""
    _cmd = 'trxrs transform'
    input_spec = _TrxTransformInputSpec
    output_spec = _TrxTransformOutputSpec

    @property
    def cmd(self):
        return 'trxrs'

    def _list_outputs(self):
        import os
        return {'out_file': os.path.abspath(self.inputs.out_file)}


# --- trxrs convert ----------------------------------------------------------
class _TrxConvertInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0,
                   desc='input streamlines (.trk/.tck/.trx)')
    out_file = File('converted.trx', usedefault=True, argstr='%s', position=1,
                    desc='output (.trx or .tck, positional; trx-rs cannot write .trk)')


class _TrxConvertOutputSpec(TraitedSpec):
    out_file = File(desc='converted streamlines')


class TrxConvert(CommandLine):
    """Convert .trk → .trx/.tck on ingest (trx-rs reads .trk but cannot write it)."""
    _cmd = 'trxrs convert'
    input_spec = _TrxConvertInputSpec
    output_spec = _TrxConvertOutputSpec

    @property
    def cmd(self):
        return 'trxrs'

    def _list_outputs(self):
        import os
        return {'out_file': os.path.abspath(self.inputs.out_file)}


# --- giftirs transform ------------------------------------------------------
class _GiftiTransformInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0,
                   desc='input *.surf.gii (POINTSET required)')
    out_file = File('warped.surf.gii', usedefault=True, argstr='%s', position=1,
                    desc='output *.surf.gii (positional)')
    transform = File(exists=True, mandatory=True, argstr='--transform %s',
                     desc='reverse-named warp (see chain_for_point_warp); -t/--transform')
    invert = traits.Bool(False, usedefault=True, argstr='--invert',
                         desc='invert an affine-only transform')
    overwrite = traits.Bool(True, usedefault=True, argstr='--overwrite',
                            desc='overwrite the output if it exists (giftirs uses --overwrite)')


class _GiftiTransformOutputSpec(TraitedSpec):
    out_file = File(desc='warped surface geometry')


class GiftiTransform(CommandLine):
    """Warp *.surf.gii vertices with `giftirs transform` (geometry only)."""
    _cmd = 'giftirs transform'
    input_spec = _GiftiTransformInputSpec
    output_spec = _GiftiTransformOutputSpec

    @property
    def cmd(self):
        return 'giftirs'

    def _list_outputs(self):
        import os
        return {'out_file': os.path.abspath(self.inputs.out_file)}


# --- odx transform ----------------------------------------------------------
class _OdxTransformInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0,
                   desc='input .odx container (ingest FOD/MRtrix/DSI/aodf first)')
    out_file = File('resampled.odx', usedefault=True, argstr='%s', position=1,
                    desc='output .odx (positional)')
    # SAME-named image h5: to move source→target, pass from-{source}_to-{target}_xfm.
    transform = File(exists=True, mandatory=True, argstr='--transform %s',
                     desc='image-direction h5 (chain_for_image_resample), NOT the point convention')
    mode = traits.Enum('mrtrix', 'ants', usedefault=True, argstr='--mode %s',
                       desc='mrtrix: single h5, no fixel-correspondence; '
                            'ants: paired h5s, preserves fixel cardinality')
    transform_inverse = File(exists=True, argstr='--transform-inverse %s',
                             desc='required for --mode ants: from-{target}_to-{source}_xfm '
                                  '(pushes fixels source→target)')


class _OdxTransformOutputSpec(TraitedSpec):
    out_file = File(desc='resampled + SH-reoriented .odx diffusion model')


class OdxTransform(CommandLine):
    """Resample + SH-reorient a diffusion model with `odx transform` (CLI-only)."""
    _cmd = 'odx transform'
    input_spec = _OdxTransformInputSpec
    output_spec = _OdxTransformOutputSpec

    @property
    def cmd(self):
        return 'odx'

    def _list_outputs(self):
        import os
        return {'out_file': os.path.abspath(self.inputs.out_file)}


# --- odx convert (ingest to .odx) -------------------------------------------
class _OdxConvertInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0,
                   desc='FOD .nii.gz / MRtrix .mif / DSI Studio / pyAFQ aodf input')
    out_file = File('model.odx', usedefault=True, argstr='%s', position=1,
                    desc='output .odx (positional)')


class _OdxConvertOutputSpec(TraitedSpec):
    out_file = File(desc='ingested .odx container')


class OdxConvert(CommandLine):
    """Ingest a diffusion model into `.odx` with `odx convert` (verify subcommand
    flags/format-detection against `odx convert --help`; `from_mrtrix`/`from_pyafq_aodf`
    PyO3 helpers are an alternative for in-process ingest)."""
    _cmd = 'odx convert'
    input_spec = _OdxConvertInputSpec
    output_spec = _OdxConvertOutputSpec

    @property
    def cmd(self):
        return 'odx'

    def _list_outputs(self):
        import os
        return {'out_file': os.path.abspath(self.inputs.out_file)}
```

> **Correctness note on the subcommand (do not skip).** The `cmd` property override above is **only safe for the `cmd == 'trxrs'` assertion** — it is *not* a safe runtime pattern. nipype's `CommandLine.cmdline` builds the command from `self.cmd`, so overriding `cmd` to return `'trxrs'` would **drop the `transform`/`convert` subcommand at run time**, producing `trxrs input output …` (broken). Two robust options; pick one consistently:
> 1. **Subcommand as a leading positional trait** (recommended): set `_cmd = 'trxrs'` and add `subcommand = traits.Str('transform', usedefault=True, argstr='%s', position=0)` to the input spec, shifting `in_file`/`out_file` to positions 1/2. Drop the `cmd` override; `.cmd == 'trxrs'` then holds naturally and the full line is `trxrs transform <in> <out> …`. Do the same for `convert`/`giftirs transform`/`odx transform`/`odx convert`.
> 2. **Keep the multi-word `_cmd`** (`'trxrs transform'`) and **remove** the `cmd` override; the test then asserts `.cmd == 'trxrs transform'` and `.cmd.split()[0] == 'trxrs'`.
>
> Option 1 is cleaner for the positional `input output` layout these CLIs use; the code blocks above show positions 0/1 *without* the subcommand trait, so applying option 1 means bumping them to 1/2.

- [ ] **Step 3: Run test**

Run: `pytest test/test_interfaces.py::test_rust_interfaces_cmdline -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/rust.py test/test_interfaces.py
git commit -m "feat(interfaces): add CommandLine wrappers for trxrs/giftirs/odx"
```

---

## Phase 3 — Interfaces

### Task 7: `interfaces/ants.py` — `ApplyTransforms`

**Files:**
- Create: `src/bdt/interfaces/ants.py`
- Create: `test/test_interfaces.py` (start file here, Tasks 8–13 add to it)

- [ ] **Step 1: Write failing test**

```python
# test/test_interfaces.py
def test_apply_transforms_defaults():
    from bdt.interfaces.ants import ApplyTransforms
    node = ApplyTransforms()
    assert node.inputs.interpolation == 'GenericLabel'
    assert node.inputs.float is True


def test_apply_transforms_float_overridable():
    """Review §7: float=True must not be permanently forced."""
    from bdt.interfaces.ants import ApplyTransforms
    node = ApplyTransforms(float=False)
    assert node.inputs.float is False
```

Run: `pytest test/test_interfaces.py::test_apply_transforms_defaults -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/ants.py`**

> **Bug fix (review §7).** The first draft used `if not self.inputs.float: self.inputs.float = True`, which permanently forces `True` (an explicit `float=False` is falsy and gets overwritten). Use `traits.Undefined` checks so a user-supplied value — including `False` — is respected.

```python
"""ANTs interface wrappers with bdt defaults."""
from nipype.interfaces.ants import ApplyTransforms as _Base
from nipype.interfaces.base import isdefined


class ApplyTransforms(_Base):
    """ANTs ApplyTransforms with bdt label-image defaults.

    Defaults interpolation to 'GenericLabel' and float to True (so atlas label
    images resample without blending neighbouring labels), but only when the
    user has not set them explicitly.
    """

    def __init__(self, **inputs):
        super().__init__(**inputs)
        if not isdefined(self.inputs.interpolation):
            self.inputs.interpolation = 'GenericLabel'
        if not isdefined(self.inputs.float):
            self.inputs.float = True
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_interfaces.py::test_apply_transforms_defaults -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/ants.py test/test_interfaces.py
git commit -m "feat(interfaces): add ApplyTransforms wrapper with GenericLabel defaults"
```

---

### Task 8: `interfaces/nilearn.py` — `IndexImage`, `NiftiLabelsMasker`

**Files:**
- Modify: `src/bdt/interfaces/nilearn.py` (create new file)
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing tests**

```python
# add to test/test_interfaces.py
import numpy as np
import nibabel as nib
import pytest


def _make_4d_nifti(tmp_path, shape=(4, 4, 4, 10)):
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / 'bold.nii.gz'
    nib.save(img, path)
    return str(path)


def _make_label_nifti(tmp_path, shape=(4, 4, 4)):
    data = np.zeros(shape, dtype=np.int32)
    data[:2, :2, :2] = 1
    data[2:, 2:, 2:] = 2
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / 'atlas.nii.gz'
    nib.save(img, path)
    return str(path)


def test_index_image_extracts_volume(tmp_path):
    from bdt.interfaces.nilearn import IndexImage
    bold = _make_4d_nifti(tmp_path)
    res = IndexImage(in_file=bold, index=3).run()
    out = nib.load(res.outputs.out_file)
    assert out.ndim == 3


def test_index_image_correct_volume(tmp_path):
    from bdt.interfaces.nilearn import IndexImage
    bold = _make_4d_nifti(tmp_path)
    orig = nib.load(bold)
    res = IndexImage(in_file=bold, index=5).run()
    out_data = np.asanyarray(nib.load(res.outputs.out_file).dataobj)
    expected = np.asanyarray(orig.dataobj)[..., 5]
    np.testing.assert_array_equal(out_data, expected)
```

Run: `pytest test/test_interfaces.py::test_index_image_extracts_volume -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/nilearn.py`**

```python
"""Nilearn-based nipype interfaces."""
from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _IndexImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='4D NIfTI file')
    index = traits.Int(mandatory=True, desc='0-based volume index to extract')


class _IndexImageOutputSpec(TraitedSpec):
    out_file = File(desc='3D NIfTI volume')


class IndexImage(SimpleInterface):
    """Extract a single volume from a 4D NIfTI image."""

    input_spec = _IndexImageInputSpec
    output_spec = _IndexImageOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        img = nib.load(self.inputs.in_file)
        data = np.asanyarray(img.dataobj)[..., self.inputs.index]
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_file = Path(runtime.cwd) / 'indexed.nii.gz'
        nib.save(out_img, out_file)
        self._results['out_file'] = str(out_file)
        return runtime
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_interfaces.py::test_index_image_extracts_volume test/test_interfaces.py::test_index_image_correct_volume -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/nilearn.py test/test_interfaces.py
git commit -m "feat(interfaces): add IndexImage nipype interface"
```

---

### Task 9: `interfaces/censoring.py` — `Censor`

**Files:**
- Create: `src/bdt/interfaces/censoring.py`
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_interfaces.py
import pandas as pd


def _make_temporal_mask(tmp_path, n_vols=10, col='framewise_displacement'):
    mask = [True, False, True, True, False, True, True, True, False, True]
    df = pd.DataFrame({col: mask})
    path = tmp_path / 'mask.tsv'
    df.to_csv(path, sep='\t', index=False)
    return str(path), sum(mask)


def test_censor_removes_volumes(tmp_path):
    from bdt.interfaces.censoring import Censor
    bold = _make_4d_nifti(tmp_path)
    mask_file, n_kept = _make_temporal_mask(tmp_path)
    res = Censor(
        in_file=bold,
        temporal_mask=mask_file,
        mask_column='framewise_displacement',
    ).run()
    out = nib.load(res.outputs.out_file)
    assert out.shape[-1] == n_kept
```

Run: `pytest test/test_interfaces.py::test_censor_removes_volumes -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/censoring.py`**

```python
"""Volume censoring interface."""
from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _CensorInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='4D NIfTI time series')
    temporal_mask = File(exists=True, mandatory=True, desc='TSV with boolean mask column')
    mask_column = traits.Str(mandatory=True, desc='Column name in temporal_mask to use as keep-mask')


class _CensorOutputSpec(TraitedSpec):
    out_file = File(desc='Censored 4D NIfTI time series')


class Censor(SimpleInterface):
    """Remove volumes from a time series based on a boolean temporal mask."""

    input_spec = _CensorInputSpec
    output_spec = _CensorOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        import pandas as pd

        img = nib.load(self.inputs.in_file)
        df = pd.read_table(self.inputs.temporal_mask)
        keep = df[self.inputs.mask_column].astype(bool).to_numpy()
        data = np.asanyarray(img.dataobj)[..., keep]
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_file = Path(runtime.cwd) / 'censored.nii.gz'
        nib.save(out_img, out_file)
        self._results['out_file'] = str(out_file)
        return runtime
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_interfaces.py::test_censor_removes_volumes -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/censoring.py test/test_interfaces.py
git commit -m "feat(interfaces): add Censor interface for temporal masking"
```

---

### Task 10: `interfaces/plotting.py` — `PlotCiftiParcellation`

**Files:**
- Create: `src/bdt/interfaces/plotting.py`
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_interfaces.py
def test_plot_cifti_parcellation_importable():
    from bdt.interfaces.plotting import PlotCiftiParcellation  # noqa: F401
    assert PlotCiftiParcellation is not None
```

Run: `pytest test/test_interfaces.py::test_plot_cifti_parcellation_importable -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/plotting.py`**

```python
"""Visualisation interfaces for parcellated data."""
from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
)


class _PlotCiftiParcellationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Parcellated scalar CIFTI (.pscalar.nii)')
    atlas_file = File(exists=True, mandatory=True, desc='Parcellated label CIFTI (.dlabel.nii)')


class _PlotCiftiParcellationOutputSpec(TraitedSpec):
    out_file = File(desc='SVG figure')


class PlotCiftiParcellation(SimpleInterface):
    """Render parcellated CIFTI scalars as a surface figure via Connectome Workbench."""

    input_spec = _PlotCiftiParcellationInputSpec
    output_spec = _PlotCiftiParcellationOutputSpec

    def _run_interface(self, runtime):
        import subprocess
        import nibabel as nib
        import numpy as np
        import matplotlib.pyplot as plt

        out_file = Path(runtime.cwd) / 'parcellation.svg'

        # Load the pscalar CIFTI and atlas dlabel for parcel names
        scalar_img = nib.load(self.inputs.in_file)
        scalar_data = scalar_img.get_fdata(dtype=np.float32).squeeze()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(scalar_data)), scalar_data)
        ax.set_xlabel('Parcel index')
        ax.set_ylabel('Value')
        ax.set_title('Parcellated CIFTI scalar')
        fig.savefig(out_file, format='svg', bbox_inches='tight')
        plt.close(fig)

        self._results['out_file'] = str(out_file)
        return runtime
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_interfaces.py::test_plot_cifti_parcellation_importable -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/plotting.py test/test_interfaces.py
git commit -m "feat(interfaces): add PlotCiftiParcellation interface"
```

---

### Task 11: `interfaces/gifti.py` — `SurfaceVolumeParcellate`

> **Reworked (review §4).** The first draft's `GiftiParcellate` re-implemented — worse — what `wb_command -cifti-parcellate` already does (it had a dead `ndim==1` branch and fragile label-table handling), and `gifti-rs` does **not** parcellate. So:
> - **Surface data + surface atlas (same mesh):** route through the existing Connectome Workbench path (`interfaces/workbench.py` / `init_parcellate_cifti_wf`). No new nibabel interface.
> - **Surface data + surface atlas (different mesh):** add `init_label_resample_wf` (`wb_command -label-resample`) before parcellation — see Task 14.
> - **Surface data + *volume* atlas:** the genuinely new capability — warp `*.surf.gii` into the volume atlas's space with `giftirs transform` (Task 6b), then sample the label volume at each vertex. That is `SurfaceVolumeParcellate` below.

**Files:**
- Create: `src/bdt/interfaces/gifti.py`
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_interfaces.py
def _make_surf_gifti(tmp_path, coords):
    """Minimal *.surf.gii with a POINTSET darray at the given world coords."""
    import nibabel as nib
    import numpy as np
    pts = np.asarray(coords, dtype=np.float32)
    darray = nib.gifti.GiftiDataArray(
        pts, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
    )
    img = nib.GiftiImage(darrays=[darray])
    path = tmp_path / 'midthickness.surf.gii'
    nib.save(img, path)
    return str(path)


def test_surface_volume_parcellate_assigns_labels(tmp_path):
    """Each vertex gets the label of the voxel it falls in (identity affine)."""
    from bdt.interfaces.gifti import SurfaceVolumeParcellate
    import nibabel as nib, numpy as np, pandas as pd
    # 4x4x4 atlas: parcel 1 at voxel (0,0,0), parcel 2 at voxel (3,3,3)
    atlas = np.zeros((4, 4, 4), dtype=np.int32)
    atlas[0, 0, 0] = 1
    atlas[3, 3, 3] = 2
    apath = tmp_path / 'atlas.nii.gz'
    nib.save(nib.Nifti1Image(atlas, np.eye(4)), apath)
    # warped surface: one vertex in each parcel voxel
    surf = _make_surf_gifti(tmp_path, [[0, 0, 0], [3, 3, 3]])
    res = SurfaceVolumeParcellate(warped_surface=surf, atlas_file=str(apath)).run()
    labels = pd.read_table(res.outputs.out_file)['parcel_id'].tolist()
    assert set(labels) == {1, 2}
```

Run: `pytest test/test_interfaces.py::test_surface_volume_parcellate_assigns_labels -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/gifti.py`**

```python
"""Surface-by-volume parcellation: sample a label volume at surface vertices.

Pairs with GiftiTransform (Task 6b): warp a subject's *.surf.gii into a volume
atlas's space, then this interface assigns each vertex the label of the voxel it
falls in. Lossless surface parcellation by a volumetric atlas — gifti-rs cannot
parcellate, and wb_command is not needed for the volume-atlas case.
"""
from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _SurfaceVolumeParcellateInputSpec(BaseInterfaceInputSpec):
    warped_surface = File(exists=True, mandatory=True,
                          desc='*.surf.gii already warped into the atlas grid space')
    atlas_file = File(exists=True, mandatory=True, desc='NIfTI label volume atlas')
    in_file = File(exists=False, desc='optional *.func.gii/.shape.gii to parcel-average; '
                                      'if absent, only a vertex→parcel table is written')


class _SurfaceVolumeParcellateOutputSpec(TraitedSpec):
    out_file = File(desc='TSV: vertex→parcel assignment (and per-parcel means if in_file given)')


class SurfaceVolumeParcellate(SimpleInterface):
    """Assign each surface vertex the label of the atlas voxel it lies in."""

    input_spec = _SurfaceVolumeParcellateInputSpec
    output_spec = _SurfaceVolumeParcellateOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        import pandas as pd

        surf = nib.load(self.inputs.warped_surface)
        pointset = next(
            d for d in surf.darrays
            if d.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
        )
        coords = np.asarray(pointset.data, dtype=float)  # (n_vertices, 3) world mm

        atlas = nib.load(self.inputs.atlas_file)
        atlas_data = np.asanyarray(atlas.dataobj).astype(int)
        inv = np.linalg.inv(atlas.affine)
        ijk = np.rint(nib.affines.apply_affine(inv, coords)).astype(int)

        shape = np.array(atlas_data.shape)
        in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)
        vertex_labels = np.zeros(len(coords), dtype=int)
        vi = ijk[in_bounds]
        vertex_labels[in_bounds] = atlas_data[vi[:, 0], vi[:, 1], vi[:, 2]]

        out = {'vertex': np.arange(len(coords)), 'parcel_id': vertex_labels}

        if self.inputs.in_file and Path(self.inputs.in_file).exists():
            func = nib.load(self.inputs.in_file)
            data = np.column_stack([d.data for d in func.darrays])  # (n_vertices, n_t)
            rows = []
            for pid in sorted(p for p in np.unique(vertex_labels) if p != 0):
                means = data[vertex_labels == pid].mean(axis=0)
                rows.append({'parcel_id': pid, **{f't{i}': v for i, v in enumerate(means)}})
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(out)

        out_file = Path(runtime.cwd) / 'surface_volume_parcellated.tsv'
        df.to_csv(out_file, sep='\t', index=False)
        self._results['out_file'] = str(out_file)
        return runtime
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_interfaces.py::test_surface_volume_parcellate_assigns_labels -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/gifti.py test/test_interfaces.py
git commit -m "feat(interfaces): add SurfaceVolumeParcellate (vertex-sample a volume atlas)"
```

---

### Task 12: `interfaces/tractography.py` — `StreamlineConnectivity`, `BundleStats`

> **Reworked (review §3).** The first draft's `BundleAtlasIntersect` was the wrong primitive on three counts: it counted streamlines *passing through* parcels (not endpoint connectivity), it looped `density_map` per parcel — O(parcels × streamlines), catastrophic at 400 parcels × millions of streamlines — and it bypassed `trx-rs` entirely. Replace it with `StreamlineConnectivity`: read endpoints once via the `trx-rs` read-only Python bindings (`positions()` + `offsets()`), look up each endpoint's parcel in the (world→voxel) label volume with a small radial search, and accumulate an N×N matrix. This runs **after** the tractogram is warped into atlas space (Strategy B; Task 17). `BundleStats` keeps the dipy length/count summary.
>
> The streamlines are expected in **TRX/TCK** (`.trk` converted on ingest via `TrxConvert`). The `trxrs` binding API is verified (`load().positions()/.offsets()`); if a future version differs, adapt — the algorithm (endpoints → voxel → label → N×N) is the contract. An MRtrix `tck2connectome` fallback is acceptable if the bindings prove insufficient (see review §3c option 3).

**Files:**
- Create: `src/bdt/interfaces/tractography.py`
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_interfaces.py
def test_tractography_importable():
    from bdt.interfaces.tractography import StreamlineConnectivity, BundleStats  # noqa: F401
    assert StreamlineConnectivity is not None
    assert BundleStats is not None


def test_streamline_connectivity_matrix(tmp_path):
    """Two streamlines whose endpoints land in parcels (1,2) and (2,2)."""
    from bdt.interfaces.tractography import StreamlineConnectivity
    import nibabel as nib, numpy as np, pandas as pd
    from dipy.io.streamline import save_tractogram
    from dipy.io.stateful_tractogram import StatefulTractogram, Space

    # atlas: parcel 1 at voxel (0,0,0), parcel 2 at voxel (3,3,3)
    atlas = np.zeros((4, 4, 4), dtype=np.int32)
    atlas[0, 0, 0] = 1
    atlas[3, 3, 3] = 2
    apath = tmp_path / 'atlas.nii.gz'
    ref = nib.Nifti1Image(atlas, np.eye(4))
    nib.save(ref, apath)

    sl = [np.array([[0, 0, 0], [3, 3, 3]], dtype=np.float32),   # 1<->2
          np.array([[3, 3, 3], [3, 3, 3]], dtype=np.float32)]   # 2<->2
    sft = StatefulTractogram(sl, ref, Space.RASMM)
    tpath = tmp_path / 'tracts.trx'
    save_tractogram(sft, str(tpath))   # if .trx unsupported in test env, use .tck

    res = StreamlineConnectivity(tractogram=str(tpath), atlas_file=str(apath)).run()
    # BEP 017 relmat.dense.tsv is header-less (values only); mapping is in nodeindices.tsv
    mat = pd.read_table(res.outputs.out_file, header=None).to_numpy()
    # symmetric 2x2; one 1-2 edge, one 2-2 self edge
    assert mat.shape == (2, 2)
    assert mat[0, 1] == 1 and mat[1, 0] == 1
    assert mat[1, 1] == 1
    # required BEP 017 sidecars are emitted
    assert res.outputs.json_file and res.outputs.nodeindices_file
```

Run: `pytest test/test_interfaces.py::test_tractography_importable -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/tractography.py`**

```python
"""Tractography interfaces: endpoint→parcel connectivity and bundle statistics."""
from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _StreamlineConnectivityInputSpec(BaseInterfaceInputSpec):
    tractogram = File(exists=True, mandatory=True,
                      desc='TRX or TCK tractogram, already in the atlas space')
    atlas_file = File(exists=True, mandatory=True, desc='NIfTI label atlas in the same space')
    atlas_labels = File(exists=False, desc='optional atlas dseg.tsv (index,name) for nodeindices')
    measure = traits.Str('count', usedefault=True,
                         desc='BEP 017 meas- value (count/length/density/denlen)')
    search_radius_mm = traits.Float(2.0, usedefault=True,
                                    desc='radial search for unlabeled endpoints')


class _StreamlineConnectivityOutputSpec(TraitedSpec):
    out_file = File(desc='N×N connectivity matrix → BEP 017 _relmat.dense.tsv')
    json_file = File(desc='REQUIRED BEP 017 _relmat.json sidecar')
    nodeindices_file = File(desc='REQUIRED BEP 017 _nodeindices.tsv (matrix index → atlas ROI)')


class StreamlineConnectivity(SimpleInterface):
    """Assign each streamline's two endpoints to parcels → N×N connectivity matrix.

    Endpoints are read once via the trx-rs read-only bindings; per-endpoint parcel
    lookup is a world→voxel transform into the label volume plus a small radial
    search when the endpoint lands on background. O(n_streamlines), not
    O(n_parcels × n_streamlines).
    """

    input_spec = _StreamlineConnectivityInputSpec
    output_spec = _StreamlineConnectivityOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        import pandas as pd

        atlas = nib.load(self.inputs.atlas_file)
        atlas_data = np.asanyarray(atlas.dataobj).astype(int)
        inv = np.linalg.inv(atlas.affine)
        parcels = sorted(p for p in np.unique(atlas_data) if p != 0)
        index = {p: i for i, p in enumerate(parcels)}
        n = len(parcels)
        mat = np.zeros((n, n), dtype=int)

        # --- read endpoints via trx-rs read-only bindings ------------------
        # positions(): (n_points, 3) world coords; offsets(): per-streamline starts.
        endpoints = _load_endpoints(self.inputs.tractogram)  # (n_streamlines, 2, 3)

        radius_vox = int(round(self.inputs.search_radius_mm))
        for (a, b) in endpoints:
            la = _lookup_label(atlas_data, inv, a, radius_vox)
            lb = _lookup_label(atlas_data, inv, b, radius_vox)
            if la and lb:
                mat[index[la], index[lb]] += 1
                if la != lb:
                    mat[index[lb], index[la]] += 1

        import json

        # BEP 017: dense relmat is stored header-less (n×m values only); the
        # index→ROI mapping lives in the REQUIRED _nodeindices.tsv, and matrix
        # properties in the REQUIRED _relmat.json. The DerivativesDataSink that
        # consumes these renames them to
        #   …_meas-{measure}_relmat.dense.tsv / _relmat.json / _nodeindices.tsv
        out_file = Path(runtime.cwd) / 'relmat.dense.tsv'
        pd.DataFrame(mat).to_csv(out_file, sep='\t', header=False, index=False)

        # node indices: matrix row/col index → atlas parcel id (+ name if dseg.tsv given)
        names = {}
        if self.inputs.atlas_labels and Path(self.inputs.atlas_labels).exists():
            lut = pd.read_table(self.inputs.atlas_labels)
            names = dict(zip(lut['index'], lut.get('name', lut['index'])))
        nodeindices = Path(runtime.cwd) / 'nodeindices.tsv'
        pd.DataFrame({
            'index': range(1, n + 1),               # BIDS index: nonzero, 1-based
            'node_file_index': parcels,
            'name': [names.get(p, f'parcel{p}') for p in parcels],
        }).to_csv(nodeindices, sep='\t', index=False)

        json_file = Path(runtime.cwd) / 'relmat.json'
        json_file.write_text(json.dumps({
            'NodeFiles': str(self.inputs.atlas_file),
            'RelationshipMeasure': self.inputs.measure,
            'Weighted': True,
            'Directed': False,
            'ValidDiagonal': True,
            'StorageFormat': 'Full',
            'Software': 'bdt.interfaces.tractography.StreamlineConnectivity',
        }, indent=2))

        self._results['out_file'] = str(out_file)
        self._results['nodeindices_file'] = str(nodeindices)
        self._results['json_file'] = str(json_file)
        return runtime


def _load_endpoints(path):
    """Return an (n_streamlines, 2, 3) array of world-space endpoints.

    Prefer the trx-rs read-only bindings; fall back to dipy for .tck if needed.
    """
    import numpy as np
    try:
        # Verified trx-rs read-only bindings: package `trxrs`, module `trxrs._core`.
        #   trxrs.load(path) -> TrxFile with .positions() (n_points, 3) world coords
        #   and .offsets() (n_streamlines,) per-streamline start indices, sentinel
        #   already stripped. Coords are in the TRX header's RAS+ (world) space.
        import trxrs
        trx = trxrs.load(str(path))
        positions = np.asarray(trx.positions())            # (n_points, 3), world
        offsets = np.asarray(trx.offsets())                # (n_streamlines,) start idx
        starts = offsets
        ends = np.r_[offsets[1:], len(positions)] - 1
        return np.stack([positions[starts], positions[ends]], axis=1)
    except Exception:
        from dipy.io.streamline import load_tractogram
        sft = load_tractogram(str(path), 'same', bbox_valid_check=False)
        return np.array([[s[0], s[-1]] for s in sft.streamlines])


def _lookup_label(atlas_data, inv_affine, world_pt, radius_vox):
    """Voxel label at a world point, with a small radial search on background."""
    import numpy as np
    import nibabel as nib

    ijk = np.rint(nib.affines.apply_affine(inv_affine, world_pt)).astype(int)
    shape = np.array(atlas_data.shape)
    if np.all((ijk >= 0) & (ijk < shape)):
        lab = int(atlas_data[ijk[0], ijk[1], ijk[2]])
        if lab:
            return lab
    # radial search
    best = 0
    for di in range(-radius_vox, radius_vox + 1):
        for dj in range(-radius_vox, radius_vox + 1):
            for dk in range(-radius_vox, radius_vox + 1):
                p = ijk + (di, dj, dk)
                if np.all((p >= 0) & (p < shape)):
                    lab = int(atlas_data[p[0], p[1], p[2]])
                    if lab:
                        return lab
    return best


class _BundleStatsInputSpec(BaseInterfaceInputSpec):
    tractogram = File(exists=True, mandatory=True, desc='TRX or TCK tractogram file')


class _BundleStatsOutputSpec(TraitedSpec):
    out_file = File(desc='TSV: n_streamlines, mean_length_mm')


class BundleStats(SimpleInterface):
    """Per-tractogram statistics: streamline count and mean length."""

    input_spec = _BundleStatsInputSpec
    output_spec = _BundleStatsOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import pandas as pd
        from dipy.io.streamline import load_tractogram
        from dipy.tracking.utils import length

        tractogram = load_tractogram(self.inputs.tractogram, 'same', bbox_valid_check=False)
        streamlines = list(tractogram.streamlines)
        lengths = list(length(streamlines)) if streamlines else []
        df = pd.DataFrame({
            'n_streamlines': [len(streamlines)],
            'mean_length_mm': [float(np.mean(lengths)) if lengths else 0.0],
        })
        out_file = Path(runtime.cwd) / 'bundle_stats.tsv'
        df.to_csv(out_file, sep='\t', index=False)
        self._results['out_file'] = str(out_file)
        return runtime
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_interfaces.py::test_tractography_importable -v`
Expected: PASS (the full `test_streamline_connectivity_matrix` may need a `.tck` fixture if `.trx` write isn't available in the test env — mark `@pytest.mark.integration` if so)

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/tractography.py test/test_interfaces.py
git commit -m "feat(interfaces): add StreamlineConnectivity (endpoint→parcel) and BundleStats"
```

---

### Task 13: `interfaces/atlas.py` — `AtlasIntersect`, `AtlasUnion`, `AtlasOuterProduct`

> **Bug fixes (review §7).** In the first draft `AtlasIntersect` and `AtlasOuterProduct` were **byte-for-byte identical** (both computed `(d1-1)*n2 + d2`), and `AtlasUnion` **hard-errored on any voxel overlap** (real cortical+subcortical atlases overlap at boundaries). Fixes:
> - **`AtlasIntersect` = restrict:** keep atlas-1's labels only where atlas-2 is also labeled (`result[both] = d1[both]`). Distinct from outer product.
> - **`AtlasOuterProduct` = Cartesian sub-parcels:** `result[both] = (d1-1)*n2 + d2`. Unchanged formula, but now the *only* interface using it.
> - **`AtlasUnion` = precedence, not crash:** combine labels; where they overlap, a `precedence` input decides the winner (default: first atlas). No `ValueError` on overlap.
>
> **BIDS conformance add-ons (design §8 BAT, §14):** `AtlasUnion` and `AtlasOuterProduct` additionally emit an **`out_nodelabels`** TSV (columns `index, SourceAtlasName, SourceAtlasIndex, SourceAtlasLabel`, [BEP 017](#) §4.3) so the offset-relabeling's source provenance is recoverable; the `dseg.tsv` MUST have `index` + `name` columns (merged BIDS `SegmentationLookup`). Also add a small **`WriteAtlasDescription`** interface here that writes the REQUIRED root `atlas-<label>_description.json` (`AtlasName` + `License`); BAT output is a validator error without it. (Wired in Task 21.)

**Files:**
- Create: `src/bdt/interfaces/atlas.py`
- Modify: `test/test_interfaces.py`

- [ ] **Step 1: Write failing tests**

```python
# add to test/test_interfaces.py
def _make_nonoverlap_atlases(tmp_path):
    """Two 4×4×4 atlases with non-overlapping labels."""
    import nibabel as nib
    import numpy as np
    data1 = np.zeros((4, 4, 4), dtype=np.int32)
    data1[:2, :2, :2] = 1   # parcel A
    data2 = np.zeros((4, 4, 4), dtype=np.int32)
    data2[2:, 2:, 2:] = 1   # parcel B (no overlap)
    affine = np.eye(4)
    p1 = tmp_path / 'atlas1.nii.gz'
    p2 = tmp_path / 'atlas2.nii.gz'
    nib.save(nib.Nifti1Image(data1, affine), p1)
    nib.save(nib.Nifti1Image(data2, affine), p2)
    return str(p1), str(p2)


def _make_overlap_atlases(tmp_path):
    """Two 4×4×4 atlases with partially overlapping labels."""
    import nibabel as nib
    import numpy as np
    data1 = np.zeros((4, 4, 4), dtype=np.int32)
    data1[:3, :3, :3] = 1
    data2 = np.zeros((4, 4, 4), dtype=np.int32)
    data2[1:, 1:, 1:] = 2  # label 2; overlaps data1 in [1:3,1:3,1:3]
    affine = np.eye(4)
    p1 = tmp_path / 'ov1.nii.gz'
    p2 = tmp_path / 'ov2.nii.gz'
    nib.save(nib.Nifti1Image(data1, affine), p1)
    nib.save(nib.Nifti1Image(data2, affine), p2)
    return str(p1), str(p2)


def test_atlas_union_non_overlapping(tmp_path):
    from bdt.interfaces.atlas import AtlasUnion
    p1, p2 = _make_nonoverlap_atlases(tmp_path)
    res = AtlasUnion(in_file1=p1, in_file2=p2).run()
    import nibabel as nib, numpy as np
    out = nib.load(res.outputs.out_file)
    data = np.asanyarray(out.dataobj)
    assert set(np.unique(data)) == {0, 1, 2}


def test_atlas_union_overlap_precedence(tmp_path):
    """Overlap must NOT crash; first atlas wins by default."""
    from bdt.interfaces.atlas import AtlasUnion
    p1, p2 = _make_overlap_atlases(tmp_path)
    res = AtlasUnion(in_file1=p1, in_file2=p2).run()   # no ValueError
    import nibabel as nib, numpy as np
    out = np.asanyarray(nib.load(res.outputs.out_file).dataobj)
    # overlap voxel [1,1,1] keeps atlas-1's parcel (label 1), not atlas-2's
    assert out[1, 1, 1] == 1


def test_intersect_and_outer_product_differ(tmp_path):
    """Regression for the byte-identical bug: results must differ."""
    from bdt.interfaces.atlas import AtlasIntersect, AtlasOuterProduct
    import nibabel as nib, numpy as np
    # atlas1: two parcels; atlas2: two parcels; overlapping region spans both crosses
    d1 = np.zeros((4, 4, 4), dtype=np.int32); d1[:, :2, :] = 1; d1[:, 2:, :] = 2
    d2 = np.zeros((4, 4, 4), dtype=np.int32); d2[:2, :, :] = 1; d2[2:, :, :] = 2
    a = tmp_path / 'a.nii.gz'; b = tmp_path / 'b.nii.gz'
    nib.save(nib.Nifti1Image(d1, np.eye(4)), a)
    nib.save(nib.Nifti1Image(d2, np.eye(4)), b)
    inter = np.asanyarray(nib.load(
        AtlasIntersect(in_file1=str(a), in_file2=str(b)).run().outputs.out_file).dataobj)
    outer = np.asanyarray(nib.load(
        AtlasOuterProduct(in_file1=str(a), in_file2=str(b)).run().outputs.out_file).dataobj)
    # intersect retains atlas-1 labels {1,2}; outer product yields up to 4 sub-parcels
    assert set(np.unique(inter)) <= {0, 1, 2}
    assert outer.max() > inter.max()
    assert not np.array_equal(inter, outer)
```

Run: `pytest test/test_interfaces.py::test_atlas_union_non_overlapping -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/atlas.py`**

```python
"""Atlas algebra interfaces for BAT."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _AtlasBinaryInputSpec(BaseInterfaceInputSpec):
    in_file1 = File(exists=True, mandatory=True, desc='First atlas NIfTI label image')
    in_file2 = File(exists=True, mandatory=True, desc='Second atlas NIfTI label image')


class _AtlasUnionInputSpec(_AtlasBinaryInputSpec):
    precedence = traits.Enum('first', 'second', usedefault=True,
                             desc='which atlas wins where labels overlap')


class _AtlasOutputSpec(TraitedSpec):
    out_file = File(desc='Output atlas NIfTI label image')
    out_tsv = File(desc='TSV label table for the output atlas')


def _write_label_tsv(result, cwd, stem):
    import pandas as pd
    out_tsv = Path(cwd) / f'{stem}.tsv'
    labels = sorted(i for i in np.unique(result) if i != 0)
    pd.DataFrame({'index': labels, 'name': [f'parcel{i}' for i in labels]}).to_csv(
        out_tsv, sep='\t', index=False
    )
    return str(out_tsv)


class AtlasUnion(SimpleInterface):
    """Combine two atlas label images; resolve overlap by a precedence rule.

    Labels from in_file2 are offset by max(in_file1) so all labels stay unique.
    Where both atlases label a voxel, ``precedence`` decides the winner (default
    'first'). Overlap is expected for real cortical+subcortical atlases, so this
    does NOT raise.
    """

    input_spec = _AtlasUnionInputSpec
    output_spec = _AtlasOutputSpec

    def _run_interface(self, runtime):
        img1 = nib.load(self.inputs.in_file1)
        img2 = nib.load(self.inputs.in_file2)
        d1 = np.asanyarray(img1.dataobj).astype(np.int32)
        d2 = np.asanyarray(img2.dataobj).astype(np.int32)

        offset = int(d1.max())
        d2_offset = np.where(d2 > 0, d2 + offset, 0)

        union = np.zeros_like(d1)
        if self.inputs.precedence == 'first':
            # start from atlas-2, then overlay atlas-1 so it wins on overlap
            union = np.where(d2_offset > 0, d2_offset, union)
            union = np.where(d1 > 0, d1, union)
        else:
            union = np.where(d1 > 0, d1, union)
            union = np.where(d2_offset > 0, d2_offset, union)

        out_file = Path(runtime.cwd) / 'atlas_union.nii.gz'
        nib.save(nib.Nifti1Image(union, img1.affine, img1.header), out_file)
        self._results['out_file'] = str(out_file)
        self._results['out_tsv'] = _write_label_tsv(union, runtime.cwd, 'atlas_union')
        return runtime


class AtlasIntersect(SimpleInterface):
    """Restrict atlas-1 to voxels also labeled in atlas-2.

    Output keeps atlas-1's parcel labels, but only where atlas-2 is non-zero.
    This is a *restriction*, distinct from the Cartesian outer product.
    """

    input_spec = _AtlasBinaryInputSpec
    output_spec = _AtlasOutputSpec

    def _run_interface(self, runtime):
        img1 = nib.load(self.inputs.in_file1)
        img2 = nib.load(self.inputs.in_file2)
        d1 = np.asanyarray(img1.dataobj).astype(np.int32)
        d2 = np.asanyarray(img2.dataobj).astype(np.int32)

        both = (d1 > 0) & (d2 > 0)
        result = np.zeros_like(d1)
        result[both] = d1[both]   # restriction: retain atlas-1 labels

        out_file = Path(runtime.cwd) / 'atlas_intersect.nii.gz'
        nib.save(nib.Nifti1Image(result, img1.affine, img1.header), out_file)
        self._results['out_file'] = str(out_file)
        self._results['out_tsv'] = _write_label_tsv(result, runtime.cwd, 'atlas_intersect')
        return runtime


class AtlasOuterProduct(SimpleInterface):
    """Cartesian product of two atlas label sets.

    On the overlap, every parcel pair (i, j) gets a unique combined label
    ``(i-1)*n2 + j`` — sub-parcels from the cross of both parcellations.
    This is the *only* interface using the Cartesian formula.
    """

    input_spec = _AtlasBinaryInputSpec
    output_spec = _AtlasOutputSpec

    def _run_interface(self, runtime):
        img1 = nib.load(self.inputs.in_file1)
        img2 = nib.load(self.inputs.in_file2)
        d1 = np.asanyarray(img1.dataobj).astype(np.int32)
        d2 = np.asanyarray(img2.dataobj).astype(np.int32)

        n2 = int(d2.max())
        both = (d1 > 0) & (d2 > 0)
        result = np.zeros_like(d1)
        result[both] = (d1[both] - 1) * n2 + d2[both]

        out_file = Path(runtime.cwd) / 'atlas_outer_product.nii.gz'
        nib.save(nib.Nifti1Image(result, img1.affine, img1.header), out_file)
        self._results['out_file'] = str(out_file)
        self._results['out_tsv'] = _write_label_tsv(result, runtime.cwd, 'atlas_outer_product')
        return runtime
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_interfaces.py::test_atlas_union_non_overlapping test/test_interfaces.py::test_atlas_intersect test/test_interfaces.py::test_atlas_outer_product -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/interfaces/atlas.py test/test_interfaces.py
git commit -m "feat(interfaces): add AtlasIntersect, AtlasUnion, AtlasOuterProduct for BAT"
```

---

## Phase 4 — BDT Workflows

### Task 13b: `init_load_atlases_wf` — the grid atlas-warping workflow

> **New (review §7).** `init_load_atlases_wf` is the heart of volumetric atlas warping and is imported by Tasks 15–16, but **no task created it** (the original "known gaps" missed it). Define it in `workflows/parcellation.py`, modeled on xcp_d's `init_load_atlases_wf`. It uses `chain_for_image_resample` (Task 6, **grid/pull** direction) and the fixed `ApplyTransforms` (Task 7, `GenericLabel`).

**Files:**
- Modify: `src/bdt/workflows/parcellation.py`
- Create/extend: `test/test_bdt_workflows.py`

- [ ] **Step 1: Write failing test**

```python
# test/test_bdt_workflows.py
def test_load_atlases_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.parcellation import init_load_atlases_wf
    with mock_config():
        wf = init_load_atlases_wf()
    assert wf is not None
    names = wf.list_node_names()
    assert any('apply_transforms' in n or 'warp' in n for n in names)
```

Run: `pytest test/test_bdt_workflows.py::test_load_atlases_wf_builds -v`
Expected: ImportError — `init_load_atlases_wf` does not exist

- [ ] **Step 2: Append `init_load_atlases_wf` to `src/bdt/workflows/parcellation.py`**

Build the transform graph at workflow-build time, resolve each atlas's chain with the **image** query, and warp with `ApplyTransforms` (`GenericLabel`). Atlas space and data space are read from BIDS entities / image headers; mirror xcp_d's structure (collect → warp → output a list of atlases in data space). The node that resolves chains should call:

```python
from bdt.utils.transforms import build_transform_graph, chain_for_image_resample
graph = build_transform_graph(config.execution.datasets)
files, tool, invert = chain_for_image_resample(graph, atlas_space, data_space)
# tool == 'ApplyTransforms'; feed `files` (and invert flags) to the ApplyTransforms node
```

> Keep the inputs/outputs compatible with the way Tasks 15–16 wire it:
> `inputnode.name_source`, `inputnode.bold_file` (the data file, source of the target grid) →
> `outputnode.atlas_files` (list of atlases resampled into data space). If xcp_d's signature differs, prefer matching xcp_d and update Tasks 15–16 to match.

- [ ] **Step 3: Run test**

Run: `pytest test/test_bdt_workflows.py::test_load_atlases_wf_builds -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/workflows/parcellation.py test/test_bdt_workflows.py
git commit -m "feat(workflows): add init_load_atlases_wf (grid atlas warping)"
```

---

### Task 14: GIFTI parcellation workflows (surface atlas + cross-mesh + volume atlas)

> **Reworked (review §4).** Three surface paths, none of which is the old hand-rolled `init_parcellate_gifti_wf`:
> 1. **surface data + same-mesh surface atlas:** reuse the existing `init_parcellate_cifti_wf` (wb_command).
> 2. **surface data + different-mesh surface atlas:** prepend `init_label_resample_wf` (`wb_command -label-resample`, needs registration spheres) then parcellate as in (1).
> 3. **surface data + volume atlas:** `init_surface_volume_parcellate_wf` — `GiftiTransform` (warp `*.surf.gii` into the atlas grid via `chain_for_point_warp`) → `SurfaceVolumeParcellate`.

**Files:**
- Modify: `src/bdt/workflows/parcellation.py`
- Create: `test/test_bdt_workflows.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_bdt_workflows.py
def test_surface_volume_parcellate_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.parcellation import init_surface_volume_parcellate_wf
    with mock_config():
        wf = init_surface_volume_parcellate_wf()
    assert wf is not None
    names = wf.list_node_names()
    assert any('gifti_transform' in n for n in names)
    assert any('surface_volume' in n or 'parcellate' in n for n in names)


def test_label_resample_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.parcellation import init_label_resample_wf
    with mock_config():
        wf = init_label_resample_wf()
    assert wf is not None
```

Run: `pytest test/test_bdt_workflows.py -k 'surface_volume or label_resample' -v`
Expected: ImportError

- [ ] **Step 2: Append to `src/bdt/workflows/parcellation.py`**

```python
def init_label_resample_wf(name='label_resample_wf'):
    """Resample a surface .label.gii atlas onto the data's mesh.

    Wraps `wb_command -label-resample` (BARYCENTRIC) using current- and
    new-sphere registration meshes. Used when the atlas mesh != data mesh.

    Inputs
    ------
    label_file : str          # atlas .label.gii (on atlas mesh)
    current_sphere : str      # atlas-mesh registration sphere
    new_sphere : str          # data-mesh registration sphere

    Outputs
    -------
    resampled_label : str     # atlas .label.gii on the data mesh
    """
    from nipype.interfaces import utility as niu
    from bdt.interfaces.workbench import LabelResample  # wrap wb_command -label-resample

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['label_file', 'current_sphere', 'new_sphere']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['resampled_label']), name='outputnode')
    resample = pe.Node(LabelResample(method='BARYCENTRIC'), name='label_resample')
    workflow.connect([
        (inputnode, resample, [
            ('label_file', 'label_in'),
            ('current_sphere', 'current_sphere'),
            ('new_sphere', 'new_sphere'),
        ]),
        (resample, outputnode, [('label_out', 'resampled_label')]),
    ])
    return workflow


def init_surface_volume_parcellate_wf(name='surface_volume_parcellate_wf'):
    """Parcellate surface data with a VOLUME atlas via giftirs vertex-warp.

    GiftiTransform warps the subject's *.surf.gii into the volume atlas's grid
    (point-warp direction; see chain_for_point_warp), then SurfaceVolumeParcellate
    samples the label volume at each vertex.

    Inputs
    ------
    surf_file : str       # subject *.surf.gii (e.g. midthickness in T1w)
    transform : str       # from-{atlas}_to-{surf} warp (point-warp convention)
    atlas_file : str      # NIfTI label volume
    in_file : str         # optional *.func.gii/.shape.gii to parcel-average

    Outputs
    -------
    timeseries : str
    """
    from nipype.interfaces import utility as niu
    from bdt.interfaces.rust import GiftiTransform
    from bdt.interfaces.gifti import SurfaceVolumeParcellate

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['surf_file', 'transform', 'atlas_file', 'in_file']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    gifti_transform = pe.Node(GiftiTransform(), name='gifti_transform')
    parcellate = pe.Node(SurfaceVolumeParcellate(), name='surface_volume_parcellate')

    workflow.connect([
        (inputnode, gifti_transform, [('surf_file', 'in_file'), ('transform', 'transform')]),
        (gifti_transform, parcellate, [('out_file', 'warped_surface')]),
        (inputnode, parcellate, [('atlas_file', 'atlas_file'), ('in_file', 'in_file')]),
        (parcellate, outputnode, [('out_file', 'timeseries')]),
    ])
    return workflow
```

> `LabelResample` is a thin `wb_command -label-resample` wrapper; add it to `interfaces/workbench.py` (or reuse an existing niworkflows/xcp_d wrapper if one is already imported). Same-mesh surface-atlas parcellation reuses `init_parcellate_cifti_wf` — no new workflow.

- [ ] **Step 3: Run tests**

Run: `pytest test/test_bdt_workflows.py -k 'surface_volume or label_resample' -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/workflows/parcellation.py src/bdt/interfaces/workbench.py test/test_bdt_workflows.py
git commit -m "feat(workflows): add surface-volume parcellation and label-resample workflows"
```

---

### Task 15: `workflows/bdt/timeseries.py` — `init_timeseries_run_wf`

**Files:**
- Create: `src/bdt/workflows/bdt/timeseries.py`
- Modify: `test/test_bdt_workflows.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_bdt_workflows.py
def test_timeseries_run_wf_builds_nifti():
    from tests.tests import mock_config
    from bdt.workflows.bdt.timeseries import init_timeseries_run_wf
    atlas_filters = [{'atlas': 'HCPMMP1'}]
    with mock_config():
        wf = init_timeseries_run_wf(
            source_file='sub-01_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
            atlas_filters=atlas_filters,
            operations=['parcellate_timeseries'],
        )
    assert wf is not None
```

Run: `pytest test/test_bdt_workflows.py::test_timeseries_run_wf_builds_nifti -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/workflows/bdt/timeseries.py`**

```python
"""BDT workflow for parcellating 4D time series (BOLD, ASL 4D)."""
from __future__ import annotations

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from bdt import config

LOGGER = logging.getLogger('nipype.workflow')

_TIMESERIES_EXTENSIONS = {'.nii.gz', '.nii', '.func.gii', '.dtseries.nii'}


def _detect_format(source_file: str) -> str:
    """Return 'nifti', 'gifti', or 'cifti' based on file extension."""
    from pathlib import Path
    name = Path(source_file).name
    if name.endswith('.dtseries.nii') or name.endswith('.dscalar.nii'):
        return 'cifti'
    if name.endswith('.func.gii') or name.endswith('.shape.gii'):
        return 'gifti'
    return 'nifti'


def init_timeseries_run_wf(
    source_file: str,
    atlas_filters: list[dict],
    operations: list[str],
    name: str | None = None,
):
    """Build a workflow to parcellate a 4D time series with one or more atlases.

    Parameters
    ----------
    source_file : str
        Path to the derivative BOLD or ASL 4D file (NIfTI, GIFTI, or CIFTI).
    atlas_filters : list of dict
        Entity-filter dicts; each passed to ``collect_atlases``.
    operations : list of str
        Subset of ``['parcellate_timeseries', 'functional_connectivity']``.
    name : str, optional
        Workflow name.

    Inputs
    ------
    source_file : str
    atlas_files : list[str]

    Outputs
    -------
    timeseries : list[str]  (one TSV per atlas)
    correlations : list[str]  (one TSV per atlas, if functional_connectivity requested)
    """
    from bdt.utils.bids import extract_entities
    from bdt.utils.utils import _get_wf_name

    file_format = _detect_format(source_file)
    if name is None:
        name = _get_wf_name(source_file, 'timeseries_run')

    workflow = Workflow(name=name)
    workflow.__desc__ = ''

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'atlas_files']),
        name='inputnode',
    )
    inputnode.inputs.source_file = source_file

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['timeseries', 'correlations']),
        name='outputnode',
    )

    if file_format == 'nifti':
        from bdt.workflows.parcellation import init_load_atlases_wf, init_parcellate_nifti_wf

        load_atlases_wf = init_load_atlases_wf()
        parcellate_wf = init_parcellate_nifti_wf()

        workflow.connect([
            (inputnode, load_atlases_wf, [
                ('source_file', 'inputnode.name_source'),
                ('source_file', 'inputnode.bold_file'),
            ]),
            (load_atlases_wf, parcellate_wf, [
                ('outputnode.atlas_files', 'inputnode.atlas_files'),
            ]),
            (inputnode, parcellate_wf, [('source_file', 'inputnode.bold_file')]),
            (parcellate_wf, outputnode, [('outputnode.timeseries', 'timeseries')]),
        ])

        if 'functional_connectivity' in operations:
            from bdt.workflows.connectivity import init_functional_connectivity_nifti_wf
            fc_wf = init_functional_connectivity_nifti_wf(
                mem_gb=2.0,
                has_multiple_runs=False,
            )
            workflow.connect([
                (parcellate_wf, fc_wf, [
                    ('outputnode.timeseries', 'inputnode.timeseries'),
                ]),
                (fc_wf, outputnode, [('outputnode.correlations', 'correlations')]),
            ])

    elif file_format == 'gifti':
        # Route by ATLAS geometry (review §4):
        #   surface atlas (.label.gii), same mesh  → init_parcellate_cifti_wf (wb_command)
        #   surface atlas, different mesh           → init_label_resample_wf first
        #   volume atlas (.nii.gz)                  → init_surface_volume_parcellate_wf
        # Atlas geometry is detected from the matched atlas file extension at build time.
        from bdt.workflows.parcellation import (
            init_parcellate_cifti_wf,
            init_surface_volume_parcellate_wf,
        )

        # (Atlas selection happens via collect_atlases; assume the first atlas's
        #  extension determines the branch. Real impl loops over atlases.)
        parcellate_wf = init_parcellate_cifti_wf()  # surface-atlas default
        workflow.connect([
            (inputnode, parcellate_wf, [
                ('source_file', 'inputnode.bold_file'),
                ('atlas_files', 'inputnode.atlas_files'),
            ]),
            (parcellate_wf, outputnode, [('outputnode.timeseries', 'timeseries')]),
        ])
        # For a volumetric atlas, swap in init_surface_volume_parcellate_wf (needs the
        # subject *.surf.gii and a chain_for_point_warp transform; see Task 14).

    else:  # cifti
        from bdt.workflows.parcellation import init_parcellate_cifti_wf

        parcellate_wf = init_parcellate_cifti_wf()
        workflow.connect([
            (inputnode, parcellate_wf, [
                ('source_file', 'inputnode.bold_file'),
                ('atlas_files', 'inputnode.atlas_files'),
            ]),
            (parcellate_wf, outputnode, [('outputnode.timeseries', 'timeseries')]),
        ])

    return workflow
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_bdt_workflows.py::test_timeseries_run_wf_builds_nifti -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/workflows/bdt/timeseries.py test/test_bdt_workflows.py
git commit -m "feat(workflows): add init_timeseries_run_wf for BDT"
```

---

### Task 16: `workflows/bdt/scalar.py` — `init_scalar_run_wf`

**Files:**
- Create: `src/bdt/workflows/bdt/scalar.py`
- Modify: `test/test_bdt_workflows.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_bdt_workflows.py
def test_scalar_run_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.bdt.scalar import init_scalar_run_wf
    with mock_config():
        wf = init_scalar_run_wf(
            source_file='sub-01_space-MNI152NLin6Asym_res-2_desc-mean_cbf.nii.gz',
            atlas_filters=[{'atlas': 'HCPMMP1'}],
            operations=['parcellate_scalar'],
        )
    assert wf is not None
```

Run: `pytest test/test_bdt_workflows.py::test_scalar_run_wf_builds -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/workflows/bdt/scalar.py`**

```python
"""BDT workflow for parcellating 3D scalar maps (CBF, FA, thickness, etc.)."""
from __future__ import annotations

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from bdt import config
from bdt.workflows.bdt.timeseries import _detect_format

LOGGER = logging.getLogger('nipype.workflow')


def init_scalar_run_wf(
    source_file: str,
    atlas_filters: list[dict],
    operations: list[str],
    name: str | None = None,
):
    """Build a workflow to extract parcel-mean values from a 3D scalar map.

    Parameters
    ----------
    source_file : str
        Path to the derivative scalar NIfTI/GIFTI/CIFTI file.
    atlas_filters : list of dict
        Entity-filter dicts; each passed to ``collect_atlases``.
    operations : list of str
        Must contain ``'parcellate_scalar'``.
    name : str, optional
        Workflow name.

    Inputs
    ------
    source_file : str
    atlas_files : list[str]

    Outputs
    -------
    scalars : list[str]  (one TSV per atlas with per-parcel mean values)
    """
    from bdt.utils.utils import _get_wf_name

    file_format = _detect_format(source_file)
    if name is None:
        name = _get_wf_name(source_file, 'scalar_run')

    workflow = Workflow(name=name)
    workflow.__desc__ = ''

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'atlas_files']),
        name='inputnode',
    )
    inputnode.inputs.source_file = source_file

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['scalars']),
        name='outputnode',
    )

    if file_format == 'nifti':
        from bdt.workflows.parcellation import init_load_atlases_wf, init_parcellate_nifti_wf

        load_atlases_wf = init_load_atlases_wf()
        parcellate_wf = init_parcellate_nifti_wf()

        workflow.connect([
            (inputnode, load_atlases_wf, [
                ('source_file', 'inputnode.name_source'),
                ('source_file', 'inputnode.bold_file'),
            ]),
            (load_atlases_wf, parcellate_wf, [
                ('outputnode.atlas_files', 'inputnode.atlas_files'),
            ]),
            (inputnode, parcellate_wf, [('source_file', 'inputnode.bold_file')]),
            (parcellate_wf, outputnode, [('outputnode.timeseries', 'scalars')]),
        ])

    elif file_format == 'gifti':
        # Route by atlas geometry (see Task 15 / review §4). Surface atlas →
        # init_parcellate_cifti_wf; volume atlas → init_surface_volume_parcellate_wf.
        from bdt.workflows.parcellation import init_parcellate_cifti_wf

        parcellate_wf = init_parcellate_cifti_wf()
        workflow.connect([
            (inputnode, parcellate_wf, [
                ('source_file', 'inputnode.bold_file'),
                ('atlas_files', 'inputnode.atlas_files'),
            ]),
            (parcellate_wf, outputnode, [('outputnode.timeseries', 'scalars')]),
        ])

    else:  # cifti (dscalar.nii)
        from bdt.workflows.parcellation import init_parcellate_cifti_wf

        parcellate_wf = init_parcellate_cifti_wf()
        workflow.connect([
            (inputnode, parcellate_wf, [
                ('source_file', 'inputnode.bold_file'),
                ('atlas_files', 'inputnode.atlas_files'),
            ]),
            (parcellate_wf, outputnode, [('outputnode.timeseries', 'scalars')]),
        ])

    return workflow
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_bdt_workflows.py::test_scalar_run_wf_builds -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bdt/workflows/bdt/scalar.py test/test_bdt_workflows.py
git commit -m "feat(workflows): add init_scalar_run_wf for BDT scalar maps"
```

---

### Task 17: `workflows/bdt/streamlines.py` — `init_streamlines_run_wf` (Strategy B)

> **Reworked (review §3, Decision 1 = Strategy B).** Streamlines are warped **into the atlas's space** (lossless point warp), not the atlas into streamline space. Pipeline: `TrxConvert` (if `.trk`) → `TrxTransform` (warp tracts→atlas, using `chain_for_point_warp(streamline_space → atlas_space)`) → `init_streamline_connectivity_wf` (endpoint→parcel N×N) and `BundleStats`. The atlas is a standard MNI parcellation (e.g. Schaefer400), not "already in bundle space." Tractography I/O is TRX/TCK.

**Files:**
- Create: `src/bdt/workflows/bdt/streamlines.py`
- Modify: `src/bdt/workflows/connectivity.py` (add `init_streamline_connectivity_wf`)
- Modify: `test/test_bdt_workflows.py`

- [ ] **Step 1: Write failing test**

```python
# add to test/test_bdt_workflows.py
def test_streamlines_run_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.bdt.streamlines import init_streamlines_run_wf
    with mock_config():
        wf = init_streamlines_run_wf(
            # BEP 046 suffix `tractogram`; .trk still triggers the convert→.trx ingest
            source_file='sub-01_space-MNI152NLin2009cAsym_tract-wholebrain_track-ifod_tractogram.trk',
            atlas_filters=[{'atlas': 'Schaefer2018', 'seg': '17networks', 'scale': '400'}],
            operations=['streamline_connectivity', 'bundle_stats'],
        )
    assert wf is not None
    names = wf.list_node_names()
    assert any('convert' in n for n in names)      # .trk ingest
    assert any('transform' in n for n in names)    # warp tracts→atlas
```

Run: `pytest test/test_bdt_workflows.py::test_streamlines_run_wf_builds -v`
Expected: ImportError

- [ ] **Step 2: Add `init_streamline_connectivity_wf` to `src/bdt/workflows/connectivity.py`**

```python
def init_streamline_connectivity_wf(name='streamline_connectivity_wf'):
    """Endpoint→parcel connectivity for a tractogram already in atlas space.

    Inputs
    ------
    tractogram : str   (TRX/TCK, in atlas space)
    atlas_file : str   (NIfTI label atlas, same space)

    Outputs
    -------
    relmat : str        (N×N matrix → BEP 017 _relmat.dense.tsv, header-less)
    relmat_json : str   (REQUIRED BEP 017 _relmat.json sidecar)
    nodeindices : str   (REQUIRED BEP 017 _nodeindices.tsv)
    bundle_stats : str  (per-tractogram count/length summary — NOT a relmat)
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from bdt.interfaces.tractography import StreamlineConnectivity, BundleStats

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['tractogram', 'atlas_file', 'atlas_labels']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['relmat', 'relmat_json', 'nodeindices', 'bundle_stats']),
        name='outputnode')

    connectivity = pe.Node(StreamlineConnectivity(), name='connectivity')
    stats = pe.Node(BundleStats(), name='stats')
    workflow.connect([
        (inputnode, connectivity, [
            ('tractogram', 'tractogram'),
            ('atlas_file', 'atlas_file'),
            ('atlas_labels', 'atlas_labels'),
        ]),
        (inputnode, stats, [('tractogram', 'tractogram')]),
        (connectivity, outputnode, [
            ('out_file', 'relmat'),
            ('json_file', 'relmat_json'),
            ('nodeindices_file', 'nodeindices'),
        ]),
        (stats, outputnode, [('out_file', 'bundle_stats')]),
    ])
    return workflow
```

> **Output naming (BEP 017).** The DerivativesDataSink wiring these outputs (in `init_streamlines_run_wf` / `init_single_subject_wf`) MUST name them with the `meas-<label>` entity and the `relmat.dense`/`relmat`/`nodeindices` suffixes, in canonical order `space- → atlas- → seg- → scale- → meas-`: e.g. `…_atlas-Schaefer2018_seg-17networks_scale-400_meas-count_relmat.dense.tsv` (+ `_relmat.json`, `_nodeindices.tsv`). Pass the atlas's `dseg.tsv` as `atlas_labels` so node names propagate. See design §8 (Streamlines) and §14.

- [ ] **Step 3: Create `src/bdt/workflows/bdt/streamlines.py`**

```python
"""BDT streamline workflow (Strategy B: warp tracts into atlas space)."""
from __future__ import annotations

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

LOGGER = logging.getLogger('nipype.workflow')


def init_streamlines_run_wf(
    source_file: str,
    atlas_filters: list[dict],
    operations: list[str],
    name: str | None = None,
):
    """Warp a tractogram into atlas space, then compute endpoint connectivity.

    Steps
    -----
    1. TrxConvert       — .trk → .trx on ingest (trx-rs cannot write .trk)
    2. TrxTransform     — warp tracts streamline_space → atlas_space, using
                          chain_for_point_warp (the *reverse*-named warp file)
    3. init_streamline_connectivity_wf — endpoint→parcel N×N + bundle stats

    The atlas stays in its own (e.g. MNI) space; the data moves to it.
    """
    from pathlib import Path
    from bdt.utils.utils import _get_wf_name
    from bdt.interfaces.rust import TrxConvert, TrxTransform
    from bdt.workflows.connectivity import init_streamline_connectivity_wf

    if name is None:
        name = _get_wf_name(source_file, 'streamlines_run')

    workflow = Workflow(name=name)
    workflow.__desc__ = ''

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'atlas_files', 'point_warp']),
        name='inputnode',
    )
    inputnode.inputs.source_file = source_file
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['relmat', 'bundle_stats']), name='outputnode')

    # 1. convert .trk → .trx on ingest (no-op pass-through if already .trx/.tck)
    convert = pe.Node(TrxConvert(), name='trx_convert')

    # 2. warp tracts into atlas space; the transform file comes from
    #    chain_for_point_warp(streamline_space → atlas_space) resolved at build time
    #    and injected via inputnode.point_warp (see init_single_subject_wf / Task 18).
    transform = pe.Node(TrxTransform(), name='trx_transform')

    # 3. endpoint connectivity in atlas space
    conn_wf = init_streamline_connectivity_wf()

    workflow.connect([
        (inputnode, convert, [('source_file', 'in_file')]),
        (convert, transform, [('out_file', 'in_file')]),
        (inputnode, transform, [('point_warp', 'transform')]),
        (transform, conn_wf, [('out_file', 'inputnode.tractogram')]),
        (inputnode, conn_wf, [('atlas_files', 'inputnode.atlas_file')]),
        (conn_wf, outputnode, [
            ('outputnode.relmat', 'relmat'),
            ('outputnode.bundle_stats', 'bundle_stats'),
        ]),
    ])
    return workflow
```

> The `point_warp` transform must be resolved with `chain_for_point_warp` (Task 6) at build time, where streamline space is read from the tractogram's BIDS `space-` entity and atlas space from the atlas's `tpl-`/`space-` entity. If only the wrong-direction warp exists, that call raises (review §5 / design §10.4) — surface the error at build time rather than producing mirrored tracts.

- [ ] **Step 4: Run test**

Run: `pytest test/test_bdt_workflows.py::test_streamlines_run_wf_builds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bdt/workflows/bdt/streamlines.py src/bdt/workflows/connectivity.py test/test_bdt_workflows.py
git commit -m "feat(workflows): add Strategy-B streamline workflow (warp tracts→atlas + endpoint connectivity)"
```

---

### Task 17b: `workflows/bdt/diffusion.py` + odx interfaces (`interfaces/odx.py`)

> **New (Decision 3 = odx-rs in scope for v1).** Parcellate SH/ODF/fixel diffusion models. `odx transform` resamples + SH-reorients into the target space (via the odx PyO3 bindings or the `OdxTransform` CLI wrapper), then `FixelParcellate`/`OdfParcellate` summarize per parcel. Fixel→parcel is **not** a plain mean (respect fixel cardinality/direction); ODF parcellation summarizes SH coefficients per parcel.

**Files:**
- Create: `src/bdt/interfaces/odx.py`
- Create: `src/bdt/workflows/bdt/diffusion.py`
- Add `init_parcellate_fixel_wf` / `init_parcellate_odf_wf` to `workflows/parcellation.py`
- Modify: `test/test_interfaces.py`, `test/test_bdt_workflows.py`

- [ ] **Step 1: Write failing tests**

```python
# add to test/test_interfaces.py
def test_odx_interfaces_importable():
    from bdt.interfaces.odx import FixelParcellate, OdfParcellate  # noqa: F401
    assert FixelParcellate is not None and OdfParcellate is not None

# add to test/test_bdt_workflows.py
def test_diffusion_model_run_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.bdt.diffusion import init_diffusion_model_run_wf
    with mock_config():
        wf = init_diffusion_model_run_wf(
            # BEP 016 diffusion model: suffix `dwimap` + model-/param- (not `fod`)
            source_file='sub-01_space-MNI152NLin6Asym_model-csd_param-wm_dwimap.nii.gz',
            atlas_filters=[{'atlas': 'HCPMMP1'}],
            operations=['parcellate_odf'],
        )
    assert wf is not None
```

Run: `pytest test/test_interfaces.py::test_odx_interfaces_importable -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/interfaces/odx.py`**

Use the `odx` PyO3 bindings to read SH data; reduce per parcel. The reader API is **verified** against `odx-rs/python/src/lib.rs`: the input is an **`.odx` container** (not a NIfTI), and `PyOdx` is *compact/masked* — one row per in-mask voxel. Relevant accessors: `odx.load(path)`, `.sh_names()`, `.sh(name)` → `(nb_voxels, n_coeffs)`, `.compact_to_ijk()` → `(nb_voxels, 3)` voxel indices, `.affine()` → 4×4 voxel→world, `.nb_voxels()`. The contract is: `.odx` model already resampled into the atlas grid (via `odx transform`) + label volume on that grid → per-parcel mean-SH TSV.

```python
"""odx-rs based diffusion-model parcellation interfaces."""
from __future__ import annotations

from pathlib import Path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, File, SimpleInterface, TraitedSpec, traits,
)


class _OdfParcellateInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='.odx SH/ODF container, already resampled into the atlas grid')
    atlas_file = File(exists=True, mandatory=True, desc='NIfTI label atlas, same grid as the .odx')


class _OdfParcellateOutputSpec(TraitedSpec):
    out_file = File(desc='TSV: per-parcel ODF/SH summary')


class OdfParcellate(SimpleInterface):
    """Summarize SH coefficients per atlas parcel (mean SH vector per parcel).

    Reads the compact .odx via the odx PyO3 bindings and indexes each in-mask
    voxel's parcel from the label volume by its (i,j,k). Assumes the .odx grid
    matches the atlas grid (guaranteed by the preceding `odx transform`).
    """
    input_spec = _OdfParcellateInputSpec
    output_spec = _OdfParcellateOutputSpec

    def _run_interface(self, runtime):
        import numpy as np, nibabel as nib, pandas as pd
        import odx  # PyO3 bindings (module odx._odx)

        model = odx.load(self.inputs.in_file)
        ijk = np.asarray(model.compact_to_ijk()).astype(int)     # (nb_voxels, 3)
        sh = np.asarray(model.sh(model.sh_names()[0]))           # (nb_voxels, n_coeffs)

        atlas = np.asanyarray(nib.load(self.inputs.atlas_file).dataobj).astype(int)
        vox_labels = atlas[ijk[:, 0], ijk[:, 1], ijk[:, 2]]      # parcel per compact voxel

        rows = []
        for pid in sorted(p for p in np.unique(vox_labels) if p != 0):
            coeffs = sh[vox_labels == pid].mean(axis=0)
            rows.append({'parcel_id': pid, **{f'sh{i}': c for i, c in enumerate(coeffs)}})
        out_file = Path(runtime.cwd) / 'odf_parcellated.tsv'
        pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)
        self._results['out_file'] = str(out_file)
        return runtime


class _FixelParcellateInputSpec(BaseInterfaceInputSpec):
    fixel_dir = traits.Directory(exists=True, mandatory=True, desc='MRtrix fixel directory')
    atlas_file = File(exists=True, mandatory=True, desc='NIfTI label atlas, same grid')
    metric = traits.Str('fd', usedefault=True, desc='fixel metric (fd/fc/fdc)')


class _FixelParcellateOutputSpec(TraitedSpec):
    out_file = File(desc='TSV: per-parcel fixel summary')


class FixelParcellate(SimpleInterface):
    """Per-parcel fixel summary. NOT a plain voxel mean — aggregate over fixels
    within each parcel honoring fixel cardinality (multiple fixels per voxel)."""
    input_spec = _FixelParcellateInputSpec
    output_spec = _FixelParcellateOutputSpec

    def _run_interface(self, runtime):
        # Read the fixel index/data via odx (or MRtrix fixel format); map each
        # fixel to its voxel's parcel and reduce the chosen metric per parcel.
        raise NotImplementedError(
            'Implement against odx-rs fixel reader; see design §8 (fixel→parcel '
            'is not a simple mean).'
        )
```

> `FixelParcellate` is intentionally stubbed with the correct contract and a `NotImplementedError` — fixel aggregation needs the odx fixel reader and a deliberate reduction. `OdfParcellate` (SH mean per parcel) is the v1 default; ship it first and add fixel support once the odx fixel API is confirmed.

- [ ] **Step 3: Create `init_parcellate_{fixel,odf}_wf` and `init_diffusion_model_run_wf`**

`init_diffusion_model_run_wf`: `OdxConvert` (ingest the FOD `.nii.gz` / MRtrix `.mif` / DSI Studio / pyAFQ aodf source into an `.odx` container — skip if the input is already `.odx`) → `OdxTransform` (resample + SH-reorient the `.odx` into the **atlas grid**, using `chain_for_image_resample` — the *image/same-named* h5, **not** `chain_for_point_warp`; `--mode mrtrix` for the v1 SH path) → `init_parcellate_odf_wf` (label volume already in that grid via `init_load_atlases_wf`, or use the atlas's native grid as the transform target so no atlas warp is needed) → `init_parcellate_fixel_wf` is the deferred fixel path (needs `--mode ants` with both h5 directions; see design §7.4). Mirror the structure of `init_scalar_run_wf`.

> **Direction reminder:** unlike streamlines/surfaces, the diffusion-model transform is a *grid resample* and uses the **image** query (`chain_for_image_resample`, same-named `from-{source}_to-{target}` h5). Routing it through `chain_for_point_warp` would fetch the wrong-direction h5 and silently mis-resample. This is the one geometry where "warp the data, not the atlas" applies on a grid — because SH reorientation is intrinsic to the model and can't be done by warping a label image.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_interfaces.py::test_odx_interfaces_importable test/test_bdt_workflows.py::test_diffusion_model_run_wf_builds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bdt/interfaces/odx.py src/bdt/workflows/bdt/diffusion.py src/bdt/workflows/parcellation.py test/
git commit -m "feat: add odx diffusion-model parcellation (odf shipped, fixel stubbed)"
```

---

### Task 18: Refactor `workflows/bdt/base.py` — spec-driven `init_single_subject_wf`

Move `init_bdt_wf` and `init_single_subject_wf` from `workflows/base.py` to `workflows/bdt/base.py`. Rewrite `init_single_subject_wf` to load the spec, call `collect_derivatives` per source entry, and dispatch each source file to the appropriate run workflow.

**Files:**
- Create: `src/bdt/workflows/bdt/base.py`
- Modify: `src/bdt/workflows/base.py` (add re-exports for backward compat)
- Modify: `test/test_base.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_base.py
def test_bdt_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.bdt.base import init_bdt_wf
    with mock_config():
        wf = init_bdt_wf()
    assert wf is not None


def test_single_subject_wf_builds():
    from tests.tests import mock_config
    from bdt.workflows.bdt.base import init_single_subject_wf
    with mock_config():
        wf = init_single_subject_wf('01')
    assert wf is not None
```

Run: `pytest test/test_base.py -v`
Expected: ImportError — `bdt.workflows.bdt.base` does not exist

- [ ] **Step 2: Create `src/bdt/workflows/bdt/base.py`**

```python
"""Top-level BDT workflow: iterates over subjects and dispatches per source entry."""
from __future__ import annotations

import os
import sys
from copy import deepcopy

import yaml
from nipype.pipeline import engine as pe
from packaging.version import Version

from bdt import config
from bdt.utils.utils import _get_wf_name, update_dict


def init_bdt_wf():
    """Build BDT's top-level pipeline (one sub-workflow per participant)."""
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)
    bdt_wf = Workflow(name=f'bdt_{ver.major}_{ver.minor}_wf')
    bdt_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)
        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        bdt_wf.add_nodes([single_subject_wf])

        log_dir = (
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'bdt.toml')

    return bdt_wf


def init_single_subject_wf(subject_id: str):
    """Organise BDT processing for a single subject.

    Reads ``config.execution.spec``, collects files for each source entry, and
    wires the appropriate run workflow (timeseries / scalar / streamlines).
    Raises ``RuntimeError`` if a source entry matches zero or >1 files.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo

    from bdt.interfaces.bids import DerivativesDataSink
    from bdt.interfaces.reportlets import AboutSummary, SubjectSummary
    from bdt.utils.spec import load_bdt_spec

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = (
        f'Results included in this manuscript come from postprocessing '
        f'performed using *BDT* {config.environment.version}.\n\n'
    )

    # ------------------------------------------------------------------
    # Load spec and collect files
    # ------------------------------------------------------------------
    if config.execution.spec is None:
        raise RuntimeError('No spec file provided. Set --spec on the command line.')

    spec = load_bdt_spec(config.execution.spec)
    entities = {'subject': subject_id}

    # ------------------------------------------------------------------
    # Summary reportlets (reuse existing pattern)
    # ------------------------------------------------------------------
    bids_info = pe.Node(
        BIDSInfo(bids_dir=config.execution.bids_dir, bids_validate=False),
        name='bids_info',
    )

    summary = pe.Node(
        SubjectSummary(std_spaces=[], nstd_spaces=[]),
        name='summary',
        run_without_submitting=True,
    )
    workflow.connect([(bids_info, summary, [('subject', 'subject_id')])])

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    # ------------------------------------------------------------------
    # Dispatch one run workflow per (source_entry × matched_file)
    # ------------------------------------------------------------------
    for source_idx, source in enumerate(spec.sources):
        # Determine which datasets to search
        dataset_dirs = {
            k: v for k, v in config.execution.datasets.items() if k in source.datasets
        }
        if not dataset_dirs:
            dataset_dirs = config.execution.datasets

        # Collect files matching this source entry across all named datasets
        all_files = []
        for ds_name, ds_path in dataset_dirs.items():
            from bids.layout import BIDSLayout
            ds_layout = BIDSLayout(str(ds_path), validate=False)
            files = ds_layout.get(
                return_type='file',
                subject=subject_id,
                suffix=source.suffix,
            )
            all_files.extend(files)

        if not all_files:
            raise RuntimeError(
                f'Source entry #{source_idx} (suffix={source.suffix!r}) matched no files '
                f'for sub-{subject_id}. Check your spec and dataset paths.'
            )

        # Determine which run workflow to use based on operations
        _streamline_ops = {'streamline_connectivity', 'bundle_stats'}
        _diffusion_ops = {'parcellate_fixel', 'parcellate_odf'}
        if _streamline_ops & set(source.operations):
            from bdt.workflows.bdt.streamlines import init_streamlines_run_wf as _wf_fn
        elif _diffusion_ops & set(source.operations):
            from bdt.workflows.bdt.diffusion import init_diffusion_model_run_wf as _wf_fn
        elif 'parcellate_scalar' in source.operations:
            from bdt.workflows.bdt.scalar import init_scalar_run_wf as _wf_fn
        else:
            from bdt.workflows.bdt.timeseries import init_timeseries_run_wf as _wf_fn

        for source_file in all_files:
            run_wf = _wf_fn(
                source_file=source_file,
                atlas_filters=source.atlases,
                operations=source.operations,
            )
            workflow.add_nodes([run_wf])

    from bdt.workflows.base import clean_datasinks
    return clean_datasinks(workflow)
```

- [ ] **Step 3: Update `src/bdt/workflows/base.py` to re-export from new location**

Add at the bottom of the existing `workflows/base.py`:
```python
# Re-export from new location for backwards compatibility
from bdt.workflows.bdt.base import init_bdt_wf, init_single_subject_wf  # noqa: F401
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bdt/workflows/bdt/base.py src/bdt/workflows/base.py test/test_base.py
git commit -m "feat(workflows): add spec-driven init_single_subject_wf in workflows/bdt/base.py"
```

---

## Phase 5 — BAT

### Task 19: BAT CLI

**Files:**
- Create: `src/bdt/cli/bat_run.py`
- Create: `src/bdt/cli/bat_parser.py`
- Create: `src/bdt/cli/bat_workflow.py`

- [ ] **Step 1: Write failing test**

```python
# test/test_cli.py — add
def test_bat_run_importable():
    from bdt.cli.bat_run import main  # noqa: F401

def test_bat_parser_builds():
    from bdt.cli.bat_parser import _build_parser
    parser = _build_parser()
    assert parser is not None
```

Run: `pytest test/test_cli.py::test_bat_run_importable -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/cli/bat_parser.py`**

```python
"""BAT command-line argument parser."""
from __future__ import annotations

import sys

from bdt import config


def _build_parser(**kwargs):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Action
    from pathlib import Path
    from bdt.cli.bdt_version import check_latest, is_flagged

    class ToDict(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            d = {}
            for spec in values:
                try:
                    name, loc = spec.split('=')
                    loc = Path(loc)
                except ValueError:
                    loc = Path(spec)
                    name = loc.name
                if name in d:
                    raise ValueError(f'Duplicate dataset name: {name}')
                d[name] = loc
            setattr(namespace, self.dest, d)

    def _path_exists(path, parser):
        if path is None or not Path(path).exists():
            raise parser.error(f'Path does not exist: <{path}>.')
        return Path(path).absolute()

    parser = ArgumentParser(
        description='BAT: BIDS Atlas Transformer',
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    parser.add_argument('bids_dir', type=lambda p: _path_exists(p, parser),
                        help='Input BIDS Atlas dataset directory.')
    parser.add_argument('output_dir', type=Path, help='Output directory.')
    parser.add_argument('analysis_level', choices=['dataset'],
                        help='BAT operates at the dataset level only.')
    parser.add_argument('--spec', required=True, type=lambda p: _path_exists(p, parser),
                        help='Path to bat_spec.yaml.')
    parser.add_argument('--datasets', nargs='+', action=ToDict, metavar='NAME=PATH',
                        help='Additional atlas datasets (for multi-dataset union/intersect).')
    parser.add_argument('-w', '--work-dir', type=Path, default=Path('work').absolute(),
                        help='Working directory.')
    parser.add_argument('--notrack', action='store_true', default=False,
                        help='Disable usage tracking.')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    return parser


def parse_args(args=None, namespace=None):
    """Parse arguments and initialise BAT config."""
    from pathlib import Path
    parser = _build_parser()
    opts = parser.parse_args(args, namespace)

    config.execution.log_level = max(25 - 5 * opts.verbose, logging_WARNING := 30)
    config.execution.bids_dir = opts.bids_dir
    config.execution.output_dir = opts.output_dir.absolute()
    config.execution.work_dir = opts.work_dir.absolute()
    config.execution.spec = opts.spec
    config.execution.datasets = opts.datasets or {}
    config.execution.notrack = opts.notrack

    config.execution.output_dir.mkdir(parents=True, exist_ok=True)
    config.execution.work_dir.mkdir(parents=True, exist_ok=True)
    config.loggers.init()
```

- [ ] **Step 3: Create `src/bdt/cli/bat_workflow.py`**

```python
"""BAT workflow builder (called in a subprocess by bat_run.py)."""
from __future__ import annotations

from pathlib import Path


def build_workflow(config_file, retval):
    """Load config and build the BAT nipype workflow."""
    from bdt import config
    from bdt.workflows.bat.base import init_bat_wf

    config.load(config_file)
    config.loggers.init()

    try:
        bat_wf = init_bat_wf()
    except Exception as exc:
        retval['return_code'] = 1
        config.loggers.workflow.critical('BAT workflow build failed: %s', exc)
        raise
    else:
        retval['workflow'] = bat_wf
        retval['return_code'] = 0
    return retval
```

- [ ] **Step 4: Create `src/bdt/cli/bat_run.py`**

```python
"""BAT entry point."""
from bdt import config

EXITCODE: int = -1


def main():
    """BAT entry point."""
    import gc
    import sys
    from multiprocessing import Manager, Process
    from os import EX_SOFTWARE

    from bdt.cli.bat_parser import parse_args
    from bdt.cli.bat_workflow import build_workflow

    parse_args()

    config_file = config.execution.work_dir / config.execution.run_uuid / 'config.toml'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config.to_filename(config_file)

    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(str(config_file), retval))
        p.start()
        p.join()
        retval = dict(retval.items())
        if p.exitcode:
            retval['return_code'] = p.exitcode

    global EXITCODE
    EXITCODE = retval.get('return_code', 0)
    bat_wf = retval.get('workflow', None)
    config.load(config_file)

    EXITCODE = EXITCODE or (bat_wf is None) * EX_SOFTWARE
    if EXITCODE != 0:
        sys.exit(EXITCODE)

    gc.collect()
    try:
        bat_wf.run(**config.nipype.get_plugin())
    except Exception as e:
        config.loggers.workflow.critical('BAT failed: %s', e)
        raise
    else:
        config.loggers.workflow.log(25, 'BAT finished successfully!')
        errno = 0
    finally:
        from bdt.utils.bids import write_derivative_description, write_bidsignore
        write_derivative_description(
            input_dir=config.execution.bids_dir,
            output_dir=config.execution.output_dir,
            dataset_links=config.execution.dataset_links,
        )
        write_bidsignore(config.execution.output_dir)
        sys.exit(int(errno > 0))


if __name__ == '__main__':
    raise RuntimeError(
        'bat_run.py should not be run directly; use the `bat` command.'
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest test/test_cli.py::test_bat_run_importable test/test_cli.py::test_bat_parser_builds -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/bdt/cli/bat_run.py src/bdt/cli/bat_parser.py src/bdt/cli/bat_workflow.py test/test_cli.py
git commit -m "feat(cli): add bat_run, bat_parser, bat_workflow entry points"
```

---

### Task 20: BAT base workflow

**Files:**
- Create: `src/bdt/workflows/bat/base.py`
- Create: `test/test_bat_workflows.py`

- [ ] **Step 1: Write failing tests**

```python
# test/test_bat_workflows.py
def test_bat_wf_builds():
    from tests.tests import mock_config
    from bdt import config
    from bdt.workflows.bat.base import init_bat_wf
    import tempfile, textwrap
    from pathlib import Path
    spec_yaml = textwrap.dedent("""
        operations:
          - name: corticalSubcortical
            operation: union
            inputs:
              - atlas: HCPMMP1
              - atlas: Tian
                seg: S2
            output_entities:
              atlas: HCPMMPTian
    """)
    with mock_config():
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            f.write(spec_yaml)
            spec_path = f.name
        config.execution.spec = spec_path
        wf = init_bat_wf()
    assert wf is not None


def test_bat_dataset_wf_builds():
    from tests.tests import mock_config
    from bdt import config
    from bdt.workflows.bat.base import init_bat_dataset_wf
    import tempfile, textwrap
    spec_yaml = textwrap.dedent("""
        operations:
          - name: networkParcels
            operation: intersect
            inputs:
              - atlas: Schaefer400
              - atlas: RSN
            output_entities:
              atlas: Schaefer400RSN
    """)
    with mock_config():
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            f.write(spec_yaml)
            spec_path = f.name
        config.execution.spec = spec_path
        wf = init_bat_dataset_wf()
    assert wf is not None
```

Run: `pytest test/test_bat_workflows.py -v`
Expected: ImportError

- [ ] **Step 2: Create `src/bdt/workflows/bat/base.py`**

```python
"""BAT top-level workflow: dataset-level atlas algebra."""
from __future__ import annotations

from copy import deepcopy

from nipype.pipeline import engine as pe
from packaging.version import Version

from bdt import config


def init_bat_wf():
    """Build BAT's top-level pipeline (dataset level, no subject loop)."""
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)
    bat_wf = Workflow(name=f'bat_{ver.major}_{ver.minor}_wf')
    bat_wf.base_dir = config.execution.work_dir

    dataset_wf = init_bat_dataset_wf()
    dataset_wf.config['execution']['crashdump_dir'] = str(
        config.execution.output_dir / 'log' / config.execution.run_uuid
    )
    for node in dataset_wf._get_all_nodes():
        node.config = deepcopy(dataset_wf.config)
    bat_wf.add_nodes([dataset_wf])

    log_dir = config.execution.output_dir / 'log' / config.execution.run_uuid
    log_dir.mkdir(exist_ok=True, parents=True)
    config.to_filename(log_dir / 'bat.toml')
    return bat_wf


def init_bat_dataset_wf(name='bat_dataset_wf'):
    """Build one algebra workflow per operation in the BAT spec.

    Reads ``config.execution.spec`` (bat_spec.yaml), collects input atlas
    files for each operation via ``collect_atlases``, and wires the
    appropriate algebra workflow (intersect / union / outer_product).
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from bdt.utils.spec import load_bat_spec

    workflow = Workflow(name=name)

    if config.execution.spec is None:
        raise RuntimeError('No spec file provided. Set --spec on the command line.')

    spec = load_bat_spec(config.execution.spec)

    # Build atlas layout from bids_dir (and any additional --datasets)
    from bids.layout import BIDSLayout
    atlas_layout = BIDSLayout(str(config.execution.bids_dir), validate=False)

    from bdt.workflows.bat.algebra import (
        init_intersect_wf,
        init_outer_product_wf,
        init_union_wf,
    )

    _WF_MAP = {
        'union': init_union_wf,
        'intersect': init_intersect_wf,
        'outer_product': init_outer_product_wf,
    }

    for op_idx, operation in enumerate(spec.operations):
        from bdt.utils.atlas import collect_atlases
        input_files = collect_atlases(atlas_layout, operation.inputs)

        algebra_wf = _WF_MAP[operation.operation](
            input_files=input_files,
            output_entities=operation.output_entities,
            output_dir=config.execution.output_dir,
            name=f'bat_{operation.name}_wf',
        )
        workflow.add_nodes([algebra_wf])

    return workflow
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_bat_workflows.py -v`
Expected: ImportError (algebra workflows not yet created — expected at this point)

- [ ] **Step 4: Commit (partial — base scaffolded)**

```bash
git add src/bdt/workflows/bat/base.py test/test_bat_workflows.py
git commit -m "feat(workflows): add BAT base workflow scaffolding"
```

---

### Task 21: BAT algebra workflows

**Files:**
- Create: `src/bdt/workflows/bat/algebra.py`

- [ ] **Step 1: (tests already written in Task 20)**

Run: `pytest test/test_bat_workflows.py -v`
Expected: ImportError — `bdt.workflows.bat.algebra` does not exist

- [ ] **Step 2: Create `src/bdt/workflows/bat/algebra.py`**

```python
"""BAT atlas algebra workflows: union, intersect, outer_product."""
from __future__ import annotations

from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def _make_algebra_wf(
    interface_cls,
    input_files: list[str],
    output_entities: dict,
    output_dir,
    name: str,
):
    """Generic wiring for a two-input atlas algebra workflow."""
    if len(input_files) != 2:
        raise ValueError(
            f'{name}: exactly 2 input atlases required, got {len(input_files)}.'
        )

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file1', 'in_file2']),
        name='inputnode',
    )
    inputnode.inputs.in_file1 = input_files[0]
    inputnode.inputs.in_file2 = input_files[1]

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file', 'out_tsv']),
        name='outputnode',
    )

    algebra = pe.Node(interface_cls(), name='algebra')

    # DataSink to write BIDS atlas outputs
    from bdt.interfaces.bids import DerivativesDataSink

    # Determine template entity from first input file (atlas files carry tpl- entity)
    import re
    tpl_match = re.search(r'tpl-(\w+)', input_files[0])
    tpl = tpl_match.group(1) if tpl_match else 'unknown'

    atlas_label = output_entities.get('atlas', 'unknown')
    # BIDS-Atlas outputs use the tpl- entity (and tpl-<label>/ directory), NOT space-.
    # Canonical entity order is atlas → seg → scale → res → desc; DerivativesDataSink
    # orders from the BIDS config, so just pass the entities through.
    extra_entities = {k: v for k, v in output_entities.items() if k != 'atlas'}

    ds_atlas = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir),
            template=tpl,          # → tpl-<label>/…  (NOT space=tpl, which yields space-<label>)
            atlas=atlas_label,
            suffix='dseg',
            extension='.nii.gz',
            **extra_entities,
        ),
        name='ds_atlas',
    )
    ds_tsv = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir),
            template=tpl,
            atlas=atlas_label,
            suffix='dseg',
            extension='.tsv',
            **extra_entities,
        ),
        name='ds_tsv',
    )

    # REQUIRED by BIDS (ATLAS_DESCRIPTION_REQUIRED, validator error): a root-level
    # atlas-<label>_description.json with at least AtlasName + License.
    write_desc = pe.Node(
        WriteAtlasDescription(
            output_dir=str(output_dir),
            atlas_label=atlas_label,
            atlas_name=output_entities.get('atlas', atlas_label),
            sources=input_files,
        ),
        name='write_atlas_description',
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, algebra, [('in_file1', 'in_file1'), ('in_file2', 'in_file2')]),
        (algebra, outputnode, [('out_file', 'out_file'), ('out_tsv', 'out_tsv')]),
        (algebra, ds_atlas, [('out_file', 'in_file')]),
        (algebra, ds_tsv, [('out_tsv', 'in_file')]),
    ])

    # union / outer_product lose source provenance under offset-relabeling, so
    # emit a BEP 017 §4.3 nodelabels.tsv mapping each output label → source atlas.
    if getattr(algebra.interface, 'emits_nodelabels', False) or interface_cls.__name__ in (
        'AtlasUnion', 'AtlasOuterProduct'
    ):
        ds_nodelabels = pe.Node(
            DerivativesDataSink(
                base_directory=str(output_dir),
                template=tpl, atlas=atlas_label,
                suffix='nodelabels', extension='.tsv', **extra_entities,
            ),
            name='ds_nodelabels',
        )
        workflow.connect([(algebra, ds_nodelabels, [('out_nodelabels', 'in_file')])])

    return workflow
```

> **Three BIDS-conformance dependencies introduced above (design §8 BAT, §14):**
> 1. **`WriteAtlasDescription`** (add to `interfaces/atlas.py`): a `SimpleInterface` that writes `output_dir/atlas-<label>_description.json` with the **REQUIRED** `AtlasName` + `License` fields (and `Description`/`DerivedFrom` summarizing the source atlases + operation). Without it, BAT output fails validation (`ATLAS_DESCRIPTION_REQUIRED`, error). `License` should be configurable (CLI/spec); default to a clear placeholder that the validator accepts.
> 2. **`out_nodelabels` output** on `AtlasUnion`/`AtlasOuterProduct` (Task 13): a `nodelabels.tsv` with columns `index, SourceAtlasName, SourceAtlasIndex, SourceAtlasLabel` (BEP 017 §4.3), recording which source parcel each combined label came from. `AtlasIntersect` keeps atlas-1's labels, so its provenance is trivial — `out_nodelabels` optional there.
> 3. **`template=` vs subject paths.** The standard niworkflows `DerivativesDataSink` is built for `sub-<label>/…` trees; BAT must emit `tpl-<label>/[<datatype>/]…`. Either extend the sink to honor a `template`/`tpl` entity and the `tpl-<label>/` directory, or write a thin `BIDSAtlasDataSink`. The `space=tpl` in the first draft was wrong — it produces `space-<label>`, not the required `tpl-<label>`.

```python
def init_union_wf(
    input_files: list[str],
    output_entities: dict,
    output_dir,
    name: str = 'bat_union_wf',
):
    """Combine two non-overlapping atlases into one."""
    from bdt.interfaces.atlas import AtlasUnion
    return _make_algebra_wf(AtlasUnion, input_files, output_entities, output_dir, name)


def init_intersect_wf(
    input_files: list[str],
    output_entities: dict,
    output_dir,
    name: str = 'bat_intersect_wf',
):
    """Restrict atlas to voxels present in both input atlases."""
    from bdt.interfaces.atlas import AtlasIntersect
    return _make_algebra_wf(AtlasIntersect, input_files, output_entities, output_dir, name)


def init_outer_product_wf(
    input_files: list[str],
    output_entities: dict,
    output_dir,
    name: str = 'bat_outer_product_wf',
):
    """Create a Cartesian product atlas from two input label sets."""
    from bdt.interfaces.atlas import AtlasOuterProduct
    return _make_algebra_wf(AtlasOuterProduct, input_files, output_entities, output_dir, name)
```

- [ ] **Step 3: Run all BAT workflow tests**

Run: `pytest test/test_bat_workflows.py -v`
Expected: PASS (2 tests — `test_bat_wf_builds` may still fail if test atlas dataset isn't available; `test_bat_dataset_wf_builds` should pass since the layout is mocked via BIDSLayout on a temp path — if collect_atlases fails due to no real atlas data, both tests should be marked as expected failures with `@pytest.mark.xfail(reason="requires real atlas data")`)

- [ ] **Step 4: Run the full test suite (non-integration)**

Run: `pytest test/ -v -m "not integration"`
Expected: All non-integration tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/bdt/workflows/bat/algebra.py
git commit -m "feat(workflows): add BAT algebra workflows (union, intersect, outer_product)"
```

---

## Phase 6 — Packaging

### Task 22: Docker Rust build stage (`trxrs`, `giftirs`, `odx` on `PATH`)

> **New (review §2/§8).** The point-warp and diffusion-model paths shell out to Rust binaries that aren't pip-installable. The container needs a build (or download) stage so `trxrs`, `giftirs`, and `odx` are on `PATH`. Without this, every streamline/surface-by-volume/diffusion integration test fails at run time.

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Add a Rust builder stage** (multi-stage build; copy binaries into the final image)

```dockerfile
# --- Rust binaries (trxrs, giftirs, odx) ---
# Verified against the repos: the `cargo install` positional is the *package*
# name (workspace member), not the binary; odx-rs ships several bins so it needs
# --bin odx; gifti-rs's binary lives in the `giftirs-cli` package.
FROM rust:1-bookworm AS rustbins
RUN cargo install --git https://github.com/tee-ar-ex/trx-rs  --bin trxrs   trx-rs       \
 && cargo install --git https://github.com/PennLINC/gifti-rs --bin giftirs giftirs-cli  \
 && cargo install --git https://github.com/PennLINC/odx-rs   --bin odx     odx-rs
# (Pin each to a tag/commit with --tag/--rev once releases are cut. itk-transforms-rs
#  is a git dependency of all three and is fetched transitively.)

# --- final image ---
# ... existing BDT image ...
COPY --from=rustbins /usr/local/cargo/bin/trxrs   /usr/local/bin/trxrs
COPY --from=rustbins /usr/local/cargo/bin/giftirs /usr/local/bin/giftirs
COPY --from=rustbins /usr/local/cargo/bin/odx     /usr/local/bin/odx
```

> Prefer pre-built release binaries from each repo's GitHub releases if available (smaller image, no Rust toolchain in the build). Pin versions. If the repos publish musl static binaries, a plain `COPY` of downloaded artifacts is simplest. Note `odx-rs` requires HDF5 (`hdf5-metno` with the `static` feature) — the static feature avoids a system libhdf5 dependency in the final image, but the *builder* stage still needs a C toolchain (present in `rust:1-bookworm`).

- [ ] **Step 2: Verify the binaries resolve in the built image**

```bash
docker build -t bdt:dev .
docker run --rm bdt:dev bash -lc 'trxrs --version && giftirs --version && odx --version'
```
Expected: each binary prints a version (exact flag may be `-V`/`--help`; verify).

- [ ] **Step 3: Smoke-test a CommandLine interface end to end** (optional, needs a tiny `.trk`)

```bash
docker run --rm -v "$PWD/test/data:/data" bdt:dev \
  python -c "from bdt.interfaces.rust import TrxConvert; \
             print(TrxConvert(in_file='/data/tiny.trk', out_file='/tmp/o.trx').run().outputs.out_file)"
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "build: add Rust build stage so trxrs/giftirs/odx are on PATH"
```

---

## Self-Review Checklist

- [x] **Task 1** — CLI rename: all four files renamed, internal cross-references updated, pyproject.toml updated (+ `odx`/`trxrs` deps)
- [x] **Task 2** — config: `bdt_workflow` and `bat_workflow` added; `execution.spec` added; `workflow.dummy_scans` removed
- [x] **Task 3** — spec parser: both `BDTSpec` and `BATSpec` covered, error on unknown operations *(update `VALID_BDT_OPERATIONS` to the new set: `streamline_connectivity`, `bundle_stats`, `parcellate_fixel`, `parcellate_odf`)*
- [x] **Task 4** — atlas utils: `collect_atlases` with ambiguity and no-match errors
- [x] **Task 5** — `bdt.tests` package: conftest imports will resolve after this task
- [x] **Task 6** — transforms: two typed queries (`chain_for_image_resample` / `chain_for_point_warp`), widened regex (`.mat`/no-`mode`), invertibility guard
- [x] **Task 6b** — Rust `CommandLine` wrappers (`trxrs`/`giftirs`/`odx`)
- [x] **Tasks 7–13** — `ApplyTransforms` float bug fixed; GIFTI routes through wb_command + new `SurfaceVolumeParcellate`; streamlines use `StreamlineConnectivity` (endpoint, not density_map); `AtlasIntersect`/`AtlasOuterProduct` made distinct; `AtlasUnion` overlap → precedence
- [x] **Task 13b** — `init_load_atlases_wf` defined (was referenced but never created)
- [x] **Task 14** — three GIFTI paths (same-mesh wb_command, cross-mesh `-label-resample`, volume-atlas vertex-warp)
- [x] **Tasks 15–17b** — run workflows: timeseries, scalar, streamlines (Strategy B), diffusion (odx)
- [x] **Task 18** — `init_single_subject_wf` spec-driven dispatch incl. streamline/diffusion ops; `workflows/base.py` re-exports for compat
- [x] **Tasks 19–21** — BAT CLI + base + algebra workflows covered
- [x] **Task 22** — Docker Rust build stage so binaries are on `PATH`

**Known gaps / follow-up:**
- **CLI/parser changes for new geometries:** the BDT parser/spec accepts the new operations and BEP-typed sources (`suffix: tractogram` + `tract-`/`track-`, BEP 046; `suffix: dwimap` + `model-`/`param-`, BEP 016) via the generic `filters` dict added to `BDTSource` (Task 3, Rev 02c). **pybids gotcha:** these BEP entities are not in the default BIDS config, so BDT must ship a derivatives entity config (a pybids `--config`/`.bids` JSON) registering `model`, `param`, `tract`, `track`, `scale`, `meas` for `layout.get()` to honor them.
- **`init_parcellate_nifti_wf`** is referenced in Tasks 15–16/13b but isn't in the existing `parcellation.py` (only `init_parcellate_cifti_wf`). Extract it from the CIFTI workflow or the `connectivity.py` NIfTI path before Task 13b.
- **Rust API/CLI surfaces — verified 2026-06-02** against the cloned `trx-rs`/`gifti-rs`/`odx-rs` sources and baked into Tasks 1, 6b, 12, 17b, 22: binaries `trxrs`/`giftirs`/`odx`; positional `input output` (no `--output`); `trxrs` overwrite `--force`, `giftirs` `--overwrite`; `odx transform` `--mode {mrtrix,ants}` + `--transform-inverse`, operating on `.odx`; Python package `trxrs` (`load().positions()/.offsets()`), `odx` reader (`load`/`sh`/`compact_to_ijk`/`affine`); `odx transform` is CLI-only. **Still to confirm at impl time:** the exact `odx convert` subcommand flags and format auto-detection (or whether to ingest via the `from_mrtrix`/`from_pyafq_aodf` PyO3 helpers instead), and whether nipype honors the chosen subcommand pattern (see the Task 6b correctness note — do *not* override `cmd` to drop the subcommand).
- **`StreamlineConnectivity` / `.trx` test fixtures:** if `.trx` write isn't available in the unit-test env, fall back to `.tck` fixtures or mark `@pytest.mark.integration`.
- **`FixelParcellate` is stubbed** (`NotImplementedError`) — `OdfParcellate` ships first; implement fixel reduction once the odx fixel reader API is confirmed.
- **BAT workflow tests (Task 20–21)** assume `collect_atlases` finds atlas files; mark `@pytest.mark.integration` or add fixture atlases under `test/data/`.
- **`config.workflow.spaces`** references in existing code: verify after `workflow` class is simplified in Task 2.
- **Point-warp directionality is the top correctness risk:** ensure every streamline/surface warp goes through `chain_for_point_warp` (never the image query) and that the build-time call raises when only the wrong-direction warp exists.

---

## BIDS / BEP Conformance Checklist (Rev 02c — see design §14)

Folded into the tasks above; verify each with the integration BIDS-validation test (design §11).

- [ ] **Spec entities (Task 3):** `BDTSource.filters` passes through `model`/`param`/`tract`/`track`/`seg`/`scale`; ship a pybids derivatives entity config registering them. — *BEP 012/016/046, atlas*
- [ ] **Time series output (Task 15):** parcellated series use the **`timeseries`** suffix (not `bold`); every atlas-derived file carries `space-`; entity order `space → atlas → seg → scale → res → den → desc`. — *BEP 012*
- [ ] **Scalar output (Task 16):** ALFF/ReHo are `stat-<label>` (not an `alff` suffix); dMRI scalars carry `model-`/`param-` on `dwimap`; anatomical stays `morph` (BEP 011 columns). — *BEP 011/012/016*
- [ ] **Diffusion model (Task 17b):** source is `suffix: dwimap` + `model-`/`param-`; SH/ODF is a plain NIfTI in BIDS (`.odx`/`.mif` tool-internal only). — *BEP 016*
- [ ] **Streamlines (Tasks 12/17):** connectivity is `…_meas-<label>_relmat.dense.tsv` (header-less) + REQUIRED `_relmat.json` + `_nodeindices.tsv`; `bundle_stats` is a plain TSV, not a relmat; tractogram source is `tractogram`/TRX. — *BEP 017/046*
- [ ] **FC (Task 15):** functional connectivity emitted as `_meas-<label>_relmat.dense.tsv` + sidecars (not a bare `correlations.tsv`); optional CIFTI `.pconn.nii` is non-canonical. — *BEP 017*
- [ ] **BAT outputs (Tasks 13/21):** canonical order `atlas → seg → scale → res → desc` (not `desc`-before-`seg`); `tpl-` output entity (not `space-`); REQUIRED `atlas-<label>_description.json` (`AtlasName`+`License`) via `WriteAtlasDescription`; `dseg.tsv` has `index`+`name`; union/outer_product emit `nodelabels.tsv`. — *merged BIDS atlas spec + BEP 017 §4.3*
- [ ] **DerivativesDataSink:** extend (or add `BIDSAtlasDataSink`) to honor the `tpl-`/`template` entity and `tpl-<label>/[<datatype>/]` layout for BAT (the standard sink is `sub-<label>/`-oriented). — *merged BIDS atlas spec*
- [ ] **Validation test (Task 21+):** run the schema/`bids-validator` over BDT and BAT outputs; assert zero errors (esp. no `ATLAS_DESCRIPTION_REQUIRED`).

**Stability note:** the merged-BIDS items (entity order, `space-`, `atlas-…_description.json`, `dseg.tsv` columns) are validator-enforceable today. The BEP items are unmerged and subject to change — but `tractogram` (BEP 046) and `relmat.dense`/`meas-` + sidecars (BEP 017) are stable and co-authored by the BDT team (Cieslak on 046, Salo leading 041), and `model-`/`param-` (BEP 016) is stable. Adopt now; track the BEP branches for drift.
