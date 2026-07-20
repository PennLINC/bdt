# Tractogram-to-pseg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `init_tractogram_to_pseg_wf` turn a grouped list of bundle-wise `.tck.gz` tractograms into a 4D `probseg` (or, when `threshold` is set, a binarized `dseg`) NIfTI plus a BIDS label TSV, wired end-to-end through the action registry and output plan.

**Architecture:** Three new nibabel/pandas `SimpleInterface`s (`ConcatenateNiftis`, `ThresholdNifti`, `EntitiesToSegTSV`) plus reused `nipype` interfaces (`Gunzip`, `ComputeTDI`) form the compute chain. The factory resolves the ACPC reference grid at build time via `FactoryContext.find_reference`. The action's output naming (probseg vs dseg) and its label-TSV sink are driven by two small, generic additions to the output registry — `OutputSpec.dynamic_suffix` and `ExtraProduct.match_primary_suffix` — so no action-specific branching enters `build_sink_plan`.

**Tech Stack:** Python, nipype (pipeline + interfaces), nibabel, numpy, pandas, mrtrix3's `tckmap` (via nipype `ComputeTDI`), pytest.

## Global Constraints

- Run all Python/pytest via the `bdtenv` micromamba env: `micromamba run -n bdtenv <cmd>`.
- Do **not** create git commits; the workspace is not a git repo and the maintainer manages version control. Each task ends with a **test-suite checkpoint** instead.
- Every source file starts with the existing NiPreps Apache-2.0 header (copy the header block verbatim from any sibling file such as `src/bdt/interfaces/connectivity.py`).
- New interfaces are `nipype` `SimpleInterface`s built only on already-vendored deps (`nibabel`, `numpy`, `pandas`); add no new heavy dependency.
- Interface input/output field name is `out_file` for the produced file, matching the codebase convention (`connectivity.py`).
- The factory's `outputnode` MUST expose the primary segmentation on field `out` (the compiler at `src/bdt/engine/workflow.py:123` wires downstream consumers and passthrough sinks from `outputnode.out`).

---

### Task 1: `ConcatenateNiftis` interface

**Files:**
- Create: `src/bdt/interfaces/tractography.py`
- Test: `test/engine/test_tractography.py`

**Interfaces:**
- Consumes: nothing (leaf interface).
- Produces: `ConcatenateNiftis` — `SimpleInterface`; inputs `in_files: list[str]` (existing NIfTIs, mandatory), `normalize: bool = True`, `out_file: str = 'concatenated.nii.gz'`; output `out_file: str` (a 4D NIfTI, one input volume per 4th-dim index, each peak-normalized to `[0, 1]` when `normalize`).

- [ ] **Step 1: Write the failing test**

Create `test/engine/test_tractography.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Unit tests for bdt.interfaces.tractography."""

import nibabel as nb
import numpy as np
import pytest

pytest.importorskip('nipype')


def _nii(tmp_path, name, data):
    img = nb.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))
    path = str(tmp_path / name)
    img.to_filename(path)
    return path


def test_concatenate_niftis_normalizes_and_stacks(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    a = np.zeros((2, 2, 2), dtype=np.float32)
    a[0, 0, 0] = 4.0  # peak 4 -> normalizes to 1.0
    b = np.zeros((2, 2, 2), dtype=np.float32)
    b[1, 1, 1] = 2.0  # peak 2 -> normalizes to 1.0

    res = ConcatenateNiftis(
        in_files=[_nii(tmp_path, 'a.nii.gz', a), _nii(tmp_path, 'b.nii.gz', b)],
        normalize=True,
    ).run()

    out = nb.load(res.outputs.out_file)
    data = out.get_fdata()
    assert data.shape == (2, 2, 2, 2)
    assert data[..., 0].max() == pytest.approx(1.0)
    assert data[..., 1].max() == pytest.approx(1.0)
    assert data[0, 0, 0, 0] == pytest.approx(1.0)
    assert data[1, 1, 1, 1] == pytest.approx(1.0)


def test_concatenate_niftis_no_normalize_keeps_counts(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    a = np.zeros((2, 2, 2), dtype=np.float32)
    a[0, 0, 0] = 4.0
    res = ConcatenateNiftis(in_files=[_nii(tmp_path, 'a.nii.gz', a)], normalize=False).run()
    assert nb.load(res.outputs.out_file).get_fdata()[0, 0, 0, 0] == pytest.approx(4.0)


def test_concatenate_niftis_all_zero_stays_zero(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    z = np.zeros((2, 2, 2), dtype=np.float32)
    res = ConcatenateNiftis(in_files=[_nii(tmp_path, 'z.nii.gz', z)], normalize=True).run()
    assert float(nb.load(res.outputs.out_file).get_fdata().max()) == 0.0


def test_concatenate_niftis_shape_mismatch_raises(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    a = np.zeros((2, 2, 2), dtype=np.float32)
    b = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='[Ss]hape'):
        ConcatenateNiftis(
            in_files=[_nii(tmp_path, 'a.nii.gz', a), _nii(tmp_path, 'b.nii.gz', b)],
        ).run()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdt.interfaces.tractography'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/bdt/interfaces/tractography.py` (copy the Apache-2.0 header block verbatim from `src/bdt/interfaces/connectivity.py`, then):

```python
"""nipype interfaces for building a bundle-wise segmentation from tractograms."""

import os

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _ConcatenateNiftisInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='3D maps to stack (one volume per bundle)'
    )
    normalize = traits.Bool(
        True, usedefault=True, desc='peak-normalize each map to [0, 1] before stacking'
    )
    out_file = File('concatenated.nii.gz', usedefault=True, desc='output 4D file name')


class _ConcatenateNiftisOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='4D NIfTI, one input volume per 4th-dim index')


class ConcatenateNiftis(SimpleInterface):
    """Stack 3D maps into a 4D volume, optionally peak-normalizing each to [0, 1]."""

    input_spec = _ConcatenateNiftisInputSpec
    output_spec = _ConcatenateNiftisOutputSpec

    def _run_interface(self, runtime):
        first = nb.load(self.inputs.in_files[0])
        vols = []
        for path in self.inputs.in_files:
            img = nb.load(path)
            data = np.asarray(img.dataobj, dtype=np.float32)
            if data.shape != first.shape:
                raise ValueError(
                    f'Shape mismatch: {path} has {data.shape}, expected {first.shape}.'
                )
            if self.inputs.normalize:
                peak = float(data.max())
                if peak > 0:
                    data = data / peak
            vols.append(data)
        stacked = np.stack(vols, axis=-1).astype(np.float32)
        out_file = os.path.abspath(self.inputs.out_file)
        out_img = nb.Nifti1Image(stacked, first.affine, first.header)
        out_img.header.set_data_dtype(np.float32)
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Test-suite checkpoint**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -q`
Expected: all green. (No commit — maintainer handles version control.)

---

### Task 2: `ThresholdNifti` interface

**Files:**
- Modify: `src/bdt/interfaces/tractography.py` (append)
- Test: `test/engine/test_tractography.py` (append)

**Interfaces:**
- Consumes: nothing (leaf interface).
- Produces: `ThresholdNifti` — `SimpleInterface`; inputs `in_file: str` (existing NIfTI, mandatory), `threshold: float` (mandatory), `binarize: bool = True`, `out_file: str = 'thresholded.nii.gz'`; output `out_file: str`. Emits `data > threshold` — as a uint8 0/1 mask when `binarize`, else the masked values — preserving input shape (incl. 4D).

- [ ] **Step 1: Write the failing test**

Append to `test/engine/test_tractography.py`:

```python
def test_threshold_nifti_binarizes_4d(tmp_path):
    from bdt.interfaces.tractography import ThresholdNifti

    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    data[0, 0, 0, 0] = 0.8
    data[1, 1, 1, 1] = 0.2
    path = _nii(tmp_path, 'p.nii.gz', data)

    res = ThresholdNifti(in_file=path, threshold=0.5, binarize=True).run()
    out = nb.load(res.outputs.out_file)
    out_data = out.get_fdata()
    assert out_data.shape == (2, 2, 2, 2)
    assert out_data[0, 0, 0, 0] == 1
    assert out_data[1, 1, 1, 1] == 0
    assert out.get_data_dtype() == np.uint8


def test_threshold_nifti_zero_threshold_keeps_any_nonzero(tmp_path):
    from bdt.interfaces.tractography import ThresholdNifti

    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 0.01
    path = _nii(tmp_path, 'q.nii.gz', data)

    res = ThresholdNifti(in_file=path, threshold=0.0, binarize=True).run()
    assert nb.load(res.outputs.out_file).get_fdata()[0, 0, 0, 0] == 1


def test_threshold_nifti_no_binarize_keeps_values(tmp_path):
    from bdt.interfaces.tractography import ThresholdNifti

    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 0.8
    path = _nii(tmp_path, 'r.nii.gz', data)

    res = ThresholdNifti(in_file=path, threshold=0.5, binarize=False).run()
    assert nb.load(res.outputs.out_file).get_fdata()[0, 0, 0, 0] == pytest.approx(0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -k threshold -v`
Expected: FAIL — `ImportError: cannot import name 'ThresholdNifti'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/bdt/interfaces/tractography.py`:

```python
class _ThresholdNiftiInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='image to threshold')
    threshold = traits.Float(mandatory=True, desc='keep values strictly greater than this')
    binarize = traits.Bool(
        True, usedefault=True, desc='emit a 0/1 mask instead of the masked values'
    )
    out_file = File('thresholded.nii.gz', usedefault=True, desc='output file name')


class _ThresholdNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='thresholded image')


class ThresholdNifti(SimpleInterface):
    """Threshold an image at ``value > threshold``; optionally binarize to a 0/1 mask."""

    input_spec = _ThresholdNiftiInputSpec
    output_spec = _ThresholdNiftiOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = np.asarray(img.dataobj, dtype=np.float32)
        mask = data > self.inputs.threshold
        if self.inputs.binarize:
            out_data = mask.astype(np.uint8)
            dtype = np.uint8
        else:
            out_data = np.where(mask, data, 0).astype(np.float32)
            dtype = np.float32
        out_file = os.path.abspath(self.inputs.out_file)
        out_img = nb.Nifti1Image(out_data, img.affine, img.header)
        out_img.header.set_data_dtype(dtype)
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -k threshold -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Test-suite checkpoint**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -q`
Expected: all green (7 tests total).

---

### Task 3: `EntitiesToSegTSV` interface

**Files:**
- Modify: `src/bdt/interfaces/tractography.py` (append)
- Test: `test/engine/test_tractography.py` (append)

**Interfaces:**
- Consumes: nothing (leaf interface).
- Produces: `EntitiesToSegTSV` — `SimpleInterface`; inputs `in_files: list[str]` (existing files, mandatory), `entity: str = 'bundle'`, `out_file: str = 'dseg.tsv'`; output `out_file: str`. Writes a tab-separated table with columns `index` (1-based, matching input order) and `name` (the parsed entity value). Raises `ValueError` if a file lacks the entity.

- [ ] **Step 1: Write the failing test**

Append to `test/engine/test_tractography.py`:

```python
def _touch(tmp_path, name):
    path = tmp_path / name
    path.touch()
    return str(path)


def test_entities_to_seg_tsv_orders_by_input(tmp_path):
    import pandas as pd

    from bdt.interfaces.tractography import EntitiesToSegTSV

    files = [
        _touch(tmp_path, 'sub-01_bundle-CST_space-ACPC_streamlines.tck.gz'),
        _touch(tmp_path, 'sub-01_bundle-AF_space-ACPC_streamlines.tck.gz'),
    ]
    res = EntitiesToSegTSV(in_files=files, entity='bundle').run()

    df = pd.read_table(res.outputs.out_file)
    assert list(df.columns) == ['index', 'name']
    assert df['index'].tolist() == [1, 2]
    assert df['name'].tolist() == ['CST', 'AF']


def test_entities_to_seg_tsv_missing_entity_raises(tmp_path):
    from bdt.interfaces.tractography import EntitiesToSegTSV

    files = [_touch(tmp_path, 'sub-01_space-ACPC_streamlines.tck.gz')]
    with pytest.raises(ValueError, match='bundle'):
        EntitiesToSegTSV(in_files=files, entity='bundle').run()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -k entities -v`
Expected: FAIL — `ImportError: cannot import name 'EntitiesToSegTSV'`.

- [ ] **Step 3: Write minimal implementation**

Add `import re` to the top-of-file imports (below `import os`) and `import pandas as pd` beside the other third-party imports in `src/bdt/interfaces/tractography.py`, then append:

```python
class _EntitiesToSegTSVInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='files whose <entity>-<value> key names each segment (one per volume)',
    )
    entity = traits.Str('bundle', usedefault=True, desc='BIDS entity key to read from each name')
    out_file = File('dseg.tsv', usedefault=True, desc='output BIDS label TSV')


class _EntitiesToSegTSVOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='BIDS label TSV with columns index, name')


class EntitiesToSegTSV(SimpleInterface):
    """Build a BIDS ``index``/``name`` label TSV from a BIDS entity in each filename.

    Row order follows ``in_files`` order, so index ``i`` (1-based) names the ``i``-th
    volume of the matching 4D segmentation.
    """

    input_spec = _EntitiesToSegTSVInputSpec
    output_spec = _EntitiesToSegTSVOutputSpec

    def _run_interface(self, runtime):
        key = self.inputs.entity
        pattern = re.compile(rf'(?:^|[_/]){re.escape(key)}-([a-zA-Z0-9]+)')
        rows = []
        for i, path in enumerate(self.inputs.in_files, start=1):
            name = os.path.basename(path)
            match = pattern.search(name)
            if match is None:
                raise ValueError(f'No {key!r} entity found in filename {name!r}.')
            rows.append({'index': i, 'name': match.group(1)})
        out_file = os.path.abspath(self.inputs.out_file)
        pd.DataFrame(rows, columns=['index', 'name']).to_csv(out_file, sep='\t', index=False)
        self._results['out_file'] = out_file
        return runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -k entities -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Test-suite checkpoint**

Run: `micromamba run -n bdtenv pytest test/engine/test_tractography.py -q`
Expected: all green (9 tests total).

---

### Task 4: Threshold-aware naming + label-TSV product in the output registry

**Files:**
- Modify: `src/bdt/spec/actions.py` (`ExtraProduct`, `OutputSpec`, `_o`, the `tractogram_to_pseg` `ActionSpec`)
- Modify: `src/bdt/outputs/plan.py` (`build_sink_plan`)
- Test: `test/engine/test_output_plan.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (for Task 5's context, and the sink layer): the `tractogram_to_pseg` action now plans a primary product whose suffix is `probseg` when `threshold` is `None`/absent and `dseg` when `threshold` is set, plus a `.tsv` label product read from `outputnode.tsv`.
  - `OutputSpec.dynamic_suffix: Callable[[dict], str] | None = None`
  - `ExtraProduct.match_primary_suffix: bool = False`
  - `_o(..., dynamic_suffix=None)` passthrough.

- [ ] **Step 1: Write the failing test**

Append to `test/engine/test_output_plan.py`:

```python
def _pseg_spec(threshold):
    params = {} if threshold is None else {'threshold': threshold}
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_bundles',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {
                        'suffix': 'streamlines',
                        'extension': '.tck.gz',
                        'space': 'ACPC',
                    },
                },
                {
                    'name': 'bundle_rois',
                    'action': 'tractogram_to_pseg',
                    'inputs': {'tractograms': 'load_bundles'},
                    'parameters': params,
                    'write_outputs': True,
                },
            ]
        }
    )
    resolved = {
        'load_bundles': Match(
            path='/x/sub-01_bundle-CST_space-ACPC_streamlines.tck.gz',
            entities={'sub': '01', 'space': 'ACPC', 'suffix': 'streamlines'},
        )
    }
    return spec, resolved


def test_pseg_probseg_suffix_when_unthresholded():
    spec, resolved = _pseg_spec(threshold=None)
    plan = build_sink_plan(spec, resolved, {})
    products = plan['bundle_rois']
    primary = next(p for p in products if p.extension == '.nii.gz')
    assert primary.suffix == 'probseg'


def test_pseg_dseg_suffix_when_thresholded():
    spec, resolved = _pseg_spec(threshold=0.0)
    plan = build_sink_plan(spec, resolved, {})
    products = plan['bundle_rois']
    primary = next(p for p in products if p.extension == '.nii.gz')
    assert primary.suffix == 'dseg'


def test_pseg_emits_label_tsv_matching_primary_suffix():
    spec, resolved = _pseg_spec(threshold=0.0)
    plan = build_sink_plan(spec, resolved, {})
    tsvs = [p for p in plan['bundle_rois'] if p.extension == '.tsv']
    assert len(tsvs) == 1
    assert tsvs[0].source_field == 'tsv'
    assert tsvs[0].suffix == 'dseg'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n bdtenv pytest test/engine/test_output_plan.py -k pseg -v`
Expected: FAIL — primary suffix is the static `'dseg'` for both cases (so `test_pseg_probseg_suffix_when_unthresholded` fails), and no `.tsv` product exists.

- [ ] **Step 3: Write minimal implementation**

In `src/bdt/spec/actions.py`, add a field to `ExtraProduct` (after `cifti_only`):

```python
    cifti_only: bool = True
    match_primary_suffix: bool = False  # label TSVs follow the resolved primary suffix
```

Add a field to `OutputSpec` (after `output_is_cifti`):

```python
    output_is_cifti: bool = False  # primary product is CIFTI regardless of input
    dynamic_suffix: object = None  # optional Callable[[dict params], str] overriding suffix
```

Extend `_o` to accept and forward it — add `dynamic_suffix=None` to the signature and pass `dynamic_suffix=dynamic_suffix` into the `OutputSpec(...)` constructor:

```python
def _o(
    suffix,
    extension,
    datatype,
    primary_role=None,
    cifti_suffix=None,
    cifti_extension=None,
    extra=(),
    preserve_source=False,
    emit_tsv=True,
    output_is_cifti=False,
    dynamic_suffix=None,
    **entities,
) -> OutputSpec:
    return OutputSpec(
        suffix=suffix,
        extension=extension,
        datatype=datatype,
        primary_role=primary_role,
        entities=entities,
        cifti_suffix=cifti_suffix,
        cifti_extension=cifti_extension,
        extra=tuple(extra),
        preserve_source=preserve_source,
        emit_tsv=emit_tsv,
        output_is_cifti=output_is_cifti,
        dynamic_suffix=dynamic_suffix,
    )
```

Replace the `tractogram_to_pseg` `ActionSpec` (currently `src/bdt/spec/actions.py:391-398`) with:

```python
    ActionSpec(
        'tractogram_to_pseg',
        PROCESSING,
        'atlas',
        roles=(_r('tractograms', 'streamlines', fan_out=False),),
        parameters=frozenset({'threshold'}),
        out=_o(
            'probseg',
            '.nii.gz',
            'dwi',
            primary_role='tractograms',
            dynamic_suffix=lambda params: (
                'dseg' if params.get('threshold') is not None else 'probseg'
            ),
            extra=(
                ExtraProduct(
                    'tsv',
                    'probseg',
                    '.tsv',
                    cifti_only=False,
                    match_primary_suffix=True,
                ),
            ),
        ),
    ),
```

In `src/bdt/outputs/plan.py`, inside `build_sink_plan`, compute the primary suffix after the existing `tsv_suffix = node_ent.pop('suffix', out.suffix)` line (around `plan.py:256`):

```python
        tsv_suffix = node_ent.pop('suffix', out.suffix)
        datatype = node_ent.pop('datatype', out.datatype)
        # Threshold-aware (or otherwise parameter-driven) primary suffix.
        primary_suffix = tsv_suffix
        if out.dynamic_suffix is not None:
            primary_suffix = out.dynamic_suffix(node.parameters)
```

Use `primary_suffix` for the volumetric primary product. Change the `else` branch (around `plan.py:295-303`) so its `suffix=tsv_suffix` becomes `suffix=primary_suffix`:

```python
        else:
            # Volumetric / already-tabular: the primary product only.
            products.append(
                OutputProduct(
                    derive=PASSTHROUGH,
                    suffix=primary_suffix,
                    extension=out.extension,
                    entities=dict(mid),
                    sidecar=dict(sidecar),
                    **common,
                )
            )
```

In the extra-products loop (around `plan.py:307-323`), honour `match_primary_suffix`:

```python
        for ep in out.extra:
            if ep.cifti_only and not cifti_by_node.get(node.name):
                continue
            ep_ent = dict(mid)
            if ep.stat is not None:
                ep_ent['stat'] = ep.stat
            ep_suffix = primary_suffix if ep.match_primary_suffix else ep.suffix
            products.append(
                OutputProduct(
                    derive=PASSTHROUGH,
                    suffix=ep_suffix,
                    extension=ep.extension,
                    entities=ep_ent,
                    sidecar=dict(sidecar),
                    source_field=ep.source_field,
                    **common,
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n bdtenv pytest test/engine/test_output_plan.py -k pseg -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Test-suite checkpoint**

Run: `micromamba run -n bdtenv pytest test/engine/test_output_plan.py -q`
Expected: all green — the existing CIFTI/parcellation product tests still pass (the new fields default to prior behaviour).

---

### Task 5: Rewrite `init_tractogram_to_pseg_wf`

**Files:**
- Modify: `src/bdt/engine/factories.py` (replace the `init_tractogram_to_pseg_wf` body, currently `src/bdt/engine/factories.py:618-691`)
- Test: `test/engine/test_nipype_workflow.py` (append)

**Interfaces:**
- Consumes: `ConcatenateNiftis`, `ThresholdNifti`, `EntitiesToSegTSV` (Tasks 1-3); reused `nipype.algorithms.misc.Gunzip` and `nipype.interfaces.mrtrix3.ComputeTDI`; `FactoryContext.find_reference` / `role_session` (existing).
- Produces: a `pe.Workflow` whose `inputnode` field is `tractograms` (a list) and whose `outputnode` fields are `out` (the seg) and `tsv` (the label table). A `binarize` (`ThresholdNifti`) node exists iff `node.parameters['threshold'] is not None`.

- [ ] **Step 1: Write the failing test**

Append to `test/engine/test_nipype_workflow.py`:

```python
def _pseg_spec_wf(threshold=None):
    params = {} if threshold is None else {'threshold': threshold}
    return parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_bundles',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {
                        'suffix': 'streamlines',
                        'extension': '.tck.gz',
                        'space': 'ACPC',
                    },
                },
                {
                    'name': 'bundle_rois',
                    'action': 'tractogram_to_pseg',
                    'inputs': {'tractograms': 'load_bundles'},
                    'parameters': params,
                    'write_outputs': True,
                },
            ]
        }
    )


def _pseg_context(spec, tmp_path):
    from bdt.engine.factories import FactoryContext
    from bdt.engine.selection import DictDataProvider, Match

    ref = tmp_path / 'sub-01_space-ACPC_dwiref.nii.gz'
    ref.touch()
    provider = DictDataProvider(
        {'qsiprep': [Match(str(ref), {'suffix': 'dwiref', 'space': 'ACPC'})]}
    )
    return FactoryContext(provider=provider, subject='01', spec=spec, datasets=['qsiprep'])


def test_tractogram_to_pseg_probseg_no_threshold(tmp_path):
    from bdt.engine.factories import init_tractogram_to_pseg_wf

    spec = _pseg_spec_wf(threshold=None)
    node = spec.by_name()['bundle_rois']
    wf = init_tractogram_to_pseg_wf(node, context=_pseg_context(spec, tmp_path))

    names = set(wf.list_node_names())
    assert 'gunzip' in names
    assert 'tck_to_tdi' in names
    assert 'concatenate' in names
    assert 'bundles_to_tsv' in names
    assert 'binarize' not in names  # no threshold -> no binarize node
    # reference grid was resolved and fixed on the TDI node
    assert wf.get_node('tck_to_tdi').inputs.reference.endswith('space-ACPC_dwiref.nii.gz')
    # outputnode contract
    out = wf.get_node('outputnode')
    assert set(out.inputs.copyable_trait_names()) >= {'out', 'tsv'}


def test_tractogram_to_pseg_dseg_with_threshold(tmp_path):
    from bdt.engine.factories import init_tractogram_to_pseg_wf

    spec = _pseg_spec_wf(threshold=0.0)
    node = spec.by_name()['bundle_rois']
    wf = init_tractogram_to_pseg_wf(node, context=_pseg_context(spec, tmp_path))

    assert 'binarize' in set(wf.list_node_names())
    assert wf.get_node('binarize').inputs.threshold == 0.0


def test_tractogram_to_pseg_requires_provider():
    from bdt.engine.factories import init_tractogram_to_pseg_wf

    spec = _pseg_spec_wf(threshold=None)
    node = spec.by_name()['bundle_rois']
    with pytest.raises(ValueError, match='provider'):
        init_tractogram_to_pseg_wf(node, context=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n bdtenv pytest test/engine/test_nipype_workflow.py -k tractogram_to_pseg -v`
Expected: FAIL — the current factory body references undefined names (`Tckmap`, `ConcatenateNiftis`, …) and uses `context.role_space(node, 'threshold')`, so building the workflow raises `NameError`/`AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Replace the entire `init_tractogram_to_pseg_wf` function (`src/bdt/engine/factories.py:618-691`) with:

```python
@workflow_factory('tractogram_to_pseg')
def init_tractogram_to_pseg_wf(node, name=None, context=None) -> pe.Workflow:
    """Build a 4D bundle segmentation (probseg / dseg) from bundle-wise tractograms.

    ``inputnode.tractograms`` is the grouped list of per-bundle ``.tck.gz`` files (all
    in ACPC space).  Each is decompressed (``Gunzip``) and turned into a track-density
    image on a shared ACPC reference grid (``tckmap`` via nipype ``ComputeTDI``); the
    per-bundle maps are peak-normalized to ``[0, 1]`` and stacked into a 4D
    ``probseg`` (:class:`~bdt.interfaces.tractography.ConcatenateNiftis`).  When a
    ``threshold`` parameter is given, the stack is binarized (``value > threshold``)
    into a 4D ``dseg``.  A BIDS ``index``/``name`` label TSV (one row per volume, in
    input order) is emitted on ``outputnode.tsv``; the segmentation is on
    ``outputnode.out``.

    The ACPC reference grid is not wired by the spec; it is resolved for the
    subject/session from ``context`` (``find_reference``), mirroring
    :func:`init_map_scalar_to_surface_wf`.
    """
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces.mrtrix3 import ComputeTDI

    from bdt.interfaces.tractography import (
        ConcatenateNiftis,
        EntitiesToSegTSV,
        ThresholdNifti,
    )

    context = context or FactoryContext()
    if context.provider is None:
        raise ValueError(
            f'tractogram_to_pseg node {node.name!r} needs a FactoryContext with a '
            'provider to resolve the ACPC reference grid.'
        )
    threshold = node.parameters.get('threshold')
    tract_ses = context.role_session(node, 'tractograms')
    reference = context.find_reference(
        {'suffix': 'dwiref', 'space': 'ACPC', 'extension': '.nii.gz'},
        session=tract_ses,
    )

    wf = pe.Workflow(name=name or node.name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['tractograms']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out', 'tsv']), name='outputnode')

    # per-bundle: .tck.gz -> .tck -> track-density image on the ACPC grid
    gunzip = pe.MapNode(Gunzip(), iterfield=['in_file'], name='gunzip')
    tck_to_tdi = pe.MapNode(
        ComputeTDI(reference=reference, out_file='tdi.nii.gz'),
        iterfield=['in_file'],
        name='tck_to_tdi',
    )
    # stack + peak-normalize -> 4D probseg
    concatenate = pe.Node(
        ConcatenateNiftis(normalize=True, out_file='pseg.nii.gz'), name='concatenate'
    )
    # label table (volume index -> bundle name), same order as inputnode.tractograms
    bundles_to_tsv = pe.Node(
        EntitiesToSegTSV(entity='bundle', out_file='dseg.tsv'), name='bundles_to_tsv'
    )

    wf.connect([
        (inputnode, gunzip, [('tractograms', 'in_file')]),
        (gunzip, tck_to_tdi, [('out_file', 'in_file')]),
        (tck_to_tdi, concatenate, [('out_file', 'in_files')]),
        (inputnode, bundles_to_tsv, [('tractograms', 'in_files')]),
        (bundles_to_tsv, outputnode, [('out_file', 'tsv')]),
    ])  # fmt:skip

    if threshold is not None:
        binarize = pe.Node(
            ThresholdNifti(
                threshold=float(threshold), binarize=True, out_file='dseg.nii.gz'
            ),
            name='binarize',
        )
        wf.connect([
            (concatenate, binarize, [('out_file', 'in_file')]),
            (binarize, outputnode, [('out_file', 'out')]),
        ])  # fmt:skip
    else:
        wf.connect([(concatenate, outputnode, [('out_file', 'out')])])

    return wf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n bdtenv pytest test/engine/test_nipype_workflow.py -k tractogram_to_pseg -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Test-suite checkpoint**

Run: `micromamba run -n bdtenv pytest test/engine/test_nipype_workflow.py -q`
Expected: all green (existing compiler/factory tests unaffected).

---

### Task 6: End-to-end compile — grouped list reaches the factory

Verifies the design's "no compiler change needed" claim: a single `select_data` matching many bundle files is delivered as a list into `inputnode.tractograms` and its `MapNode`s, through the real `init_bdt_wf` path.

**Files:**
- Test: `test/engine/test_nipype_workflow.py` (append)

**Interfaces:**
- Consumes: `init_bdt_wf(spec, selections, context=...)` (existing), the Task 5 factory, and the Task 4 registry.
- Produces: no source change — a regression test only.

- [ ] **Step 1: Write the failing test**

Append to `test/engine/test_nipype_workflow.py`:

```python
def test_tractogram_to_pseg_grouped_list_compiles(tmp_path):
    """A multi-match bundle selection compiles into the factory as a grouped list."""
    from bdt.engine.factories import FactoryContext
    from bdt.engine.selection import DictDataProvider, Match

    spec = _pseg_spec_wf(threshold=0.0)

    ref = tmp_path / 'sub-01_space-ACPC_dwiref.nii.gz'
    ref.touch()
    provider = DictDataProvider(
        {'qsiprep': [Match(str(ref), {'suffix': 'dwiref', 'space': 'ACPC'})]}
    )
    context = FactoryContext(provider=provider, subject='01', spec=spec, datasets=['qsiprep'])

    # grouped selection -> the source node carries the full list of bundle paths
    bundle_paths = [
        f'/a/sub-01_bundle-{b}_space-ACPC_streamlines.tck.gz' for b in ('CST', 'AF', 'IFOF')
    ]
    selections = {'load_bundles': bundle_paths}

    wf = init_bdt_wf(spec, selections, context=context)

    # the grouped source node holds the list unchanged, and it feeds the factory inputnode
    assert wf.get_node('load_bundles').inputs.out == bundle_paths
    assert 'bundle_rois.gunzip' in set(wf.list_node_names())
    assert 'bundle_rois.tck_to_tdi' in set(wf.list_node_names())
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `micromamba run -n bdtenv pytest test/engine/test_nipype_workflow.py -k grouped_list -v`
Expected: PASS on the first run (the grouped-list mechanism already exists via `_classify_selections`/`_combinations`; Task 5 supplies the factory). If it FAILS, the failure localizes a real gap in grouped-selection delivery — fix `init_bdt_wf` selection wiring before proceeding, do not weaken the test.

- [ ] **Step 3: (only if Step 2 failed) implement the targeted compiler fix**

If and only if Step 2 failed, make `init_bdt_wf` (`src/bdt/engine/workflow.py`) forward a grouped selection's match list as a list onto the consuming `inputnode.<role>` (the source node already stores whatever `selections[node.name]` holds; ensure the caller passes the list). Re-run Step 2 to green.

- [ ] **Step 4: Full-suite checkpoint**

Run: `micromamba run -n bdtenv pytest test/ -q`
Expected: all green across interfaces, output-plan, and workflow suites.

---

## Self-Review

**Spec coverage:**
- Factory rewrite (outputnode `out`/`tsv`, threshold from `node.parameters`, `find_reference` grid, Gunzip→ComputeTDI→ConcatenateNiftis→[ThresholdNifti]) → Task 5. ✔
- `ConcatenateNiftis` (peak-normalize + stack) → Task 1. ✔
- `ThresholdNifti` (`value > threshold`, binarize) → Task 2. ✔
- `EntitiesToSegTSV` (index/name, input order, missing-entity raise) → Task 3. ✔
- Reuse `Gunzip` + `ComputeTDI` → Task 5 (imports + nodes). ✔
- `OutputSpec.dynamic_suffix` (probseg vs dseg) + `ExtraProduct.match_primary_suffix` label TSV → Task 4. ✔
- Normalize-to-[0,1], threshold on normalized → Tasks 1 + 2 + 5 (`normalize=True`, `binarize` on the normalized stack). ✔
- Grouped-list-already-works verification → Task 6. ✔
- Tests mirroring `map_scalar_to_surface` + `test_output_plan` patterns → Tasks 4-6. ✔

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code and exact commands. ✔

**Type consistency:** `out_file` is the output field on all three interfaces and the reused nodes; the factory reads `('out_file', ...)` from each. `ConcatenateNiftis(in_files=..., normalize=...)`, `ThresholdNifti(in_file=..., threshold=..., binarize=...)`, `EntitiesToSegTSV(in_files=..., entity=...)` match between the interface tasks (1-3) and the factory (Task 5). `dynamic_suffix`/`match_primary_suffix` names match between `actions.py` and `plan.py` (Task 4). ✔

## Out of scope (from the spec)

- General fan-out over multi-match selections / list-valued roles (multiple upstream nodes into one role) — unrelated compiler follow-up.
- BIDS strictness of a 4D `dseg`/`probseg` label file.
- Whether `parcellate_scalar_as_roi` needs the label TSV alongside the atlas NIfTI downstream.
