# XCP-D Parcellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every NIfTI and CIFTI parcellation path through a real nilearn/Workbench masker with XCP-D's coverage definition and output format, deleting the hand-rolled numpy.

**Architecture:** The parcellate factory dispatches on atlas form — CIFTI to the existing XCP-D port, 3D dseg to the vendored `NiftiParcellate`, 4D to a new `ProbSegParcellate` that computes a mask-restricted weighted mean. Coverage everywhere is `|parcel ∩ brain_mask| / |parcel|`, computed from the atlas alone. The brain mask is auto-discovered; atlas labels arrive from an on-disk sidecar (selection atlases) or a new secondary compiler edge (processing atlases).

**Tech Stack:** Python 3.12, nipype, nilearn 0.14.0, nibabel, pandas, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-xcpd-parcellation-design.md`.
- Environment: **`bdtenv`**, not `lincapps`. Every command runs as `micromamba run -n bdtenv <cmd>` from `/mnt/c/Users/tsalo/Documents/linc/bdt`.
- **No git commits.** This project is edited in place; the user manages their own version control. Every task ends with a verification step, never a commit.
- Coverage is computed from the atlas and brain mask only. The data file must never enter a coverage calculation.
- Every `NiftiMasker` / `NiftiLabelsMasker` construction passes `standardize=None` (not `False`, which raises `FutureWarning` on nilearn 0.14.0) and `resampling_target=None`.
- Tests run clean under `-W error::UserWarning`.
- The CIFTI path (`_init_parcellate_cifti_wf`) is not modified by any task.

---

### Task 1: Fix the uint8 coverage bug in the vendored NiftiParcellate

`NiftiLabelsMasker(strategy='sum')` returns 0 for `uint8` input on nilearn 0.14.0, so the coverage denominator is zero and coverage evaluates to `inf`. Since `inf < min_coverage` is `False`, coverage thresholding silently never fires. This must be fixed before anything consumes the interface.

**Files:**
- Modify: `src/bdt/interfaces/connectivity.py:104-108`
- Test: `test/engine/test_nifti_parcellate.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: a working `NiftiParcellate(filtered_file, mask, atlas, atlas_labels, min_coverage)` with outputs `timeseries` and `coverage`. Tasks 5 and 7 rely on it.

- [ ] **Step 1: Write the failing test**

Create `test/engine/test_nifti_parcellate.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the vendored XCP-D NiftiParcellate."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.connectivity import NiftiParcellate  # noqa: E402


def _fixture(tmp_path):
    """Two parcels of 256 voxels; the mask covers 192 of parcel 1 and 64 of parcel 2.

    Expected coverage is therefore 192/256 = 0.75 and 64/256 = 0.25.  NaNs are
    placed in the *data* to prove coverage does not depend on it.
    """
    aff = np.eye(4)
    shape = (8, 8, 8)
    atlas = np.zeros(shape, 'int16')
    atlas[0:4, :, :] = 1
    atlas[4:8, :, :] = 2
    mask = np.zeros(shape, 'uint8')
    mask[0:3, :, :] = 1
    mask[4:8, 0:2, :] = 1
    rng = np.random.default_rng(0)
    data = (rng.random(shape + (4,)) * 10).astype('float32')
    data[3, :, :, :] = np.nan

    nb.Nifti1Image(atlas, aff).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(mask, aff).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(data, aff).to_filename(tmp_path / 'data.nii.gz')
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )
    return tmp_path


def test_coverage_is_mask_over_atlas(tmp_path, monkeypatch):
    """Coverage is |parcel n mask| / |parcel|, computed from the atlas alone.

    Regression: the binarized atlas was built as uint8, for which nilearn 0.14.0's
    ``strategy='sum'`` returns 0 -- the denominator was zero, coverage came out inf,
    and ``inf < min_coverage`` is False so no parcel was ever NaN-masked.
    """
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = NiftiParcellate(
        filtered_file=str(tmp_path / 'data.nii.gz'),
        mask=str(tmp_path / 'mask.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        min_coverage=0.5,
    ).run()

    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']
    assert np.isfinite(coverage).all(), f'non-finite coverage: {coverage.to_dict()}'
    assert coverage['A'] == pytest.approx(0.75)
    assert coverage['B'] == pytest.approx(0.25)


def test_parcels_below_min_coverage_are_nan(tmp_path, monkeypatch):
    """Parcel B (0.25 coverage) is NaN-masked at min_coverage=0.5; A (0.75) is kept."""
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = NiftiParcellate(
        filtered_file=str(tmp_path / 'data.nii.gz'),
        mask=str(tmp_path / 'mask.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        min_coverage=0.5,
    ).run()

    ts = pd.read_table(res.outputs.timeseries)
    assert list(ts.columns) == ['A', 'B']
    assert ts['B'].isna().all(), 'B is below min_coverage and must be NaN'
    assert ts['A'].notna().all(), 'A is above min_coverage and must be retained'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_nifti_parcellate.py -v`

Expected: both FAIL. `test_coverage_is_mask_over_atlas` fails the `np.isfinite` assertion (coverage is `inf`); `test_parcels_below_min_coverage_are_nan` fails because nothing is NaN-masked.

- [ ] **Step 3: Fix the dtype**

In `src/bdt/interfaces/connectivity.py`, replace:

```python
        atlas_img_bin = nb.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.uint8),
            atlas_img.affine,
            atlas_img.header,
        )
```

with:

```python
        # NOTE: deliberate divergence from upstream XCP-D, which uses np.uint8 here.
        # nilearn 0.14.0's NiftiLabelsMasker(strategy='sum') returns 0 for uint8 input
        # (correct for float), which zeroes the coverage denominator below -> coverage
        # becomes inf -> `parcel_coverage < min_coverage` is never True -> coverage
        # thresholding silently no-ops.  float32 is exact for the 0/1 values involved.
        atlas_img_bin = nb.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.float32),
            atlas_img.affine,
            atlas_img.header,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_nifti_parcellate.py -v`

Expected: 2 passed.

- [ ] **Step 5: Verify no regression**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: the 5 known pre-existing failures only (`test_outputs` x2, `test_pipeline`, `test_pybids_provider` x2); zero new failures.

---

### Task 2: ProbSegParcellate interface for 4D atlases

Handles both 4D sub-cases behind a `binarize` flag. Computes a brain-mask-restricted weighted mean using a single `NiftiMasker` call.

**Files:**
- Create: `src/bdt/interfaces/probseg.py`
- Test: `test/engine/test_probseg_parcellate.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `ProbSegParcellate` (a `SimpleInterface`) with inputs `data`, `atlas`, `atlas_labels`, `mask`, `binarize` (bool, default False), `min_coverage` (float, default 0.5) and outputs `timeseries`, `coverage`. Task 5 wires it; Task 7 asserts on it. Output files are named `timeseries.tsv` and `coverage.tsv` in the node cwd, matching `NiftiParcellate`.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_probseg_parcellate.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for ProbSegParcellate (4D probabilistic / binarized atlases)."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.probseg import ProbSegParcellate  # noqa: E402


def _write(tmp_path, atlas, mask, data, names):
    aff = np.eye(4)
    nb.Nifti1Image(atlas, aff).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(mask, aff).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(data, aff).to_filename(tmp_path / 'data.nii.gz')
    pd.DataFrame({'index': range(1, len(names) + 1), 'name': names}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )


def _run(tmp_path, **kwargs):
    return ProbSegParcellate(
        data=str(tmp_path / 'data.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        mask=str(tmp_path / 'mask.nii.gz'),
        **kwargs,
    ).run()


def test_weighted_mean_within_mask_matches_brute_force(tmp_path, monkeypatch):
    """Value is sum(w*d)/sum(w) over voxels inside the brain mask only."""
    rng = np.random.default_rng(0)
    shape = (6, 6, 6)
    atlas = rng.random(shape + (3,)).astype('float32')
    data = (rng.random(shape + (5,)) * 10).astype('float32')
    mask = np.zeros(shape, 'uint8')
    mask[1:5, 1:5, 1:5] = 1
    _write(tmp_path, atlas, mask, data, ['a', 'b', 'c'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.0)
    got = pd.read_table(res.outputs.timeseries)

    inside = mask > 0
    for i, name in enumerate(['a', 'b', 'c']):
        w = atlas[..., i][inside]
        for t in range(5):
            d = data[..., t][inside]
            assert got[name][t] == pytest.approx(float((w * d).sum() / w.sum()), rel=1e-5)


def test_binarize_equals_plain_mean_within_parcel_and_mask(tmp_path, monkeypatch):
    """With binarize=True the value is the plain mean over (parcel n mask).

    This is exactly what a per-volume NiftiLabelsMasker returns for a binary label
    image -- verified numerically during design (5.254389 vs 5.254390).
    """
    rng = np.random.default_rng(1)
    shape = (6, 6, 6)
    atlas = rng.random(shape + (2,)).astype('float32')
    data = (rng.random(shape + (1,)) * 10).astype('float32')
    mask = np.zeros(shape, 'uint8')
    mask[1:5, 1:5, 1:5] = 1
    _write(tmp_path, atlas, mask, data, ['a', 'b'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, binarize=True, min_coverage=0.0)
    got = pd.read_table(res.outputs.timeseries)

    inside = mask > 0
    for i, name in enumerate(['a', 'b']):
        sel = (atlas[..., i] > 0) & inside
        assert got[name][0] == pytest.approx(float(data[..., 0][sel].mean()), rel=1e-5)


def test_coverage_is_masked_weight_over_total_weight(tmp_path, monkeypatch):
    """Coverage is sum(w*mask)/sum(w) -- from the atlas and mask, never the data."""
    shape = (4, 4, 4)
    atlas = np.zeros(shape + (1,), 'float32')
    atlas[:, :, :, 0] = 1.0  # total weight 64
    mask = np.zeros(shape, 'uint8')
    mask[0:2, :, :] = 1  # 32 of 64 -> coverage 0.5
    data = np.full(shape + (1,), 3.0, 'float32')
    data[3, :, :, :] = np.nan  # outside the mask; must not affect coverage
    _write(tmp_path, atlas, mask, data, ['a'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.0)
    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']
    assert coverage['a'] == pytest.approx(0.5)


def test_parcels_below_min_coverage_are_nan(tmp_path, monkeypatch):
    """A parcel under min_coverage is NaN in the timeseries but keeps its coverage."""
    shape = (4, 4, 4)
    atlas = np.zeros(shape + (2,), 'float32')
    atlas[0:2, :, :, 0] = 1.0  # fully inside the mask -> coverage 1.0
    atlas[2:4, :, :, 1] = 1.0  # fully outside the mask -> coverage 0.0
    mask = np.zeros(shape, 'uint8')
    mask[0:2, :, :] = 1
    data = np.full(shape + (1,), 2.0, 'float32')
    _write(tmp_path, atlas, mask, data, ['keep', 'drop'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.5)
    ts = pd.read_table(res.outputs.timeseries)
    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']

    assert list(ts.columns) == ['keep', 'drop']
    assert ts['keep'][0] == pytest.approx(2.0)
    assert np.isnan(ts['drop'][0])
    assert coverage['drop'] == pytest.approx(0.0)


def test_scalar_input_produces_one_row(tmp_path, monkeypatch):
    """A 3D scalar is the n_timepoints == 1 case -- a 1-row wide TSV."""
    shape = (4, 4, 4)
    atlas = np.ones(shape + (1,), 'float32')
    mask = np.ones(shape, 'uint8')
    data = np.full(shape, 7.0, 'float32')  # 3D, no time axis
    _write(tmp_path, atlas, mask, data, ['a'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.0)
    ts = pd.read_table(res.outputs.timeseries)
    assert ts.shape == (1, 1)
    assert ts['a'][0] == pytest.approx(7.0)


def test_volume_count_must_match_label_count(tmp_path, monkeypatch):
    """A labels table that disagrees with the atlas volume count is a hard error."""
    shape = (4, 4, 4)
    atlas = np.ones(shape + (2,), 'float32')
    mask = np.ones(shape, 'uint8')
    data = np.ones(shape + (1,), 'float32')
    _write(tmp_path, atlas, mask, data, ['only_one'])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match='2 volumes.*1 label'):
        _run(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_probseg_parcellate.py -v`

Expected: all FAIL at import with `ModuleNotFoundError: No module named 'bdt.interfaces.probseg'`.

- [ ] **Step 3: Write the implementation**

Create `src/bdt/interfaces/probseg.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Parcellation of 4D (probabilistic or binarized) volumetric atlases.

XCP-D's :class:`~bdt.interfaces.connectivity.NiftiParcellate` covers 3D integer
label atlases via ``NiftiLabelsMasker``, which cannot represent the *overlapping*
parcels a 4D atlas encodes (one volume per region).  This module is the 4D
counterpart, sharing XCP-D's coverage definition and output format.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

LOGGER = logging.getLogger('nipype.interface')


class _ProbSegParcellateInputSpec(BaseInterfaceInputSpec):
    data = File(
        exists=True,
        mandatory=True,
        desc='3D scalar or 4D timeseries NIfTI, already on the atlas grid',
    )
    atlas = File(
        exists=True, mandatory=True, desc='4D atlas: one volume per region (pseg or dseg)'
    )
    atlas_labels = File(
        exists=True, mandatory=True, desc='BIDS dseg.tsv with index/name columns'
    )
    mask = File(exists=True, mandatory=True, desc='brain mask on the data grid')
    binarize = traits.Bool(
        False,
        usedefault=True,
        desc=(
            'Binarize each atlas volume before averaging.  Set for a thresholded '
            '(dseg) 4D atlas, where the weighted mean reduces to the plain mean '
            'over the parcel -- identical to a per-volume NiftiLabelsMasker.'
        ),
    )
    min_coverage = traits.Float(
        0.5,
        usedefault=True,
        desc='Parcels with coverage below this are replaced with NaN.',
    )


class _ProbSegParcellateOutputSpec(TraitedSpec):
    timeseries = File(exists=True, desc='Parcellated time series file.')
    coverage = File(exists=True, desc='Parcel-wise coverage file.')


class ProbSegParcellate(SimpleInterface):
    """Mask-restricted weighted mean of a 4D atlas' regions.

    ``value = sum(w * d) / sum(w)`` over voxels inside the brain mask, where ``w``
    is the region's probability map.  ``coverage = sum(w * mask) / sum(w)``, taken
    from the atlas and mask alone -- the data never enters it, on the assumption
    (see the design doc) that the brain mask already excludes NaN and
    zero-variance voxels.

    ``NiftiMapsMasker`` is deliberately not used: its extraction is a least-squares
    unmixing that returns ``sum(w*d)/sum(w**2)`` even one map at a time, which is a
    scaling coefficient rather than a mean.
    """

    input_spec = _ProbSegParcellateInputSpec
    output_spec = _ProbSegParcellateOutputSpec

    def _run_interface(self, runtime):
        atlas_img = nb.load(self.inputs.atlas)
        if atlas_img.ndim != 4:
            raise ValueError(
                f'ProbSegParcellate expects a 4D atlas, got {atlas_img.ndim}D: '
                f'{self.inputs.atlas}'
            )
        n_parcels = atlas_img.shape[3]

        labels_df = pd.read_table(self.inputs.atlas_labels).sort_values(by='index')
        names = labels_df['name'].astype(str).tolist()
        if len(names) != n_parcels:
            raise ValueError(
                f'Atlas {self.inputs.atlas} has {n_parcels} volumes but '
                f'{self.inputs.atlas_labels} has {len(names)} labels; they must agree '
                '(a 4D atlas volume maps to the 1-based index in the labels table).'
            )

        # standardize=None: standardize=False raises a FutureWarning on nilearn 0.14.
        masker = NiftiMasker(mask_img=self.inputs.mask, standardize=None)
        weights = np.atleast_2d(masker.fit_transform(self.inputs.atlas))  # (n_parcels, n_vox)
        data = np.atleast_2d(masker.transform(self.inputs.data))  # (n_t, n_vox)

        total = np.asarray(atlas_img.dataobj, dtype='float64')
        total = total.reshape(-1, n_parcels)
        if self.inputs.binarize:
            weights = (weights > 0).astype('float64')
            total = (total > 0).astype('float64')
        total_weight = total.sum(axis=0)

        covered = weights.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(total_weight > 0, covered / total_weight, 0.0)

        n_t = data.shape[0]
        values = np.full((n_t, n_parcels), np.nan, dtype='float64')
        usable = (covered > 0) & (coverage >= self.inputs.min_coverage)
        if usable.any():
            # (n_parcels, n_vox) @ (n_vox, n_t) -> (n_parcels, n_t), then transpose
            num = weights[usable] @ data.T
            values[:, usable] = (num / covered[usable][:, None]).T

        n_dropped = int((~usable).sum())
        if n_dropped:
            LOGGER.warning(
                '%d/%d parcels fall below min_coverage=%.2f and are set to NaN.',
                n_dropped,
                n_parcels,
                self.inputs.min_coverage,
            )

        self._results['timeseries'] = os.path.join(runtime.cwd, 'timeseries.tsv')
        pd.DataFrame(values, columns=names).to_csv(
            self._results['timeseries'], sep='\t', na_rep='n/a', index=False
        )

        self._results['coverage'] = os.path.join(runtime.cwd, 'coverage.tsv')
        pd.DataFrame(
            coverage.astype(np.float32), index=names, columns=['coverage']
        ).to_csv(self._results['coverage'], sep='\t', na_rep='n/a', index_label='Node')

        return runtime
```

- [ ] **Step 4: Register the module**

In `src/bdt/interfaces/__init__.py`, add `probseg` to the import line and `__all__`, keeping alphabetical order:

```python
from . import bids, probseg, reportlets, transforms

__all__ = ['bids', 'probseg', 'reportlets', 'transforms']
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_probseg_parcellate.py -v`

Expected: 6 passed.

- [ ] **Step 6: Verify the suite is clean**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_probseg_parcellate.py -W error::UserWarning -q`

Expected: 6 passed, no warnings escalated to errors.

---

### Task 3: FactoryContext helpers for atlas form, labels, and datatype

**Files:**
- Modify: `src/bdt/engine/factories.py` (add methods to `FactoryContext`, after `role_suffix` at ~line 107)
- Test: `test/engine/test_factory_context_transforms.py` (append)

**Interfaces:**
- Consumes: `FactoryContext._role_entity`, `FactoryContext.resolved` (a `{node_name: Match}` dict of *selection* matches only).
- Produces:
  - `FactoryContext.role_datatype(node, role, default=None) -> str | None`
  - `FactoryContext.role_atlas_ndim(node, role='atlas') -> int | None` — 3 or 4
  - `FactoryContext.role_atlas_labels(node, role='atlas') -> str | None` — sidecar path for a selection atlas, `None` for a processing atlas (whose labels arrive over the wire from Task 4)

  Task 5 uses all three.

- [ ] **Step 1: Write the failing tests**

Append to `test/engine/test_factory_context_transforms.py`:

```python
def test_role_atlas_ndim_reads_header_for_selection_atlas(tmp_path):
    """A selected atlas exists on disk at build time, so read its header."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / 'tpl-X_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), 'int16'), np.eye(4)).to_filename(path)
    ctx = FactoryContext(resolved={'atlas_sel': _StubMatch(str(path), {'suffix': 'dseg'})})
    node = _node({'atlas': ['atlas_sel']})
    assert ctx.role_atlas_ndim(node, 'atlas') == 3


def test_role_atlas_ndim_is_four_for_processing_atlas():
    """A processing-node atlas does not exist yet at build time, but is 4D by
    construction: tractogram_to_pseg is the only atlas-producing action and it
    stacks bundles via ConcatenateNiftis."""
    ctx = FactoryContext(resolved={})
    node = _node({'atlas': ['bundle_rois']})
    assert ctx.role_atlas_ndim(node, 'atlas') == 4


def test_role_atlas_labels_finds_sibling_sidecar(tmp_path):
    """AtlasPack ships tpl-..._dseg.tsv beside tpl-..._dseg.nii.gz."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / 'tpl-X_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), 'int16'), np.eye(4)).to_filename(path)
    sidecar = tmp_path / 'tpl-X_atlas-Y_dseg.tsv'
    sidecar.write_text('index\tname\n1\tA\n')
    ctx = FactoryContext(resolved={'atlas_sel': _StubMatch(str(path), {})})
    node = _node({'atlas': ['atlas_sel']})
    assert ctx.role_atlas_labels(node, 'atlas') == str(sidecar)


def test_role_atlas_labels_errors_when_sidecar_missing(tmp_path):
    """A selected atlas with no sidecar is a hard error naming the expected path."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / 'tpl-X_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), 'int16'), np.eye(4)).to_filename(path)
    ctx = FactoryContext(resolved={'atlas_sel': _StubMatch(str(path), {})})
    node = _node({'atlas': ['atlas_sel']})
    with pytest.raises(ValueError, match='no labels sidecar'):
        ctx.role_atlas_labels(node, 'atlas')


def test_role_atlas_labels_none_for_processing_atlas():
    """A processing-node atlas gets its labels over the wire, not from disk."""
    ctx = FactoryContext(resolved={})
    node = _node({'atlas': ['bundle_rois']})
    assert ctx.role_atlas_labels(node, 'atlas') is None


def test_role_datatype_reads_resolved_entities():
    ctx = FactoryContext(
        resolved={'load_bold': _StubMatch('bold.nii.gz', {'datatype': 'func'})}
    )
    node = _node({'timeseries': ['load_bold']})
    assert ctx.role_datatype(node, 'timeseries') == 'func'
    assert ctx.role_datatype(node, 'atlas') is None
```

Add `import pytest` to the top of the file (after the module docstring, before the `from types import SimpleNamespace` line).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py -v -k "atlas_ndim or atlas_labels or role_datatype"`

Expected: 6 FAIL with `AttributeError: 'FactoryContext' object has no attribute 'role_atlas_ndim'` (and similar).

- [ ] **Step 3: Implement the helpers**

In `src/bdt/engine/factories.py`, insert immediately after the `role_suffix` method:

```python
    def role_datatype(self, node, role: str, default: str | None = None) -> str | None:
        """The ``datatype`` entity of the file feeding ``node``'s ``role``."""
        return self._role_entity(node, role, 'datatype', default)

    def role_atlas_ndim(self, node, role: str = 'atlas') -> int | None:
        """Dimensionality of the atlas feeding ``role``, resolved at *build* time.

        A **selected** atlas exists on disk while the graph is built, so its header
        is read (a header read, not a data read).  A **processing** atlas does not
        exist yet, but is 4D by construction: ``tractogram_to_pseg`` is the only
        atlas-producing action and it stacks bundles via ``ConcatenateNiftis``.
        Warping preserves dimensionality, so a reading taken from the original
        selection stays valid for the warped atlas that reaches the masker.
        """
        import nibabel as nb

        for up in node.inputs.get(role, []):
            match = (self.resolved or {}).get(up)
            if match is None:
                return 4
            try:
                return int(nb.load(match.path).ndim)
            except Exception as exc:  # unreadable header -> name it, never guess
                raise ValueError(
                    f'Could not read the atlas header for role {role!r} of node '
                    f'{getattr(node, "name", node)!r}: {match.path}'
                ) from exc
        return None

    def role_atlas_labels(self, node, role: str = 'atlas') -> str | None:
        """Path to the BIDS ``dseg.tsv`` describing the atlas feeding ``role``.

        For a **selected** atlas this is the sibling sidecar (AtlasPack ships
        ``tpl-..._dseg.tsv`` beside ``tpl-..._dseg.nii.gz``).  It must be resolved
        from the *original selection*: the warped atlas in the node cwd has no
        sidecar beside it.  For a **processing** atlas this returns ``None`` --
        those labels arrive over a wired ``tsv`` edge instead.
        """
        import os

        for up in node.inputs.get(role, []):
            match = (self.resolved or {}).get(up)
            if match is None:
                return None
            path = match.path
            for ext in ('.nii.gz', '.nii'):
                if path.endswith(ext):
                    sidecar = path[: -len(ext)] + '.tsv'
                    if os.path.exists(sidecar):
                        return sidecar
                    raise ValueError(
                        f'Atlas {path} has no labels sidecar at {sidecar}. '
                        'A volumetric atlas needs a BIDS dseg.tsv (index/name) to name '
                        'its parcels and to detect parcels lost when the atlas is warped.'
                    )
            raise ValueError(f'Unrecognized atlas extension (expected .nii/.nii.gz): {path}')
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py -v`

Expected: all pass (the 6 new ones plus the pre-existing ones).

---

### Task 4: Carry a secondary `tsv` edge through role wiring

The one compiler change. Strictly additive: the edge is wired only when the upstream exposes `outputnode.tsv` **and** the downstream declares `inputnode.<role>_labels`. Nothing else in the graph changes.

**Files:**
- Modify: `src/bdt/engine/workflow.py:118-124`
- Test: `test/engine/test_workflow_label_wiring.py` (create)

**Interfaces:**
- Consumes: `built` (`{node_name: (kind, workflow_or_node)}`) from `init_bdt_wf`.
- Produces: the convention that a subworkflow wanting a processing atlas' labels declares an `inputnode` field named `f'{role}_labels'` (so, for the `atlas` role: `atlas_labels`). Task 5 relies on this.

- [ ] **Step 1: Write the failing test**

Create `test/engine/test_workflow_label_wiring.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The secondary labels (``tsv``) edge carried alongside a role's primary output."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from bdt.engine.workflow import _identity_fields


def _wf(inputs=None, outputs=None):
    wf = pe.Workflow(name='sub')
    if inputs:
        wf.add_nodes([pe.Node(niu.IdentityInterface(fields=inputs), name='inputnode')])
    if outputs:
        wf.add_nodes([pe.Node(niu.IdentityInterface(fields=outputs), name='outputnode')])
    return wf


def test_identity_fields_reads_declared_fields():
    wf = _wf(inputs=['atlas', 'atlas_labels'], outputs=['out', 'tsv'])
    assert _identity_fields(wf, 'inputnode') == {'atlas', 'atlas_labels'}
    assert _identity_fields(wf, 'outputnode') == {'out', 'tsv'}


def test_identity_fields_missing_node_is_empty():
    assert _identity_fields(_wf(), 'inputnode') == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_workflow_label_wiring.py -v`

Expected: FAIL with `ImportError: cannot import name '_identity_fields'`.

- [ ] **Step 3: Implement the helper and the wiring**

In `src/bdt/engine/workflow.py`, add this module-level function just above `init_bdt_wf`'s definition (or above `_attach_sinks` if that reads better in context):

```python
def _identity_fields(wf_obj, node_name: str) -> set[str]:
    """The field names of an ``IdentityInterface`` node inside ``wf_obj``.

    Returns an empty set when the node is absent, so callers can probe for an
    optional convention without branching on existence.
    """
    node = wf_obj.get_node(node_name)
    if node is None:
        return set()
    return set(getattr(node.interface, '_fields', ()) or ())
```

Then replace the role-wiring loop body:

```python
        for role, upstream_names in node.inputs.items():
            up_kind, up_obj = built[upstream_names[0]]  # single-match; fan-out is a follow-up
            src_field = 'out' if up_kind == 'selection' else 'outputnode.out'
            wf.connect(up_obj, src_field, downstream, f'inputnode.{role}')
```

with:

```python
        for role, upstream_names in node.inputs.items():
            up_kind, up_obj = built[upstream_names[0]]  # single-match; fan-out is a follow-up
            src_field = 'out' if up_kind == 'selection' else 'outputnode.out'
            wf.connect(up_obj, src_field, downstream, f'inputnode.{role}')

            # Secondary labels edge.  A processing node that produces a segmentation
            # also produces its label table (e.g. tractogram_to_pseg -> EntitiesToSegTSV
            # on outputnode.tsv); a consumer that needs it declares inputnode.<role>_labels.
            # Strictly additive: wired only when both sides opt in, so every other
            # action's graph is byte-identical.
            if up_kind == 'processing' and 'tsv' in _identity_fields(up_obj, 'outputnode'):
                if f'{role}_labels' in _identity_fields(downstream, 'inputnode'):
                    wf.connect(
                        up_obj, 'outputnode.tsv', downstream, f'inputnode.{role}_labels'
                    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_workflow_label_wiring.py -v`

Expected: 2 passed.

- [ ] **Step 5: Verify no existing graph changed**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: the 5 known pre-existing failures only; zero new. No current subworkflow declares `<role>_labels`, so this task must be a no-op for every existing graph.

---

### Task 5: Route the parcellate factories by atlas form

Replaces `_init_parcellate_volumetric_wf` and `_init_parcellate_volumetric_timeseries_wf` with a single builder that dispatches on `role_atlas_ndim` and wires the discovered brain mask and atlas labels.

**Files:**
- Modify: `src/bdt/engine/factories.py` — replace `_init_parcellate_volumetric_wf` (~line 517) and `_init_parcellate_volumetric_timeseries_wf` (~line 389)
- Test: `test/engine/test_parcellate_routing_masker.py` (create)

**Interfaces:**
- Consumes: `ProbSegParcellate` (Task 2); `role_atlas_ndim`, `role_atlas_labels`, `role_datatype` (Task 3); the `<role>_labels` inputnode convention (Task 4); `NiftiParcellate` (Task 1); the existing `_warp_atlas_field(wf, node, context, inputnode, data_role)`.
- Produces: `_init_parcellate_volumetric_wf(node, name, context, data_role)` building a subworkflow whose `outputnode` has fields `['out', 'coverage']` for both scalar and timeseries. Task 7 asserts on node names `parcellate`, `warp_atlas`.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_parcellate_routing_masker.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Volumetric parcellate routing: 3D dseg -> NiftiParcellate, 4D -> ProbSegParcellate."""

from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pytest

from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf
from bdt.engine.selection import DictDataProvider, Match


def _atlas(tmp_path, ndim, name='tpl-MNI152NLin6Asym_atlas-Y_dseg.nii.gz'):
    shape = (4, 4, 4) if ndim == 3 else (4, 4, 4, 2)
    path = tmp_path / name
    nb.Nifti1Image(np.zeros(shape, 'float32'), np.eye(4)).to_filename(path)
    (tmp_path / name.replace('.nii.gz', '.tsv')).write_text('index\tname\n1\tA\n2\tB\n')
    return str(path)


def _node(**parameters):
    return SimpleNamespace(
        name='parcellate_bold',
        inputs={'timeseries': ['load_bold'], 'atlas': ['atlas_sel']},
        parameters=parameters,
        desc=None,
    )


def _context(atlas_path, mask_path, space='MNI152NLin6Asym', suffix='dseg'):
    resolved = {
        'load_bold': Match(
            '/d/sub-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz',
            {'space': space, 'suffix': 'bold', 'datatype': 'func', 'extension': '.nii.gz'},
        ),
        'atlas_sel': Match(
            atlas_path,
            {'space': space, 'suffix': suffix, 'extension': '.nii.gz'},
        ),
    }
    provider = DictDataProvider({
        'fmriprep': [
            Match(mask_path, {'space': space, 'suffix': 'mask', 'desc': 'brain',
                              'datatype': 'func'}),
        ],
    })
    return FactoryContext(
        resolved=resolved, provider=provider, subject='01', datasets=['fmriprep']
    )


def _mask(tmp_path):
    path = tmp_path / 'sub-01_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(path)
    return str(path)


def test_3d_dseg_atlas_uses_xcpd_nifti_parcellate(tmp_path):
    from bdt.interfaces.connectivity import NiftiParcellate

    ctx = _context(_atlas(tmp_path, 3), _mask(tmp_path))
    wf = init_parcellate_timeseries_wf(_node(min_coverage=0.5), context=ctx)
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, NiftiParcellate)
    assert parcellate.inputs.min_coverage == 0.5
    # the sidecar is resolved from the selection, not the warped atlas
    assert parcellate.inputs.atlas_labels.endswith('_dseg.tsv')


def test_4d_probseg_atlas_uses_probseg_parcellate_unbinarized(tmp_path):
    from bdt.interfaces.probseg import ProbSegParcellate

    ctx = _context(_atlas(tmp_path, 4), _mask(tmp_path), suffix='probseg')
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, ProbSegParcellate)
    assert parcellate.inputs.binarize is False


def test_4d_dseg_atlas_binarizes(tmp_path):
    """A thresholded 4D atlas (suffix dseg) is binarized before averaging."""
    from bdt.interfaces.probseg import ProbSegParcellate

    ctx = _context(_atlas(tmp_path, 4), _mask(tmp_path), suffix='dseg')
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, ProbSegParcellate)
    assert parcellate.inputs.binarize is True


def test_brain_mask_is_discovered_and_wired(tmp_path):
    mask = _mask(tmp_path)
    ctx = _context(_atlas(tmp_path, 3), mask)
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    assert wf.get_node('parcellate').inputs.mask == mask


def test_missing_brain_mask_is_a_hard_error(tmp_path):
    """No data-derived fallback: a missing mask names the failed query and stops."""
    ctx = _context(_atlas(tmp_path, 3), _mask(tmp_path))
    ctx.provider = DictDataProvider({'fmriprep': []})
    with pytest.raises(ValueError, match="'suffix': 'mask'"):
        init_parcellate_timeseries_wf(_node(), context=ctx)


def test_outputnode_exposes_out_and_coverage(tmp_path):
    ctx = _context(_atlas(tmp_path, 3), _mask(tmp_path))
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    from bdt.engine.workflow import _identity_fields

    assert _identity_fields(wf, 'outputnode') == {'out', 'coverage'}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_routing_masker.py -v`

Expected: FAIL — the current factory builds `ParcellateVolumetricTimeseries`, has no `mask` input, and its scalar variant exposes only `out`.

- [ ] **Step 3: Replace the two volumetric builders**

In `src/bdt/engine/factories.py`, delete `_init_parcellate_volumetric_wf` and `_init_parcellate_volumetric_timeseries_wf` entirely and add this single replacement in their place:

```python
def _discover_brain_mask(context, node, data_role: str) -> str:
    """The brain mask matching the data's space, session, and datatype.

    Deliberately has no data-derived fallback: synthesizing a mask from the data's
    finite/non-zero support would make coverage depend on the data, contradicting
    XCP-D's definition (|parcel n mask| / |parcel|).  A space with no matching mask
    is a spec problem to fix, so this raises with the failed query named.
    """
    filters = {'suffix': 'mask', 'desc': 'brain', 'space': context.role_space(node, data_role)}
    datatype = context.role_datatype(node, data_role)
    if datatype:
        filters['datatype'] = datatype
    return context.find_reference(filters, session=context.role_session(node, data_role))


def _init_parcellate_volumetric_wf(node, name, context, data_role: str) -> pe.Workflow:
    """Volumetric parcellation, dispatched on atlas form.

    3D integer-label atlas -> XCP-D's :class:`NiftiParcellate` (``NiftiLabelsMasker``).
    4D atlas (one volume per region, possibly overlapping) ->
    :class:`~bdt.interfaces.probseg.ProbSegParcellate`, binarized when the atlas is a
    thresholded ``dseg``.  ``outputnode`` exposes ``out`` (the wide TSV; a scalar is
    the one-row case) and ``coverage``.
    """
    from bdt.interfaces.connectivity import NiftiParcellate
    from bdt.interfaces.probseg import ProbSegParcellate

    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))

    fields = [data_role, 'atlas']
    labels_path = context.role_atlas_labels(node, 'atlas')
    if labels_path is None:
        # processing-node atlas: labels arrive over the secondary edge (workflow.py)
        fields.append('atlas_labels')
    inputnode = pe.Node(niu.IdentityInterface(fields=fields), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out', 'coverage']), name='outputnode'
    )

    ndim = context.role_atlas_ndim(node, 'atlas')
    mask = _discover_brain_mask(context, node, data_role)

    if ndim == 3:
        parcellate = pe.Node(
            NiftiParcellate(mask=mask, min_coverage=min_coverage), name='parcellate'
        )
        data_field = 'filtered_file'
    else:
        atlas_suffix = context.role_suffix(node, 'atlas', default='probseg')
        parcellate = pe.Node(
            ProbSegParcellate(
                mask=mask,
                min_coverage=min_coverage,
                binarize=atlas_suffix == 'dseg',
            ),
            name='parcellate',
        )
        data_field = 'data'

    if labels_path is not None:
        parcellate.inputs.atlas_labels = labels_path
    else:
        wf.connect([(inputnode, parcellate, [('atlas_labels', 'atlas_labels')])])

    atlas_node, atlas_field = _warp_atlas_field(wf, node, context, inputnode, data_role)
    wf.connect([
        (inputnode, parcellate, [(data_role, data_field)]),
        (atlas_node, parcellate, [(atlas_field, 'atlas')]),
        (parcellate, outputnode, [('timeseries', 'out'), ('coverage', 'coverage')]),
    ])  # fmt:skip
    return wf
```

- [ ] **Step 4: Update the two call sites**

In `init_parcellate_scalar_wf`, replace the volumetric return with:

```python
    return _init_parcellate_volumetric_wf(node, name, context, 'scalar')
```

In `init_parcellate_timeseries_wf`, replace the volumetric return with:

```python
    return _init_parcellate_volumetric_wf(node, name, context, 'timeseries')
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_routing_masker.py -v`

Expected: 6 passed.

- [ ] **Step 6: Run the pre-existing routing tests**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_scalar_routing.py test/engine/test_parcellate_timeseries_routing.py -v`

Expected failures are confined to assertions that name the deleted interfaces. Apply exactly these substitutions:

- `from bdt.interfaces.parcellate import ParcellateVolumetric` → `from bdt.interfaces.connectivity import NiftiParcellate`, and the corresponding `isinstance(..., ParcellateVolumetric)` → `isinstance(..., NiftiParcellate)`.
- `ParcellateVolumetricTimeseries` → `NiftiParcellate` for 3D-dseg fixtures, or `ProbSegParcellate` (from `bdt.interfaces.probseg`) for 4D fixtures.
- Any fixture whose atlas `Match` has no on-disk file must now point at a real temporary NIfTI plus its `.tsv` sidecar, because `role_atlas_ndim` reads the header and `role_atlas_labels` requires the sidecar. Use the `_atlas()` helper from `test_parcellate_routing_masker.py`.
- Any context lacking a provider must gain a `DictDataProvider` supplying a `desc-brain` mask in the data's space, since `_discover_brain_mask` now runs on every volumetric build.

Do **not** weaken the cross-space, ACPC-bridge, or multi-session assertions — `_warp_atlas_field` is untouched by this task and they must still pass unchanged. If one of them fails, that is a real regression, not a fixture problem.

- [ ] **Step 7: Write the failing test for the scalar coverage product**

Append to `test/engine/test_output_plan.py`:

```python
def test_volumetric_parcellate_scalar_plans_a_coverage_tsv():
    """parcellate_scalar writes coverage as a derivative, like parcellate_timeseries.

    Volumetric -> .tsv (via ExtraProduct.volumetric_extension); CIFTI -> .pscalar.nii.
    """
    spec = load_spec('scripts/tract_parcellate.yml')
    plan = build_sink_plan(spec, resolved={}, cifti_by_node={})
    products = [p for node_products in plan.values() for p in node_products]
    coverage = [p for p in products if p.stat == 'coverage']
    assert coverage, 'parcellate_scalar must plan a coverage product'
    assert all(p.extension == '.tsv' for p in coverage)
```

Adapt the `load_spec` / `build_sink_plan` call signature and the plan's return shape to whatever the surrounding tests in that file already use — do not invent a new API.

- [ ] **Step 8: Add the ExtraProduct**

In `src/bdt/spec/actions.py`, in the `parcellate_scalar` `ActionSpec`'s `out=_o(...)`, add an `extra` tuple mirroring `parcellate_timeseries`:

```python
            extra=(
                ExtraProduct(
                    'coverage', 'map', '.pscalar.nii',
                    volumetric_extension='.tsv', cifti_only=False, stat='coverage',
                ),
            ),
```

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_output_plan.py -v`

Expected: the new test passes. Other tests in the file may need their expected-product counts incremented by one for `parcellate_scalar` nodes — that is a legitimate consequence of this decision, not a regression.

---

### Task 6: Functional connectivity via XCP-D's TSVConnect

**Files:**
- Modify: `src/bdt/engine/factories.py` — `init_functional_connectivity_wf` volumetric branch (~line 560)
- Test: `test/engine/test_functional_connectivity_routing.py` (modify)

**Interfaces:**
- Consumes: `TSVConnect` from `src/bdt/interfaces/connectivity.py` — input `timeseries`, output `correlations`.
- Produces: the volumetric FC branch builds a node named `correlate` (not `correlate_tsv`) whose `outputnode.out` is a `Node`-labelled relmat TSV. Task 7 asserts on it.

- [ ] **Step 1: Write the failing test**

In `test/engine/test_functional_connectivity_routing.py`, replace the `test_fc_volumetric_uses_tsv_correlation` test with:

```python
def test_fc_volumetric_uses_xcpd_tsv_connect():
    """A volumetric parcellated TSV correlates via XCP-D's TSVConnect."""
    from bdt.interfaces.connectivity import TSVConnect

    wf = init_functional_connectivity_wf(_node(), context=_volumetric_context())
    correlate = wf.get_node('correlate')
    assert isinstance(correlate.interface, TSVConnect)
    assert 'correlate_tsv' not in set(wf.list_node_names())
```

Keep the existing CIFTI test unchanged. Reuse whatever `_node()` / `_volumetric_context()` helpers the file already defines.

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_functional_connectivity_routing.py -v`

Expected: FAIL — `wf.get_node('correlate')` is `None` in the volumetric branch, which still builds `correlate_tsv`.

- [ ] **Step 3: Swap the implementation**

In `init_functional_connectivity_wf`, replace the whole volumetric branch:

```python
    from bdt.utils.cifti import tsv_correlation

    correlate = pe.Node(
        niu.Function(
            function=tsv_correlation,
            input_names=['tsv_path', 'out_path'],
            output_names=['out'],
        ),
        name='correlate_tsv',
    )
    correlate.inputs.out_path = 'correlations.tsv'
    wf.connect([
        (inputnode, correlate, [('timeseries', 'tsv_path')]),
        (correlate, outputnode, [('out', 'out')]),
    ])  # fmt:skip
    return wf
```

with:

```python
    from bdt.interfaces.connectivity import TSVConnect

    correlate = pe.Node(TSVConnect(), name='correlate')
    wf.connect([
        (inputnode, correlate, [('timeseries', 'timeseries')]),
        (correlate, outputnode, [('correlations', 'out')]),
    ])  # fmt:skip
    return wf
```

Also update the docstring's second paragraph to read:

```
    CIFTI ptseries -> :class:`CiftiCorrelation` -> pconn; a volumetric parcellated
    TSV -> XCP-D's :class:`~bdt.interfaces.connectivity.TSVConnect` -> a relmat TSV
    with ``Node`` row labels.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_functional_connectivity_routing.py -v`

Expected: all pass.

---

### Task 7: Delete the hand-rolled code and add acceptance tests

**Files:**
- Delete: `src/bdt/interfaces/parcellate.py`, `test/engine/test_parcellate_volumetric.py`
- Modify: `src/bdt/utils/cifti.py` (remove `tsv_correlation`), `test/engine/test_cifti_utils.py` (remove its two tests)
- Test: `test/engine/test_parcellate_routing_masker.py` (append acceptance tests)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing acceptance tests**

Append to `test/engine/test_parcellate_routing_masker.py`:

```python
def test_nifti_parcellate_yml_routes_to_xcpd_maskers(tmp_path):
    """Acceptance: scripts/nifti_parcellate.yml compiles onto the XCP-D maskers.

    The bold is MNI152NLin6Asym and the AtlasPack atlas is MNI152NLin2009cAsym, so
    warp_atlas must be present; the atlas is a 3D dseg, so NiftiParcellate handles
    it; and FC must use TSVConnect.  Entities are as a real BIDSDataProvider returns
    them (tpl-, no space entity).
    """
    from bdt.engine.workflow import init_bdt_wf
    from bdt.interfaces.connectivity import NiftiParcellate, TSVConnect
    from bdt.spec import load_spec

    atlas = _atlas(tmp_path, 3, name='tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_dseg.nii.gz')
    mask = _mask(tmp_path)
    spec = load_spec('scripts/nifti_parcellate.yml')

    resolved = {
        'load_bold': Match(
            '/d/sub-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz',
            {'space': 'MNI152NLin6Asym', 'desc': 'preproc', 'suffix': 'bold',
             'datatype': 'func', 'extension': '.nii.gz'},
        ),
        'atlas_4s456': Match(
            atlas,
            {'template': 'MNI152NLin2009cAsym', 'atlas': '4S456Parcels',
             'suffix': 'dseg', 'extension': '.nii.gz', 'res': '01'},
        ),
    }
    provider = DictDataProvider({
        'fmriprep': [
            Match(mask, {'space': 'MNI152NLin6Asym', 'suffix': 'mask',
                         'desc': 'brain', 'datatype': 'func'}),
        ],
        'atlases': [],
    })
    context = FactoryContext(
        spec=spec, resolved=resolved, provider=provider, subject='01',
        datasets=['fmriprep', 'atlases'],
    )
    wf = init_bdt_wf(spec, context=context, selections={})
    names = set(wf.list_node_names())

    parcellate = wf.get_node('parcellate_bold').get_node('parcellate')
    assert isinstance(parcellate.interface, NiftiParcellate)
    assert wf.get_node('parcellate_bold').get_node('warp_atlas') is not None
    assert isinstance(wf.get_node('fc_bold').get_node('correlate').interface, TSVConnect)
    assert not any('vertex_mask' in n or 'restrict_atlas' in n for n in names)


def test_hand_rolled_parcellate_module_is_gone():
    """The bespoke numpy implementation must not survive alongside the maskers."""
    import pytest

    with pytest.raises(ModuleNotFoundError):
        import bdt.interfaces.parcellate  # noqa: F401

    import bdt.utils.cifti as cifti

    assert not hasattr(cifti, 'tsv_correlation')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_routing_masker.py -v -k "yml_routes or hand_rolled"`

Expected: `test_hand_rolled_parcellate_module_is_gone` FAILS (the module still imports).

- [ ] **Step 3: Delete the hand-rolled code**

```bash
rm src/bdt/interfaces/parcellate.py test/engine/test_parcellate_volumetric.py
```

In `src/bdt/utils/cifti.py`, delete the entire `tsv_correlation` function. In `test/engine/test_cifti_utils.py`, delete `test_tsv_correlation` and `test_tsv_correlation_runs_as_a_nipype_function_node`, and remove `tsv_correlation` from the `from bdt.utils.cifti import (...)` block (leave `os` imported only if still used; remove it otherwise).

- [ ] **Step 4: Run the acceptance tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_routing_masker.py -v`

Expected: all pass.

- [ ] **Step 5: Confirm no stale references remain**

Run: `grep -rn "ParcellateVolumetric\|tsv_correlation\|interfaces.parcellate\|correlate_tsv" src/ test/ scripts/`

Expected: no output.

- [ ] **Step 6: Full verification**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: the 5 known pre-existing failures only (`test_outputs` x2, `test_pipeline`, `test_pybids_provider` x2); zero new.

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -W error::UserWarning -q`

Expected: the same 5; no warnings escalated.

- [ ] **Step 7: Record the outcome**

Append a Spec-4 section to `.superpowers/sdd/progress.md` recording per-task status, the `uint8` coverage fix as a deliberate upstream divergence, and the `tract_parcellate.yml` output-format change (tidy long -> wide + separate coverage).

---

## Notes for the implementer

- **`tract_parcellate.yml` output format changes.** Its parcellated scalar becomes a one-row wide TSV plus a separate coverage TSV. This is intended and approved; do not add a compatibility shim.
- **The binarized branch.** The spec's thresholded-4D case was described as "`NiftiLabelsMasker` per binarized volume". Task 2 implements it as the same weighted mean with binary weights, which is mathematically identical (verified: 5.254389 vs 5.254390) and avoids fitting one masker per bundle. `test_binarize_equals_plain_mean_within_parcel_and_mask` pins the equivalence. If a reviewer objects to the deviation, the fix is to loop `NiftiLabelsMasker` per volume — the test does not change.
- **Never let the data into coverage.** If a task tempts you to compute coverage from `isfinite(data)` or `data != 0`, stop: that was the old hand-rolled semantics this work exists to remove.

## Resolved: parcellate_scalar emits coverage (decided 2026-07-21)

`parcellate_scalar` gains a coverage `ExtraProduct` mirroring
`parcellate_timeseries`, so both paths write coverage as a real derivative. This
adds one output file per `parcellate_scalar` node, **including on the CIFTI scalar
path**, which will now emit a `.pscalar.nii` coverage map it does not produce
today. That is intended.

Implemented as Task 5, Steps 7-8 below.
