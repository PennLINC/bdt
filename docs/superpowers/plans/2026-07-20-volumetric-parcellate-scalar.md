# Volumetric `parcellate_scalar` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the volumetric (NIfTI) Strategy-A path to the existing `parcellate_scalar` action so `fa_roi` (same-space) and `cbf_roi` (cross-space, atlas warped ACPC→MNI) run, consuming Spec 1's `ResolveApplyTransforms`.

**Architecture:** `init_parcellate_scalar_wf` branches at build time on the scalar's file type: CIFTI keeps the existing coverage-aware CIFTI subworkflow (untouched); volumetric routes through a new subworkflow that (optionally) warps the atlas into the scalar's space via `ResolveApplyTransforms` and then parcellates it with a new coverage-aware `ParcellateVolumetric` interface. The atlas warp reuses the established `init_map_scalar_to_surface_wf` cross-space pattern (computed ACPC↔T1w bridge + transform discovery), but resolves an arbitrary-length chain via the Spec 1 graph.

**Tech Stack:** Python 3.12, nipype, nitransforms 25.1.0, nibabel, numpy, pandas; micromamba env `bdtenv`.

## Global Constraints

- Environment: `bdtenv` micromamba env. Run tests via `micromamba run -n bdtenv python -m pytest`.
- **No git commits** — this project edits in place; the user manages version control. The "Commit" steps below are replaced by a "Verify (no commit)" step in every task. Do **not** run any `git` command.
- **Reuse, don't rewrite:** Spec 1 (`src/bdt/transforms/graph.py`, `queries.py`, `templateflow.py`, and `bdt.interfaces.transforms`) is complete and its tests pass — only Task 1 modifies `bdt.interfaces.transforms`, and only additively. The CIFTI parcellation path (`_init_parcellate_cifti_wf`) must stay byte-for-byte unchanged.
- **Mirror the established cross-space pattern** in `init_map_scalar_to_surface_wf` (`src/bdt/engine/factories.py:468-556`): `_register_acpc_to_t1w` for the ACPC↔T1w bridge, `context.find_reference` for its references, and the "BDT computes its own ACPC↔T1w bridge; QSIPrep's stored ACPC↔anat transforms are deliberately not used" policy.
- **Verified nitransforms fact:** `nitransforms.resampling.apply` raises `RuntimeError: invalid shape for coordinate array` on a 4D moving image. A 4D image must be warped per-3D-volume (`nibabel.funcs.four_to_three` → `apply` each → `nibabel.funcs.concat_images`); resolve the transform chain **once** and reuse it across volumes. Pull semantics: a +2 world-x ITK translation moves a marker at voxel x=3 to output voxel x=1.
- **Label convention:** `EntitiesToSegTSV` (`src/bdt/interfaces/tractography.py:236`) writes the 4D segmentation's `dseg.tsv` with **1-based** `index` (volume *i* ↔ `index == i+1` when 0-based, i.e. rows start at 1). The volumetric parcellation's 4D region indices must match (1-based).
- **Atlas forms:** 4D dseg (binary per-region masks, treated like pseg), 4D pseg (probabilistic per-region weights), 3D dseg (integer labels). 4D → voxel-value-weighted mean per volume; 3D → per-label mean. Detected by image `ndim` (no explicit kind flag in the parcellation interface).
- **Output TSV columns (exact order):** `index`, `name`, `mean`, `coverage`. One row per region. `mean` is NaN when `coverage < min_coverage`. `min_coverage` default `0.5`.

---

### Task 1: 4D moving support in `ResolveApplyTransforms`

**Files:**
- Modify: `src/bdt/interfaces/transforms.py` (only `ResolveApplyTransforms._run_interface`)
- Test: `test/engine/test_transforms_interface.py` (append)

**Interfaces:**
- Consumes: existing `ResolveApplyTransforms` inputs (`source`, `target`, `moving`, `reference`, `local_transforms`, `bridges`, `interpolation`, `out_file`); `_build_chain` (Spec 1); `nitransforms.resampling.apply`.
- Produces: no signature change. `moving` may now be a 3D **or** 4D image; a 4D moving yields a 4D `out_file` on the reference grid (each volume warped independently by the single resolved chain).

- [ ] **Step 1: Write the failing test**

Append to `test/engine/test_transforms_interface.py` (reuses the existing `_itk_affine` and `_label_img` helpers at the top of that file):

```python
@pytest.mark.filterwarnings('ignore:Reference space not set')
def test_resolve_apply_warps_4d_atlas_per_volume(tmp_path, monkeypatch):
    import nibabel as nb

    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms

    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    xfm = _itk_affine(tmp_path / 'sub-01_from-ATLAS_to-GRID_mode-image_xfm.mat', 2.0)
    # 4D atlas: volume 0 marks voxel (3,3,3); volume 1 marks voxel (2,2,2).
    data = np.zeros((6, 6, 6, 2), dtype=np.int16)
    data[3, 3, 3, 0] = 1
    data[2, 2, 2, 1] = 1
    moving = str(tmp_path / 'atlas4d.nii.gz')
    nb.Nifti1Image(data, np.eye(4)).to_filename(moving)
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    res = ResolveApplyTransforms(
        source='ATLAS',
        target='GRID',
        moving=moving,
        reference=reference,
        local_transforms=[xfm],
        interpolation='nearest',
        out_file=str(tmp_path / 'out.nii.gz'),
    ).run()

    out = np.asarray(nb.load(res.outputs.out_file).dataobj)
    assert out.shape == (6, 6, 6, 2)
    # +2 world-x pull: vol0 marker 3->1, vol1 marker 2->0.
    assert np.argwhere(out[..., 0] > 0).tolist() == [[1, 3, 3]]
    assert np.argwhere(out[..., 1] > 0).tolist() == [[0, 2, 2]]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py::test_resolve_apply_warps_4d_atlas_per_volume -v`
Expected: FAIL — `RuntimeError: invalid shape for coordinate array` from the 4D `apply` call.

- [ ] **Step 3: Add the 4D branch**

In `src/bdt/interfaces/transforms.py`, locate the tail of `ResolveApplyTransforms._run_interface` (the block that currently does the single `apply(...)` and `resampled.to_filename(out_file)`). Replace **only** that resample-and-write block with:

```python
        import nibabel as nb
        from nibabel.funcs import concat_images, four_to_three

        moving_img = nb.load(self.inputs.moving)
        if moving_img.ndim > 3:
            # nitransforms.apply rejects 4D; warp each 3D volume with the single
            # resolved chain, then restack onto the (3D) reference grid.
            warped = [
                apply(
                    transform, vol, reference=self.inputs.reference,
                    order=order, mode='constant', cval=0.0,
                )
                for vol in four_to_three(moving_img)
            ]
            resampled = concat_images(warped)
        else:
            resampled = apply(
                transform, self.inputs.moving, reference=self.inputs.reference,
                order=order, mode='constant', cval=0.0,
            )
        resampled.to_filename(out_file)
```

(The lines above this block — building `transform`, computing `order`, and resolving `out_file` — are unchanged. `apply` is already imported at the top of `_run_interface`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py -q -W error::UserWarning`
Expected: PASS — all interface tests (the existing 3D warp/identity/no-path/load/chain tests plus the new 4D test), output pristine.

- [ ] **Step 5: Verify (no commit — edit in place)**

Run: `micromamba run -n bdtenv python -m pytest test/spec/test_transforms.py test/spec/test_templateflow_edges.py test/engine/test_transforms_interface.py -q`
Expected: PASS (the full Spec 1 subsystem + the new test). Do **not** commit.

---

### Task 2: `ParcellateVolumetric` interface (coverage-aware, all three atlas forms)

**Files:**
- Create: `src/bdt/interfaces/parcellate.py`
- Test: `test/engine/test_parcellate_volumetric.py`

**Interfaces:**
- Consumes: `nibabel`, `numpy`, `pandas`; `nipype.interfaces.base` (`BaseInterfaceInputSpec`, `File`, `SimpleInterface`, `TraitedSpec`, `traits`).
- Produces:
  - `_parcellate_volumetric(scalar_path, atlas_path, out_path, min_coverage=0.5, labels=None) -> str` — pure function. `labels` is an optional `{int index: str name}` mapping. Writes the tidy TSV and returns `out_path`.
  - `ParcellateVolumetric(SimpleInterface)` — inputs `scalar` (File, mandatory), `atlas` (File, mandatory), `atlas_labels` (File, optional — a BIDS `dseg.tsv` with `index`/`name` columns), `min_coverage` (Float, default 0.5), `out_file` (Str, default `'parcellated.tsv'`); output `out_file` (File).

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_parcellate_volumetric.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for bdt.interfaces.parcellate (volumetric coverage-aware parcellation)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('nibabel')


def _nii(path, data):
    import nibabel as nb

    nb.Nifti1Image(np.asarray(data), np.eye(4)).to_filename(str(path))
    return str(path)


def test_4d_dseg_binary_masks_give_in_mask_mean(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    # scalar: a 4x1x1 row of values [1, 2, 3, 4].
    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1))
    # 4D dseg, 2 regions: region0 covers voxels 0&1, region1 covers voxels 2&3.
    atlas = np.zeros((4, 1, 1, 2), dtype=np.int16)
    atlas[0, 0, 0, 0] = 1
    atlas[1, 0, 0, 0] = 1
    atlas[2, 0, 0, 1] = 1
    atlas[3, 0, 0, 1] = 1
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.0)
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['index', 'name', 'mean', 'coverage']
    assert df['index'].tolist() == [1, 2]  # 1-based, matches EntitiesToSegTSV
    assert df['mean'].tolist() == [1.5, 3.5]
    assert df['coverage'].tolist() == [1.0, 1.0]


def test_4d_pseg_weights_diverge_from_plain_mean(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([10.0, 20.0]).reshape(2, 1, 1))
    # one probabilistic region weighting voxel0 by 0.25 and voxel1 by 0.75.
    atlas = np.zeros((2, 1, 1, 1), dtype=np.float32)
    atlas[0, 0, 0, 0] = 0.25
    atlas[1, 0, 0, 0] = 0.75
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.0)
    df = pd.read_csv(out, sep='\t')
    # weighted: (0.25*10 + 0.75*20) / (0.25+0.75) = 17.5, not the plain mean 15.
    assert df['mean'].tolist() == [17.5]


def test_3d_dseg_label_means_with_and_without_labels(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([5.0, 7.0, 9.0]).reshape(3, 1, 1))
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', np.array([1, 1, 2], dtype=np.int16).reshape(3, 1, 1))

    # no labels -> name is the string of the label value.
    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.0)
    df = pd.read_csv(out, sep='\t', dtype={'name': str})
    assert df['index'].tolist() == [1, 2]
    assert df['name'].tolist() == ['1', '2']
    assert df['mean'].tolist() == [6.0, 9.0]  # (5+7)/2, 9

    # with labels -> names mapped.
    out2 = _parcellate_volumetric(
        scalar, atlas_path, str(tmp_path / 'out2.tsv'),
        min_coverage=0.0, labels={1: 'CST', 2: 'AF'},
    )
    df2 = pd.read_csv(out2, sep='\t')
    assert df2['name'].tolist() == ['CST', 'AF']


def test_low_coverage_region_is_nan_masked(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    # region covers 4 voxels; only 1 has valid (finite, nonzero) data -> coverage 0.25.
    scalar = _nii(
        tmp_path / 'scalar.nii.gz', np.array([8.0, 0.0, 0.0, 0.0]).reshape(4, 1, 1)
    )
    atlas = np.zeros((4, 1, 1, 1), dtype=np.int16)
    atlas[:, 0, 0, 0] = 1
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.5)
    df = pd.read_csv(out, sep='\t')
    assert df['coverage'].tolist() == [0.25]
    assert np.isnan(df['mean'].tolist()[0])


def test_interface_reads_labels_sidecar(tmp_path):
    from bdt.interfaces.parcellate import ParcellateVolumetric

    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([5.0, 7.0, 9.0]).reshape(3, 1, 1))
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', np.array([1, 1, 2], dtype=np.int16).reshape(3, 1, 1))
    labels_tsv = tmp_path / 'dseg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['CST', 'AF']}).to_csv(labels_tsv, sep='\t', index=False)

    res = ParcellateVolumetric(
        scalar=scalar, atlas=atlas_path, atlas_labels=str(labels_tsv),
        min_coverage=0.0, out_file=str(tmp_path / 'out.tsv'),
    ).run()
    df = pd.read_csv(res.outputs.out_file, sep='\t')
    assert df['name'].tolist() == ['CST', 'AF']
    assert df['mean'].tolist() == [6.0, 9.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_volumetric.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdt.interfaces.parcellate'`.

- [ ] **Step 3: Write the module**

Create `src/bdt/interfaces/parcellate.py`:

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
"""Coverage-aware volumetric (NIfTI) parcellation of a scalar over a label atlas.

Handles three atlas forms: a 4D dseg (per-region binary masks), a 4D pseg
(per-region probabilistic weights), and a 3D dseg (integer labels).  The 4D
forms share one voxel-value-**weighted** mean per volume (a binary dseg reduces
to the plain in-mask mean); the 3D form is the per-label mean.  Per region, a
coverage fraction (region weight over voxels with valid data, out of the region's
total weight) NaN-masks regions below ``min_coverage`` — mirroring the CIFTI
parcellation path (:func:`bdt.engine.factories._init_parcellate_cifti_wf`).
"""

from __future__ import annotations

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


def _region_row(index, weight, scalar, valid, min_coverage, labels):
    """One output row for a region given its voxel weight map."""
    import numpy as np

    total = float(weight.sum())
    covered = float((weight * valid).sum())
    coverage = covered / total if total > 0 else 0.0
    if covered > 0:
        scalar0 = np.where(valid, scalar, 0.0)  # zero out non-valid (avoids NaN*0)
        mean = float((weight * scalar0).sum() / covered)
    else:
        mean = float('nan')
    if coverage < min_coverage:
        mean = float('nan')
    name = str(index)
    if labels is not None and index in labels:
        name = labels[index]
    return {'index': int(index), 'name': name, 'mean': mean, 'coverage': coverage}


def _parcellate_volumetric(scalar_path, atlas_path, out_path, min_coverage=0.5, labels=None):
    """Parcellate ``scalar_path`` over ``atlas_path`` into a tidy per-region TSV.

    ``labels`` is an optional ``{index: name}`` mapping; region names default to
    the string of the region index when absent.  Columns: ``index``, ``name``,
    ``mean``, ``coverage`` (one row per region).  A 4D atlas is weighted per
    volume (regions 1-based, matching ``EntitiesToSegTSV``); a 3D atlas is a
    per-integer-label mean.
    """
    import nibabel as nb
    import numpy as np
    import pandas as pd

    scalar = np.asarray(nb.load(scalar_path).dataobj, dtype='float64')
    atlas_img = nb.load(atlas_path)
    atlas = np.asarray(atlas_img.dataobj, dtype='float64')
    valid = np.isfinite(scalar) & (scalar != 0)

    rows = []
    if atlas_img.ndim > 3:
        for i in range(atlas.shape[3]):
            rows.append(
                _region_row(i + 1, atlas[..., i], scalar, valid, min_coverage, labels)
            )
    else:
        for value in np.unique(atlas):
            if value == 0:
                continue
            weight = (atlas == value).astype('float64')
            rows.append(_region_row(int(value), weight, scalar, valid, min_coverage, labels))

    pd.DataFrame(rows, columns=['index', 'name', 'mean', 'coverage']).to_csv(
        out_path, sep='\t', index=False
    )
    return str(out_path)


class _ParcellateVolumetricInputSpec(BaseInterfaceInputSpec):
    scalar = File(exists=True, mandatory=True, desc='scalar NIfTI in the atlas grid/space')
    atlas = File(exists=True, mandatory=True, desc='3D dseg or 4D dseg/pseg label atlas')
    atlas_labels = File(
        exists=True, desc='optional BIDS dseg.tsv (index/name) naming the regions'
    )
    min_coverage = traits.Float(0.5, usedefault=True, desc='NaN-mask regions below this')
    out_file = traits.Str('parcellated.tsv', usedefault=True, desc='output TSV filename')


class _ParcellateVolumetricOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy per-region TSV: index, name, mean, coverage')


class ParcellateVolumetric(SimpleInterface):
    """Coverage-aware volumetric parcellation of a scalar over a label atlas."""

    input_spec = _ParcellateVolumetricInputSpec
    output_spec = _ParcellateVolumetricOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        labels = None
        if self.inputs.atlas_labels:
            table = pd.read_csv(self.inputs.atlas_labels, sep='\t')
            labels = {int(i): str(n) for i, n in zip(table['index'], table['name'])}

        out_file = self.inputs.out_file
        if not os.path.isabs(out_file):
            out_file = os.path.join(runtime.cwd, out_file)
        _parcellate_volumetric(
            self.inputs.scalar, self.inputs.atlas, out_file,
            min_coverage=self.inputs.min_coverage, labels=labels,
        )
        self._results['out_file'] = out_file
        return runtime
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_volumetric.py -q -W error::UserWarning`
Expected: PASS (5 tests), output pristine.

- [ ] **Step 5: Verify (no commit — edit in place)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_volumetric.py -q`
Expected: PASS. Do **not** commit.

---

### Task 3: `FactoryContext` helpers — file-type/suffix + transform discovery

**Files:**
- Modify: `src/bdt/engine/factories.py` (add three methods to `FactoryContext`)
- Test: `test/engine/test_factory_context_transforms.py`

**Interfaces:**
- Consumes: existing `FactoryContext` (`resolved`, `datasets`, `provider`, `subject`, `_role_entity`, `_select_scoped`); `bdt.transforms.graph.parse_xfm_filename`.
- Produces (methods on `FactoryContext`):
  - `role_extension(node, role, default=None) -> str | None` — the `extension` entity of the file feeding `role`.
  - `role_suffix(node, role, default=None) -> str | None` — the `suffix` entity.
  - `discover_transforms(session=None) -> list[str]` — sorted BIDS `_xfm` file paths across all `datasets` for the subject, **excluding** any transform with `ACPC` as an endpoint (the ACPC↔T1w hop is always the computed bridge). Returns `[]` when there is no provider.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_factory_context_transforms.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for FactoryContext file-type + transform-discovery helpers."""

from types import SimpleNamespace

from bdt.engine.factories import FactoryContext


class _StubMatch:
    def __init__(self, path, entities=None):
        self.path = path
        self.entities = entities or {}


class _StubProvider:
    """Returns preset matches per dataset, ignoring filters/session."""

    def __init__(self, by_dataset):
        self._by_dataset = by_dataset

    def select(self, dataset, filters, exclude, subject=None):
        return list(self._by_dataset.get(dataset, []))


def _node(role_to_upstreams):
    return SimpleNamespace(inputs=role_to_upstreams)


def test_role_extension_and_suffix_read_resolved_entities():
    ctx = FactoryContext(
        resolved={'load_cbf': _StubMatch('cbf.nii.gz', {'extension': '.nii.gz', 'suffix': 'cbf'})}
    )
    node = _node({'scalar': ['load_cbf']})
    assert ctx.role_extension(node, 'scalar') == '.nii.gz'
    assert ctx.role_suffix(node, 'scalar') == 'cbf'
    assert ctx.role_extension(node, 'atlas', default='') == ''  # role absent -> default


def test_discover_transforms_excludes_acpc_endpoints_and_sorts():
    provider = _StubProvider({
        'aslprep': [_StubMatch('/d/sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5')],
        'qsiprep': [
            _StubMatch('/d/sub-01_from-ACPC_to-T1w_mode-image_xfm.mat'),   # excluded (ACPC)
            _StubMatch('/d/sub-01_from-T1w_to-ACPC_mode-image_xfm.mat'),   # excluded (ACPC)
        ],
    })
    ctx = FactoryContext(provider=provider, subject='01', datasets=['aslprep', 'qsiprep'])
    got = ctx.discover_transforms()
    assert got == ['/d/sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5']


def test_discover_transforms_no_provider_returns_empty():
    assert FactoryContext(datasets=['aslprep']).discover_transforms() == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py -v`
Expected: FAIL — `AttributeError: 'FactoryContext' object has no attribute 'role_extension'`.

- [ ] **Step 3: Add the methods**

In `src/bdt/engine/factories.py`, add these methods to the `FactoryContext` class, immediately after the existing `role_session` method (near line 99):

```python
    def role_extension(self, node, role: str, default: str | None = None) -> str | None:
        """The ``extension`` entity of the file feeding ``node``'s ``role``."""
        return self._role_entity(node, role, 'extension', default)

    def role_suffix(self, node, role: str, default: str | None = None) -> str | None:
        """The ``suffix`` entity of the file feeding ``node``'s ``role``."""
        return self._role_entity(node, role, 'suffix', default)

    def discover_transforms(self, session: str | None = None) -> list[str]:
        """Subject transform files across all datasets, for Spec 1's ``local_transforms``.

        Enumerates BIDS ``_xfm`` files (via the provider) and **excludes** any whose
        endpoints include ``ACPC``: the ACPC↔T1w hop is always the transform BDT
        computes itself (the rigid bridge), never QSIPrep's stored one — matching the
        locked decision in :func:`init_map_scalar_to_surface_wf`.  Returns ``[]`` when
        no provider is configured (the build-time stub path).
        """
        from bdt.transforms.graph import parse_xfm_filename

        if self.provider is None:
            return []
        paths: list[str] = []
        for dataset in self.datasets or []:
            for match in self._select_scoped(dataset, {'suffix': 'xfm'}, None, session):
                xfm = parse_xfm_filename(match.path)
                if xfm is None or 'ACPC' in (xfm.frm, xfm.to):
                    continue
                paths.append(match.path)
        return sorted(paths)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify (no commit — edit in place)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py -q`
Expected: PASS. Do **not** commit.

---

### Task 4: `parcellate_scalar` routing + volumetric subworkflow

**Files:**
- Modify: `src/bdt/engine/factories.py` (rewrite `init_parcellate_scalar_wf`; add `_init_parcellate_volumetric_wf`)
- Test: `test/engine/test_parcellate_scalar_routing.py`

**Interfaces:**
- Consumes: `ParcellateVolumetric` (Task 2); `ResolveApplyTransforms` (Task 1); `FactoryContext.role_extension`/`role_suffix`/`role_space`/`role_session`/`discover_transforms`/`find_reference` (Task 3 + existing); `_register_acpc_to_t1w` (existing); `_init_parcellate_cifti_wf` (existing, unchanged).
- Produces: `init_parcellate_scalar_wf(node, name=None, context=None)` routes CIFTI→`_init_parcellate_cifti_wf` / volumetric→`_init_parcellate_volumetric_wf`; the latter builds `inputnode(scalar, atlas)` → optional `ResolveApplyTransforms` warp → `ParcellateVolumetric` → `outputnode(out)`.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_parcellate_scalar_routing.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Build-time routing tests for parcellate_scalar (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_parcellate_scalar_wf


def _match(path, entities):
    from bdt.engine.selection import Match

    return Match(path, entities)


def _node(name, role_to_upstreams, parameters=None):
    return SimpleNamespace(
        name=name, inputs=role_to_upstreams, parameters=parameters or {}
    )


def test_cifti_scalar_routes_to_cifti_wf():
    # a .dscalar.nii scalar -> the existing CIFTI path (has a vertex_mask node,
    # which the volumetric path never builds).
    node = _node('alff_parc', {'scalar': ['load_alff'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        resolved={'load_alff': _match('alff.dscalar.nii', {'extension': '.dscalar.nii'})}
    )
    names = set(init_parcellate_scalar_wf(node, context=ctx).list_node_names())
    assert 'vertex_mask' in names


def test_volumetric_same_space_has_no_warp_node():
    # FA scalar (ACPC) + atlas (ACPC): same space -> parcellate directly, no warp.
    node = _node('fa_roi', {'scalar': ['load_fa'], 'atlas': ['bundle_rois']})
    ctx = FactoryContext(
        resolved={
            'load_fa': _match('fa.nii.gz', {'extension': '.nii.gz', 'space': 'ACPC'}),
            'bundle_rois': _match('rois.nii.gz', {'extension': '.nii.gz', 'space': 'ACPC'}),
        }
    )
    names = set(init_parcellate_scalar_wf(node, context=ctx).list_node_names())
    assert 'parcellate' in names
    assert 'warp_atlas' not in names


def test_volumetric_cross_space_non_acpc_builds_warp_without_bridge():
    # CBF (MNI152NLin2009cAsym) + atlas (MNI152NLin6Asym): cross-space but no ACPC,
    # so a ResolveApplyTransforms warp is inserted and NO registration node is built.
    from bdt.engine.selection import DictDataProvider

    node = _node('roi', {'scalar': ['load_cbf'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider({'aslprep': []}),
        subject='01',
        datasets=['aslprep'],
        resolved={
            'load_cbf': _match(
                'cbf.nii.gz', {'extension': '.nii.gz', 'space': 'MNI152NLin2009cAsym'}
            ),
            'load_atlas': _match(
                'atlas.nii.gz',
                {'extension': '.nii.gz', 'space': 'MNI152NLin6Asym', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_scalar_wf(node, context=ctx).list_node_names())
    assert 'warp_atlas' in names
    assert 'register_acpc' not in names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_scalar_routing.py -v`
Expected: FAIL — `test_volumetric_*` fail (the current `init_parcellate_scalar_wf` always routes to the CIFTI path, so there is no `parcellate`/`warp_atlas` node); `test_cifti_*` passes.

- [ ] **Step 3: Rewrite the factory + add the volumetric subworkflow**

In `src/bdt/engine/factories.py`, replace the existing `init_parcellate_scalar_wf` (the `@workflow_factory('parcellate_scalar')` function, currently at ~lines 274-281) with the following two functions:

```python
@workflow_factory('parcellate_scalar')
def init_parcellate_scalar_wf(node, name=None, context=None) -> pe.Workflow:
    """Parcellate a scalar with a label atlas, coverage-aware.

    Routes on the scalar's file type: a CIFTI dscalar uses the surface/grayordinate
    path (:func:`_init_parcellate_cifti_wf`, unchanged); a volumetric NIfTI uses the
    Strategy-A path (:func:`_init_parcellate_volumetric_wf`), warping the atlas into
    the scalar's space when they differ.  An unknown/absent extension defaults to the
    CIFTI path (build-time safe for the existing CIFTI stories).
    """
    context = context or FactoryContext()
    extension = context.role_extension(node, 'scalar', default='') or ''
    if extension in ('.nii', '.nii.gz'):
        return _init_parcellate_volumetric_wf(node, name, context)
    return _init_parcellate_cifti_wf(node, name, 'scalar', 'parcellated.pscalar.nii')


def _init_parcellate_volumetric_wf(node, name, context) -> pe.Workflow:
    """Volumetric (NIfTI) coverage-aware parcellation, Strategy A.

    ``inputnode`` takes the ``scalar`` and ``atlas`` (a 3D dseg or 4D dseg/pseg).
    When the atlas and scalar spaces differ, the atlas is warped into the scalar's
    space with :class:`~bdt.interfaces.transforms.ResolveApplyTransforms` (nearest
    for a dseg, linear for a pseg); the ACPC↔T1w hop, when needed, is the computed
    rigid bridge (mirrors :func:`init_map_scalar_to_surface_wf`).  ``outputnode.out``
    is the tidy per-region TSV.
    """
    from bdt.interfaces.parcellate import ParcellateVolumetric
    from bdt.interfaces.transforms import ResolveApplyTransforms

    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))

    inputnode = pe.Node(niu.IdentityInterface(fields=['scalar', 'atlas']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')

    atlas_space = context.role_space(node, 'atlas')
    scalar_space = context.role_space(node, 'scalar')
    cross_space = (
        atlas_space is not None and scalar_space is not None and atlas_space != scalar_space
    )

    parcellate = pe.Node(
        ParcellateVolumetric(min_coverage=min_coverage, out_file='parcellated.tsv'),
        name='parcellate',
    )
    wf.connect([
        (inputnode, parcellate, [('scalar', 'scalar')]),
        (parcellate, outputnode, [('out_file', 'out')]),
    ])  # fmt:skip

    if not cross_space:
        wf.connect([(inputnode, parcellate, [('atlas', 'atlas')])])
        return wf

    atlas_suffix = context.role_suffix(node, 'atlas', default='dseg') or 'dseg'
    interpolation = 'linear' if atlas_suffix in ('probseg', 'pseg') else 'nearest'
    warp = pe.Node(
        ResolveApplyTransforms(
            source=atlas_space,
            target=scalar_space,
            interpolation=interpolation,
            local_transforms=context.discover_transforms(
                session=context.role_session(node, 'atlas')
            ),
            out_file='atlas_in_scalar_space.nii.gz',
        ),
        name='warp_atlas',
    )
    wf.connect([
        (inputnode, warp, [('atlas', 'moving'), ('scalar', 'reference')]),
        (warp, parcellate, [('out_file', 'atlas')]),
    ])  # fmt:skip

    if 'ACPC' in (atlas_space, scalar_space):
        if context.provider is None:
            raise ValueError(
                f'parcellate_scalar node {node.name!r} is cross-space through ACPC and '
                'needs a FactoryContext provider to resolve the bridge references.'
            )
        # ACPC↔T1w bridge references (mirrors init_map_scalar_to_surface_wf): fixed =
        # the native T1w anatomical (+ brain mask); moving = the space-ACPC anatomical
        # (+ brain mask), at the atlas's anat level.
        atlas_ses = context.role_session(node, 'atlas')
        fixed_img = context.find_reference(
            {'suffix': ['T1w', 'T2w'], 'desc': 'preproc', 'space': None, 'datatype': 'anat'}
        )
        fixed_mask = context.find_reference(
            {'suffix': 'mask', 'desc': 'brain', 'space': None, 'datatype': 'anat'}
        )
        moving_img = context.find_reference(
            {'suffix': ['T1w', 'T2w'], 'desc': 'preproc', 'space': 'ACPC', 'datatype': 'anat'},
            session=atlas_ses,
        )
        moving_mask = context.find_reference(
            {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'},
            session=atlas_ses,
        )
        register = pe.Node(
            niu.Function(
                function=_register_acpc_to_t1w,
                input_names=['fixed_image', 'fixed_mask', 'moving_image', 'moving_mask'],
                output_names=['out'],
            ),
            name='register_acpc',
            n_procs=4,
        )
        register.inputs.fixed_image = fixed_img
        register.inputs.fixed_mask = fixed_mask
        register.inputs.moving_image = moving_img
        register.inputs.moving_mask = moving_mask
        # bridges is a List[File]; wrap the single .mat via Merge(1).
        bridge_list = pe.Node(niu.Merge(1), name='bridge_list')
        wf.connect([
            (register, bridge_list, [('out', 'in1')]),
            (bridge_list, warp, [('out', 'bridges')]),
        ])  # fmt:skip

    return wf
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_scalar_routing.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify the CIFTI path is unchanged + no regressions (no commit)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_scalar_routing.py test/engine/test_parcellate_volumetric.py test/engine/test_transforms_interface.py -q`
Expected: PASS. Then run the existing CIFTI parcellation / spec assembly tests to confirm no regression:
Run: `micromamba run -n bdtenv python -m pytest test/engine -q`
Expected: only the 5 known pre-existing failures (`test_outputs` write_participant/collision, `test_pipeline` collision, `test_pybids_provider` ×2); zero new failures. Do **not** commit.

---

### Task 5: ACPC cross-space bridge assembly + real-pipeline compile + ledger

**Files:**
- Test: `test/engine/test_parcellate_scalar_routing.py` (append the ACPC-bridge assembly test)
- Modify: `.superpowers/sdd/progress.md` (append the Spec 2 section)
- Verify: `scripts/tract_parcellate.yml` compiles end-to-end (no YAML change expected)

**Interfaces:**
- Consumes: the full stack from Tasks 1-4; `bdt.engine.selection.DictDataProvider`/`Match`.
- Produces: coverage of the ACPC cross-space path (`warp_atlas` + `register_acpc` + `bridge_list`, with the registration wired to the resolved anatomicals) — the one path Task 4's non-ACPC test deliberately skips — plus a real-pipeline compile check.

- [ ] **Step 1: Write the failing ACPC-bridge assembly test**

Append to `test/engine/test_parcellate_scalar_routing.py` (this mirrors `_map_context` +
`test_map_scalar_to_surface_cross_space` in `test/engine/test_nipype_workflow.py`):

```python
def test_volumetric_cross_space_acpc_builds_bridge(tmp_path):
    # CBF (MNI152NLin6Asym) + atlas (ACPC): the atlas is warped ACPC->MNI, and the
    # ACPC->T1w hop is the computed rigid bridge -> register_acpc + bridge_list wired
    # to the resolved anatomicals. ASLPrep's T1w->MNI xfm is discovered; QSIPrep's
    # ACPC<->T1w xfm is excluded (the bridge is authoritative).
    from bdt.engine.selection import DictDataProvider, Match

    def touch(rel):
        (tmp_path / rel).touch()
        return str(tmp_path / rel)

    provider = DictDataProvider({
        'anat': [
            Match(touch('sub-01_desc-preproc_T1w.nii.gz'),
                  {'suffix': 'T1w', 'desc': 'preproc', 'datatype': 'anat'}),
            Match(touch('sub-01_desc-brain_mask.nii.gz'),
                  {'suffix': 'mask', 'desc': 'brain', 'datatype': 'anat'}),
        ],
        'qsiprep': [
            Match(touch('sub-01_space-ACPC_desc-preproc_T1w.nii.gz'),
                  {'suffix': 'T1w', 'desc': 'preproc', 'space': 'ACPC', 'datatype': 'anat'}),
            Match(touch('sub-01_space-ACPC_desc-brain_mask.nii.gz'),
                  {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'}),
            Match(touch('sub-01_from-ACPC_to-T1w_mode-image_xfm.mat'),
                  {'suffix': 'xfm', 'from': 'ACPC', 'to': 'T1w'}),  # excluded by discovery
        ],
        'aslprep': [
            Match(touch('sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5'),
                  {'suffix': 'xfm', 'from': 'T1w', 'to': 'MNI152NLin6Asym'}),
        ],
    })
    node = _node('cbf_roi', {'scalar': ['load_cbf'], 'atlas': ['bundle_rois']})
    ctx = FactoryContext(
        provider=provider,
        subject='01',
        datasets=['anat', 'qsiprep', 'aslprep'],
        resolved={
            'load_cbf': _match('cbf.nii.gz', {'extension': '.nii.gz', 'space': 'MNI152NLin6Asym'}),
            'bundle_rois': _match(
                'rois.nii.gz', {'extension': '.nii.gz', 'space': 'ACPC', 'suffix': 'dseg'}
            ),
        },
    )
    wf = init_parcellate_scalar_wf(node, context=ctx)
    names = set(wf.list_node_names())
    for want in ('warp_atlas', 'register_acpc', 'bridge_list', 'parcellate'):
        assert want in names, f'missing {want}'
    reg = wf.get_node('register_acpc')
    assert reg.inputs.fixed_image.endswith('desc-preproc_T1w.nii.gz')
    assert reg.inputs.moving_image.endswith('space-ACPC_desc-preproc_T1w.nii.gz')
    # nearest interpolation for a dseg atlas
    assert wf.get_node('warp_atlas').inputs.interpolation == 'nearest'
    # the ASLPrep T1w->MNI xfm is discovered; the QSIPrep ACPC xfm is not.
    locals_ = wf.get_node('warp_atlas').inputs.local_transforms
    assert any('to-MNI152NLin6Asym' in p for p in locals_)
    assert not any('ACPC' in p for p in locals_)
```

- [ ] **Step 2: Run it to verify it fails, then passes**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_scalar_routing.py::test_volumetric_cross_space_acpc_builds_bridge -v`
Expected before Task 4 is complete: FAIL. With Tasks 1-4 done, this test exercises code already written in Task 4 — run it and expect PASS. If it fails, the defect is in Task 4's ACPC block (fix there, not by loosening the test).

- [ ] **Step 3: Confirm the real pipeline compiles**

The tractogram work ran `scripts/tract_parcellate.yml` against the ds008325 fixtures. Locate the compile entry point the acceptance test uses:

Run: `micromamba run -n bdtenv python -c "from bdt.engine.pipeline import run_spec; import inspect; print(inspect.signature(run_spec))"`
and read `test/spec/test_spec.py` for how a story YAML is loaded into a `Spec` and compiled (the `STORY_3_4` acceptance path from the tractogram work).

Compile `scripts/tract_parcellate.yml` through that entry point with the ds008325 fixtures if present; assert it builds without error (the `fa_roi`/`cbf_roi` nodes now resolve to volumetric subworkflows). If the fixtures are absent in this environment, record in the report that runtime verification is pending and rely on the assembly tests (Tasks 4-5 Step 1).

- [ ] **Step 4: Run the full subsystem suite (no commit)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py test/engine/test_parcellate_volumetric.py test/engine/test_factory_context_transforms.py test/engine/test_parcellate_scalar_routing.py test/spec/test_transforms.py test/spec/test_templateflow_edges.py -q`
Expected: PASS. Then `micromamba run -n bdtenv python -m pytest test/engine -q` and confirm only the 5 known pre-existing failures remain (zero new). Do **not** commit.

- [ ] **Step 5: Update the progress ledger**

Append a `# Progress ledger: volumetric parcellate_scalar (Spec 2)` section to `.superpowers/sdd/progress.md` summarizing: routing by scalar file type (`role_extension`); `ParcellateVolumetric` (3 atlas forms via `ndim`, coverage NaN-mask, tidy `index/name/mean/coverage` TSV); the 4D-per-volume warp added to `ResolveApplyTransforms`; `discover_transforms` (ACPC-endpoint exclusion); the ACPC-bridge cross-space wiring mirroring `map_scalar_to_surface`; and the **known limitations**: (a) region names are index-based because the atlas's `dseg.tsv` sidecar isn't carried by role wiring (`atlas_labels` input exists + tested, factory leaves it unwired pending compiler support for secondary role outputs); (b) the ACPC-bridge `find_reference` for the native T1w assumes exactly one native T1w across datasets (raises clearly otherwise — a dataset-anchor is a follow-up). Do **not** commit.

---

## Notes for the executor

- **Region naming limitation (carry into review):** the `atlas` role wiring delivers only the upstream `outputnode.out` (the image), not `bundle_rois`' `outputnode.tsv` label sidecar. So in the real pipeline, `ParcellateVolumetric` names regions by their 1-based index, not by bundle name. The interface's optional `atlas_labels` input (tested in Task 2) is the seam for real names once the compiler can carry a secondary role output — that wiring is out of scope here and noted in the design's out-of-scope list.
- **Bridge is build-tested, not run-tested:** the ACPC↔T1w registration (`_register_acpc_to_t1w`, ANTsPy) needs real anatomicals and is not exercised by unit tests. The cross-space unit test in Task 4 deliberately uses a non-ACPC pairing so no bridge is built; Task 5 assembles the ACPC path (touched reference files) but does not run the registration.
- **Native-T1w bridge reference is resolved with `find_reference` (exactly-one across all datasets).** Unlike `init_map_scalar_to_surface_wf` (which anchors `fixed` to the surfaces' dataset), `parcellate_scalar` has no anat-anchoring role, so it resolves the native T1w across all `--datasets`. If more than one dataset ships a `space`-less `desc-preproc_T1w`, `find_reference` raises a clear "matched N" error at build time — a dataset-anchor for the fixed image is the follow-up if that arises on real data. Do not paper over it by picking the first match.
- **Do not touch** `_init_parcellate_cifti_wf`, `graph.py`, `queries.py`, or `templateflow.py`. Task 1's edit to `bdt.interfaces.transforms` is purely additive (a 4D branch).
```
