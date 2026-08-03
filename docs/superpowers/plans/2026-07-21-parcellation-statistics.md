# Parcellation Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `parcellate_scalar` compute several per-parcel statistics, emit them as one tidy TSV in both modalities, and give CIFTI one `.pscalar.nii` per statistic.

**Architecture:** A shared vocabulary module normalizes statistic names and composes `stat-` entities. The volumetric path gains one interface producing the tidy table for both 3D and 4D atlases; the CIFTI path builds one Workbench parcellation per statistic and merges the results into the same tidy table. The sink plan multiplies only the native CIFTI product across statistics — the TSV stays singular and is passed through from the subworkflow rather than converted by the sink.

**Tech Stack:** Python 3.12, nipype, nilearn 0.14.0, nibabel, pandas, Connectome Workbench, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-parcellation-statistics-design.md`.
- Environment: **`bdtenv`**. Every command is `micromamba run -n bdtenv <cmd>` from `/mnt/c/Users/tsalo/Documents/linc/bdt`.
- **No git commits.** Edited in place; the user manages version control. Tasks end with verification, never a commit.
- **Do not run ruff.** It is not installed in `bdtenv` and must not be borrowed from another environment — the user runs `pipx run ruff` themselves. Match surrounding style by hand.
- Supported statistics are exactly `mean` and `standard_deviation`. Any other value is a build-time error naming the supported set.
- `standard_deviation` is the **population** SD (ddof=0), matching `numpy.std()`, nilearn's `strategy='standard_deviation'`, and Workbench's `-method STDEV`.
- Entity values: normalize to alphanumerics (`standard_deviation` -> `standarddeviation`); join with a source statistic using `+`, source first (`stat-alff+mean`).
- TSV **column headers** keep the readable parameter spelling (`standard_deviation`). Only entity values are normalized.
- Coverage semantics are unchanged: computed from the atlas and brain mask only, never the data.
- `parcellate_timeseries` must be byte-identical after every task.
- Entities are keyed by pybids entity name throughout (`statistic`, not `stat`).
- The suite must stay at **177 passed, 0 failed** (plus new tests).

---

### Task 1: Statistic vocabulary and entity helpers

Pure functions, no nipype. Everything later depends on these.

**Files:**
- Create: `src/bdt/utils/statistics.py`
- Test: `test/engine/test_statistics_helpers.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SUPPORTED_STATISTICS: tuple[str, ...]` — `('mean', 'standard_deviation')`
  - `parse_statistics(parameters: dict) -> list[str]` — reads `parameters['statistics']`, defaults to `['mean']`, validates, preserves order, rejects duplicates
  - `normalize_statistic(name: str) -> str` — `'standard_deviation'` -> `'standarddeviation'`
  - `compose_statistic_entity(source: str | None, statistic: str) -> str` — `('alff', 'mean')` -> `'alff+mean'`; `(None, 'mean')` -> `'mean'`

  Tasks 2, 4, 5 and 6 all use these.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_statistics_helpers.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Statistic vocabulary, entity normalization, and stat- composition."""

import pytest

from bdt.utils.statistics import (
    SUPPORTED_STATISTICS,
    compose_statistic_entity,
    normalize_statistic,
    parse_statistics,
)


def test_default_is_mean_alone():
    assert parse_statistics({}) == ['mean']
    assert parse_statistics({'min_coverage': 0.5}) == ['mean']


def test_requested_order_is_preserved():
    got = parse_statistics({'statistics': ['standard_deviation', 'mean']})
    assert got == ['standard_deviation', 'mean']


def test_a_single_string_is_accepted():
    assert parse_statistics({'statistics': 'mean'}) == ['mean']


def test_unsupported_statistic_names_the_supported_set():
    with pytest.raises(ValueError, match='median'):
        parse_statistics({'statistics': ['mean', 'median']})
    with pytest.raises(ValueError, match='standard_deviation'):
        parse_statistics({'statistics': ['median']})


def test_duplicates_are_rejected():
    """A repeated statistic would emit two identically-named outputs."""
    with pytest.raises(ValueError, match='duplicate'):
        parse_statistics({'statistics': ['mean', 'mean']})


def test_empty_list_is_rejected():
    with pytest.raises(ValueError, match='at least one'):
        parse_statistics({'statistics': []})


def test_normalize_strips_non_alphanumerics():
    assert normalize_statistic('standard_deviation') == 'standarddeviation'
    assert normalize_statistic('mean') == 'mean'


def test_normalize_keeps_plus_signs():
    """'+' is legal in a BIDS entity value and is the composition separator."""
    assert normalize_statistic('alff+mean') == 'alff+mean'


def test_compose_joins_source_first_with_a_plus():
    assert compose_statistic_entity('alff', 'mean') == 'alff+mean'
    assert compose_statistic_entity('alff', 'standard_deviation') == 'alff+standarddeviation'


def test_compose_without_a_source_statistic():
    assert compose_statistic_entity(None, 'mean') == 'mean'
    assert compose_statistic_entity('', 'standard_deviation') == 'standarddeviation'


def test_supported_set_is_exactly_mean_and_sd():
    assert SUPPORTED_STATISTICS == ('mean', 'standard_deviation')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_statistics_helpers.py -v`

Expected: all FAIL at import with `ModuleNotFoundError: No module named 'bdt.utils.statistics'`.

- [ ] **Step 3: Write the implementation**

Create `src/bdt/utils/statistics.py`:

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
"""The per-parcel statistic vocabulary, shared by the plan and the factories.

Statistic names are chosen to match nilearn's ``strategy`` vocabulary so the
volumetric path can pass them straight to ``NiftiLabelsMasker``.  ``mean`` and
``standard_deviation`` are the only ones implemented; nilearn and Workbench both
offer more, and adding one is a matter of extending
:data:`SUPPORTED_STATISTICS` plus the two backend lookup tables.
"""

from __future__ import annotations

import re

#: Statistics a spec may request.  ``standard_deviation`` is the *population* SD
#: (ddof=0) across a parcel's voxels/vertices — what ``numpy.std()``,
#: ``NiftiLabelsMasker(strategy='standard_deviation')`` and Workbench's
#: ``-method STDEV`` all return.
SUPPORTED_STATISTICS = ('mean', 'standard_deviation')

_ENTITY_ILLEGAL = re.compile(r'[^A-Za-z0-9+]')


def parse_statistics(parameters: dict) -> list[str]:
    """The statistics a node requests, in the order requested.

    Defaults to ``['mean']``.  A bare string is accepted for the common
    single-statistic case.  Order is preserved because it sets the column order of
    the tidy table.
    """
    requested = (parameters or {}).get('statistics', ['mean'])
    if isinstance(requested, str):
        requested = [requested]
    requested = list(requested)

    if not requested:
        raise ValueError(
            'statistics must name at least one statistic; supported: '
            f'{", ".join(SUPPORTED_STATISTICS)}.'
        )
    unsupported = [s for s in requested if s not in SUPPORTED_STATISTICS]
    if unsupported:
        raise ValueError(
            f'Unsupported statistic(s) {", ".join(map(str, unsupported))}. '
            f'Supported: {", ".join(SUPPORTED_STATISTICS)}.'
        )
    if len(set(requested)) != len(requested):
        raise ValueError(
            f'statistics contains duplicate entries ({", ".join(requested)}); each '
            'statistic names one output, so repeats would collide.'
        )
    return requested


def normalize_statistic(name: str) -> str:
    """A statistic name as a BIDS entity value.

    Entity values are alphanumeric, plus ``+`` which BIDS allows and which we use
    as the composition separator — so ``standard_deviation`` becomes
    ``standarddeviation``.  The readable spelling is kept for TSV column headers.
    """
    return _ENTITY_ILLEGAL.sub('', str(name))


def compose_statistic_entity(source: str | None, statistic: str) -> str:
    """The ``statistic`` entity for a product, source statistic first.

    A parcellated ALFF map is both an ALFF map and a mean, and the filename should
    say so: ``stat-alff`` + ``mean`` -> ``stat-alff+mean``.  With no source
    statistic (e.g. CBF) the result is just the parcellation statistic.
    """
    statistic = normalize_statistic(statistic)
    if not source:
        return statistic
    return f'{normalize_statistic(source)}+{statistic}'
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_statistics_helpers.py -v`

Expected: 10 passed.

---

### Task 2: Volumetric scalar statistics interface

One interface covering both atlas forms, emitting the tidy table. The existing `NiftiParcellate` and `ProbSegParcellate` stay exactly as they are — `parcellate_timeseries` still uses them and must not change.

**Files:**
- Create: `src/bdt/interfaces/parcellate_stats.py`
- Modify: `src/bdt/interfaces/__init__.py`
- Test: `test/engine/test_parcellate_stats.py` (create)

**Interfaces:**
- Consumes: `bdt.utils.statistics.SUPPORTED_STATISTICS` (Task 1).
- Produces: `ParcellateScalarStatistics` (a `SimpleInterface`).
  - Inputs: `scalar`, `atlas`, `atlas_labels`, `mask` (all `File`, mandatory); `statistics` (`List(Str)`, default `['mean']`); `binarize` (`Bool`, default `False`); `min_coverage` (`Float`, default `0.5`).
  - Outputs: `out_file` (the tidy TSV), `coverage` (unchanged `Node`/`coverage` table).
  - Writes `parcellated.tsv` and `coverage.tsv` in the node cwd.

  Task 6 wires it.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_parcellate_stats.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tidy multi-statistic parcellation of a volumetric scalar."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.parcellate_stats import ParcellateScalarStatistics  # noqa: E402


def _write(tmp_path, atlas, mask, scalar, names):
    aff = np.eye(4)
    nb.Nifti1Image(atlas, aff).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(mask, aff).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(scalar, aff).to_filename(tmp_path / 'scalar.nii.gz')
    pd.DataFrame({'index': range(1, len(names) + 1), 'name': names}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )


def _run(tmp_path, **kwargs):
    return ParcellateScalarStatistics(
        scalar=str(tmp_path / 'scalar.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        mask=str(tmp_path / 'mask.nii.gz'),
        **kwargs,
    ).run()


def _dseg_fixture(tmp_path):
    """3D integer-label atlas: two parcels, whole volume in the brain mask."""
    shape = (6, 6, 6)
    atlas = np.zeros(shape, 'int16')
    atlas[:3] = 1
    atlas[3:] = 2
    mask = np.ones(shape, 'uint8')
    rng = np.random.default_rng(0)
    scalar = rng.random(shape).astype('float32')
    _write(tmp_path, atlas, mask, scalar, ['A', 'B'])
    return atlas, mask, scalar


def test_default_is_a_mean_column_only(tmp_path, monkeypatch):
    atlas, _, scalar = _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(_run(tmp_path, min_coverage=0.0).outputs.out_file)

    assert list(df.columns) == ['node', 'mean']
    assert df['node'].tolist() == ['A', 'B']
    assert df['mean'][0] == pytest.approx(float(scalar[atlas == 1].mean()), rel=1e-5)


def test_dseg_standard_deviation_is_the_population_sd(tmp_path, monkeypatch):
    """ddof=0, matching numpy.std() -- not the sample SD."""
    atlas, _, scalar = _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0)
        .outputs.out_file
    )

    assert list(df.columns) == ['node', 'mean', 'standard_deviation']
    for row, value in enumerate((1, 2)):
        vals = scalar[atlas == value]
        assert df['standard_deviation'][row] == pytest.approx(float(vals.std()), rel=1e-5)
        assert df['standard_deviation'][row] != pytest.approx(
            float(vals.std(ddof=1)), rel=1e-9
        )


def test_column_order_follows_the_request(tmp_path, monkeypatch):
    _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(tmp_path, statistics=['standard_deviation', 'mean'], min_coverage=0.0)
        .outputs.out_file
    )
    assert list(df.columns) == ['node', 'standard_deviation', 'mean']


def test_dseg_statistics_respect_the_brain_mask(tmp_path, monkeypatch):
    """Voxels outside the mask contribute to neither statistic."""
    shape = (6, 6, 6)
    atlas = np.ones(shape, 'int16')
    mask = np.zeros(shape, 'uint8')
    mask[:3] = 1
    rng = np.random.default_rng(1)
    scalar = rng.random(shape).astype('float32')
    scalar[3:] = 1000.0  # outside the mask; would dominate if included
    _write(tmp_path, atlas, mask, scalar, ['A'])
    monkeypatch.chdir(tmp_path)

    df = pd.read_table(
        _run(tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0)
        .outputs.out_file
    )
    inside = scalar[:3]
    assert df['mean'][0] == pytest.approx(float(inside.mean()), rel=1e-5)
    assert df['standard_deviation'][0] == pytest.approx(float(inside.std()), rel=1e-5)


def test_low_coverage_parcels_are_nan_in_every_column(tmp_path, monkeypatch):
    shape = (6, 6, 6)
    atlas = np.zeros(shape, 'int16')
    atlas[:3] = 1  # fully inside the mask
    atlas[3:] = 2  # fully outside
    mask = np.zeros(shape, 'uint8')
    mask[:3] = 1
    scalar = np.ones(shape, 'float32')
    _write(tmp_path, atlas, mask, scalar, ['keep', 'drop'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.5)
    df = pd.read_table(res.outputs.out_file)
    assert df['node'].tolist() == ['keep', 'drop']
    assert np.isnan(df['mean'][1])
    assert np.isnan(df['standard_deviation'][1])
    assert not np.isnan(df['mean'][0])

    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']
    assert coverage['keep'] == pytest.approx(1.0)
    assert coverage['drop'] == pytest.approx(0.0)


def _pseg_fixture(tmp_path):
    """4D probabilistic atlas: two overlapping maps, mask covers part of the volume."""
    shape = (6, 6, 6)
    rng = np.random.default_rng(2)
    atlas = rng.random(shape + (2,)).astype('float32')
    mask = np.zeros(shape, 'uint8')
    mask[1:5, 1:5, 1:5] = 1
    scalar = (rng.random(shape) * 10).astype('float32')
    _write(tmp_path, atlas, mask, scalar, ['a', 'b'])
    return atlas, mask, scalar


def test_pseg_weighted_mean_and_sd_match_brute_force(tmp_path, monkeypatch):
    """Weighted population SD: sqrt(sum(w*(d-mu)^2)/sum(w)) inside the mask."""
    atlas, mask, scalar = _pseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0)
        .outputs.out_file
    )

    inside = mask > 0
    d = scalar[inside]
    for row in (0, 1):
        w = atlas[..., row][inside]
        mu = float((w * d).sum() / w.sum())
        sd = float(np.sqrt((w * (d - mu) ** 2).sum() / w.sum()))
        assert df['mean'][row] == pytest.approx(mu, rel=1e-5)
        assert df['standard_deviation'][row] == pytest.approx(sd, rel=1e-5)


def test_pseg_binarize_matches_the_plain_masked_statistics(tmp_path, monkeypatch):
    """With binarize=True the weights are 0/1, so both statistics reduce to the
    plain mean and population SD over (parcel n mask)."""
    atlas, mask, scalar = _pseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(
            tmp_path, statistics=['mean', 'standard_deviation'], binarize=True,
            min_coverage=0.0,
        ).outputs.out_file
    )

    inside = mask > 0
    for row in (0, 1):
        sel = (atlas[..., row] > 0) & inside
        vals = scalar[sel]
        assert df['mean'][row] == pytest.approx(float(vals.mean()), rel=1e-5)
        assert df['standard_deviation'][row] == pytest.approx(float(vals.std()), rel=1e-5)


def test_unsupported_statistic_is_rejected(tmp_path, monkeypatch):
    _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='median'):
        _run(tmp_path, statistics=['median'])


def test_label_count_must_match_a_4d_atlas(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    atlas = np.ones(shape + (2,), 'float32')
    mask = np.ones(shape, 'uint8')
    scalar = np.ones(shape, 'float32')
    _write(tmp_path, atlas, mask, scalar, ['only_one'])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='2 volumes.*1 label'):
        _run(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_stats.py -v`

Expected: all FAIL at import with `ModuleNotFoundError: No module named 'bdt.interfaces.parcellate_stats'`.

- [ ] **Step 3: Write the implementation**

Create `src/bdt/interfaces/parcellate_stats.py`:

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
"""Multi-statistic parcellation of a volumetric *scalar* into a tidy table.

``parcellate_timeseries`` keeps XCP-D's wide (timepoints x parcels) layout via
:class:`~bdt.interfaces.connectivity.NiftiParcellate` and
:class:`~bdt.interfaces.probseg.ProbSegParcellate`.  A parcellated *scalar* has no
time axis, so it is reported tidily instead — a row per parcel, a column per
requested statistic — matching the along-tract profile tables.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from bdt.utils.statistics import SUPPORTED_STATISTICS

LOGGER = logging.getLogger('nipype.interface')

#: statistic -> nilearn ``NiftiLabelsMasker`` strategy, for a 3D label atlas.
_NILEARN_STRATEGY = {'mean': 'mean', 'standard_deviation': 'standard_deviation'}


class _ParcellateScalarStatisticsInputSpec(BaseInterfaceInputSpec):
    scalar = File(exists=True, mandatory=True, desc='3D scalar on the atlas grid')
    atlas = File(exists=True, mandatory=True, desc='3D label or 4D per-region atlas')
    atlas_labels = File(exists=True, mandatory=True, desc='BIDS dseg.tsv (index/name)')
    mask = File(exists=True, mandatory=True, desc='brain mask on the scalar grid')
    statistics = traits.List(
        traits.Str,
        value=['mean'],
        usedefault=True,
        desc='per-parcel statistics to compute, in output column order',
    )
    binarize = traits.Bool(
        False, usedefault=True, desc='binarize each volume of a 4D atlas first'
    )
    min_coverage = traits.Float(
        0.5, usedefault=True, desc='parcels below this coverage are NaN in every column'
    )


class _ParcellateScalarStatisticsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy TSV: node + one column per statistic')
    coverage = File(exists=True, desc='Parcel-wise coverage file.')


class ParcellateScalarStatistics(SimpleInterface):
    """Per-parcel statistics of a scalar, as a tidy table.

    A **3D** atlas delegates to ``NiftiLabelsMasker(strategy=...)``, once per
    statistic.  A **4D** atlas (one volume per region, possibly overlapping) is
    weighted: the mean is ``sum(w*d)/sum(w)`` and the standard deviation the
    matching weighted population SD ``sqrt(sum(w*(d-mu)^2)/sum(w))``, both over
    voxels inside the brain mask.

    Coverage is ``|parcel n mask| / |parcel|`` in both cases — from the atlas and
    the mask alone, never from the data.
    """

    input_spec = _ParcellateScalarStatisticsInputSpec
    output_spec = _ParcellateScalarStatisticsOutputSpec

    def _run_interface(self, runtime):
        statistics = list(self.inputs.statistics)
        unsupported = [s for s in statistics if s not in SUPPORTED_STATISTICS]
        if unsupported:
            raise ValueError(
                f'Unsupported statistic(s) {", ".join(unsupported)}. '
                f'Supported: {", ".join(SUPPORTED_STATISTICS)}.'
            )

        labels_df = pd.read_table(self.inputs.atlas_labels).sort_values(by='index')
        atlas_img = nb.load(self.inputs.atlas)

        if atlas_img.ndim == 3:
            names, values, coverage = self._label_atlas(labels_df, atlas_img, statistics)
        else:
            names, values, coverage = self._weighted_atlas(labels_df, atlas_img, statistics)

        usable = coverage >= self.inputs.min_coverage
        n_dropped = int((~usable).sum())
        if n_dropped:
            LOGGER.warning(
                '%d/%d parcels fall below min_coverage=%.2f and are set to NaN.',
                n_dropped, len(names), self.inputs.min_coverage,
            )

        table = {'node': names}
        for stat in statistics:
            column = np.asarray(values[stat], dtype='float64')
            table[stat] = np.where(usable, column, np.nan)

        self._results['out_file'] = os.path.join(runtime.cwd, 'parcellated.tsv')
        pd.DataFrame(table).to_csv(
            self._results['out_file'], sep='\t', na_rep='n/a', index=False
        )

        self._results['coverage'] = os.path.join(runtime.cwd, 'coverage.tsv')
        pd.DataFrame(
            coverage.astype(np.float32), index=names, columns=['coverage']
        ).to_csv(self._results['coverage'], sep='\t', na_rep='n/a', index_label='Node')
        return runtime

    def _label_atlas(self, labels_df, atlas_img, statistics):
        """3D integer-label atlas -> nilearn, one masker per statistic."""
        lut = labels_df[['index', 'name']].reset_index(drop=True)
        names = lut['name'].astype(str).tolist()

        # Coverage from the binarized atlas alone: masked voxel count over total.
        # float32 (not uint8) -- nilearn 0.14's strategy='sum' returns 0 for uint8.
        binary = nb.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.float32), atlas_img.affine
        )
        counts = {}
        for key, mask_img in (('covered', self.inputs.mask), ('total', None)):
            masker = NiftiLabelsMasker(
                labels_img=atlas_img, lut=lut, background_label=0, mask_img=mask_img,
                strategy='sum', resampling_target=None, keep_masked_labels=True,
                standardize=None,
            )
            counts[key] = np.squeeze(masker.fit_transform(binary))
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(counts['total'] > 0, counts['covered'] / counts['total'], 0.0)

        values = {}
        for stat in statistics:
            masker = NiftiLabelsMasker(
                labels_img=atlas_img, lut=lut, background_label=0,
                mask_img=self.inputs.mask, strategy=_NILEARN_STRATEGY[stat],
                resampling_target=None, keep_masked_labels=True, standardize=None,
            )
            values[stat] = np.squeeze(masker.fit_transform(self.inputs.scalar))
        return names, values, np.atleast_1d(coverage)

    def _weighted_atlas(self, labels_df, atlas_img, statistics):
        """4D per-region atlas -> mask-restricted weighted statistics."""
        n_parcels = atlas_img.shape[3]
        names = labels_df['name'].astype(str).tolist()
        if len(names) != n_parcels:
            raise ValueError(
                f'Atlas {self.inputs.atlas} has {n_parcels} volumes but '
                f'{self.inputs.atlas_labels} has {len(names)} labels; they must agree.'
            )

        masker = NiftiMasker(mask_img=self.inputs.mask, standardize=None)
        weights = np.atleast_2d(masker.fit_transform(self.inputs.atlas))
        data = np.atleast_2d(masker.transform(self.inputs.scalar))[0]

        total = np.asarray(atlas_img.dataobj, dtype='float64').reshape(-1, n_parcels)
        if self.inputs.binarize:
            weights = (weights > 0).astype('float64')
            total = (total > 0).astype('float64')
        total_weight = total.sum(axis=0)
        covered = weights.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(total_weight > 0, covered / total_weight, 0.0)

        safe = np.where(covered > 0, covered, np.nan)
        means = (weights @ data) / safe
        values = {}
        if 'mean' in statistics:
            values['mean'] = means
        if 'standard_deviation' in statistics:
            deviation = data[np.newaxis, :] - means[:, np.newaxis]
            variance = (weights * deviation**2).sum(axis=1) / safe
            values['standard_deviation'] = np.sqrt(variance)
        return names, values, coverage
```

- [ ] **Step 4: Register the module**

In `src/bdt/interfaces/__init__.py`, add `parcellate_stats` to the import line and `__all__`, keeping alphabetical order:

```python
from . import bids, parcellate_stats, probseg, reportlets, transforms

__all__ = ['bids', 'parcellate_stats', 'probseg', 'reportlets', 'transforms']
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_stats.py -v`

Expected: 9 passed.

- [ ] **Step 6: Confirm nothing else moved**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: 177 passed plus the 9 new ones; 0 failed.

---

### Task 3: Merge per-statistic pscalars into the tidy table

The CIFTI counterpart of Task 2's table. Kept as its own interface so it is testable without a workflow.

**Files:**
- Create: `src/bdt/interfaces/cifti_stats.py`
- Modify: `src/bdt/interfaces/__init__.py`
- Test: `test/engine/test_cifti_stats.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `PscalarsToTidyTsv` (a `SimpleInterface`).
  - Inputs: `in_files` (`InputMultiObject(File)`, mandatory — one parcellated CIFTI per statistic, in the same order as `statistics`); `statistics` (`List(Str)`, mandatory — the column names); `out_file` (`File`, default `'parcellated.tsv'`).
  - Output: `out_file` — a tidy TSV whose first column is `node` (the CIFTI parcel names) followed by one column per statistic.

  Task 4 wires it.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_cifti_stats.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Merging per-statistic parcellated CIFTIs into one tidy table."""

import pytest

pytest.importorskip('nibabel')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.cifti_stats import PscalarsToTidyTsv  # noqa: E402


def _pscalar(path, values, names=('P1', 'P2', 'P3')):
    """A minimal parcellated CIFTI: one map over ``names`` parcels."""
    from nibabel.cifti2 import cifti2_axes as cax

    parcels = cax.ParcelsAxis(
        name=list(names),
        voxels=[np.array([[0, 0, 0]]) for _ in names],
        vertices=[{} for _ in names],
        affine=np.eye(4),
        volume_shape=(2, 2, 2),
        nvertices={},
    )
    scalars = cax.ScalarAxis(name=['x'])
    hdr = nb.cifti2.Cifti2Header.from_axes((scalars, parcels))
    nb.Cifti2Image(np.asarray(values, dtype=float)[None, :], hdr).to_filename(str(path))
    return str(path)


def test_columns_are_node_plus_each_statistic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, 2.0, 3.0])
    b = _pscalar(tmp_path / 'sd.pscalar.nii', [0.1, 0.2, 0.3])

    res = PscalarsToTidyTsv(
        in_files=[a, b], statistics=['mean', 'standard_deviation']
    ).run()
    df = pd.read_table(res.outputs.out_file)

    assert list(df.columns) == ['node', 'mean', 'standard_deviation']
    assert df['node'].tolist() == ['P1', 'P2', 'P3']
    assert df['mean'].tolist() == [1.0, 2.0, 3.0]
    assert df['standard_deviation'].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_single_statistic_gives_one_value_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [4.0, 5.0, 6.0])
    res = PscalarsToTidyTsv(in_files=[a], statistics=['mean']).run()
    df = pd.read_table(res.outputs.out_file)
    assert list(df.columns) == ['node', 'mean']


def test_nan_parcels_round_trip_as_na(tmp_path, monkeypatch):
    """Low-coverage parcels are NaN upstream and must stay missing, not become 0."""
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, np.nan, 3.0])
    res = PscalarsToTidyTsv(in_files=[a], statistics=['mean']).run()
    df = pd.read_table(res.outputs.out_file)
    assert np.isnan(df['mean'][1])
    assert 'n/a' in (tmp_path / 'parcellated.tsv').read_text()


def test_file_and_statistic_counts_must_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='1 file.*2 statistic'):
        PscalarsToTidyTsv(
            in_files=[a], statistics=['mean', 'standard_deviation']
        ).run()


def test_mismatched_parcel_names_are_rejected(tmp_path, monkeypatch):
    """Every statistic must describe the same parcels, in the same order."""
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, 2.0, 3.0])
    b = _pscalar(tmp_path / 'sd.pscalar.nii', [1.0, 2.0], names=('P1', 'P2'))
    with pytest.raises(ValueError, match='parcel'):
        PscalarsToTidyTsv(
            in_files=[a, b], statistics=['mean', 'standard_deviation']
        ).run()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_cifti_stats.py -v`

Expected: all FAIL at import with `ModuleNotFoundError: No module named 'bdt.interfaces.cifti_stats'`.

- [ ] **Step 3: Write the implementation**

Create `src/bdt/interfaces/cifti_stats.py`:

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
"""Fold per-statistic parcellated CIFTIs into one tidy table.

A CIFTI scalar map holds exactly one value per parcel, so several statistics need
several files.  The *table*, though, should look the same as the volumetric one —
a row per parcel, a column per statistic — so the two modalities are readable side
by side.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _PscalarsToTidyTsvInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='parcellated CIFTIs, one per statistic, ordered like ``statistics``',
    )
    statistics = traits.List(
        traits.Str, mandatory=True, desc='column name for each file, in order'
    )
    out_file = File('parcellated.tsv', usedefault=True, desc='output tidy TSV')


class _PscalarsToTidyTsvOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy TSV: node + one column per statistic')


class PscalarsToTidyTsv(SimpleInterface):
    """Merge parcellated CIFTIs into a ``node`` + one-column-per-statistic table."""

    input_spec = _PscalarsToTidyTsvInputSpec
    output_spec = _PscalarsToTidyTsvOutputSpec

    def _run_interface(self, runtime):
        in_files = list(self.inputs.in_files)
        statistics = list(self.inputs.statistics)
        if len(in_files) != len(statistics):
            raise ValueError(
                f'Got {len(in_files)} file(s) for {len(statistics)} statistic(s); '
                'each statistic needs exactly one parcellated CIFTI.'
            )

        table = {}
        names = None
        for path, stat in zip(in_files, statistics, strict=True):
            parcels, values = _read_pscalar(path)
            if names is None:
                names = parcels
            elif parcels != names:
                raise ValueError(
                    f'{path} describes different parcels than the first input; every '
                    'statistic must cover the same parcels in the same order.'
                )
            table[stat] = values

        out_file = os.path.abspath(self.inputs.out_file)
        pd.DataFrame({'node': names, **table}).to_csv(
            out_file, sep='\t', na_rep='n/a', index=False
        )
        self._results['out_file'] = out_file
        return runtime


def _read_pscalar(path):
    """``(parcel names, values)`` from a parcellated CIFTI."""
    img = nb.load(str(path))
    data = np.asarray(img.get_fdata())
    axes = [img.header.get_axis(i) for i in range(data.ndim)]
    parc_idx = next(
        (i for i, ax in enumerate(axes) if isinstance(ax, nb.cifti2.ParcelsAxis)), None
    )
    if parc_idx is None:
        raise ValueError(f'{path} is not a parcellated CIFTI (no ParcelsAxis).')
    names = list(axes[parc_idx].name)
    if parc_idx == 0:
        data = data.T
    return names, np.squeeze(data)
```

- [ ] **Step 4: Register the module**

In `src/bdt/interfaces/__init__.py`, add `cifti_stats` to the import line and `__all__`, keeping alphabetical order:

```python
from . import bids, cifti_stats, parcellate_stats, probseg, reportlets, transforms

__all__ = [
    'bids',
    'cifti_stats',
    'parcellate_stats',
    'probseg',
    'reportlets',
    'transforms',
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_cifti_stats.py -v`

Expected: 5 passed.

---

### Task 4: Per-statistic CIFTI parcellation

**Files:**
- Modify: `src/bdt/engine/factories.py` — `_init_parcellate_cifti_wf`
- Test: `test/engine/test_parcellate_statistics_routing.py` (create)

**Interfaces:**
- Consumes: `parse_statistics` (Task 1); `PscalarsToTidyTsv` (Task 3).
- Produces: `_init_parcellate_cifti_wf(node, name, in_role, out_file, statistics=None)`. When `statistics` is `None` (the `parcellate_timeseries` call) the workflow is **exactly** as it is today: nodes `parcellate_data` / `mask`, `outputnode` fields `['out', 'coverage']`. When a list is given, it additionally builds `parcellate_data_<stat>` / `mask_<stat>` per statistic, a `to_tsv` node, and `outputnode` gains `out_<stat>` per statistic plus `tsv`; `out` stays the **first** requested statistic so role wiring downstream is unaffected.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_parcellate_statistics_routing.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Per-statistic parcellation wiring, CIFTI and volumetric."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import (  # noqa: E402
    FactoryContext,
    init_parcellate_scalar_wf,
    init_parcellate_timeseries_wf,
)
from bdt.engine.selection import Match  # noqa: E402
from bdt.engine.workflow import _identity_fields  # noqa: E402


def _node(name='parc', **parameters):
    return SimpleNamespace(
        name=name,
        inputs={'scalar': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters=parameters,
        desc=None,
    )


def _cifti_context():
    return FactoryContext(resolved={
        'load_scalar': Match('s.dscalar.nii', {'extension': '.dscalar.nii'}),
        'load_atlas': Match('a.dlabel.nii', {'extension': '.dlabel.nii'}),
    })


def test_cifti_default_builds_one_mean_parcellation():
    wf = init_parcellate_scalar_wf(_node(), context=_cifti_context())
    names = set(wf.list_node_names())
    assert 'parcellate_data_mean' in names
    assert 'parcellate_data_standard_deviation' not in names
    assert _identity_fields(wf, 'outputnode') == {'out', 'out_mean', 'coverage', 'tsv'}


def test_cifti_two_statistics_build_two_parcellations():
    wf = init_parcellate_scalar_wf(
        _node(statistics=['mean', 'standard_deviation']), context=_cifti_context()
    )
    names = set(wf.list_node_names())
    assert 'parcellate_data_mean' in names
    assert 'parcellate_data_standard_deviation' in names
    assert wf.get_node('parcellate_data_mean').inputs.cor_method == 'MEAN'
    assert wf.get_node('parcellate_data_standard_deviation').inputs.cor_method == 'STDEV'
    assert _identity_fields(wf, 'outputnode') == {
        'out', 'out_mean', 'out_standard_deviation', 'coverage', 'tsv'
    }


def test_cifti_tidy_tsv_node_receives_every_statistic():
    from bdt.interfaces.cifti_stats import PscalarsToTidyTsv

    wf = init_parcellate_scalar_wf(
        _node(statistics=['mean', 'standard_deviation']), context=_cifti_context()
    )
    to_tsv = wf.get_node('to_tsv')
    assert isinstance(to_tsv.interface, PscalarsToTidyTsv)
    assert to_tsv.inputs.statistics == ['mean', 'standard_deviation']

    # each masked pscalar reaches the merger, and the merger feeds the table --
    # nipype needs a Merge node to build the in_files list, so the masks do not
    # connect to to_tsv directly
    edges = {(u.name, d.name): data['connect'] for u, d, data in wf._graph.edges(data=True)}
    assert edges[('mask_mean', 'statistic_list')] == [('out_file', 'in1')]
    assert edges[('mask_standard_deviation', 'statistic_list')] == [('out_file', 'in2')]
    assert edges[('statistic_list', 'to_tsv')] == [('out', 'in_files')]


def test_parcellate_timeseries_cifti_is_untouched():
    """The timeseries path keeps its original node names and outputnode fields."""
    node = SimpleNamespace(
        name='parc', inputs={'timeseries': ['load_bold'], 'atlas': ['load_atlas']},
        parameters={}, desc=None,
    )
    ctx = FactoryContext(resolved={
        'load_bold': Match('b.dtseries.nii', {'extension': '.dtseries.nii'}),
        'load_atlas': Match('a.dlabel.nii', {'extension': '.dlabel.nii'}),
    })
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    names = set(wf.list_node_names())
    assert 'parcellate_data' in names
    assert not any(n.startswith('parcellate_data_') for n in names)
    assert _identity_fields(wf, 'outputnode') == {'out', 'coverage'}


def test_unsupported_statistic_fails_at_build_time():
    with pytest.raises(ValueError, match='median'):
        init_parcellate_scalar_wf(_node(statistics=['median']), context=_cifti_context())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_statistics_routing.py -v`

Expected: the CIFTI tests FAIL (`parcellate_data_mean` absent — the node is still called `parcellate_data`); `test_parcellate_timeseries_cifti_is_untouched` PASSES already and must keep passing.

- [ ] **Step 3: Rework the CIFTI workflow**

In `src/bdt/engine/factories.py`, change the signature of `_init_parcellate_cifti_wf` to accept `statistics=None`:

```python
def _init_parcellate_cifti_wf(node, name, in_role, out_file, statistics=None) -> pe.Workflow:
```

Immediately after `min_coverage` is read, add the statistic plumbing:

```python
    # ``statistics=None`` -> the historical single-mean workflow (parcellate_timeseries).
    # A list -> one Workbench parcellation per statistic plus the merged tidy table.
    per_statistic = list(statistics) if statistics else None
    method = {'mean': 'MEAN', 'standard_deviation': 'STDEV'}
```

Replace the `outputnode` construction with:

```python
    fields = ['out', 'coverage']
    if per_statistic:
        fields += [f'out_{s}' for s in per_statistic] + ['tsv']
    outputnode = pe.Node(niu.IdentityInterface(fields=fields), name='outputnode')
```

Leave `restrict_atlas`, `vertex_mask`, `parcellate_coverage` and `threshold` exactly as they are, and delete the single `parcellate_data` / `mask` pair plus their four connections **only when `per_statistic` is set**. Concretely, replace the `parcellate_data` node, the `mask` node, and the `wf.connect([...])` block with:

```python
    wf.connect([
        (inputnode, restrict_atlas, [(in_role, 'template_cifti'), ('atlas', 'label')]),
        (inputnode, vertex_mask, [(in_role, 'in_file')]),
        (restrict_atlas, parcellate_coverage, [('out_file', 'atlas_label')]),
        (vertex_mask, parcellate_coverage, [('mask_file', 'in_file')]),
        (parcellate_coverage, threshold, [('out_file', 'data')]),
        (parcellate_coverage, outputnode, [('out_file', 'coverage')]),
    ])  # fmt:skip

    for index, stat in enumerate(per_statistic or ['mean']):
        suffix = f'_{stat}' if per_statistic else ''
        parcellate_data = pe.Node(
            CiftiParcellateWorkbench(
                direction='COLUMN',
                only_numeric=True,
                cor_method=method[stat] if per_statistic else 'MEAN',
                out_file=out_file if not per_statistic else f'parcellated_{stat}.pscalar.nii',
            ),
            name=f'parcellate_data{suffix}',
        )
        mask = pe.Node(CiftiMask(), name=f'mask{suffix}')
        wf.connect([
            (inputnode, parcellate_data, [(in_role, 'in_file')]),
            (restrict_atlas, parcellate_data, [('out_file', 'atlas_label')]),
            (vertex_mask, parcellate_data, [('mask_file', 'cifti_weights')]),
            (parcellate_data, mask, [('out_file', 'in_file')]),
            (threshold, mask, [('out_file', 'mask')]),
        ])  # fmt:skip
        if per_statistic:
            wf.connect([(mask, outputnode, [('out_file', f'out_{stat}')])])
            if index == 0:
                # ``out`` stays the first requested statistic so role wiring into a
                # downstream node is unchanged by asking for more statistics.
                wf.connect([(mask, outputnode, [('out_file', 'out')])])
        else:
            wf.connect([(mask, outputnode, [('out_file', 'out')])])

    if per_statistic:
        to_tsv = pe.Node(
            PscalarsToTidyTsv(statistics=per_statistic, out_file='parcellated.tsv'),
            name='to_tsv',
        )
        merge = pe.Node(niu.Merge(len(per_statistic)), name='statistic_list')
        for index, stat in enumerate(per_statistic, start=1):
            wf.connect([(wf.get_node(f'mask_{stat}'), merge, [('out_file', f'in{index}')])])
        wf.connect([
            (merge, to_tsv, [('out', 'in_files')]),
            (to_tsv, outputnode, [('out_file', 'tsv')]),
        ])  # fmt:skip

    return wf
```

Add the import at the top of the function body, beside the other local imports:

```python
    from bdt.interfaces.cifti_stats import PscalarsToTidyTsv
```

- [ ] **Step 4: Pass the statistics in from the scalar factory**

In `init_parcellate_scalar_wf`, replace the CIFTI branch's call with one that parses and forwards the parameter:

```python
    if context.role_is_cifti(node, 'scalar'):
        from bdt.utils.statistics import parse_statistics

        return _init_parcellate_cifti_wf(
            node, name, 'scalar', 'parcellated.pscalar.nii',
            statistics=parse_statistics(node.parameters),
        )
```

Leave `init_parcellate_timeseries_wf`'s call unchanged — it must keep passing no `statistics`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_statistics_routing.py -v`

Expected: 5 passed.

- [ ] **Step 6: Confirm the timeseries path is untouched**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: 0 failed. Any failure in `test_parcellate_timeseries_routing.py` or `test_nipype_workflow.py` means the `statistics=None` branch is not byte-identical — fix the branch, do not adjust those tests.

---

### Task 5: Plan the per-statistic products

**Files:**
- Modify: `src/bdt/spec/actions.py` — `OutputSpec`, and `parcellate_scalar`
- Modify: `src/bdt/outputs/plan.py` — `build_sink_plan`
- Test: `test/engine/test_output_plan.py` (append)

**Interfaces:**
- Consumes: `parse_statistics`, `compose_statistic_entity` (Task 1); `outputnode` fields `out_<stat>` and `tsv` (Task 4).
- Produces: `OutputSpec.tsv_source_field: str | None = None`. When set and the node is CIFTI, the TSV product is a `PASSTHROUGH` reading that `outputnode` field instead of a `CIFTI_TO_TSV` conversion.

- [ ] **Step 1: Write the failing tests**

Append to `test/engine/test_output_plan.py`:

```python
def _cifti_scalar_spec(**parameters):
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_alff',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'suffix': 'boldmap', 'statistic': 'alff'},
                },
                {
                    'name': 'atlas',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': '4S1056Parcels'},
                },
                {
                    'name': 'alff_parc',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 'load_alff', 'atlas': 'atlas'},
                    'parameters': parameters,
                    'write_outputs': True,
                },
            ]
        }
    )
    resolved = {
        'load_alff': Match(
            path='/x/sub-01_stat-alff_boldmap.dscalar.nii',
            entities={
                'subject': '01', 'space': 'fsLR', 'den': '91k', 'statistic': 'alff',
                'suffix': 'boldmap', 'datatype': 'func', 'extension': '.dscalar.nii',
            },
        ),
        'atlas': Match(path='/a/atlas.dlabel.nii', entities={'atlas': '4S1056Parcels'}),
    }
    return spec, resolved


def test_cifti_scalar_plans_one_pscalar_per_statistic_and_one_tsv():
    spec, resolved = _cifti_scalar_spec(statistics=['mean', 'standard_deviation'])
    prods = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/a'})['alff_parc']

    pscalars = [p for p in prods if p.extension == '.pscalar.nii' and p.suffix == 'boldmap']
    tsvs = [p for p in prods if p.extension == '.tsv' and p.suffix == 'boldmap']
    assert len(pscalars) == 2
    assert len(tsvs) == 1, 'the tidy table is singular, whatever the statistic count'

    # source statistic first, joined with '+', normalized to alphanumerics
    assert [p.entities['statistic'] for p in pscalars] == [
        'alff+mean', 'alff+standarddeviation'
    ]
    # each reads its own outputnode field
    assert [p.source_field for p in pscalars] == ['out_mean', 'out_standard_deviation']
    # the tidy table keeps the source's own statistic and is passed through, not converted
    assert tsvs[0].entities['statistic'] == 'alff'
    assert tsvs[0].source_field == 'tsv'
    assert tsvs[0].derive == PASSTHROUGH


def test_cifti_scalar_default_plans_a_single_mean_pscalar():
    spec, resolved = _cifti_scalar_spec()
    prods = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/a'})['alff_parc']
    pscalars = [p for p in prods if p.extension == '.pscalar.nii' and p.suffix == 'boldmap']
    assert len(pscalars) == 1
    assert pscalars[0].entities['statistic'] == 'alff+mean'
    assert pscalars[0].source_field == 'out_mean'


def test_volumetric_scalar_plans_one_tsv_and_no_pscalar():
    spec, resolved = _cifti_scalar_spec(statistics=['mean', 'standard_deviation'])
    resolved['load_alff'] = Match(
        path='/x/sub-01_stat-alff_boldmap.nii.gz',
        entities={
            'subject': '01', 'space': 'MNI152NLin6Asym', 'statistic': 'alff',
            'suffix': 'boldmap', 'datatype': 'func', 'extension': '.nii.gz',
        },
    )
    prods = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/a'})['alff_parc']
    assert not [p for p in prods if p.extension == '.pscalar.nii']
    primary = [p for p in prods if p.suffix == 'boldmap' and p.extension == '.tsv']
    assert len(primary) == 1
    assert primary[0].entities['statistic'] == 'alff'  # the table holds every statistic
    assert primary[0].source_field == 'out'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_output_plan.py -v -k "statistic"`

Expected: the three new tests FAIL — one pscalar is planned regardless of statistics, and the TSV is a `CIFTI_TO_TSV`.

- [ ] **Step 3: Add the OutputSpec field**

In `src/bdt/spec/actions.py`, add to `OutputSpec` beside the other optional fields:

```python
    #: When set, the sub-workflow already produces the tabular form on this
    #: ``outputnode`` field, so the sink passes it through instead of converting the
    #: native CIFTI.  Used where the table holds several statistics at once and so
    #: cannot be derived from any single CIFTI file.
    tsv_source_field: str | None = None
```

In the `parcellate_scalar` `ActionSpec`, add `statistics` to the accepted parameters and point the TSV at the merged field:

```python
        parameters=frozenset({'min_coverage', 'statistics'}),
```

and inside its `out=_o(...)`:

```python
            tsv_source_field='tsv',
```

- [ ] **Step 4: Multiply the CIFTI product in the plan**

In `src/bdt/outputs/plan.py`, add the import at the top:

```python
from bdt.utils.statistics import compose_statistic_entity, parse_statistics
```

Replace the native-CIFTI block with one that expands across statistics:

```python
        if cifti_by_node.get(node.name) and cifti_suffix and out.cifti_extension:
            # One native CIFTI per statistic: a parcellated CIFTI holds a single
            # value per parcel, so several statistics need several files.  Actions
            # that do not accept ``statistics`` keep exactly one product.
            if 'statistics' in aspec.parameters:
                requested = parse_statistics(node.parameters)
            else:
                requested = []
            if requested:
                source_statistic = mid.get('statistic')
                for stat in requested:
                    stat_ent = dict(mid)
                    stat_ent['statistic'] = compose_statistic_entity(source_statistic, stat)
                    products.append(
                        OutputProduct(
                            derive=PASSTHROUGH,
                            suffix=cifti_suffix,
                            extension=out.cifti_extension,
                            entities=stat_ent,
                            sidecar=dict(sidecar),
                            source_field=f'out_{stat}',
                            **common,
                        )
                    )
            else:
                products.append(
                    OutputProduct(
                        derive=PASSTHROUGH,
                        suffix=cifti_suffix,
                        extension=out.cifti_extension,
                        entities=dict(mid),
                        sidecar=dict(sidecar),
                        **common,
                    )
                )
            # ... plus a flattened TSV for tabular (parcellated) outputs, but not for
            # a dense CIFTI (a resampled/mapped surface scalar has no table form).
            # The table holds every statistic at once, so it keeps the source's own
            # ``statistic`` entity and is read straight off the sub-workflow when the
            # action builds it there.
            if out.emit_tsv:
                products.append(
                    OutputProduct(
                        derive=PASSTHROUGH if out.tsv_source_field else CIFTI_TO_TSV,
                        suffix=tsv_suffix,
                        extension=out.extension,
                        entities=dict(mid),
                        sidecar=dict(sidecar),
                        source_field=out.tsv_source_field or 'out',
                        **common,
                    )
                )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_output_plan.py -v`

Expected: all pass. `test_parcellate_scalar_preserves_source_naming` asserts the CIFTI pscalar list — update its expected products to the two-entry form only if it requested two statistics; with the default it should still see exactly one pscalar, now carrying `statistic == 'alff+mean'`.

- [ ] **Step 6: Full suite**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: 0 failed.

---

### Task 6: Wire the volumetric factory and accept end to end

**Files:**
- Modify: `src/bdt/engine/factories.py` — `_init_parcellate_volumetric_wf`
- Test: `test/engine/test_parcellate_statistics_routing.py` (append)

**Interfaces:**
- Consumes: `ParcellateScalarStatistics` (Task 2); `parse_statistics` (Task 1).
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing tests**

Append to `test/engine/test_parcellate_statistics_routing.py`:

```python
def _volumetric_context(tmp_path, ndim=3):
    import nibabel as nb
    import numpy as np

    from bdt.engine.selection import DictDataProvider

    shape = (4, 4, 4) if ndim == 3 else (4, 4, 4, 2)
    atlas = tmp_path / 'tpl-MNI152NLin6Asym_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros(shape, 'float32'), np.eye(4)).to_filename(atlas)
    (tmp_path / 'tpl-MNI152NLin6Asym_atlas-Y_dseg.tsv').write_text('index\tname\n1\tA\n2\tB\n')
    mask = tmp_path / 'sub-01_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(mask)

    return FactoryContext(
        provider=DictDataProvider({'fmriprep': [Match(str(mask), {
            'space': 'MNI152NLin6Asym', 'suffix': 'mask', 'desc': 'brain',
            'datatype': 'func',
        })]}),
        subject='01', datasets=['fmriprep'],
        resolved={
            'load_scalar': Match('/d/sub-01_space-MNI152NLin6Asym_cbf.nii.gz', {
                'space': 'MNI152NLin6Asym', 'suffix': 'cbf', 'datatype': 'func',
                'extension': '.nii.gz',
            }),
            'load_atlas': Match(str(atlas), {
                'space': 'MNI152NLin6Asym', 'suffix': 'dseg', 'extension': '.nii.gz',
            }),
        },
    )


def test_volumetric_scalar_uses_the_statistics_interface(tmp_path):
    from bdt.interfaces.parcellate_stats import ParcellateScalarStatistics

    wf = init_parcellate_scalar_wf(
        _node(statistics=['mean', 'standard_deviation']),
        context=_volumetric_context(tmp_path),
    )
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, ParcellateScalarStatistics)
    assert parcellate.inputs.statistics == ['mean', 'standard_deviation']


def test_volumetric_scalar_defaults_to_mean(tmp_path):
    wf = init_parcellate_scalar_wf(_node(), context=_volumetric_context(tmp_path))
    assert wf.get_node('parcellate').inputs.statistics == ['mean']


def test_volumetric_4d_atlas_binarizes_a_thresholded_dseg(tmp_path):
    """The 4D branch keeps its binarize signal when statistics are requested."""
    ctx = _volumetric_context(tmp_path, ndim=4)
    wf = init_parcellate_scalar_wf(_node(statistics=['mean']), context=ctx)
    assert wf.get_node('parcellate').inputs.binarize is True


def test_volumetric_timeseries_still_uses_nifti_parcellate(tmp_path):
    """Only the scalar action gains statistics; the timeseries path is unchanged."""
    from bdt.interfaces.connectivity import NiftiParcellate

    ctx = _volumetric_context(tmp_path)
    node = SimpleNamespace(
        name='parc', inputs={'timeseries': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters={}, desc=None,
    )
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    assert isinstance(wf.get_node('parcellate').interface, NiftiParcellate)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_statistics_routing.py -v -k volumetric`

Expected: the first three FAIL (the scalar path still builds `NiftiParcellate` / `ProbSegParcellate`, which have no `statistics` input); the timeseries one passes.

- [ ] **Step 3: Route the volumetric scalar path**

In `_init_parcellate_volumetric_wf`, the `data_role` distinguishes the two actions: `'scalar'` gets the statistics interface, `'timeseries'` keeps the existing pair. Replace the interface-selection block with:

```python
    ndim = context.role_atlas_ndim(node, 'atlas')
    mask = _discover_brain_mask(context, node, data_role)
    atlas_suffix = context.role_suffix(node, 'atlas', default='probseg')

    if data_role == 'scalar':
        # A parcellated scalar is reported tidily, one row per parcel and one column
        # per statistic, for both atlas forms.
        from bdt.utils.statistics import parse_statistics

        parcellate = pe.Node(
            ParcellateScalarStatistics(
                mask=mask,
                min_coverage=min_coverage,
                statistics=parse_statistics(node.parameters),
                binarize=ndim != 3 and atlas_suffix == 'dseg',
            ),
            name='parcellate',
        )
        data_field = 'scalar'
    elif ndim == 3:
        parcellate = pe.Node(
            NiftiParcellate(mask=mask, min_coverage=min_coverage), name='parcellate'
        )
        data_field = 'filtered_file'
    else:
        parcellate = pe.Node(
            ProbSegParcellate(
                mask=mask,
                min_coverage=min_coverage,
                binarize=atlas_suffix == 'dseg',
            ),
            name='parcellate',
        )
        data_field = 'data'
```

Add `ParcellateScalarStatistics` to the local imports at the top of the function:

```python
    from bdt.interfaces.parcellate_stats import ParcellateScalarStatistics
```

The `outputnode` connection differs by interface — `ParcellateScalarStatistics` exposes `out_file`, the other two `timeseries`. Replace the final `wf.connect` block's outputnode line accordingly:

```python
    primary_field = 'out_file' if data_role == 'scalar' else 'timeseries'
    wf.connect([
        (inputnode, parcellate, [(data_role, data_field)]),
        (atlas_node, parcellate, [(atlas_field, 'atlas')]),
        (parcellate, outputnode, [(primary_field, 'out'), ('coverage', 'coverage')]),
    ])  # fmt:skip
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_statistics_routing.py -v`

Expected: 9 passed.

- [ ] **Step 5: Acceptance against the real spec**

Add `statistics: [mean, standard_deviation]` to `cbf_roi` in `scripts/tract_parcellate.yml`:

```yaml
- name: cbf_roi
  action: parcellate_scalar
  inputs:
    scalar: load_cbf
    atlas: bundle_rois  # warps ROI dseg -> CBF space
  parameters:
    statistics: [mean, standard_deviation]
  write_outputs: true
```

Then confirm the spec still validates and compiles:

Run: `micromamba run -n bdtenv python -m pytest test/spec -q`

Expected: 0 failed.

- [ ] **Step 6: Full verification**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`

Expected: 0 failed.

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -W error::UserWarning -q`

Expected: 0 failed; no warnings escalated.

- [ ] **Step 7: Record the outcome**

Append a Spec-5 section to `.superpowers/sdd/progress.md` covering per-task status, the population-vs-sample SD decision, and the CIFTI-emits-N-pscalars / one-TSV asymmetry.

---

## Notes for the implementer

- **Do not touch `parcellate_timeseries`.** Its wide table and its use of `NiftiParcellate` / `ProbSegParcellate` are deliberate. Several tasks have a test asserting this; if one fails, the fix is in your change, not the test.
- **Do not run ruff.** It is absent from `bdtenv` and must not be borrowed from another env.
- **`standard_deviation` is population (ddof=0).** If a test comparing against `numpy.std(ddof=1)` seems to fail by a small amount, the expectation is wrong, not the code.
- **Statistic order is the column order.** `parse_statistics` preserves it deliberately; do not sort.
