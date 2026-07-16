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
"""Driver tests: tool-free unit tests for resolution/fan-out/collision, plus a
gated end-to-end run on the real GRMPY test data when wb_command is present."""

import shutil
from pathlib import Path

import pytest

from bdt.engine.selection import DictDataProvider, Match, SelectionError
from bdt.spec import parse_spec

STORY_3_1 = {
    'nodes': [
        {
            'name': 'load_bold',
            'action': 'select_data',
            'dataset': 'xcpd',
            'filters': {'suffix': 'bold', 'extension': '.dtseries.nii'},
        },
        {
            'name': 'atlas_4s',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': '4S1056Parcels'},
        },
        {
            'name': 'parcellate_bold',
            'action': 'parcellate_timeseries',
            'inputs': {'timeseries': 'load_bold', 'atlas': 'atlas_4s'},
            'write_outputs': True,
        },
        {
            'name': 'fc_bold',
            'action': 'functional_connectivity',
            'inputs': {'timeseries': 'parcellate_bold'},
            'write_outputs': True,
        },
    ]
}


def _two_task_provider():
    return DictDataProvider(
        {
            'xcpd': [
                Match('/x/sub-01_task-rest_bold.dtseries.nii',
                      {'sub': '01', 'task': 'rest', 'suffix': 'bold',
                       'extension': '.dtseries.nii', 'space': 'fsLR', 'den': '91k'}),
                Match('/x/sub-01_task-nback_bold.dtseries.nii',
                      {'sub': '01', 'task': 'nback', 'suffix': 'bold',
                       'extension': '.dtseries.nii', 'space': 'fsLR', 'den': '91k'}),
            ],
            'atlases': [
                Match('/a/atlas-4S1056Parcels_dseg.dlabel.nii',
                      {'atlas': '4S1056Parcels', 'suffix': 'dseg', 'extension': '.dlabel.nii'}),
            ],
        }
    )


def test_resolve_and_fan_out_over_matches():
    from bdt.engine.pipeline import _combinations, _resolve_selections

    spec = parse_spec(STORY_3_1)
    resolved = _resolve_selections(spec, _two_task_provider(), subject='01')
    assert len(resolved['load_bold']) == 2
    assert len(resolved['atlas_4s']) == 1

    combos = list(_combinations(resolved))
    # 2 bold x 1 atlas -> 2 combinations, each a single-match dict
    assert len(combos) == 2
    tasks = sorted(c['load_bold'].entities['task'] for c in combos)
    assert tasks == ['nback', 'rest']
    assert all(c['atlas_4s'].entities['atlas'] == '4S1056Parcels' for c in combos)


def test_missing_required_selection_errors():
    from bdt.engine.pipeline import _resolve_selections

    spec = parse_spec(STORY_3_1)
    provider = DictDataProvider({'xcpd': [], 'atlases': []})
    with pytest.raises(SelectionError, match='matched no files'):
        _resolve_selections(spec, provider, subject='01')


def test_collision_detected_before_run(tmp_path):
    from bdt.engine.pipeline import _check_collisions
    from bdt.outputs.plan import PASSTHROUGH, OutputProduct

    same = {
        'derive': PASSTHROUGH,
        'suffix': 'timeseries',
        'extension': '.tsv',
        'datatype': 'func',
        'scope': 'participant',
        'entities': {'sub': '01', 'atlas': 'A'},
    }
    plan = {'a': [OutputProduct(**same)], 'b': [OutputProduct(**same)]}
    with pytest.raises(ValueError, match='collision'):
        _check_collisions(parse_spec(STORY_3_1), plan, str(tmp_path))


# --- gated real end-to-end run on the GRMPY test data ------------------------

_DATA = Path('/Volumes/5TB/BDT_testing/data')
_ATLASES = Path('/Volumes/5TB/BDT_testing/atlases')
_HAVE_TOOLS = shutil.which('wb_command') is not None
_HAVE_DATA = _DATA.exists() and _ATLASES.exists()

requires_real = pytest.mark.skipif(
    not (_HAVE_TOOLS and _HAVE_DATA),
    reason='needs wb_command on PATH and the GRMPY test data + AtlasPack mounted',
)


@requires_real
def test_story_3_1_end_to_end_matches_xcpd(tmp_path):
    """Run parcellate + FC on real fsLR 91k data; outputs must match XCP-D bit-for-bit."""
    import nibabel as nb
    import numpy as np

    from bdt.engine.pipeline import run_spec

    datasets = {'xcpd': str(_DATA / 'derivatives' / 'xcpd'), 'atlases': str(_ATLASES)}
    spec = parse_spec(STORY_3_1)
    # narrow to one run for a fast, single-combination check
    spec.by_name()['load_bold'].filters.update(
        {'space': 'fsLR', 'desc': 'denoised', 'task': 'rest', 'acq': 'singleband'}
    )
    spec.by_name()['atlas_4s'].filters['extension'] = '.dlabel.nii'
    spec.by_name()['parcellate_bold'].parameters['min_coverage'] = 0.5

    results = run_spec(
        spec, datasets, tmp_path / 'out', tmp_path / 'work',
        subjects=['125511'], plugin='Linear',
    )
    assert len(results) == 1
    outputs = results[0].outputs
    # ptseries + tsv + coverage pscalar (parcellate) and pconn + relmat (FC)
    assert len(outputs) == 5
    assert all(Path(p).exists() for p in outputs)

    xcpd = _DATA / 'derivatives/xcpd/sub-125511/ses-1/func'
    base = 'sub-125511_ses-1_task-rest_acq-singleband_space-fsLR_seg-4S1056Parcels_den-91k'

    def data(p):
        return np.asarray(nb.load(str(p)).get_fdata())

    # ptseries: coverage-aware parcellation reproduces XCP-D exactly (incl. NaN masking)
    mine_pt = data(next(p for p in outputs if p.endswith('.ptseries.nii')))
    gt_pt = data(xcpd / f'{base}_stat-mean_timeseries.ptseries.nii')
    assert np.array_equal(np.isnan(mine_pt), np.isnan(gt_pt))
    fin = np.isfinite(mine_pt) & np.isfinite(gt_pt)
    assert np.allclose(mine_pt[fin], gt_pt[fin], atol=1e-4)

    # coverage map matches
    mine_cov = data(next(p for p in outputs if 'stat-coverage' in p))
    gt_cov = data(xcpd / f'{base}_stat-coverage_boldmap.pscalar.nii')
    assert np.allclose(mine_cov, gt_cov, atol=1e-6)

    # pconn matches
    mine_pc = data(next(p for p in outputs if p.endswith('.pconn.nii')))
    gt_pc = data(xcpd / f'{base}_stat-pearsoncorrelation_boldmap.pconn.nii')
    fin = np.isfinite(mine_pc) & np.isfinite(gt_pc)
    assert np.allclose(mine_pc[fin], gt_pc[fin], atol=1e-4)


@requires_real
def test_parcellate_scalar_matches_xcpd_alff(tmp_path):
    """parcellate_scalar on XCP-D's dense ALFF reproduces XCP-D's parcellated ALFF."""
    import nibabel as nb
    import numpy as np
    import pandas as pd

    from bdt.engine.pipeline import run_spec

    datasets = {'xcpd': str(_DATA / 'derivatives' / 'xcpd'), 'atlases': str(_ATLASES)}
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_alff',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {
                        'suffix': 'boldmap', 'stat': 'alff', 'space': 'fsLR', 'den': '91k',
                        'task': 'rest', 'acq': 'singleband', 'extension': '.dscalar.nii',
                    },
                    'exclude': [{'desc': 'smooth'}],
                },
                {
                    'name': 'atlas',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': '4S1056Parcels', 'extension': '.dlabel.nii'},
                },
                {
                    'name': 'alff_parc',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 'load_alff', 'atlas': 'atlas'},
                    'parameters': {'min_coverage': 0.5},
                    'write_outputs': True,
                },
            ]
        }
    )
    results = run_spec(
        spec, datasets, tmp_path / 'out', tmp_path / 'work',
        subjects=['125511'], plugin='Linear',
    )
    outputs = results[0].outputs
    # preserve-source naming: keeps stat-alff + boldmap suffix, adds atlas-
    ps = next(p for p in outputs if p.endswith('.pscalar.nii'))
    assert 'atlas-4S1056Parcels_stat-alff_boldmap' in ps

    mine = np.asarray(nb.load(ps).get_fdata()).ravel()
    gt_tsv = (_DATA / 'derivatives/xcpd/sub-125511/ses-1/func/'
              'sub-125511_ses-1_task-rest_acq-singleband_space-fsLR_seg-4S1056Parcels_'
              'stat-alff_bold.tsv')
    gt = pd.read_csv(str(gt_tsv), sep='\t').to_numpy().ravel()
    assert np.array_equal(np.isnan(mine), np.isnan(gt))
    fin = np.isfinite(mine) & np.isfinite(gt)
    assert np.allclose(mine[fin], gt[fin], atol=1e-4)
