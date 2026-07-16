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
"""End-to-end builder tests over synthetic NIfTI (no wb_command / ANTs needed)."""

import pytest

pytest.importorskip('nibabel')
pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.actions.connectivity import build_functional_connectivity  # noqa: E402
from bdt.actions.parcellate import build_parcellate_timeseries  # noqa: E402
from bdt.engine.result import BuildContext, NodeResult  # noqa: E402
from bdt.spec.model import Node  # noqa: E402


def _nifti(path, array, affine=None):
    nb.Nifti1Image(array, affine if affine is not None else np.eye(4)).to_filename(str(path))
    return str(path)


@pytest.fixture
def bold_and_atlas(tmp_path):
    bold = _nifti(tmp_path / 'bold.nii.gz', np.random.default_rng(0).random((4, 4, 4, 6)))
    atlas_arr = np.zeros((4, 4, 4), dtype=np.int16)
    atlas_arr[:2] = 1
    atlas_arr[2:] = 2
    atlas = _nifti(tmp_path / 'atlas.nii.gz', atlas_arr)
    return bold, atlas


def _timeseries_result(bold):
    return NodeResult(
        node='load_bold',
        action='select_data',
        fmt='timeseries',
        files=[bold],
        entities={'sub': '01', 'task': 'rest', 'space': 'T1w'},
        sources=['bids:xcpd:sub-01/func/sub-01_task-rest_bold.nii.gz'],
        space='T1w',
    )


def _atlas_result(atlas, space='T1w'):
    return NodeResult(
        node='atlas',
        action='select_atlases',
        fmt='atlas',
        files=[atlas],
        entities={'atlas': 'MyAtlas', 'space': space},
        sources=['bids:atlases:tpl-T1w_atlas-MyAtlas_dseg.nii.gz'],
        space=space,
    )


def test_parcellate_timeseries_nifti(tmp_path, bold_and_atlas):
    bold, atlas = bold_and_atlas
    node = Node(
        name='parc',
        action='parcellate_timeseries',
        inputs={'timeseries': ['load_bold'], 'atlas': ['atlas']},
    )
    resolved = {'timeseries': [_timeseries_result(bold)], 'atlas': [_atlas_result(atlas)]}
    ctx = BuildContext(work_dir=str(tmp_path / 'work'))

    results = build_parcellate_timeseries(ctx, node, resolved)
    assert len(results) == 1
    res = results[0]
    assert res.fmt == 'parcellated_timeseries'
    assert res.entities['atlas'] == 'MyAtlas'
    assert res.entities['stat'] == 'mean'
    df = pd.read_csv(res.files[0], sep='\t')
    assert df.shape == (6, 2)  # 6 timepoints x 2 regions
    # provenance threads both the bold and atlas sources
    assert any('bold' in s for s in res.sources)
    assert any('atlas' in s for s in res.sources)


def test_functional_connectivity_from_parcellated(tmp_path, bold_and_atlas):
    bold, atlas = bold_and_atlas
    parc_node = Node(
        name='parc',
        action='parcellate_timeseries',
        inputs={'timeseries': ['load_bold'], 'atlas': ['atlas']},
    )
    ctx = BuildContext(work_dir=str(tmp_path / 'work'))
    parc = build_parcellate_timeseries(
        ctx,
        parc_node,
        {'timeseries': [_timeseries_result(bold)], 'atlas': [_atlas_result(atlas)]},
    )[0]

    fc_node = Node(name='fc', action='functional_connectivity', inputs={'timeseries': ['parc']})
    fc = build_functional_connectivity(ctx, fc_node, {'timeseries': [parc]})
    assert len(fc) == 1
    assert fc[0].fmt == 'relmat'
    assert fc[0].entities['stat'] == 'pearsoncorrelation'
    mat = pd.read_csv(fc[0].files[0], sep='\t')
    assert mat.shape == (2, 2)  # region x region
    assert np.allclose(np.diag(mat.to_numpy()), 1.0)


def test_parcellate_cross_space_nifti_not_implemented(tmp_path, bold_and_atlas):
    bold, atlas = bold_and_atlas
    node = Node(
        name='parc',
        action='parcellate_timeseries',
        inputs={'timeseries': ['load_bold'], 'atlas': ['atlas']},
    )
    # atlas in a different space than the data -> needs an ANTs warp (not yet wired)
    resolved = {
        'timeseries': [_timeseries_result(bold)],
        'atlas': [_atlas_result(atlas, space='MNI152NLin6Asym')],
    }
    ctx = BuildContext(work_dir=str(tmp_path / 'work'))
    with pytest.raises(NotImplementedError, match='Strategy-A ANTs warp'):
        build_parcellate_timeseries(ctx, node, resolved)
