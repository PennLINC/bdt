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
"""Tests for the pybids-backed data provider (requires pybids)."""

import json

import pytest

pytest.importorskip('bids')

from bdt.engine.pybids_provider import BIDSDataProvider  # noqa: E402


def _make_dataset(root, files, derivative=True):
    root.mkdir(parents=True, exist_ok=True)
    desc = {'Name': 'fixture', 'BIDSVersion': '1.9.0'}
    if derivative:
        desc.update({'DatasetType': 'derivative', 'GeneratedBy': [{'Name': 'bdt'}]})
    (root / 'dataset_description.json').write_text(json.dumps(desc))
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'')


@pytest.fixture
def datasets(tmp_path):
    _make_dataset(
        tmp_path / 'xcpd',
        [
            'sub-01/func/sub-01_task-rest_space-fsLR_den-32k_desc-denoised_bold.dtseries.nii',
            'sub-02/func/sub-02_task-rest_space-fsLR_den-32k_desc-denoised_bold.dtseries.nii',
        ],
    )
    _make_dataset(
        tmp_path / 'qsirecon',
        [
            'sub-01/dwi/sub-01_model-tensor_param-fa_space-ACPC_dwimap.nii.gz',
            'sub-01/dwi/sub-01_model-noddi_param-icvf_space-ACPC_dwimap.nii.gz',
            'sub-01/dwi/sub-01_model-noddi_param-od_space-ACPC_dwimap.nii.gz',
        ],
    )
    _make_dataset(
        tmp_path / 'atlases',
        ['tpl-fsLR/atlas-HCPMMP1/tpl-fsLR_atlas-HCPMMP1_den-32k_dseg.dlabel.nii'],
    )
    return {
        'xcpd': str(tmp_path / 'xcpd'),
        'qsirecon': str(tmp_path / 'qsirecon'),
        'atlases': str(tmp_path / 'atlases'),
    }


def test_select_by_suffix_and_entities(datasets):
    p = BIDSDataProvider(datasets)
    matches = p.select('xcpd', {'suffix': 'bold', 'space': 'fsLR', 'desc': 'denoised'})
    assert len(matches) == 2  # both subjects
    e = matches[0].entities
    assert e['sub'] in {'01', '02'}
    assert e['space'] == 'fsLR'
    assert e['den'] == '32k'
    assert e['desc'] == 'denoised'
    assert e['extension'] == '.dtseries.nii'


def test_subject_scoping(datasets):
    p = BIDSDataProvider(datasets)
    matches = p.select('xcpd', {'suffix': 'bold'}, [], subject='01')
    assert [m.entities['sub'] for m in matches] == ['01']


def test_custom_entities_atlas_and_param(datasets):
    p = BIDSDataProvider(datasets)
    # 'atlas' is a BDT-registered entity; without bdt_entities.json it would not parse.
    atl = p.select('atlases', {'atlas': 'HCPMMP1'})
    assert len(atl) == 1
    assert atl[0].entities['atlas'] == 'HCPMMP1'
    # 'param' is likewise BDT-registered
    fa = p.select('qsirecon', {'suffix': 'dwimap', 'param': 'fa'})
    assert len(fa) == 1
    assert fa[0].entities['param'] == 'fa'
    assert fa[0].entities['model'] == 'tensor'


def test_exclude(datasets):
    p = BIDSDataProvider(datasets)
    # all dwimap for sub-01 except the NODDI ones (story 3.3 pattern)
    matches = p.select('qsirecon', {'suffix': 'dwimap'}, [{'model': 'noddi'}], subject='01')
    models = sorted(m.entities['model'] for m in matches)
    assert models == ['tensor']


def test_relpath_and_subjects(datasets):
    p = BIDSDataProvider(datasets)
    assert p.subjects('xcpd') == ['01', '02']
    m = p.select('xcpd', {'suffix': 'bold'}, [], subject='01')[0]
    assert p.relpath('xcpd', m.path).startswith('sub-01/func/')
