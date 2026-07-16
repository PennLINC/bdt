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
"""Full-stack integration test: run_spec over a real pybids dataset (no mocks)."""

import json

import pytest

pytest.importorskip('bids')

from bdt.engine import run_spec  # noqa: E402


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


def test_run_spec_story_3_1_over_pybids(tmp_path):
    """select bold + atlas -> parcellate -> FC -> write, driven end-to-end by pybids."""
    _make_dataset(
        tmp_path / 'xcpd',
        ['sub-01/func/sub-01_task-rest_space-fsLR_desc-denoised_bold.dtseries.nii'],
    )
    _make_dataset(
        tmp_path / 'atlases',
        ['tpl-fsLR/atlas-4S456Parcels/tpl-fsLR_atlas-4S456Parcels_den-32k_dseg.dlabel.nii'],
    )
    datasets = {'xcpd': str(tmp_path / 'xcpd'), 'atlases': str(tmp_path / 'atlases')}
    spec = {
        'nodes': [
            {
                'name': 'load_bold',
                'action': 'select_data',
                'dataset': 'xcpd',
                'filters': {'suffix': 'bold', 'space': 'fsLR', 'desc': 'denoised'},
            },
            {
                'name': 'atlas',
                'action': 'select_atlases',
                'dataset': 'atlases',
                'filters': {'atlas': '4S456Parcels'},
            },
            {
                'name': 'parcellate_bold',
                'action': 'parcellate_timeseries',
                'inputs': {'timeseries': 'load_bold', 'atlas': 'atlas'},
                'parameters': {'min_coverage': 0.5},
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
    from bdt.spec import parse_spec

    out_dir = tmp_path / 'out'
    run_spec(
        parse_spec(spec),
        datasets,
        out_dir,
        participant_labels=['01'],
        use_builtin_actions=False,  # structural test: passthrough, no wb_command
    )

    outs = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob('*') if p.is_file())
    # the source desc- propagates per the section 1.7 compose rule (desc sorts last)
    assert any(
        'atlas-4S456Parcels' in o and 'stat-mean' in o and o.endswith('timeseries.tsv')
        for o in outs
    )
    assert any(
        'atlas-4S456Parcels' in o and 'stat-pearsoncorrelation' in o and o.endswith('relmat.tsv')
        for o in outs
    )
    assert all(o.startswith('sub-01/func/') for o in outs)


def test_run_spec_discovers_subjects(tmp_path):
    _make_dataset(
        tmp_path / 'xcpd',
        [
            'sub-01/func/sub-01_task-rest_desc-denoised_bold.dtseries.nii',
            'sub-02/func/sub-02_task-rest_desc-denoised_bold.dtseries.nii',
        ],
    )
    _make_dataset(
        tmp_path / 'atlases',
        ['tpl-fsLR/atlas-A/tpl-fsLR_atlas-A_dseg.dlabel.nii'],
    )
    datasets = {'xcpd': str(tmp_path / 'xcpd'), 'atlases': str(tmp_path / 'atlases')}
    from bdt.spec import parse_spec

    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'b',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'suffix': 'bold'},
                },
                {
                    'name': 'a',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'A'},
                },
                {
                    'name': 'parc',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'b', 'atlas': 'a'},
                    'write_outputs': True,
                },
            ]
        }
    )
    out_dir = tmp_path / 'out'
    # subjects auto-discovered from the datasets; passthrough (no external tools)
    results = run_spec(spec, datasets, out_dir, use_builtin_actions=False)
    assert set(results) == {'01', '02'}
    outs = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob('*.tsv'))
    assert any(o.startswith('sub-01/') for o in outs)
    assert any(o.startswith('sub-02/') for o in outs)
