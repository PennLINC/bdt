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
"""End-to-end tests for the node-graph executor with the passthrough builder.

The passthrough builder does no real computation, so these tests exercise the
engine's structural behaviour: dependency ordering, fan-out over list-valued /
multi-match roles, grouping of ``surfaces``-style roles, output naming, dataset
vs participant scope, and collision detection.
"""

import json

import pytest

from bdt.engine import DictDataProvider, Executor, Match
from bdt.outputs import DerivativeSink, OutputCollisionError
from bdt.spec import parse_spec


def _mk(tmp_path, name, text='x'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _outputs(out_dir):
    return sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob('*') if p.is_file())


def test_story_3_1_end_to_end(tmp_path):
    out_dir = tmp_path / 'out'
    provider = DictDataProvider(
        {
            'xcpd': [
                Match(
                    _mk(tmp_path, 'bold.dtseries.nii'),
                    {
                        'sub': '01',
                        'task': 'rest',
                        'space': 'fsLR',
                        'suffix': 'bold',
                        'extension': '.dtseries.nii',
                    },
                )
            ],
            'atlases': [
                Match(
                    _mk(tmp_path, 'atlas.dlabel.nii'),
                    {'atlas': '4S456Parcels', 'extension': '.dlabel.nii'},
                )
            ],
        }
    )
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_bold',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'suffix': 'bold'},
                },
                {
                    'name': 'atlas_4s456',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': '4S456Parcels'},
                },
                {
                    'name': 'parcellate_bold',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'load_bold', 'atlas': 'atlas_4s456'},
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
    )
    Executor(spec, provider, DerivativeSink(out_dir), datasets={'xcpd', 'atlases'}).run()

    outs = _outputs(out_dir)
    assert (
        'sub-01/func/sub-01_task-rest_space-fsLR_atlas-4S456Parcels_stat-mean_timeseries.tsv'
        in outs
    )
    assert (
        'sub-01/func/'
        'sub-01_task-rest_space-fsLR_atlas-4S456Parcels_stat-pearsoncorrelation_relmat.tsv' in outs
    )
    # provenance sidecar carries GeneratedBy + the original bold Source
    stem = 'sub-01_task-rest_space-fsLR_atlas-4S456Parcels_stat-mean_timeseries'
    sidecar = json.loads((out_dir / 'sub-01' / 'func' / f'{stem}.json').read_text())
    assert sidecar['GeneratedBy'][0]['Action'] == 'parcellate_timeseries'
    assert any('bold' in s for s in sidecar['Sources'])


def test_atlas_list_fans_out(tmp_path):
    out_dir = tmp_path / 'out'
    provider = DictDataProvider(
        {
            'qsirecon': [Match(_mk(tmp_path, 'noddi.nii.gz'), {'sub': '01', 'suffix': 'dwimap'})],
            'atlases': [
                Match(_mk(tmp_path, 'a.dlabel.nii'), {'atlas': 'HCPMMP1'}),
                Match(_mk(tmp_path, 'b.dlabel.nii'), {'atlas': 'Schaefer400'}),
            ],
        }
    )
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 's',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {'suffix': 'dwimap'},
                },
                {
                    'name': 'a',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'HCPMMP1'},
                },
                {
                    'name': 'b',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'Schaefer400'},
                },
                {
                    'name': 'parc',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 's', 'atlas': ['a', 'b']},
                    'write_outputs': True,
                },
            ]
        }
    )
    Executor(spec, provider, DerivativeSink(out_dir)).run()
    outs = _outputs(out_dir)
    tsvs = [o for o in outs if o.endswith('.tsv')]
    assert len(tsvs) == 2
    assert any('atlas-HCPMMP1' in o for o in tsvs)
    assert any('atlas-Schaefer400' in o for o in tsvs)


def test_surfaces_group_does_not_multiply(tmp_path):
    """scalar fans over 2 params; the 6-file surfaces set is one group -> 2 outputs."""
    out_dir = tmp_path / 'out'
    surf_files = [
        Match(
            _mk(tmp_path, f'{h}_{s}.surf.gii'),
            {'sub': '01', 'hemi': h, 'suffix': s, 'extension': '.surf.gii'},
        )
        for h in ('L', 'R')
        for s in ('pial', 'white', 'midthickness')
    ]
    provider = DictDataProvider(
        {
            'qsirecon': [
                Match(
                    _mk(tmp_path, 'p1.nii.gz'), {'sub': '01', 'param': 'icvf', 'suffix': 'dwimap'}
                ),
                Match(
                    _mk(tmp_path, 'p2.nii.gz'), {'sub': '01', 'param': 'od', 'suffix': 'dwimap'}
                ),
            ],
            'smriprep': surf_files,
        }
    )
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'scalars',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {'suffix': 'dwimap'},
                },
                {
                    'name': 'surfaces',
                    'action': 'select_data',
                    'dataset': 'smriprep',
                    'filters': {'extension': '.surf.gii'},
                },
                {
                    'name': 'on_surf',
                    'action': 'map_scalar_to_surface',
                    'inputs': {'scalar': 'scalars', 'surfaces': 'surfaces'},
                    'write_outputs': True,
                },
            ]
        }
    )
    Executor(spec, provider, DerivativeSink(out_dir)).run()
    outs = [o for o in _outputs(out_dir) if o.endswith('.dscalar.nii')]
    assert len(outs) == 2
    assert any('param-icvf' in o for o in outs)
    assert any('param-od' in o for o in outs)


def test_desc_prepend_compose(tmp_path):
    """Dataset-scope gene-expression parcellation: source desc + node desc compose."""
    out_dir = tmp_path / 'out'
    provider = DictDataProvider(
        {
            'neuromaps': [
                Match(
                    _mk(tmp_path, 'ge.dscalar.nii'),
                    {'desc': 'geneexpression', 'space': 'fsLR', 'den': '32k', 'suffix': 'map'},
                )
            ],
            'atlases': [Match(_mk(tmp_path, 'a.dlabel.nii'), {'atlas': 'HCPMMP1'})],
        }
    )
    spec = parse_spec(
        {
            'dataset': [
                {
                    'name': 'ge',
                    'action': 'select_data',
                    'dataset': 'neuromaps',
                    'filters': {'suffix': 'map'},
                },
                {
                    'name': 'atlas',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'HCPMMP1'},
                },
                {
                    'name': 'ge_parc',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 'ge', 'atlas': 'atlas'},
                    'desc': 'strict',
                    'write_outputs': True,
                },
            ]
        }
    )
    Executor(spec, provider, DerivativeSink(out_dir)).run()
    tsvs = [o for o in _outputs(out_dir) if o.endswith('.tsv')]
    assert len(tsvs) == 1
    assert tsvs[0].startswith('tpl-fsLR/')
    assert 'desc-geneexpressionStrict' in tsvs[0]


def test_output_collision_raises(tmp_path):
    out_dir = tmp_path / 'out'
    provider = DictDataProvider(
        {
            'qsirecon': [Match(_mk(tmp_path, 'noddi.nii.gz'), {'sub': '01', 'suffix': 'dwimap'})],
            'atlases': [Match(_mk(tmp_path, 'a.dlabel.nii'), {'atlas': 'HCPMMP1'})],
        }
    )
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 's',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {'suffix': 'dwimap'},
                },
                {
                    'name': 'a',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'HCPMMP1'},
                },
                {
                    'name': 'parc1',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 's', 'atlas': 'a'},
                    'write_outputs': True,
                },
                {
                    'name': 'parc2',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 's', 'atlas': 'a'},
                    'write_outputs': True,
                },
            ]
        }
    )
    with pytest.raises(OutputCollisionError):
        Executor(spec, provider, DerivativeSink(out_dir)).run()
