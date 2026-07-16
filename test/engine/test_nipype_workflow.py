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
"""Assembly-level tests for the nipype workflow compiler (no tools executed)."""

import pytest

pytest.importorskip('nipype')
pytest.importorskip('niworkflows')

from bdt.engine.workflow import init_bdt_wf  # noqa: E402
from bdt.spec import parse_spec  # noqa: E402

STORY_3_1 = {
    'nodes': [
        {
            'name': 'load_bold',
            'action': 'select_data',
            'dataset': 'xcpd',
            'filters': {'suffix': 'bold', 'extension': '.dtseries.nii'},
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


def test_compile_story_3_1_to_nipype_workflow():
    spec = parse_spec(STORY_3_1)
    selections = {
        'load_bold': '/data/sub-01_bold.dtseries.nii',
        'atlas_4s456': '/data/atlas.dlabel.nii',
    }
    wf = init_bdt_wf(spec, selections)

    names = set(wf.list_node_names())
    # selection source nodes
    assert 'load_bold' in names
    assert 'atlas_4s456' in names
    # processing sub-workflow internals (proves the factories ran + connected)
    assert 'parcellate_bold.parcellate_data' in names
    assert 'parcellate_bold.vertex_mask' in names  # coverage-aware pipeline
    assert 'parcellate_bold.inputnode' in names
    assert 'fc_bold.correlate' in names

    # the selection file paths are set on the source nodes
    assert wf.get_node('load_bold').inputs.out == '/data/sub-01_bold.dtseries.nii'


def test_role_wiring_edges():
    """The graph connects load_bold -> parcellate_bold.inputnode.timeseries etc."""
    spec = parse_spec(STORY_3_1)
    wf = init_bdt_wf(spec, {'load_bold': '/x/b.dtseries.nii', 'atlas_4s456': '/x/a.dlabel.nii'})

    # nipype's top-level graph keys edges by the (sub-)workflow node, with the
    # inner field qualified as ``inputnode.<role>`` in the connect list.
    edges = {
        (u.name, v.name): data.get('connect', []) for u, v, data in wf._graph.edges(data=True)
    }

    # load_bold.out -> parcellate_bold.inputnode.timeseries
    assert ('out', 'inputnode.timeseries') in edges[('load_bold', 'parcellate_bold')]
    # atlas_4s456.out -> parcellate_bold.inputnode.atlas
    assert ('out', 'inputnode.atlas') in edges[('atlas_4s456', 'parcellate_bold')]
    # parcellate_bold.outputnode.out -> fc_bold.inputnode.timeseries
    assert ('outputnode.out', 'inputnode.timeseries') in edges[('parcellate_bold', 'fc_bold')]


def test_sink_nodes_attached_from_plan(tmp_path):
    """A sink_plan attaches native + tsv sink nodes for write_outputs nodes."""
    from bdt.engine.selection import Match
    from bdt.outputs.plan import build_sink_plan

    spec = parse_spec(STORY_3_1)
    resolved = {
        'load_bold': Match(
            path='/data/sub-01_bold.dtseries.nii',
            entities={'sub': '01', 'space': 'fsLR', 'den': '91k', 'suffix': 'bold',
                      'extension': '.dtseries.nii'},
        ),
        'atlas_4s456': Match(
            path='/data/atlas.dlabel.nii', entities={'atlas': '4S456Parcels'}
        ),
    }
    plan = build_sink_plan(spec, resolved)
    selections = {
        'load_bold': resolved['load_bold'].path,
        'atlas_4s456': resolved['atlas_4s456'].path,
    }
    wf = init_bdt_wf(spec, selections, base_directory=str(tmp_path), sink_plan=plan)

    names = set(wf.list_node_names())
    # native CIFTI sink + tsv-convert + tsv sink for each written node
    assert 'parcellate_bold_sink0' in names
    assert 'parcellate_bold_totsv1' in names
    assert 'parcellate_bold_sink1' in names
    assert 'fc_bold_sink0' in names
    assert 'fc_bold_totsv1' in names


def test_no_sinks_without_plan():
    """Assembly-only build (no plan) attaches no sink nodes -> compute graph only."""
    spec = parse_spec(STORY_3_1)
    wf = init_bdt_wf(spec, {'load_bold': '/x/b.dtseries.nii', 'atlas_4s456': '/x/a.dlabel.nii'})
    assert not any('sink' in n for n in wf.list_node_names())


def test_missing_factory_raises():
    # atlas_union has no nipype factory yet -> clear error, not silent passthrough
    spec = parse_spec(
        {
            'dataset': [
                {
                    'name': 'a',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'A'},
                },
                {
                    'name': 'b',
                    'action': 'select_atlases',
                    'dataset': 'atlases',
                    'filters': {'atlas': 'B'},
                },
                {'name': 'u', 'action': 'atlas_union', 'inputs': {'a': 'a', 'b': 'b'}},
            ]
        }
    )
    with pytest.raises(NotImplementedError, match='atlas_union'):
        init_bdt_wf(spec, {'a': '/x/a.dlabel.nii', 'b': '/x/b.dlabel.nii'})
