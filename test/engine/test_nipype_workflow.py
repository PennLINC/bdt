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
    assert 'parcellate_bold.restrict_atlas' in names  # atlas -> data brainordinates
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
            entities={
                'sub': '01',
                'space': 'fsLR',
                'den': '91k',
                'suffix': 'bold',
                'extension': '.dtseries.nii',
            },
        ),
        'atlas_4s456': Match(path='/data/atlas.dlabel.nii', entities={'atlas': '4S456Parcels'}),
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


SURFACE_SPEC = {
    'nodes': [
        {
            'name': 'surfaces',
            'action': 'select_data',
            'dataset': 'anat',
            'filters': {
                'hemi': ['L', 'R'],
                'suffix': ['pial', 'white', 'midthickness'],
                'extension': '.surf.gii',
            },
        },
        {
            'name': 'load_thickness',
            'action': 'select_data',
            'dataset': 'anat',
            'filters': {'suffix': 'thickness', 'extension': '.shape.gii'},
        },
        {
            'name': 'thickness_fslr',
            'action': 'resample_surface_scalar',
            'inputs': {'surface_scalar': 'load_thickness', 'surfaces': 'surfaces'},
            'parameters': {'target_density': '32k'},
            'write_outputs': True,
        },
    ]
}


def test_compile_resample_surface_scalar(tmp_path):
    """Grouped surface roles + aux context compile to a per-hemi resample factory.

    Tool-free: a stub provider + fake templateflow getter resolve the aux inputs to
    touched dummy files (nipype validates File(exists) at set time), so the graph
    builds without running wb_command.
    """
    from bdt.engine.factories import FactoryContext
    from bdt.engine.selection import DictDataProvider, Match

    def touch(rel):
        p = tmp_path / rel
        p.touch()
        return str(p)

    spec = parse_spec(SURFACE_SPEC)
    provider = DictDataProvider(
        {
            'anat': [
                Match(
                    touch(f'sub-01_hemi-{h}_space-fsLR_desc-msmsulc_sphere.surf.gii'),
                    {'suffix': 'sphere', 'space': 'fsLR', 'desc': 'msmsulc', 'hemi': h},
                )
                for h in 'LR'
            ]
            + [
                Match(
                    touch(f'sub-01_hemi-{h}_space-fsLR_den-32k_midthickness.surf.gii'),
                    {'suffix': 'midthickness', 'space': 'fsLR', 'den': '32k', 'hemi': h},
                )
                for h in 'LR'
            ],
        }
    )

    def fake_tf(**kw):
        tag = 'sphere.surf.gii' if kw.get('suffix') == 'sphere' else 'nomedialwall.label.gii'
        return touch(f'tpl-fsLR_hemi-{kw["hemi"]}_{tag}')

    context = FactoryContext(provider=provider, subject='01', spec=spec, templateflow_get=fake_tf)
    # grouped selections arrive as lists of paths on the source nodes
    selections = {
        'surfaces': [
            f'/a/sub-01_hemi-{h}_{s}.surf.gii'
            for h in 'LR'
            for s in ('pial', 'white', 'midthickness')
        ],
        'load_thickness': [f'/a/sub-01_hemi-{h}_thickness.shape.gii' for h in 'LR'],
    }
    wf = init_bdt_wf(spec, selections, context=context)

    names = set(wf.list_node_names())
    for want in (
        'thickness_fslr.inputnode',
        'thickness_fslr.create_dense',
        'thickness_fslr.resample_L',
        'thickness_fslr.resample_R',
        'thickness_fslr.srcroi_L',
        'thickness_fslr.pick_midthick_R',
    ):
        assert want in names, f'missing {want}'

    # aux inputs were resolved for the subject and fixed on the resample nodes
    rl = wf.get_node('thickness_fslr').get_node('resample_L')
    assert rl.inputs.current_sphere.endswith('desc-msmsulc_sphere.surf.gii')
    assert rl.inputs.new_area.endswith('den-32k_midthickness.surf.gii')
    assert rl.inputs.method == 'ADAP_BARY_AREA'
    # grouped source node holds the L/R list unchanged
    assert wf.get_node('load_thickness').inputs.out == selections['load_thickness']


def _map_spec():
    return parse_spec(
        {
            'nodes': [
                {
                    'name': 'surfaces',
                    'action': 'select_data',
                    'dataset': 'anat',
                    'filters': {
                        'hemi': ['L', 'R'],
                        'suffix': ['pial', 'white', 'midthickness'],
                        'extension': '.surf.gii',
                    },
                },
                {
                    'name': 'load_noddi',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {
                        'suffix': 'dwimap',
                        'model': 'noddi',
                        'param': 'icvf',
                        'space': 'ACPC',
                    },
                },
                {
                    'name': 'noddi_on_surface',
                    'action': 'map_scalar_to_surface',
                    'inputs': {'scalar': 'load_noddi', 'surfaces': 'surfaces'},
                },
            ]
        }
    )


def _map_context(spec, scalar_space, tmp_path, moving_suffix='T1w'):
    """A stub FactoryContext with touched reference files for map_scalar_to_surface.

    ``moving_suffix`` is the modality of the QSIPrep ACPC anatomical (``T1w`` or, for
    a ``--anat-modality T2w`` run, ``T2w``).
    """
    from bdt.engine.factories import FactoryContext
    from bdt.engine.selection import DictDataProvider, Match

    def touch(rel):
        (tmp_path / rel).touch()
        return str(tmp_path / rel)

    provider = DictDataProvider(
        {
            'anat': [
                Match(
                    touch('sub-01_desc-preproc_T1w.nii.gz'),
                    {'suffix': 'T1w', 'desc': 'preproc', 'datatype': 'anat'},
                ),
                Match(
                    touch('sub-01_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'datatype': 'anat'},
                ),
            ],
            'qsiprep': [
                Match(
                    touch(f'sub-01_space-ACPC_desc-preproc_{moving_suffix}.nii.gz'),
                    {
                        'suffix': moving_suffix,
                        'desc': 'preproc',
                        'space': 'ACPC',
                        'datatype': 'anat',
                    },
                ),
                Match(
                    touch('sub-01_space-ACPC_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'},
                ),
            ],
        }
    )
    resolved = {
        'load_noddi': Match('/q/scalar.nii.gz', {'space': scalar_space, 'suffix': 'dwimap'}),
        'surfaces': Match(
            '/a/hemi-L_midthickness.surf.gii', {'suffix': 'midthickness', 'hemi': 'L'}
        ),
    }
    return FactoryContext(
        provider=provider,
        subject='01',
        spec=spec,
        datasets=['anat', 'qsiprep', 'qsirecon'],
        resolved=resolved,
    )


def test_map_scalar_to_surface_cross_space(tmp_path):
    """A scalar in ACPC vs T1w surfaces compiles the computed-rigid + giftirs branch."""
    from bdt.engine.factories import init_map_scalar_to_surface_wf

    spec = _map_spec()
    node = spec.by_name()['noddi_on_surface']
    wf = init_map_scalar_to_surface_wf(node, context=_map_context(spec, 'ACPC', tmp_path))
    names = set(wf.list_node_names())
    for want in (
        'register_acpc',
        'warp_white_L',
        'warp_midthickness_R',
        'vol2surf_L',
        'dilate_R',
        'merge_hemis',
    ):
        assert want in names, f'missing {want}'
    # the single ANTsPy registration node is wired to the resolved T1w (fixed) + ACPC
    # (moving) anatomicals + their brain masks
    reg = wf.get_node('register_acpc')
    assert reg.inputs.fixed_image.endswith('desc-preproc_T1w.nii.gz')
    assert reg.inputs.moving_image.endswith('space-ACPC_desc-preproc_T1w.nii.gz')
    assert reg.inputs.fixed_mask.endswith('desc-brain_mask.nii.gz')


def test_map_scalar_to_surface_accepts_t2w_acpc_anatomical(tmp_path):
    """A --anat-modality T2w QSIPrep run: the ACPC moving image is a T2w, resolved
    the same (rigid MI is contrast-agnostic; fixed stays the surfaces' T1w)."""
    from bdt.engine.factories import init_map_scalar_to_surface_wf

    spec = _map_spec()
    node = spec.by_name()['noddi_on_surface']
    ctx = _map_context(spec, 'ACPC', tmp_path, moving_suffix='T2w')
    wf = init_map_scalar_to_surface_wf(node, context=ctx)
    reg = wf.get_node('register_acpc')
    assert reg.inputs.moving_image.endswith('space-ACPC_desc-preproc_T2w.nii.gz')
    # fixed remains the T1w that matches the surfaces' space
    assert reg.inputs.fixed_image.endswith('desc-preproc_T1w.nii.gz')


def test_map_scalar_to_surface_same_space_no_registration(tmp_path):
    """A T1w scalar onto T1w surfaces maps directly — no registration / warp nodes."""
    from bdt.engine.factories import init_map_scalar_to_surface_wf

    spec = _map_spec()
    node = spec.by_name()['noddi_on_surface']
    wf = init_map_scalar_to_surface_wf(node, context=_map_context(spec, 'T1w', tmp_path))
    names = set(wf.list_node_names())
    assert 'vol2surf_L' in names
    assert 'dilate_R' in names
    assert not any('register' in n or 'warp' in n or 'extract' in n for n in names)


def test_map_scalar_to_surface_unsupported_cross_space_raises(tmp_path):
    """Only T1w<->ACPC is computed; any other cross-space pairing errors clearly."""
    from bdt.engine.factories import init_map_scalar_to_surface_wf

    spec = _map_spec()
    node = spec.by_name()['noddi_on_surface']
    with pytest.raises(NotImplementedError, match='T1w'):
        init_map_scalar_to_surface_wf(
            node, context=_map_context(spec, 'MNI152NLin2009cAsym', tmp_path)
        )


def test_reference_detection_subject_vs_session_level():
    """Anatomical reference resolution honours the subject- vs session-level anat level.

    A BIDS anatomical may be session-level (`sub-X/ses-Y/anat`) or subject-level
    (`sub-X/anat`, shared across sessions); the detection must pick the current
    session's file, fall back to a session-less one, and never grab a *different*
    session's file.
    """
    from bdt.engine.factories import FactoryContext
    from bdt.engine.selection import DictDataProvider, Match

    t1 = {'suffix': 'T1w', 'desc': 'preproc'}

    def ctx(matches):
        return FactoryContext(
            provider=DictDataProvider({'anat': matches}), subject='01', datasets=['anat']
        )

    # multi-session, session-level -> pick the current session, not the other
    multi = ctx(
        [
            Match(
                '/a/sub-01_ses-1_desc-preproc_T1w.nii.gz',
                {'sub': '01', 'ses': '1', 'suffix': 'T1w', 'desc': 'preproc'},
            ),
            Match(
                '/a/sub-01_ses-2_desc-preproc_T1w.nii.gz',
                {'sub': '01', 'ses': '2', 'suffix': 'T1w', 'desc': 'preproc'},
            ),
        ]
    )
    assert multi.aux_file('anat', t1, session='1').endswith('ses-1_desc-preproc_T1w.nii.gz')
    assert multi.aux_file('anat', t1, session='2').endswith('ses-2_desc-preproc_T1w.nii.gz')

    # subject-level anatomical (no ses) -> session-less fallback for session-scoped data
    subj = ctx(
        [
            Match(
                '/a/sub-01_desc-preproc_T1w.nii.gz',
                {'sub': '01', 'suffix': 'T1w', 'desc': 'preproc'},
            )
        ]
    )
    assert subj.aux_file('anat', t1, session='1').endswith('sub-01_desc-preproc_T1w.nii.gz')

    # never grab a different session's file: ses-1 data, only ses-2 anat -> error
    other = ctx(
        [
            Match(
                '/a/sub-01_ses-2_desc-preproc_T1w.nii.gz',
                {'sub': '01', 'ses': '2', 'suffix': 'T1w', 'desc': 'preproc'},
            )
        ]
    )
    with pytest.raises(ValueError, match='expected exactly 1'):
        other.aux_file('anat', t1, session='1')


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
                    'name': 'load_ref',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {
                        'suffix': 'dwimap',
                        'model': 'tensor',
                        'param': 'fa',
                        'space': 'ACPC',
                    },
                },
                {
                    'name': 'bundle_rois',
                    'action': 'tractogram_to_pseg',
                    'inputs': {'tractograms': 'load_bundles', 'reference': 'load_ref'},
                    'parameters': params,
                    'write_outputs': True,
                },
            ]
        }
    )


def test_tractogram_to_pseg_probseg_no_threshold():
    from bdt.engine.factories import init_tractogram_to_pseg_wf

    spec = _pseg_spec_wf(threshold=None)
    node = spec.by_name()['bundle_rois']
    wf = init_tractogram_to_pseg_wf(node)

    names = set(wf.list_node_names())
    assert 'gunzip' in names
    assert 'tck_to_tdi' in names
    assert 'concatenate' in names
    assert 'bundles_to_tsv' in names
    assert 'binarize' not in names  # no threshold -> no binarize node
    # the reference grid is a wired input, not a fixed value
    assert 'reference' in wf.get_node('inputnode').inputs.copyable_trait_names()
    # outputnode contract
    out = wf.get_node('outputnode')
    assert set(out.inputs.copyable_trait_names()) >= {'out', 'tsv'}


def test_tractogram_to_pseg_dseg_with_threshold():
    from bdt.engine.factories import init_tractogram_to_pseg_wf

    spec = _pseg_spec_wf(threshold=0.0)
    node = spec.by_name()['bundle_rois']
    wf = init_tractogram_to_pseg_wf(node)

    assert 'binarize' in set(wf.list_node_names())
    assert wf.get_node('binarize').inputs.threshold == 0.0


def test_tractogram_to_pseg_grouped_list_compiles():
    """A multi-match bundle selection compiles into the factory as a grouped list."""
    spec = _pseg_spec_wf(threshold=0.0)

    # grouped selection -> the source node carries the full list of bundle paths
    bundle_paths = [
        f'/a/sub-01_bundle-{b}_space-ACPC_streamlines.tck.gz' for b in ('CST', 'AF', 'IFOF')
    ]
    selections = {
        'load_bundles': bundle_paths,
        'load_ref': '/a/sub-01_model-tensor_param-fa_space-ACPC_dwimap.nii.gz',
    }

    wf = init_bdt_wf(spec, selections)

    # the grouped source node holds the list unchanged, and it feeds the factory inputnode
    assert wf.get_node('load_bundles').inputs.out == bundle_paths
    assert 'bundle_rois.gunzip' in set(wf.list_node_names())
    assert 'bundle_rois.tck_to_tdi' in set(wf.list_node_names())
    # the reference grid is wired from the reference selection node
    assert wf.get_node('load_ref').inputs.out == selections['load_ref']


def _profile_spec():
    return parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_bundles',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {'suffix': 'streamlines', 'extension': '.tck.gz', 'space': 'ACPC'},
                },
                {
                    'name': 'load_fa',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {'suffix': 'dwimap', 'param': 'fa', 'space': 'ACPC'},
                },
                {
                    'name': 'fa_profile',
                    'action': 'parcellate_scalar_as_tract_profile',
                    'inputs': {'scalar': 'load_fa', 'bundles': 'load_bundles'},
                    'parameters': {'n_nodes': 50},
                    'write_outputs': True,
                },
            ]
        }
    )


def test_tract_profile_factory_wires_scalar_and_bundles():
    from bdt.engine.factories import init_parcellate_scalar_as_tract_profile_wf

    spec = _profile_spec()
    node = spec.by_name()['fa_profile']
    wf = init_parcellate_scalar_as_tract_profile_wf(node)

    names = set(wf.list_node_names())
    assert 'gunzip' in names
    assert 'profile' in names
    assert wf.get_node('profile').inputs.n_nodes == 50
    inp = wf.get_node('inputnode').inputs.copyable_trait_names()
    assert 'scalar' in inp and 'bundles' in inp


def test_tract_profile_grouped_bundles_compiles():
    spec = _profile_spec()
    bundle_paths = [
        f'/a/sub-01_bundle-{b}_space-ACPC_streamlines.tck.gz' for b in ('CST', 'AF')
    ]
    selections = {
        'load_bundles': bundle_paths,
        'load_fa': '/a/sub-01_param-fa_space-ACPC_dwimap.nii.gz',
    }
    wf = init_bdt_wf(spec, selections)

    assert wf.get_node('load_bundles').inputs.out == bundle_paths
    assert 'fa_profile.gunzip' in set(wf.list_node_names())
    assert 'fa_profile.profile' in set(wf.list_node_names())
