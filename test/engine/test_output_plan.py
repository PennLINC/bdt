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
"""Unit tests for the output plan (entity composition + product list). No nipype."""

from bdt.engine.selection import Match
from bdt.outputs.plan import CIFTI_TO_TSV, PASSTHROUGH, build_sink_plan, node_output_entities
from bdt.spec import load_spec, parse_spec


def _story_3_1(bold_ext='.dtseries.nii', bold_path='/x/sub-01_bold.dtseries.nii'):
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_bold',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'suffix': 'bold', 'extension': bold_ext},
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
    )
    resolved = {
        'load_bold': Match(
            path=bold_path,
            entities={
                'subject': '01',
                'session': '1',
                'task': 'rest',
                'space': 'fsLR',
                'den': '91k',
                'desc': 'denoised',
                'suffix': 'bold',
                'extension': bold_ext,
            },
        ),
        'atlas_4s': Match(
            path='/x/atlas.dlabel.nii',
            entities={'tpl': 'fsLR', 'atlas': '4S1056Parcels', 'den': '91k'},
        ),
    }
    return spec, resolved


def test_node_output_entities_injects_atlas_and_stat():
    spec, resolved = _story_3_1()
    ent = node_output_entities(spec, resolved)

    parc = ent['parcellate_bold']
    # atlas injected from the wired atlas role; action's fixed stat added
    assert parc['atlas'] == '4S1056Parcels'
    assert parc['statistic'] == 'mean'
    # source geometry entities carried through
    assert parc['space'] == 'fsLR'
    assert parc['den'] == '91k'
    # a new-derivative action fixes its own suffix/datatype; extension is dropped
    assert parc['suffix'] == 'timeseries'
    assert parc['datatype'] == 'func'
    assert 'extension' not in parc

    # FC seeds from the parcellated series, so it inherits the atlas, and overrides stat
    fc = ent['fc_bold']
    assert fc['atlas'] == '4S1056Parcels'
    assert fc['statistic'] == 'pearsoncorrelation'


def test_cifti_input_plans_native_and_tsv():
    spec, resolved = _story_3_1(bold_ext='.dtseries.nii')
    plan = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/x'})

    parc = plan['parcellate_bold']
    assert [(p.derive, p.suffix, p.extension) for p in parc] == [
        (PASSTHROUGH, 'timeseries', '.ptseries.nii'),
        (CIFTI_TO_TSV, 'timeseries', '.tsv'),
        (PASSTHROUGH, 'boldmap', '.pscalar.nii'),  # coverage map
    ]
    # the coverage product reads outputnode.coverage and re-tags stat-coverage
    coverage = parc[2]
    assert coverage.source_field == 'coverage'
    assert coverage.entities['statistic'] == 'coverage'
    assert parc[0].entities['statistic'] == 'mean'

    fc = plan['fc_bold']
    assert [(p.derive, p.suffix, p.extension) for p in fc] == [
        (PASSTHROUGH, 'boldmap', '.pconn.nii'),
        (CIFTI_TO_TSV, 'relmat', '.tsv'),
    ]
    # provenance Sources point at the selection leaves
    assert any('bids:xcpd:' in s for s in parc[0].sidecar['Sources'])
    assert any('bids:atlases:' in s for s in parc[0].sidecar['Sources'])


def test_nifti_input_plans_tsv_only():
    spec, resolved = _story_3_1(bold_ext='.nii.gz', bold_path='/x/sub-01_bold.nii.gz')
    plan = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/x'})

    # A volumetric input yields the TSV product (no native CIFTI) plus a
    # volumetric-format coverage TSV (Task 4: coverage is now format-aware).
    assert [(p.derive, p.suffix, p.extension) for p in plan['parcellate_bold']] == [
        (PASSTHROUGH, 'timeseries', '.tsv'),
        (PASSTHROUGH, 'boldmap', '.tsv'),
    ]
    assert [(p.derive, p.suffix, p.extension) for p in plan['fc_bold']] == [
        (PASSTHROUGH, 'relmat', '.tsv'),
    ]


def test_parcellate_scalar_preserves_source_naming():
    """parcellate_scalar keeps the source suffix/datatype/stat, only adds atlas-."""
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
    plan = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/a'})
    prods = plan['alff_parc']
    # native pscalar + tsv, both keeping the source suffix (boldmap) and stat-alff,
    # plus the parcel-coverage map (Task 5: parcellate_scalar now plans coverage too).
    assert [(p.derive, p.suffix, p.extension) for p in prods] == [
        (PASSTHROUGH, 'boldmap', '.pscalar.nii'),
        (CIFTI_TO_TSV, 'boldmap', '.tsv'),
        (PASSTHROUGH, 'map', '.pscalar.nii'),  # coverage map
    ]
    assert prods[0].entities['statistic'] == 'alff'  # not overwritten with 'mean'
    assert prods[0].entities['atlas'] == '4S1056Parcels'
    assert 'suffix' not in prods[0].entities  # pulled out to the explicit field
    assert prods[2].source_field == 'coverage'
    assert prods[2].entities['statistic'] == 'coverage'


def test_only_write_outputs_nodes_are_planned():
    spec, resolved = _story_3_1()
    # mark parcellate_bold as an intermediate (not written)
    spec.by_name()['parcellate_bold'].write_outputs = False
    plan = build_sink_plan(spec, resolved)
    assert 'parcellate_bold' not in plan
    assert 'fc_bold' in plan


def _pseg_spec(threshold):
    params = {} if threshold is None else {'threshold': threshold}
    spec = parse_spec(
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
                    'filters': {'suffix': 'dwimap', 'param': 'fa', 'space': 'ACPC'},
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
    resolved = {
        'load_bundles': Match(
            path='/x/sub-01_bundle-CST_space-ACPC_streamlines.tck.gz',
            entities={'subject': '01', 'space': 'ACPC', 'suffix': 'streamlines'},
        ),
        'load_ref': Match(
            path='/x/sub-01_param-fa_space-ACPC_dwimap.nii.gz',
            entities={'subject': '01', 'space': 'ACPC', 'param': 'fa', 'suffix': 'dwimap'},
        ),
    }
    return spec, resolved


def test_pseg_probseg_suffix_when_unthresholded():
    spec, resolved = _pseg_spec(threshold=None)
    plan = build_sink_plan(spec, resolved, {})
    products = plan['bundle_rois']
    primary = next(p for p in products if p.extension == '.nii.gz')
    assert primary.suffix == 'probseg'


def test_pseg_dseg_suffix_when_thresholded():
    spec, resolved = _pseg_spec(threshold=0.0)
    plan = build_sink_plan(spec, resolved, {})
    products = plan['bundle_rois']
    primary = next(p for p in products if p.extension == '.nii.gz')
    assert primary.suffix == 'dseg'


def test_pseg_emits_label_tsv_matching_primary_suffix():
    spec, resolved = _pseg_spec(threshold=0.0)
    plan = build_sink_plan(spec, resolved, {})
    tsvs = [p for p in plan['bundle_rois'] if p.extension == '.tsv']
    assert len(tsvs) == 1
    assert tsvs[0].source_field == 'tsv'
    assert tsvs[0].suffix == 'dseg'


def _story_3_1_volumetric():
    # Volumetric analog of _story_3_1: NIfTI bold + NIfTI atlas, same space, no
    # CIFTI native form -- modeled on test_volumetric_same_space_has_no_warp_node.
    spec = parse_spec(
        {
            'nodes': [
                {
                    'name': 'load_bold',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'suffix': 'bold', 'extension': '.nii.gz'},
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
            ]
        }
    )
    resolved = {
        'load_bold': Match(
            path='/x/sub-01_bold.nii.gz',
            entities={
                'subject': '01',
                'session': '1',
                'task': 'rest',
                'space': 'MNI152NLin6Asym',
                'desc': 'denoised',
                'suffix': 'bold',
                'extension': '.nii.gz',
            },
        ),
        'atlas_4s': Match(
            path='/x/atlas.nii.gz',
            entities={
                'tpl': 'MNI152NLin6Asym',
                'atlas': '4S1056Parcels',
                'suffix': 'dseg',
                'extension': '.nii.gz',
            },
        ),
    }
    return spec, resolved


def test_volumetric_parcellate_timeseries_emits_tsv_coverage():
    spec, resolved = _story_3_1_volumetric()
    plan = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/x'})

    parc = plan['parcellate_bold']
    assert [(p.suffix, p.extension, p.source_field) for p in parc] == [
        ('timeseries', '.tsv', 'out'),
        ('boldmap', '.tsv', 'coverage'),
    ]
    coverage = parc[1]
    assert coverage.entities['statistic'] == 'coverage'


def test_volumetric_parcellate_scalar_plans_a_coverage_tsv():
    """parcellate_scalar writes coverage as a derivative, like parcellate_timeseries.

    Volumetric -> .tsv (via ExtraProduct.volumetric_extension); CIFTI -> .pscalar.nii.
    """
    spec = load_spec('scripts/tract_parcellate.yml')
    plan = build_sink_plan(spec, resolved={})
    products = [p for node_products in plan.values() for p in node_products]
    coverage = [p for p in products if p.entities.get('statistic') == 'coverage']
    assert coverage, 'parcellate_scalar must plan a coverage product'
    assert all(p.extension == '.tsv' for p in coverage)


def test_cifti_parcellate_timeseries_coverage_still_pscalar():
    spec, resolved = _story_3_1()
    plan = build_sink_plan(spec, resolved, roots={'xcpd': '/x', 'atlases': '/x'})

    parc = plan['parcellate_bold']
    assert [(p.suffix, p.extension, p.source_field) for p in parc] == [
        ('timeseries', '.ptseries.nii', 'out'),
        ('timeseries', '.tsv', 'out'),
        ('boldmap', '.pscalar.nii', 'coverage'),
    ]
