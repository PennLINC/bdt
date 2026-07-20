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
"""Tests for the BDT node-graph spec model, loader, and static validator.

The eight user-stories specs (docs/2026-07-15-bdt-user-stories-and-spec.md §3)
must all validate; the negative cases exercise each error class the validator is
required to catch (spec §1.8).
"""

import pytest

from bdt.spec import SpecValidationError, load_spec, parse_spec, validate_spec
from bdt.spec.model import SpecError

# ---------------------------------------------------------------------------
# The eight user stories, transcribed from the spec.  Each must validate.
# ---------------------------------------------------------------------------

STORY_3_1 = {
    'nodes': [
        {
            'name': 'load_bold',
            'action': 'select_data',
            'dataset': 'xcpd',
            'filters': {
                'space': 'fsLR',
                'desc': 'denoised',
                'suffix': 'bold',
                'extension': '.dtseries.nii',
            },
        },
        {
            'name': 'atlas_4s456',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': '4S456Parcels', 'extension': '.dlabel.nii'},
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
            'parameters': {'xdf_covariance': True},
            'write_outputs': True,
        },
    ]
}

STORY_3_2 = {
    'dataset': [
        {
            'name': 'atlas_hcpmmp',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': 'HCPMMP1', 'den': '32k'},
        },
        {
            'name': 'atlas_4s456',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': '4S456Parcels', 'den': '32k'},
        },
        {
            'name': 'load_geneexpr',
            'action': 'select_data',
            'dataset': 'neuromaps',
            'filters': {'suffix': 'map', 'desc': 'geneexpression', 'space': 'fsLR', 'den': '32k'},
        },
        {
            'name': 'geneexpr_parc',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'load_geneexpr', 'atlas': 'atlas_hcpmmp'},
            'write_outputs': True,
        },
    ],
    'nodes': [
        {
            'name': 'surfaces',
            'action': 'select_data',
            'dataset': 'smriprep',
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
            'filters': {'suffix': 'dwimap', 'model': 'noddi', 'space': 'ACPC'},
        },
        {
            'name': 'noddi_on_surface',
            'action': 'map_scalar_to_surface',
            'inputs': {'scalar': 'load_noddi', 'surfaces': 'surfaces'},
            'parameters': {'source_space': 'fsnative'},
        },
        {
            'name': 'noddi_fslr',
            'action': 'resample_surface_scalar',
            'inputs': {'surface_scalar': 'noddi_on_surface', 'surfaces': 'surfaces'},
            'parameters': {'target_space': 'fsLR', 'target_density': '32k'},
            'write_outputs': True,
        },
        {
            'name': 'noddi_parc',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'noddi_fslr', 'atlas': ['atlas_hcpmmp', 'atlas_4s456']},
            'write_outputs': True,
        },
        {
            'name': 'noddi_depth',
            'action': 'cortical_depth_profile',
            'inputs': {'scalar': 'load_noddi', 'surfaces': 'surfaces'},
            'parameters': {'n_surfaces': 14, 'include_pial': True, 'include_white': True},
            'write_outputs': True,
        },
        {
            'name': 'load_thickness',
            'action': 'select_data',
            'dataset': 'smriprep',
            'filters': {'suffix': 'morph', 'desc': 'thickness', 'space': 'fsnative'},
        },
        {
            'name': 'thickness_fslr',
            'action': 'resample_surface_scalar',
            'inputs': {'surface_scalar': 'load_thickness', 'surfaces': 'surfaces'},
            'parameters': {'target_space': 'fsLR', 'target_density': '32k'},
        },
        {
            'name': 'thickness_parc',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'thickness_fslr', 'atlas': 'atlas_hcpmmp'},
            'write_outputs': True,
        },
        {
            'name': 'load_alff',
            'action': 'select_data',
            'dataset': 'xcpd',
            'filters': {'suffix': 'boldmap', 'stat': 'alff', 'space': 'fsLR', 'den': '32k'},
        },
        {
            'name': 'alff_parc',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'load_alff', 'atlas': 'atlas_hcpmmp'},
            'write_outputs': True,
        },
        # whole-brain CIFTI variant
        {
            'name': 'subcort_structures',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': 'HCPSubcortical', 'space': 'MNI152NLin6Asym', 'res': 2},
        },
        {
            'name': 'noddi_subcort',
            'action': 'resample_subcortical',
            'inputs': {'scalar': 'load_noddi', 'structures': 'subcort_structures'},
            'parameters': {'target_space': 'MNI152NLin6Asym', 'resolution': 2},
        },
        {
            'name': 'noddi_cifti',
            'action': 'assemble_cifti',
            'inputs': {'surface': 'noddi_fslr', 'volume': 'noddi_subcort'},
            'write_outputs': True,
        },
    ],
}

STORY_3_3 = {
    'nodes': [
        {
            'name': 'atlas_hcpmmp',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': 'HCPMMP1', 'den': '32k'},
        },
        {
            'name': 'white_surface',
            'action': 'select_data',
            'dataset': 'smriprep',
            'filters': {'hemi': ['L', 'R'], 'suffix': 'white', 'extension': '.surf.gii'},
        },
        {
            'name': 'load_scalars',
            'action': 'select_data',
            'dataset': 'qsirecon',
            'filters': {'suffix': 'dwimap', 'space': 'T1w'},
            'exclude': [{'model': 'noddi'}],
        },
        {
            'name': 'wm_depth',
            'action': 'wm_depth_profile',
            'inputs': {'scalar': 'load_scalars', 'surfaces': 'white_surface'},
            'parameters': {
                'origin': 'white',
                'direction': 'inward',
                'distances_mm': [0.0, 0.5, 1.0, 1.5, 2.0],
            },
            'write_outputs': True,
        },
        {
            'name': 'wm_depth_parc',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'wm_depth', 'atlas': 'atlas_hcpmmp'},
            'write_outputs': True,
        },
    ]
}

STORY_3_4 = {
    'nodes': [
        {
            'name': 'load_bundles',
            'action': 'select_data',
            'dataset': 'qsirecon',
            'filters': {
                'model': 'gqi',
                'suffix': 'streamlines',
                'extension': '.trx',
                'space': 'ACPC',
            },
        },
        {
            'name': 'bundle_rois',
            'action': 'tractogram_to_pseg',
            'inputs': {'tractograms': 'load_bundles', 'reference': 'load_fa'},
            'parameters': {'threshold': 0.0},
            'write_outputs': True,
        },
        {
            'name': 'load_cbf',
            'action': 'select_data',
            'dataset': 'aslprep',
            'filters': {'suffix': 'cbf', 'desc': 'basil'},
        },
        {
            'name': 'cbf_roi',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'load_cbf', 'atlas': 'bundle_rois'},
            'write_outputs': True,
        },
        {
            'name': 'cbf_profile',
            'action': 'parcellate_scalar_as_tract_profile',
            'inputs': {'scalar': 'load_cbf', 'bundles': 'load_bundles'},
            'parameters': {'n_nodes': 100},
            'write_outputs': True,
        },
        {
            'name': 'load_fa',
            'action': 'select_data',
            'dataset': 'qsirecon',
            'filters': {'suffix': 'dwimap', 'model': 'tensor', 'param': 'fa', 'space': 'ACPC'},
        },
        {
            'name': 'fa_roi',
            'action': 'parcellate_scalar',
            'inputs': {'scalar': 'load_fa', 'atlas': 'bundle_rois'},
            'write_outputs': True,
        },
        {
            'name': 'fa_profile',
            'action': 'parcellate_scalar_as_tract_profile',
            'inputs': {'scalar': 'load_fa', 'bundles': 'load_bundles'},
            'parameters': {'n_nodes': 100},
            'write_outputs': True,
        },
    ]
}

STORY_3_5 = {
    'nodes': [
        {
            'name': 'load_bundles',
            'action': 'select_data',
            'dataset': 'qsirecon',
            'filters': {
                'model': 'gqi',
                'suffix': 'streamlines',
                'extension': '.trx',
                'space': 'ACPC',
            },
        },
        {
            'name': 'atlas_schaefer200',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': '4S200Parcels', 'extension': '.dlabel.nii'},
        },
        {
            'name': 'tract2region',
            'action': 'tract2region',
            'inputs': {'bundles': 'load_bundles', 'atlas': 'atlas_schaefer200'},
            'parameters': {'connectivity_type': ['pass', 'end'], 'connectivity_value': 'count'},
            'write_outputs': True,
        },
    ]
}

STORY_3_6 = {
    'nodes': [
        {
            'name': 'load_streamlines',
            'action': 'select_data',
            'dataset': 'qsirecon',
            'filters': {
                'model': 'msmtcsd',
                'suffix': 'streamlines',
                'extension': '.trx',
                'space': 'ACPC',
            },
        },
        {
            'name': 'load_fa',
            'action': 'select_data',
            'dataset': 'qsirecon',
            'filters': {'suffix': 'dwimap', 'model': 'tensor', 'param': 'fa', 'space': 'ACPC'},
        },
        {
            'name': 'load_cbf',
            'action': 'select_data',
            'dataset': 'aslprep',
            'filters': {'suffix': 'cbf', 'desc': 'basil'},
        },
        {
            'name': 'annotate_fa',
            'action': 'map_scalar_to_streamlines',
            'inputs': {'scalar': 'load_fa', 'streamlines': 'load_streamlines'},
            'parameters': {'name': 'fa', 'per_streamline': 'mean', 'per_vertex': True},
        },
        {
            'name': 'annotate_cbf',
            'action': 'map_scalar_to_streamlines',
            'inputs': {'scalar': 'load_cbf', 'streamlines': 'annotate_fa'},
            'parameters': {'name': 'cbf', 'per_streamline': 'mean'},
            'write_outputs': True,
        },
        {
            'name': 'atlas_schaefer400',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': 'Schaefer400', 'desc': '400Parcels17Networks'},
        },
        {
            'name': 'conn',
            'action': 'region2region',
            'inputs': {'streamlines': 'annotate_cbf', 'atlas': 'atlas_schaefer400'},
            'parameters': {
                'search_radius': 2,
                'edges': [{'weight': 'count'}, {'weight': 'sift2', 'stat_edge': 'sum'}],
            },
            'write_outputs': True,
        },
    ]
}

STORY_3_7 = {
    'nodes': [
        {
            'name': 'load_bold',
            'action': 'select_data',
            'dataset': 'xcpd',
            'filters': {
                'suffix': 'bold',
                'desc': 'denoised',
                'space': 'T1w',
                'extension': '.nii.gz',
            },
        },
        {
            'name': 'atlas_aseg',
            'action': 'select_atlases',
            'dataset': 'freesurfer',
            'filters': {'suffix': 'dseg', 'desc': 'aseg'},
        },
        {
            'name': 'parcellate_bold',
            'action': 'parcellate_timeseries',
            'inputs': {'timeseries': 'load_bold', 'atlas': 'atlas_aseg'},
            'write_outputs': True,
        },
    ]
}

STORY_3_8 = {
    'dataset': [
        {
            'name': 'cortical',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': 'HCPMMP1'},
        },
        {
            'name': 'subcortical',
            'action': 'select_atlases',
            'dataset': 'atlases',
            'filters': {'atlas': 'Tian', 'seg': 'S2'},
        },
        {
            'name': 'cortical_subcortical',
            'action': 'atlas_union',
            'inputs': {'a': 'cortical', 'b': 'subcortical'},
            'parameters': {'output_atlas': 'HCPMMPTian'},
            'write_outputs': True,
        },
    ]
}

ALL_STORIES = {
    '3.1': STORY_3_1,
    '3.2': STORY_3_2,
    '3.3': STORY_3_3,
    '3.4': STORY_3_4,
    '3.5': STORY_3_5,
    '3.6': STORY_3_6,
    '3.7': STORY_3_7,
    '3.8': STORY_3_8,
}


@pytest.mark.parametrize('story', sorted(ALL_STORIES), ids=sorted(ALL_STORIES))
def test_user_story_specs_validate(story):
    """Every user-story spec parses and passes static validation."""
    spec = parse_spec(ALL_STORIES[story])
    validate_spec(spec)  # datasets=None -> skip --datasets key check


def test_dataset_key_check():
    spec = parse_spec(STORY_3_1)
    validate_spec(spec, datasets={'xcpd', 'atlases'})
    with pytest.raises(SpecValidationError) as exc:
        validate_spec(spec, datasets={'xcpd'})  # 'atlases' missing
    assert any('atlases' in e for e in exc.value.errors)


# ---------------------------------------------------------------------------
# Negative cases — one per error class the validator must catch.
# ---------------------------------------------------------------------------


def _expect_error(doc, needle):
    spec = parse_spec(doc)
    with pytest.raises(SpecValidationError) as exc:
        validate_spec(spec)
    assert any(needle in e for e in exc.value.errors), (needle, exc.value.errors)


def test_unknown_action():
    _expect_error(
        {'nodes': [{'name': 'x', 'action': 'not_a_real_action', 'inputs': {'a': 'x'}}]},
        'unknown action',
    )


def test_duplicate_name():
    _expect_error(
        {
            'nodes': [
                {'name': 'dup', 'action': 'select_atlases', 'dataset': 'atlases'},
                {'name': 'dup', 'action': 'select_atlases', 'dataset': 'atlases'},
            ]
        },
        'Duplicate node name',
    )


def test_dangling_reference():
    _expect_error(
        {
            'nodes': [
                {'name': 'a', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 'p',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'nope', 'atlas': 'a'},
                },
            ]
        },
        'undefined node',
    )


def test_selection_with_inputs():
    _expect_error(
        {
            'nodes': [
                {'name': 'a', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 's',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'inputs': {'timeseries': 'a'},
                },
            ]
        },
        "may not use 'inputs:'",
    )


def test_processing_with_dataset():
    _expect_error(
        {
            'nodes': [
                {
                    'name': 'ts',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'extension': '.dtseries.nii'},
                },
                {'name': 'a', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 'p',
                    'action': 'parcellate_timeseries',
                    'dataset': 'xcpd',
                    'inputs': {'timeseries': 'ts', 'atlas': 'a'},
                },
            ]
        },
        "may not use 'dataset:'",
    )


def test_fc_requires_parcellated_series():
    """Feeding a dense (unparcellated) series into FC is a format error."""
    _expect_error(
        {
            'nodes': [
                {
                    'name': 'load_bold',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'suffix': 'bold', 'extension': '.dtseries.nii'},
                },
                {
                    'name': 'fc',
                    'action': 'functional_connectivity',
                    'inputs': {'timeseries': 'load_bold'},
                },  # dense, not parcellated
            ]
        },
        'produces',
    )


def test_missing_required_role():
    _expect_error(
        {
            'nodes': [
                {
                    'name': 'ts',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'extension': '.dtseries.nii'},
                },
                {
                    'name': 'p',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'ts'},
                },  # missing 'atlas'
            ]
        },
        "missing required role 'atlas'",
    )


def test_unknown_role():
    _expect_error(
        {
            'nodes': [
                {
                    'name': 'ts',
                    'action': 'select_data',
                    'dataset': 'xcpd',
                    'filters': {'extension': '.dtseries.nii'},
                },
                {'name': 'a', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 'p',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'ts', 'atlas': 'a', 'bogus': 'a'},
                },
            ]
        },
        "unknown input role 'bogus'",
    )


def test_cycle():
    _expect_error(
        {
            'nodes': [
                {
                    'name': 'a',
                    'action': 'map_scalar_to_streamlines',
                    'inputs': {'scalar': 'a', 'streamlines': 'b'},
                },
                {
                    'name': 'b',
                    'action': 'map_scalar_to_streamlines',
                    'inputs': {'scalar': 'b', 'streamlines': 'a'},
                },
            ]
        },
        'not acyclic',
    )


def test_dataset_node_references_participant():
    _expect_error(
        {
            'dataset': [
                {
                    'name': 'ds_parc',
                    'action': 'parcellate_scalar',
                    'inputs': {'scalar': 'subj_scalar', 'atlas': 'ds_atlas'},
                },
                {'name': 'ds_atlas', 'action': 'select_atlases', 'dataset': 'atlases'},
            ],
            'nodes': [
                {
                    'name': 'subj_scalar',
                    'action': 'select_data',
                    'dataset': 'qsirecon',
                    'filters': {'suffix': 'dwimap'},
                },
            ],
        },
        'references participant-level node',
    )


# ---------------------------------------------------------------------------
# Loader + shape errors.
# ---------------------------------------------------------------------------


def test_load_spec_from_yaml(tmp_path):
    import yaml

    p = tmp_path / 'spec.yaml'
    p.write_text(yaml.safe_dump(STORY_3_1))
    spec = load_spec(p)
    validate_spec(spec)
    assert [n.name for n in spec.nodes] == [
        'load_bold',
        'atlas_4s456',
        'parcellate_bold',
        'fc_bold',
    ]


def test_shape_errors():
    with pytest.raises(SpecError):
        parse_spec({'nodes': [{'action': 'select_data'}]})  # missing name
    with pytest.raises(SpecError):
        parse_spec({'nodes': [{'name': 'x'}]})  # missing action
    with pytest.raises(SpecError):
        parse_spec({'bogus': []})  # unknown top-level key
    with pytest.raises(SpecError):
        parse_spec({'nodes': [{'name': 'x', 'action': 'select_data', 'inputs': 'notadict'}]})


def test_empty_spec():
    with pytest.raises(SpecError):
        parse_spec({})
