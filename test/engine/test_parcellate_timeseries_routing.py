# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Build-time routing tests for parcellate_timeseries (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf  # noqa: E402


def _atlas(tmp_path, ndim, name='tpl-MNI152NLin6Asym_atlas-Y_dseg.nii.gz'):
    """A real on-disk atlas + its BIDS labels sidecar (role_atlas_ndim/labels need both)."""
    import nibabel as nb
    import numpy as np

    shape = (4, 4, 4) if ndim == 3 else (4, 4, 4, 2)
    path = tmp_path / name
    nb.Nifti1Image(np.zeros(shape, 'float32'), np.eye(4)).to_filename(path)
    (tmp_path / name.replace('.nii.gz', '.tsv')).write_text('index\tname\n1\tA\n2\tB\n')
    return str(path)


def _mask(tmp_path, space, basename=None):
    """A real on-disk brain mask, for the discovery `_discover_brain_mask` now runs."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / (basename or f'sub-01_space-{space}_desc-brain_mask.nii.gz')
    nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(path)
    return str(path)


def _match(path, entities):
    from bdt.engine.selection import Match

    return Match(path, entities)


def _node(name, role_to_upstreams, parameters=None):
    return SimpleNamespace(name=name, inputs=role_to_upstreams, parameters=parameters or {})


def test_cifti_timeseries_routes_to_cifti_wf():
    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(resolved={'load_bold': _match('bold.dtseries.nii', {})})
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'vertex_mask' in names  # CIFTI path node


def test_volumetric_timeseries_same_space_builds_parcellate_no_warp(tmp_path):
    from bdt.engine.selection import DictDataProvider

    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider(
            {
                'fmriprep': [
                    _match(
                        _mask(tmp_path, 'MNI152NLin6Asym'),
                        {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                    )
                ],
            }
        ),
        subject='01',
        datasets=['fmriprep'],
        resolved={
            'load_bold': _match('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
            'load_atlas': _match(
                _atlas(tmp_path, 3, 'atlas_dseg.nii.gz'),
                {'space': 'MNI152NLin6Asym', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'parcellate_mean' in names
    assert 'warp_atlas' not in names


def test_volumetric_timeseries_cross_space_builds_warp(tmp_path):
    from bdt.engine.selection import DictDataProvider

    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider(
            {
                'atlases': [
                    _match(
                        _mask(tmp_path, 'MNI152NLin6Asym'),
                        {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                    )
                ],
            }
        ),
        subject='01',
        datasets=['atlases'],
        resolved={
            'load_bold': _match('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
            'load_atlas': _match(
                _atlas(tmp_path, 3, 'atlas_dseg.nii.gz'),
                {'space': 'MNI152NLin2009cAsym', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'warp_atlas' in names
    assert 'register_acpc' not in names  # no ACPC endpoint


def test_nifti_parcellate_yml_compiles_end_to_end(tmp_path):
    """Acceptance: scripts/nifti_parcellate.yml builds a full nipype graph through
    the volumetric parcellate_timeseries + functional_connectivity path.

    The bold (MNI152NLin6Asym) and atlas (MNI152NLin2009cAsym) are in different
    standard spaces, so ``warp_atlas`` must appear (cross-space) but
    ``register_acpc`` must not (neither endpoint is ACPC). Build-only: no tool is
    run, but ``role_atlas_ndim``/``role_atlas_labels`` now read the atlas's header
    and sidecar at *build* time, so (unlike the bold, still a bare path) the atlas
    must be a real on-disk file with a BIDS ``dseg.tsv`` beside it.
    """
    from bdt.engine.selection import DictDataProvider
    from bdt.engine.workflow import init_bdt_wf
    from bdt.spec import load_spec

    spec = load_spec('scripts/nifti_parcellate.yml')

    bold_path = '/data/sub-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    cbf_path = '/data/sub-01_space-MNI152NLin6Asym_cbf.nii.gz'
    atlas_path = _atlas(
        tmp_path, 3, 'tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_res-01_dseg.nii.gz'
    )
    # a 4D probabilistic atlas, so the probseg branch is exercised alongside the dseg one
    difumo_path = _atlas(
        tmp_path, 4, 'tpl-MNI152NLin2009cAsym_atlas-DiFuMo_scale-64dimensions_probseg.nii.gz'
    )

    resolved = {
        'load_bold': _match(
            bold_path,
            {
                'space': 'MNI152NLin6Asym',
                'desc': 'preproc',
                'suffix': 'bold',
                'extension': '.nii.gz',
                'datatype': 'func',
            },
        ),
        'load_cbf': _match(
            cbf_path,
            {
                'space': 'MNI152NLin6Asym',
                'suffix': 'cbf',
                'extension': '.nii.gz',
                'datatype': 'perf',
            },
        ),
        'atlas_difumo': _match(
            difumo_path,
            {
                'template': 'MNI152NLin2009cAsym',
                'atlas': 'DiFuMo',
                'scale': '64dimensions',
                'suffix': 'probseg',
                'extension': '.nii.gz',
            },
        ),
        'atlas_4s456': _match(
            atlas_path,
            {
                # verbatim from a real BIDSDataProvider query against AtlasPack:
                # the file is ``tpl-``-named and carries NO ``space`` entity.
                'template': 'MNI152NLin2009cAsym',
                'atlas': '4S456Parcels',
                'suffix': 'dseg',
                'extension': '.nii.gz',
                'res': '01',
            },
        ),
    }
    # no local xfms and no ACPC anatomicals needed for a standard->standard warp
    provider = DictDataProvider(
        {
            'fmriprep': [
                _match(
                    _mask(tmp_path, 'MNI152NLin6Asym'),
                    {
                        'suffix': 'mask',
                        'desc': 'brain',
                        'space': 'MNI152NLin6Asym',
                        'datatype': 'func',
                    },
                )
            ],
            'atlases': [],
            # parcellate_cbf's brain mask lives beside the CBF in aslprep's tree
            'aslprep': [
                _match(
                    _mask(tmp_path, 'MNI152NLin6Asym', 'perf_mask.nii.gz'),
                    {
                        'suffix': 'mask',
                        'desc': 'brain',
                        'space': 'MNI152NLin6Asym',
                        'datatype': 'perf',
                    },
                )
            ],
        }
    )
    context = FactoryContext(
        spec=spec,
        resolved=resolved,
        provider=provider,
        subject='01',
        datasets=['fmriprep', 'atlases', 'aslprep'],
    )
    selections = {
        'load_bold': bold_path,
        'load_cbf': cbf_path,
        'atlas_4s456': atlas_path,
        'atlas_difumo': difumo_path,
    }

    wf = init_bdt_wf(spec, selections, context=context)

    names = set(wf.list_node_names())

    def has_suffix(suffix):
        return any(n.endswith(suffix) for n in names)

    # volumetric nodes present
    assert has_suffix('parcellate_bold_4s456.parcellate_mean'), names
    assert has_suffix('parcellate_bold_4s456.warp_atlas'), names
    assert has_suffix('fc_bold_4s456.correlate'), names
    # CIFTI-path nodes absent: the volumetric branch also builds a node named
    # 'correlate' (XCP-D's TSVConnect), so matching on that name alone no longer
    # proves the CIFTI path was skipped. Assert the interface type instead.
    from bdt.interfaces.connectivity import NiftiParcellate, TSVConnect

    correlate_node = wf.get_node(next(n for n in names if n.endswith('fc_bold_4s456.correlate')))
    assert isinstance(correlate_node.interface, TSVConnect), correlate_node.interface
    # the 3D dseg atlas must route through the XCP-D masker port, not the deleted
    # hand-rolled parcellator.
    parcellate_node = wf.get_node(
        next(n for n in names if n.endswith('parcellate_bold_4s456.parcellate_mean'))
    )
    assert isinstance(parcellate_node.interface, NiftiParcellate), parcellate_node.interface
    assert not has_suffix('parcellate_bold_4s456.vertex_mask'), names
    assert not has_suffix('parcellate_bold_4s456.restrict_atlas'), names
    # no ACPC bridge (neither space is ACPC)
    assert not has_suffix('parcellate_bold_4s456.register_acpc'), names

    # every statistic in the YAML becomes its own masker, with nilearn's own name
    statistics = spec.by_name()['parcellate_bold_4s456'].parameters['statistics']
    assert len(statistics) == 7, 'fixture should exercise the full vocabulary'
    for stat in statistics:
        node_name = next(n for n in names if n.endswith(f'parcellate_bold_4s456.parcellate_{stat}'))
        assert wf.get_node(node_name).inputs.strategy == stat
    # ...while parcellate_scalar merges them into ONE tidy node instead
    assert has_suffix('parcellate_cbf_4s456.parcellate'), names
    scalar_node = wf.get_node(next(n for n in names if n.endswith('parcellate_cbf_4s456.parcellate')))
    assert list(scalar_node.inputs.statistics) == statistics

    # and the sink plan reflects the asymmetry: 7 wide timeseries tables vs 1 tidy one
    from bdt.outputs.plan import build_sink_plan

    plan = build_sink_plan(spec, resolved, roots={'fmriprep': '/data', 'atlases': str(tmp_path)})
    from bdt.utils.statistics import normalize_statistic

    timeseries = [p for p in plan['parcellate_bold_4s456'] if p.suffix == 'timeseries']
    # entity values are alphanumeric, so standard_deviation -> standarddeviation
    assert [p.entities['statistic'] for p in timeseries] == [
        normalize_statistic(s) for s in statistics
    ]
    assert 'standarddeviation' in {p.entities['statistic'] for p in timeseries}
    tables = [p for p in plan['parcellate_cbf_4s456'] if p.extension == '.tsv' and p.suffix == 'cbf']
    assert len(tables) == 1, tables
    assert 'statistic' not in tables[0].entities

    # ---- the probabilistic (4D) atlas takes the weighted branch instead --------
    from bdt.interfaces.probseg import ProbSegParcellate

    probseg_node = wf.get_node(
        next(n for n in names if n.endswith('parcellate_bold_difumo.parcellate'))
    )
    assert isinstance(probseg_node.interface, ProbSegParcellate), probseg_node.interface
    # ONE node computing every statistic (the weighting work is shared), fanned out
    # by pick_<stat> -- unlike the dseg branch's one masker per statistic.
    assert list(probseg_node.inputs.statistics) == ['mean', 'standard_deviation']
    assert has_suffix('parcellate_bold_difumo.pick_mean'), names
    assert has_suffix('parcellate_bold_difumo.pick_standard_deviation'), names
    assert not has_suffix('parcellate_bold_difumo.parcellate_mean'), names

    # a probabilistic atlas rejects the statistics that have no weighted definition
    difumo = spec.by_name()['parcellate_bold_difumo']
    difumo.parameters['statistics'] = ['mean', 'median']
    with pytest.raises(ValueError, match='probabilistic'):
        init_bdt_wf(spec, selections, context=context)
