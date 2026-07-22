# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Per-statistic parcellation wiring, CIFTI and volumetric."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import (  # noqa: E402
    FactoryContext,
    init_parcellate_scalar_wf,
    init_parcellate_timeseries_wf,
)
from bdt.engine.selection import Match  # noqa: E402
from bdt.engine.workflow import _identity_fields  # noqa: E402


def _node(name='parc', **parameters):
    return SimpleNamespace(
        name=name,
        inputs={'scalar': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters=parameters,
        desc=None,
    )


def _cifti_context():
    return FactoryContext(
        resolved={
            'load_scalar': Match('s.dscalar.nii', {'extension': '.dscalar.nii'}),
            'load_atlas': Match('a.dlabel.nii', {'extension': '.dlabel.nii'}),
        }
    )


def test_cifti_default_builds_one_mean_parcellation():
    wf = init_parcellate_scalar_wf(_node(), context=_cifti_context())
    names = set(wf.list_node_names())
    assert 'parcellate_data_mean' in names
    assert 'parcellate_data_standard_deviation' not in names
    assert _identity_fields(wf, 'outputnode') == {'out', 'out_mean', 'coverage', 'tsv'}


def test_cifti_two_statistics_build_two_parcellations():
    wf = init_parcellate_scalar_wf(
        _node(statistics=['mean', 'standard_deviation']), context=_cifti_context()
    )
    names = set(wf.list_node_names())
    assert 'parcellate_data_mean' in names
    assert 'parcellate_data_standard_deviation' in names
    assert wf.get_node('parcellate_data_mean').inputs.cor_method == 'MEAN'
    assert wf.get_node('parcellate_data_standard_deviation').inputs.cor_method == 'STDEV'
    assert _identity_fields(wf, 'outputnode') == {
        'out',
        'out_mean',
        'out_standard_deviation',
        'coverage',
        'tsv',
    }


def test_cifti_tidy_tsv_node_receives_every_statistic():
    from bdt.interfaces.cifti_stats import PscalarsToTidyTsv

    wf = init_parcellate_scalar_wf(
        _node(statistics=['mean', 'standard_deviation']), context=_cifti_context()
    )
    to_tsv = wf.get_node('to_tsv')
    assert isinstance(to_tsv.interface, PscalarsToTidyTsv)
    assert to_tsv.inputs.statistics == ['mean', 'standard_deviation']

    # each masked pscalar reaches the merger, and the merger feeds the table --
    # nipype needs a Merge node to build the in_files list, so the masks do not
    # connect to to_tsv directly
    edges = {(u.name, d.name): data['connect'] for u, d, data in wf._graph.edges(data=True)}
    assert edges[('mask_mean', 'statistic_list')] == [('out_file', 'in1')]
    assert edges[('mask_standard_deviation', 'statistic_list')] == [('out_file', 'in2')]
    assert edges[('statistic_list', 'to_tsv')] == [('out', 'in_files')]


def _cifti_timeseries_node(**parameters):
    return SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load_bold'], 'atlas': ['load_atlas']},
        parameters=parameters,
        desc=None,
    )


def _cifti_timeseries_context():
    return FactoryContext(
        resolved={
            'load_bold': Match('b.dtseries.nii', {'extension': '.dtseries.nii'}),
            'load_atlas': Match('a.dlabel.nii', {'extension': '.dlabel.nii'}),
        }
    )


def test_parcellate_timeseries_cifti_default_builds_one_parcellation():
    """The default single statistic is still a single MEAN parcellation.

    It is named ``parcellate_data_mean`` now that the node fans out, and the
    outputnode gains ``out_mean`` -- but ``out`` still carries it, so role wiring
    into functional_connectivity is unaffected.
    """
    wf = init_parcellate_timeseries_wf(
        _cifti_timeseries_node(), context=_cifti_timeseries_context()
    )
    names = set(wf.list_node_names())
    assert 'parcellate_data_mean' in names
    assert [n for n in names if n.startswith('parcellate_data_')] == ['parcellate_data_mean']
    assert wf.get_node('parcellate_data_mean').inputs.cor_method == 'MEAN'
    assert _identity_fields(wf, 'outputnode') == {'out', 'coverage', 'out_mean'}


def test_parcellate_timeseries_cifti_has_no_tidy_table():
    """A wide table cannot hold statistics as columns, so no PscalarsToTidyTsv here.

    parcellate_scalar merges into one tidy table; parcellate_timeseries emits a
    separate ptseries (and TSV) per statistic instead.  Building the merger here
    would silently collapse the time axis.
    """
    wf = init_parcellate_timeseries_wf(
        _cifti_timeseries_node(statistics=['mean', 'median']),
        context=_cifti_timeseries_context(),
    )
    names = set(wf.list_node_names())
    assert 'to_tsv' not in names
    assert 'statistic_list' not in names
    assert {'parcellate_data_mean', 'parcellate_data_median'} <= names
    assert wf.get_node('parcellate_data_median').inputs.cor_method == 'MEDIAN'
    fields = _identity_fields(wf, 'outputnode')
    assert 'tsv' not in fields
    assert {'out', 'out_mean', 'out_median', 'coverage'} == fields


def test_unsupported_statistic_fails_at_build_time():
    with pytest.raises(ValueError, match='mode'):
        init_parcellate_scalar_wf(_node(statistics=['mode']), context=_cifti_context())


def _volumetric_context(tmp_path, ndim=3):
    import nibabel as nb
    import numpy as np

    from bdt.engine.selection import DictDataProvider

    shape = (4, 4, 4) if ndim == 3 else (4, 4, 4, 2)
    atlas = tmp_path / 'tpl-MNI152NLin6Asym_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros(shape, 'float32'), np.eye(4)).to_filename(atlas)
    (tmp_path / 'tpl-MNI152NLin6Asym_atlas-Y_dseg.tsv').write_text('index\tname\n1\tA\n2\tB\n')
    mask = tmp_path / 'sub-01_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(mask)

    return FactoryContext(
        provider=DictDataProvider(
            {
                'fmriprep': [
                    Match(
                        str(mask),
                        {
                            'space': 'MNI152NLin6Asym',
                            'suffix': 'mask',
                            'desc': 'brain',
                            'datatype': 'func',
                        },
                    )
                ]
            }
        ),
        subject='01',
        datasets=['fmriprep'],
        resolved={
            'load_scalar': Match(
                '/d/sub-01_space-MNI152NLin6Asym_cbf.nii.gz',
                {
                    'space': 'MNI152NLin6Asym',
                    'suffix': 'cbf',
                    'datatype': 'func',
                    'extension': '.nii.gz',
                },
            ),
            'load_atlas': Match(
                str(atlas),
                {
                    'space': 'MNI152NLin6Asym',
                    'suffix': 'dseg',
                    'extension': '.nii.gz',
                },
            ),
        },
    )


def test_volumetric_scalar_uses_the_statistics_interface(tmp_path):
    from bdt.interfaces.parcellate_stats import ParcellateScalarStatistics

    wf = init_parcellate_scalar_wf(
        _node(statistics=['mean', 'standard_deviation']),
        context=_volumetric_context(tmp_path),
    )
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, ParcellateScalarStatistics)
    assert parcellate.inputs.statistics == ['mean', 'standard_deviation']


def test_volumetric_scalar_defaults_to_mean(tmp_path):
    wf = init_parcellate_scalar_wf(_node(), context=_volumetric_context(tmp_path))
    assert wf.get_node('parcellate').inputs.statistics == ['mean']


def test_volumetric_4d_atlas_binarizes_a_thresholded_dseg(tmp_path):
    """The 4D branch keeps its binarize signal when statistics are requested."""
    ctx = _volumetric_context(tmp_path, ndim=4)
    wf = init_parcellate_scalar_wf(_node(statistics=['mean']), context=ctx)
    assert wf.get_node('parcellate').inputs.binarize is True


def test_volumetric_timeseries_still_uses_nifti_parcellate(tmp_path):
    """The timeseries path keeps NiftiParcellate; it fans out rather than switching.

    A parcellated *series* stays wide (timepoints x parcels) whatever the statistic,
    so it must not be routed to the tidy ParcellateScalarStatistics.
    """
    from bdt.interfaces.connectivity import NiftiParcellate

    ctx = _volumetric_context(tmp_path)
    node = SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters={},
        desc=None,
    )
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    parcellate = wf.get_node('parcellate_mean')
    assert isinstance(parcellate.interface, NiftiParcellate)
    assert parcellate.inputs.strategy == 'mean'


def test_volumetric_timeseries_fans_out_one_masker_per_statistic(tmp_path):
    """Each statistic is a separate NiftiParcellate with its own nilearn strategy."""
    from bdt.interfaces.connectivity import NiftiParcellate

    ctx = _volumetric_context(tmp_path)
    node = SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters={'statistics': ['median', 'variance']},
        desc=None,
    )
    wf = init_parcellate_timeseries_wf(node, context=ctx)

    for stat in ('median', 'variance'):
        parcellate = wf.get_node(f'parcellate_{stat}')
        assert isinstance(parcellate.interface, NiftiParcellate)
        # SUPPORTED_STATISTICS is nilearn's own vocabulary, passed straight through
        assert parcellate.inputs.strategy == stat
    assert 'parcellate_mean' not in set(wf.list_node_names())

    fields = _identity_fields(wf, 'outputnode')
    assert {'out', 'coverage', 'out_median', 'out_variance'} == fields
    # ``out`` mirrors the FIRST requested statistic, not an alphabetical or
    # hardcoded 'mean' one -- 'median' sorts before 'variance', so request order is
    # verified separately below.
    edges = {(u.name, d.name): data['connect'] for u, d, data in wf._graph.edges(data=True)}
    assert ('timeseries', 'out') in edges[('parcellate_median', 'outputnode')]
    assert ('coverage', 'coverage') in edges[('parcellate_median', 'outputnode')]
    assert ('timeseries', 'out') not in edges[('parcellate_variance', 'outputnode')]


def test_volumetric_timeseries_out_follows_request_order(tmp_path):
    """Reversing the request moves ``out`` -- so it is not a hardcoded statistic."""
    ctx = _volumetric_context(tmp_path)
    node = SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters={'statistics': ['variance', 'median']},
        desc=None,
    )
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    edges = {(u.name, d.name): data['connect'] for u, d, data in wf._graph.edges(data=True)}
    assert ('timeseries', 'out') in edges[('parcellate_variance', 'outputnode')]
    assert ('timeseries', 'out') not in edges[('parcellate_median', 'outputnode')]


def test_volumetric_timeseries_4d_atlas_rejects_extra_statistics(tmp_path):
    """A probabilistic atlas yields a weighted mean only; anything else is an error."""
    ctx = _volumetric_context(tmp_path, ndim=4)
    node = SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters={'statistics': ['median']},
        desc=None,
    )
    with pytest.raises(ValueError, match='probabilistic'):
        init_parcellate_timeseries_wf(node, context=ctx)


def _edges(wf):
    return {(u.name, d.name): data['connect'] for u, d, data in wf._graph.edges(data=True)}


def test_weighted_statistics_take_cifti_weights_directly():
    """The normal path is unchanged: raw data in, vertex mask as -cifti-weights."""
    wf = init_parcellate_timeseries_wf(
        _cifti_timeseries_node(statistics=['mean', 'median', 'sum', 'variance']),
        context=_cifti_timeseries_context(),
    )
    edges = _edges(wf)
    assert 'nan_mask_data' not in set(wf.list_node_names())
    for stat in ('mean', 'median', 'sum', 'variance'):
        node = f'parcellate_data_{stat}'
        assert ('mask_file', 'cifti_weights') in edges[('vertex_mask', node)]
        assert ('timeseries', 'in_file') in edges[('inputnode', node)]


def test_min_and_max_are_nan_masked_instead_of_weighted():
    """Workbench refuses -cifti-weights for MIN/MAX.

    Dropping the weights alone would be a silent wrong answer -- an uncovered
    vertex is *zero*, so it would win every minimum.  The data is NaN-masked first
    and ``-only-numeric`` (always on) does the excluding.
    """
    from bdt.interfaces.cifti import CiftiMask

    wf = init_parcellate_timeseries_wf(
        _cifti_timeseries_node(statistics=['minimum', 'maximum']),
        context=_cifti_timeseries_context(),
    )
    edges = _edges(wf)

    assert isinstance(wf.get_node('nan_mask_data').interface, CiftiMask)
    assert ('mask_file', 'mask') in edges[('vertex_mask', 'nan_mask_data')]
    for stat in ('minimum', 'maximum'):
        node = f'parcellate_data_{stat}'
        # fed the NaN-masked series, never the raw one, and never weighted
        assert ('out_file', 'in_file') in edges[('nan_mask_data', node)]
        assert ('inputnode', node) not in edges
        assert ('vertex_mask', node) not in edges
        assert wf.get_node(node).inputs.only_numeric is True


def test_the_nan_masked_series_is_built_once_for_both():
    """MIN and MAX share one masking node rather than each rebuilding it."""
    wf = init_parcellate_timeseries_wf(
        _cifti_timeseries_node(statistics=['minimum', 'maximum', 'mean']),
        context=_cifti_timeseries_context(),
    )
    names = wf.list_node_names()
    assert len([n for n in names if n.startswith('nan_mask')]) == 1
    # ...and the weighted statistic in the same request still gets the raw data
    assert ('timeseries', 'in_file') in _edges(wf)[('inputnode', 'parcellate_data_mean')]
