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
    return FactoryContext(resolved={
        'load_scalar': Match('s.dscalar.nii', {'extension': '.dscalar.nii'}),
        'load_atlas': Match('a.dlabel.nii', {'extension': '.dlabel.nii'}),
    })


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
        'out', 'out_mean', 'out_standard_deviation', 'coverage', 'tsv'
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


def test_parcellate_timeseries_cifti_is_untouched():
    """The timeseries path keeps its original node names and outputnode fields."""
    node = SimpleNamespace(
        name='parc', inputs={'timeseries': ['load_bold'], 'atlas': ['load_atlas']},
        parameters={}, desc=None,
    )
    ctx = FactoryContext(resolved={
        'load_bold': Match('b.dtseries.nii', {'extension': '.dtseries.nii'}),
        'load_atlas': Match('a.dlabel.nii', {'extension': '.dlabel.nii'}),
    })
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    names = set(wf.list_node_names())
    assert 'parcellate_data' in names
    assert not any(n.startswith('parcellate_data_') for n in names)
    assert _identity_fields(wf, 'outputnode') == {'out', 'coverage'}


def test_unsupported_statistic_fails_at_build_time():
    with pytest.raises(ValueError, match='median'):
        init_parcellate_scalar_wf(_node(statistics=['median']), context=_cifti_context())


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
        provider=DictDataProvider({'fmriprep': [Match(str(mask), {
            'space': 'MNI152NLin6Asym', 'suffix': 'mask', 'desc': 'brain',
            'datatype': 'func',
        })]}),
        subject='01', datasets=['fmriprep'],
        resolved={
            'load_scalar': Match('/d/sub-01_space-MNI152NLin6Asym_cbf.nii.gz', {
                'space': 'MNI152NLin6Asym', 'suffix': 'cbf', 'datatype': 'func',
                'extension': '.nii.gz',
            }),
            'load_atlas': Match(str(atlas), {
                'space': 'MNI152NLin6Asym', 'suffix': 'dseg', 'extension': '.nii.gz',
            }),
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
    """Only the scalar action gains statistics; the timeseries path is unchanged."""
    from bdt.interfaces.connectivity import NiftiParcellate

    ctx = _volumetric_context(tmp_path)
    node = SimpleNamespace(
        name='parc', inputs={'timeseries': ['load_scalar'], 'atlas': ['load_atlas']},
        parameters={}, desc=None,
    )
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    assert isinstance(wf.get_node('parcellate').interface, NiftiParcellate)
