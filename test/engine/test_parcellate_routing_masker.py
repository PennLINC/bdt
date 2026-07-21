# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Volumetric parcellate routing: 3D dseg -> NiftiParcellate, 4D -> ProbSegParcellate."""

from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pytest

from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf
from bdt.engine.selection import DictDataProvider, Match


def _atlas(tmp_path, ndim, name='tpl-MNI152NLin6Asym_atlas-Y_dseg.nii.gz'):
    shape = (4, 4, 4) if ndim == 3 else (4, 4, 4, 2)
    path = tmp_path / name
    nb.Nifti1Image(np.zeros(shape, 'float32'), np.eye(4)).to_filename(path)
    (tmp_path / name.replace('.nii.gz', '.tsv')).write_text('index\tname\n1\tA\n2\tB\n')
    return str(path)


def _node(**parameters):
    return SimpleNamespace(
        name='parcellate_bold',
        inputs={'timeseries': ['load_bold'], 'atlas': ['atlas_sel']},
        parameters=parameters,
        desc=None,
    )


def _context(atlas_path, mask_path, space='MNI152NLin6Asym', suffix='dseg'):
    resolved = {
        'load_bold': Match(
            '/d/sub-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz',
            {'space': space, 'suffix': 'bold', 'datatype': 'func', 'extension': '.nii.gz'},
        ),
        'atlas_sel': Match(
            atlas_path,
            {'space': space, 'suffix': suffix, 'extension': '.nii.gz'},
        ),
    }
    provider = DictDataProvider(
        {
            'fmriprep': [
                Match(
                    mask_path,
                    {'space': space, 'suffix': 'mask', 'desc': 'brain', 'datatype': 'func'},
                ),
            ],
        }
    )
    return FactoryContext(
        resolved=resolved, provider=provider, subject='01', datasets=['fmriprep']
    )


def _mask(tmp_path):
    path = tmp_path / 'sub-01_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(path)
    return str(path)


def test_3d_dseg_atlas_uses_xcpd_nifti_parcellate(tmp_path):
    from bdt.interfaces.connectivity import NiftiParcellate

    ctx = _context(_atlas(tmp_path, 3), _mask(tmp_path))
    wf = init_parcellate_timeseries_wf(_node(min_coverage=0.5), context=ctx)
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, NiftiParcellate)
    assert parcellate.inputs.min_coverage == 0.5
    # the sidecar is resolved from the selection, not the warped atlas
    assert parcellate.inputs.atlas_labels.endswith('_dseg.tsv')


def test_4d_probseg_atlas_uses_probseg_parcellate_unbinarized(tmp_path):
    from bdt.interfaces.probseg import ProbSegParcellate

    ctx = _context(_atlas(tmp_path, 4), _mask(tmp_path), suffix='probseg')
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, ProbSegParcellate)
    assert parcellate.inputs.binarize is False


def test_4d_dseg_atlas_binarizes(tmp_path):
    """A thresholded 4D atlas (suffix dseg) is binarized before averaging."""
    from bdt.interfaces.probseg import ProbSegParcellate

    ctx = _context(_atlas(tmp_path, 4), _mask(tmp_path), suffix='dseg')
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    parcellate = wf.get_node('parcellate')
    assert isinstance(parcellate.interface, ProbSegParcellate)
    assert parcellate.inputs.binarize is True


def test_brain_mask_is_discovered_and_wired(tmp_path):
    mask = _mask(tmp_path)
    ctx = _context(_atlas(tmp_path, 3), mask)
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    assert wf.get_node('parcellate').inputs.mask == mask


def test_missing_brain_mask_is_a_hard_error(tmp_path):
    """No data-derived fallback: a missing mask names the failed query and stops."""
    ctx = _context(_atlas(tmp_path, 3), _mask(tmp_path))
    ctx.provider = DictDataProvider({'fmriprep': []})
    with pytest.raises(ValueError, match="'suffix': 'mask'"):
        init_parcellate_timeseries_wf(_node(), context=ctx)


def test_outputnode_exposes_out_and_coverage(tmp_path):
    ctx = _context(_atlas(tmp_path, 3), _mask(tmp_path))
    wf = init_parcellate_timeseries_wf(_node(), context=ctx)
    from bdt.engine.workflow import _identity_fields

    assert _identity_fields(wf, 'outputnode') == {'out', 'coverage'}


def test_hand_rolled_parcellate_module_is_gone():
    """The bespoke numpy implementation must not survive alongside the maskers."""
    import pytest

    with pytest.raises(ModuleNotFoundError):
        import bdt.interfaces.parcellate  # noqa: F401

    import bdt.utils.cifti as cifti

    assert not hasattr(cifti, 'tsv_correlation')


# Entity dicts below are VERBATIM from a real BIDSDataProvider query against
# ds008325 sub-125511. pybids uses long entity names -- `acquisition`, not `acq`;
# `reconstruction`, not `rec` -- and a fixture that invents the short forms will
# pass while production fails, which is exactly how the first fix slipped through.
_REST_MB = {
    'subject': '125511',
    'session': '1',
    'task': 'rest',
    'acquisition': 'multiband',
    'space': 'MNI152NLin6Asym',
    'suffix': 'bold',
    'extension': '.nii.gz',
    'datatype': 'func',
    'desc': 'preproc',
    'res': '2',
}
_MASKS = [
    {
        'subject': '125511',
        'session': '1',
        'task': 'fracback',
        'acquisition': 'singleband',
        'space': 'MNI152NLin6Asym',
        'suffix': 'mask',
        'extension': '.nii.gz',
        'datatype': 'func',
        'desc': 'brain',
        'res': '2',
    },
    {
        'subject': '125511',
        'session': '1',
        'task': 'rest',
        'acquisition': 'multiband',
        'space': 'MNI152NLin6Asym',
        'suffix': 'mask',
        'extension': '.nii.gz',
        'datatype': 'func',
        'desc': 'brain',
        'res': '2',
    },
    {
        'subject': '125511',
        'session': '1',
        'task': 'rest',
        'acquisition': 'singleband',
        'space': 'MNI152NLin6Asym',
        'suffix': 'mask',
        'extension': '.nii.gz',
        'datatype': 'func',
        'desc': 'brain',
        'res': '2',
    },
    {
        'subject': '125511',
        'session': '1',
        'reconstruction': 'refaced',
        'space': 'MNI152NLin6Asym',
        'suffix': 'mask',
        'extension': '.nii.gz',
        'datatype': 'anat',
        'desc': 'brain',
        'res': '2',
    },
]


def _bids_name(entities, suffix):
    order = ['subject', 'session', 'task', 'acquisition', 'reconstruction', 'space', 'res', 'desc']
    keys = {
        'subject': 'sub',
        'session': 'ses',
        'task': 'task',
        'acquisition': 'acq',
        'reconstruction': 'rec',
        'space': 'space',
        'res': 'res',
        'desc': 'desc',
    }
    parts = [f'{keys[k]}-{entities[k]}' for k in order if k in entities]
    return '_'.join(parts) + f'_{suffix}.nii.gz'


def _mask_provider(tmp_path, mask_entities):
    import nibabel as nb
    import numpy as np

    matches = []
    for ent in mask_entities:
        path = tmp_path / _bids_name(ent, 'mask')
        nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(path)
        matches.append(Match(str(path), ent))
    return DictDataProvider({'fmriprep': matches})


def _bold_node_and_context(tmp_path, provider, bold_entities=None):
    node = SimpleNamespace(
        name='parcellate_bold',
        inputs={'timeseries': ['load_bold'], 'atlas': ['atlas_sel']},
        parameters={},
        desc=None,
    )
    ctx = FactoryContext(
        provider=provider,
        subject='125511',
        datasets=['fmriprep'],
        resolved={
            'load_bold': Match(
                '/d/' + _bids_name(bold_entities or _REST_MB, 'bold'),
                bold_entities or _REST_MB,
            ),
            'atlas_sel': Match(
                _atlas(tmp_path, 3),
                {
                    'space': 'MNI152NLin6Asym',
                    'suffix': 'dseg',
                    'extension': '.nii.gz',
                },
            ),
        },
    )
    return node, ctx


def test_brain_mask_query_is_scoped_to_the_data_acquisition(tmp_path):
    """One mask per BOLD run is the ordinary fMRIPrep layout, not an edge case.

    Regression: the mask query originally carried only space/session/datatype, so
    ds008325 sub-125511 matched 4 masks and ``find_reference`` -- which demands
    exactly one -- failed the whole build before any node ran.
    """
    provider = _mask_provider(tmp_path, _MASKS)
    node, ctx = _bold_node_and_context(tmp_path, provider)

    chosen = init_parcellate_timeseries_wf(node, context=ctx).get_node('parcellate').inputs.mask
    assert 'task-rest_acq-multiband' in chosen, chosen
    assert 'fracback' not in chosen, chosen
    assert 'rec-refaced' not in chosen, chosen


def test_brain_mask_falls_back_when_scoping_over_constrains(tmp_path):
    """A derivative may omit an entity the data carries; don't over-constrain to zero.

    Here the BOLD is ``part-mag`` but the single mask has no ``part``. The specific
    query matches nothing, so the base space/datatype query must still resolve it.
    """
    bold = dict(_REST_MB, part='mag')
    only_mask = dict(_MASKS[1])
    provider = _mask_provider(tmp_path, [only_mask])
    node, ctx = _bold_node_and_context(tmp_path, provider, bold_entities=bold)

    chosen = init_parcellate_timeseries_wf(node, context=ctx).get_node('parcellate').inputs.mask
    assert 'task-rest_acq-multiband' in chosen, chosen


def test_unresolvable_brain_mask_names_every_attempted_query(tmp_path):
    """Genuinely ambiguous input still fails loudly, listing what was tried."""
    ambiguous = [dict(_MASKS[1]), dict(_MASKS[1])]  # two identical-entity masks
    ambiguous[1] = dict(_MASKS[1], desc='brain')
    provider = _mask_provider(tmp_path, _MASKS[:1] + _MASKS[2:3])  # neither matches rest/multiband
    node, ctx = _bold_node_and_context(tmp_path, provider)

    with pytest.raises(ValueError, match='Could not resolve exactly one brain mask'):
        init_parcellate_timeseries_wf(node, context=ctx)


def test_anatomical_lookup_prefers_t1w_over_t2w(tmp_path):
    """T1w and T2w are a preference order, not an any-of set.

    Regression: the ACPC-bridge lookup used ``suffix: ['T1w', 'T2w']``, which on a
    subject having both -- the ds008325 layout, ``rec-refaced_desc-preproc_T1w`` plus
    ``rec-defaced_desc-preproc_T2w`` -- matched two files and failed the whole build.
    Entities are as a real BIDSDataProvider reports them.
    """
    from bdt.engine.factories import _find_anatomical

    t1 = tmp_path / 'sub-125511_ses-1_rec-refaced_desc-preproc_T1w.nii.gz'
    t2 = tmp_path / 'sub-125511_ses-1_rec-defaced_desc-preproc_T2w.nii.gz'
    for p in (t1, t2):
        nb.Nifti1Image(np.zeros((2, 2, 2), 'float32'), np.eye(4)).to_filename(p)
    provider = DictDataProvider(
        {
            'aslprep': [
                Match(
                    str(t1),
                    {
                        'subject': '125511',
                        'session': '1',
                        'reconstruction': 'refaced',
                        'desc': 'preproc',
                        'suffix': 'T1w',
                        'datatype': 'anat',
                        'extension': '.nii.gz',
                    },
                ),
                Match(
                    str(t2),
                    {
                        'subject': '125511',
                        'session': '1',
                        'reconstruction': 'defaced',
                        'desc': 'preproc',
                        'suffix': 'T2w',
                        'datatype': 'anat',
                        'extension': '.nii.gz',
                    },
                ),
            ]
        }
    )
    ctx = FactoryContext(provider=provider, subject='125511', datasets=['aslprep'])
    assert _find_anatomical(ctx, None, None) == str(t1)


def test_anatomical_lookup_falls_back_to_t2w(tmp_path):
    """A T2w-only subject (QSIPrep --anat-modality T2w) still resolves."""
    from bdt.engine.factories import _find_anatomical

    t2 = tmp_path / 'sub-01_space-ACPC_desc-preproc_T2w.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), 'float32'), np.eye(4)).to_filename(t2)
    provider = DictDataProvider(
        {
            'qsiprep': [
                Match(
                    str(t2),
                    {
                        'subject': '01',
                        'space': 'ACPC',
                        'desc': 'preproc',
                        'suffix': 'T2w',
                        'datatype': 'anat',
                        'extension': '.nii.gz',
                    },
                ),
            ]
        }
    )
    ctx = FactoryContext(provider=provider, subject='01', datasets=['qsiprep'])
    assert _find_anatomical(ctx, 'ACPC', None) == str(t2)


def test_ambiguous_same_suffix_anatomicals_raise(tmp_path):
    """Two reconstructions of the T1w is a real ambiguity -- don't silently take T2w."""
    from bdt.engine.factories import _find_anatomical

    paths = []
    for rec in ('refaced', 'defaced'):
        p = tmp_path / f'sub-01_rec-{rec}_desc-preproc_T1w.nii.gz'
        nb.Nifti1Image(np.zeros((2, 2, 2), 'float32'), np.eye(4)).to_filename(p)
        paths.append(
            Match(
                str(p),
                {
                    'subject': '01',
                    'reconstruction': rec,
                    'desc': 'preproc',
                    'suffix': 'T1w',
                    'datatype': 'anat',
                    'extension': '.nii.gz',
                },
            )
        )
    ctx = FactoryContext(
        provider=DictDataProvider({'aslprep': paths}), subject='01', datasets=['aslprep']
    )
    with pytest.raises(ValueError, match='matched 2 files'):
        _find_anatomical(ctx, None, None)
