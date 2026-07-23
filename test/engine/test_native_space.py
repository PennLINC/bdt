# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""A derivative with no ``space-`` entity is in its modality's own native space."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402

from bdt.engine.factories import (  # noqa: E402
    NATIVE_SPACE_BY_DATATYPE,
    FactoryContext,
    init_parcellate_timeseries_wf,
)
from bdt.engine.selection import DictDataProvider, Match  # noqa: E402


def _img(path, shape, zoom, labels=False):
    affine = np.eye(4) * zoom
    affine[3, 3] = 1
    data = np.ones(shape, 'int16' if labels else 'float32')
    nb.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


def _context(tmp_path, atlas_entities, data_entities, atlas_shape=(4, 4, 4), atlas_zoom=1):
    atlas = _img(tmp_path / 'atlas_dseg.nii.gz', atlas_shape, atlas_zoom, labels=True)
    (tmp_path / 'atlas_dseg.tsv').write_text('index\tname\n1\tA\n')
    data = _img(tmp_path / 'data_bold.nii.gz', (4, 4, 4), 1)
    mask = _img(tmp_path / 'mask.nii.gz', (4, 4, 4), 1)
    # the sibling brain mask carries the same datatype/space entities as the data,
    # which is the point: an unlabelled BOLD sits beside an unlabelled mask
    mask_entities = {'suffix': 'mask', 'desc': 'brain'}
    for key in ('datatype', 'space'):
        if key in data_entities:
            mask_entities[key] = data_entities[key]
    return FactoryContext(
        provider=DictDataProvider({'d': [Match(mask, mask_entities)]}),
        subject='01',
        datasets=['d'],
        resolved={
            'load_data': Match(data, data_entities),
            'load_atlas': Match(atlas, atlas_entities),
        },
    )


def _node():
    return SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load_data'], 'atlas': ['load_atlas']},
        parameters={},
        desc=None,
    )


def test_native_space_names_match_the_preprocessors_own_transforms():
    """Each name must be the one the xfm filenames use, or nothing can connect.

    Verified against real derivatives: fMRIPrep writes ``from-boldref_to-T1w``,
    ASLPrep ``from-aslref_to-T1w``.  ``dwi`` is deliberately absent -- QSIPrep names
    its preprocessed output ``space-ACPC`` explicitly, so an unlabelled dwi is raw
    and has no agreed space name.
    """
    assert NATIVE_SPACE_BY_DATATYPE == {'func': 'boldref', 'anat': 'T1w', 'perf': 'aslref'}


def test_unlabelled_func_and_anat_are_recognised_as_different_spaces(tmp_path):
    """The regression: two files with no ``space-`` used to compare equal.

    A boldref-space BOLD and a T1w-space dseg both lack ``space-``, so cross-space
    detection saw ``None == None`` and built no warp -- then nilearn failed deep in
    the masker with a field-of-view error naming neither file.
    """
    ctx = _context(
        tmp_path,
        atlas_entities={'suffix': 'dseg', 'datatype': 'anat', 'extension': '.nii.gz'},
        data_entities={'suffix': 'bold', 'datatype': 'func', 'extension': '.nii.gz'},
    )
    node = _node()
    assert ctx.role_space(node, 'timeseries') == 'boldref'
    assert ctx.role_space(node, 'atlas') == 'T1w'

    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'warp_atlas' in names


def test_an_explicit_space_still_wins(tmp_path):
    ctx = _context(
        tmp_path,
        atlas_entities={'suffix': 'dseg', 'datatype': 'anat', 'space': 'MNI152NLin6Asym'},
        data_entities={'suffix': 'bold', 'datatype': 'func', 'space': 'MNI152NLin6Asym'},
    )
    node = _node()
    assert ctx.role_space(node, 'atlas') == 'MNI152NLin6Asym'
    # same named space -> no warp, and the grids agree so the guard is satisfied
    assert 'warp_atlas' not in set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())


def test_the_brain_mask_query_asks_for_an_absent_space_not_the_resolved_one(tmp_path):
    """`space-boldref` appears in no filename: the sibling mask is unlabelled too.

    Querying the *resolved* space would match nothing; the query must use the
    ``Query.NONE`` sentinel for the entity's absence.
    """
    from bids.layout import Query

    ctx = _context(
        tmp_path,
        atlas_entities={'suffix': 'dseg', 'datatype': 'anat'},
        data_entities={'suffix': 'bold', 'datatype': 'func'},
    )
    node = _node()
    assert ctx.role_space(node, 'timeseries') == 'boldref'  # logical space
    assert ctx.role_space_entity(node, 'timeseries') is Query.NONE  # filename entity
    # ...and the mask is found, which it would not be under `space-boldref`
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    assert wf.get_node('parcellate_mean').inputs.mask.endswith('mask.nii.gz')


def test_unnameable_spaces_on_different_grids_are_a_build_time_error(tmp_path):
    """No warp is possible, so refuse rather than build a graph that cannot run."""
    ctx = _context(
        tmp_path,
        # datatype the native-space map does not cover -> space stays unknown
        atlas_entities={'suffix': 'dseg', 'datatype': 'dwi'},
        data_entities={'suffix': 'bold', 'datatype': 'dwi'},
        atlas_shape=(8, 8, 8),
        atlas_zoom=2,
    )
    with pytest.raises(ValueError, match='different voxel grids'):
        init_parcellate_timeseries_wf(_node(), context=ctx)


def test_unnameable_spaces_on_the_same_grid_are_fine(tmp_path):
    """Identical geometry is parcellatable whatever the space is called."""
    ctx = _context(
        tmp_path,
        atlas_entities={'suffix': 'dseg', 'datatype': 'dwi'},
        data_entities={'suffix': 'bold', 'datatype': 'dwi'},
    )
    names = set(init_parcellate_timeseries_wf(_node(), context=ctx).list_node_names())
    assert 'warp_atlas' not in names
    assert 'parcellate_mean' in names
