# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for bdt.interfaces.parcellate (volumetric coverage-aware parcellation)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('nibabel')


def _nii(path, data):
    import nibabel as nb

    nb.Nifti1Image(np.asarray(data), np.eye(4)).to_filename(str(path))
    return str(path)


def test_4d_dseg_binary_masks_give_in_mask_mean(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    # scalar: a 4x1x1 row of values [1, 2, 3, 4].
    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1))
    # 4D dseg, 2 regions: region0 covers voxels 0&1, region1 covers voxels 2&3.
    atlas = np.zeros((4, 1, 1, 2), dtype=np.int16)
    atlas[0, 0, 0, 0] = 1
    atlas[1, 0, 0, 0] = 1
    atlas[2, 0, 0, 1] = 1
    atlas[3, 0, 0, 1] = 1
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.0)
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['index', 'name', 'mean', 'coverage']
    assert df['index'].tolist() == [1, 2]  # 1-based, matches EntitiesToSegTSV
    assert df['mean'].tolist() == [1.5, 3.5]
    assert df['coverage'].tolist() == [1.0, 1.0]


def test_4d_pseg_weights_diverge_from_plain_mean(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([10.0, 20.0]).reshape(2, 1, 1))
    # one probabilistic region weighting voxel0 by 0.25 and voxel1 by 0.75.
    atlas = np.zeros((2, 1, 1, 1), dtype=np.float32)
    atlas[0, 0, 0, 0] = 0.25
    atlas[1, 0, 0, 0] = 0.75
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.0)
    df = pd.read_csv(out, sep='\t')
    # weighted: (0.25*10 + 0.75*20) / (0.25+0.75) = 17.5, not the plain mean 15.
    assert df['mean'].tolist() == [17.5]


def test_3d_dseg_label_means_with_and_without_labels(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([5.0, 7.0, 9.0]).reshape(3, 1, 1))
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', np.array([1, 1, 2], dtype=np.int16).reshape(3, 1, 1))

    # no labels -> name is the string of the label value.
    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.0)
    df = pd.read_csv(out, sep='\t', dtype={'name': str})
    assert df['index'].tolist() == [1, 2]
    assert df['name'].tolist() == ['1', '2']
    assert df['mean'].tolist() == [6.0, 9.0]  # (5+7)/2, 9

    # with labels -> names mapped.
    out2 = _parcellate_volumetric(
        scalar, atlas_path, str(tmp_path / 'out2.tsv'),
        min_coverage=0.0, labels={1: 'CST', 2: 'AF'},
    )
    df2 = pd.read_csv(out2, sep='\t')
    assert df2['name'].tolist() == ['CST', 'AF']


def test_low_coverage_region_is_nan_masked(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric

    # region covers 4 voxels; only 1 has valid (finite, nonzero) data -> coverage 0.25.
    scalar = _nii(
        tmp_path / 'scalar.nii.gz', np.array([8.0, 0.0, 0.0, 0.0]).reshape(4, 1, 1)
    )
    atlas = np.zeros((4, 1, 1, 1), dtype=np.int16)
    atlas[:, 0, 0, 0] = 1
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric(scalar, atlas_path, str(tmp_path / 'out.tsv'), min_coverage=0.5)
    df = pd.read_csv(out, sep='\t')
    assert df['coverage'].tolist() == [0.25]
    assert np.isnan(df['mean'].tolist()[0])


def test_interface_reads_labels_sidecar(tmp_path):
    from bdt.interfaces.parcellate import ParcellateVolumetric

    scalar = _nii(tmp_path / 'scalar.nii.gz', np.array([5.0, 7.0, 9.0]).reshape(3, 1, 1))
    atlas_path = _nii(tmp_path / 'atlas.nii.gz', np.array([1, 1, 2], dtype=np.int16).reshape(3, 1, 1))
    labels_tsv = tmp_path / 'dseg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['CST', 'AF']}).to_csv(labels_tsv, sep='\t', index=False)

    res = ParcellateVolumetric(
        scalar=scalar, atlas=atlas_path, atlas_labels=str(labels_tsv),
        min_coverage=0.0, out_file=str(tmp_path / 'out.tsv'),
    ).run()
    df = pd.read_csv(res.outputs.out_file, sep='\t')
    assert df['name'].tolist() == ['CST', 'AF']
    assert df['mean'].tolist() == [6.0, 9.0]
