# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for ProbSegParcellate (4D probabilistic / binarized atlases)."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.probseg import ProbSegParcellate  # noqa: E402


def _write(tmp_path, atlas, mask, data, names):
    aff = np.eye(4)
    nb.Nifti1Image(atlas, aff).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(mask, aff).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(data, aff).to_filename(tmp_path / 'data.nii.gz')
    pd.DataFrame({'index': range(1, len(names) + 1), 'name': names}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )


def _run(tmp_path, **kwargs):
    return ProbSegParcellate(
        data=str(tmp_path / 'data.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        mask=str(tmp_path / 'mask.nii.gz'),
        **kwargs,
    ).run()


def test_weighted_mean_within_mask_matches_brute_force(tmp_path, monkeypatch):
    """Value is sum(w*d)/sum(w) over voxels inside the brain mask only."""
    rng = np.random.default_rng(0)
    shape = (6, 6, 6)
    atlas = rng.random(shape + (3,)).astype('float32')
    data = (rng.random(shape + (5,)) * 10).astype('float32')
    mask = np.zeros(shape, 'uint8')
    mask[1:5, 1:5, 1:5] = 1
    _write(tmp_path, atlas, mask, data, ['a', 'b', 'c'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.0)
    got = pd.read_table(res.outputs.timeseries)

    inside = mask > 0
    for i, name in enumerate(['a', 'b', 'c']):
        w = atlas[..., i][inside]
        for t in range(5):
            d = data[..., t][inside]
            assert got[name][t] == pytest.approx(float((w * d).sum() / w.sum()), rel=1e-5)


def test_binarize_equals_plain_mean_within_parcel_and_mask(tmp_path, monkeypatch):
    """With binarize=True the value is the plain mean over (parcel n mask).

    This is exactly what a per-volume NiftiLabelsMasker returns for a binary label
    image -- verified numerically during design (5.254389 vs 5.254390).
    """
    rng = np.random.default_rng(1)
    shape = (6, 6, 6)
    atlas = rng.random(shape + (2,)).astype('float32')
    data = (rng.random(shape + (1,)) * 10).astype('float32')
    mask = np.zeros(shape, 'uint8')
    mask[1:5, 1:5, 1:5] = 1
    _write(tmp_path, atlas, mask, data, ['a', 'b'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, binarize=True, min_coverage=0.0)
    got = pd.read_table(res.outputs.timeseries)

    inside = mask > 0
    for i, name in enumerate(['a', 'b']):
        sel = (atlas[..., i] > 0) & inside
        assert got[name][0] == pytest.approx(float(data[..., 0][sel].mean()), rel=1e-5)


def test_coverage_is_masked_weight_over_total_weight(tmp_path, monkeypatch):
    """Coverage is sum(w*mask)/sum(w) -- from the atlas and mask, never the data."""
    shape = (4, 4, 4)
    atlas = np.zeros(shape + (1,), 'float32')
    atlas[:, :, :, 0] = 1.0  # total weight 64
    mask = np.zeros(shape, 'uint8')
    mask[0:2, :, :] = 1  # 32 of 64 -> coverage 0.5
    data = np.full(shape + (1,), 3.0, 'float32')
    data[3, :, :, :] = np.nan  # outside the mask; must not affect coverage
    _write(tmp_path, atlas, mask, data, ['a'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.0)
    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']
    assert coverage['a'] == pytest.approx(0.5)


def test_parcels_below_min_coverage_are_nan(tmp_path, monkeypatch):
    """A parcel under min_coverage is NaN in the timeseries but keeps its coverage."""
    shape = (4, 4, 4)
    atlas = np.zeros(shape + (2,), 'float32')
    atlas[0:2, :, :, 0] = 1.0  # fully inside the mask -> coverage 1.0
    atlas[2:4, :, :, 1] = 1.0  # fully outside the mask -> coverage 0.0
    mask = np.zeros(shape, 'uint8')
    mask[0:2, :, :] = 1
    data = np.full(shape + (1,), 2.0, 'float32')
    _write(tmp_path, atlas, mask, data, ['keep', 'drop'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.5)
    ts = pd.read_table(res.outputs.timeseries)
    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']

    assert list(ts.columns) == ['keep', 'drop']
    assert ts['keep'][0] == pytest.approx(2.0)
    assert np.isnan(ts['drop'][0])
    assert coverage['drop'] == pytest.approx(0.0)


def test_scalar_input_produces_one_row(tmp_path, monkeypatch):
    """A 3D scalar is the n_timepoints == 1 case -- a 1-row wide TSV."""
    shape = (4, 4, 4)
    atlas = np.ones(shape + (1,), 'float32')
    mask = np.ones(shape, 'uint8')
    data = np.full(shape, 7.0, 'float32')  # 3D, no time axis
    _write(tmp_path, atlas, mask, data, ['a'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, min_coverage=0.0)
    ts = pd.read_table(res.outputs.timeseries)
    assert ts.shape == (1, 1)
    assert ts['a'][0] == pytest.approx(7.0)


def test_volume_count_must_match_label_count(tmp_path, monkeypatch):
    """A labels table that disagrees with the atlas volume count is a hard error."""
    shape = (4, 4, 4)
    atlas = np.ones(shape + (2,), 'float32')
    mask = np.ones(shape, 'uint8')
    data = np.ones(shape + (1,), 'float32')
    _write(tmp_path, atlas, mask, data, ['only_one'])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match='2 volumes.*1 label'):
        _run(tmp_path)
