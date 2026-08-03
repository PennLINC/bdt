# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tidy multi-statistic parcellation of a volumetric scalar."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.parcellate_stats import ParcellateScalarStatistics  # noqa: E402


def _write(tmp_path, atlas, mask, scalar, names):
    aff = np.eye(4)
    nb.Nifti1Image(atlas, aff).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(mask, aff).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(scalar, aff).to_filename(tmp_path / 'scalar.nii.gz')
    pd.DataFrame({'index': range(1, len(names) + 1), 'name': names}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )


def _run(tmp_path, **kwargs):
    return ParcellateScalarStatistics(
        scalar=str(tmp_path / 'scalar.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        mask=str(tmp_path / 'mask.nii.gz'),
        **kwargs,
    ).run()


def _dseg_fixture(tmp_path):
    """3D integer-label atlas: two parcels, whole volume in the brain mask."""
    shape = (6, 6, 6)
    atlas = np.zeros(shape, 'int16')
    atlas[:3] = 1
    atlas[3:] = 2
    mask = np.ones(shape, 'uint8')
    rng = np.random.default_rng(0)
    scalar = rng.random(shape).astype('float32')
    _write(tmp_path, atlas, mask, scalar, ['A', 'B'])
    return atlas, mask, scalar


def test_default_is_a_mean_column_only(tmp_path, monkeypatch):
    atlas, _, scalar = _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(_run(tmp_path, min_coverage=0.0).outputs.out_file)

    assert list(df.columns) == ['node', 'mean']
    assert df['node'].tolist() == ['A', 'B']
    assert df['mean'][0] == pytest.approx(float(scalar[atlas == 1].mean()), rel=1e-5)


def test_dseg_standard_deviation_is_the_population_sd(tmp_path, monkeypatch):
    """ddof=0, matching numpy.std() -- not the sample SD."""
    atlas, _, scalar = _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(
            tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0
        ).outputs.out_file
    )

    assert list(df.columns) == ['node', 'mean', 'standard_deviation']
    for row, value in enumerate((1, 2)):
        vals = scalar[atlas == value]
        assert df['standard_deviation'][row] == pytest.approx(float(vals.std()), rel=1e-5)
        assert df['standard_deviation'][row] != pytest.approx(float(vals.std(ddof=1)), rel=1e-9)


def test_column_order_follows_the_request(tmp_path, monkeypatch):
    _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(
            tmp_path, statistics=['standard_deviation', 'mean'], min_coverage=0.0
        ).outputs.out_file
    )
    assert list(df.columns) == ['node', 'standard_deviation', 'mean']


def test_dseg_statistics_respect_the_brain_mask(tmp_path, monkeypatch):
    """Voxels outside the mask contribute to neither statistic."""
    shape = (6, 6, 6)
    atlas = np.ones(shape, 'int16')
    mask = np.zeros(shape, 'uint8')
    mask[:3] = 1
    rng = np.random.default_rng(1)
    scalar = rng.random(shape).astype('float32')
    scalar[3:] = 1000.0  # outside the mask; would dominate if included
    _write(tmp_path, atlas, mask, scalar, ['A'])
    monkeypatch.chdir(tmp_path)

    df = pd.read_table(
        _run(
            tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0
        ).outputs.out_file
    )
    inside = scalar[:3]
    assert df['mean'][0] == pytest.approx(float(inside.mean()), rel=1e-5)
    assert df['standard_deviation'][0] == pytest.approx(float(inside.std()), rel=1e-5)


def test_low_coverage_parcels_are_nan_in_every_column(tmp_path, monkeypatch):
    shape = (6, 6, 6)
    atlas = np.zeros(shape, 'int16')
    atlas[:3] = 1  # fully inside the mask
    atlas[3:] = 2  # fully outside
    mask = np.zeros(shape, 'uint8')
    mask[:3] = 1
    scalar = np.ones(shape, 'float32')
    _write(tmp_path, atlas, mask, scalar, ['keep', 'drop'])
    monkeypatch.chdir(tmp_path)

    res = _run(tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.5)
    df = pd.read_table(res.outputs.out_file)
    assert df['node'].tolist() == ['keep', 'drop']
    assert np.isnan(df['mean'][1])
    assert np.isnan(df['standard_deviation'][1])
    assert not np.isnan(df['mean'][0])

    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']
    assert coverage['keep'] == pytest.approx(1.0)
    assert coverage['drop'] == pytest.approx(0.0)


def _pseg_fixture(tmp_path):
    """4D probabilistic atlas: two overlapping maps, mask covers part of the volume."""
    shape = (6, 6, 6)
    rng = np.random.default_rng(2)
    atlas = rng.random(shape + (2,)).astype('float32')
    mask = np.zeros(shape, 'uint8')
    mask[1:5, 1:5, 1:5] = 1
    scalar = (rng.random(shape) * 10).astype('float32')
    _write(tmp_path, atlas, mask, scalar, ['a', 'b'])
    return atlas, mask, scalar


def test_pseg_weighted_mean_and_sd_match_brute_force(tmp_path, monkeypatch):
    """Weighted population SD: sqrt(sum(w*(d-mu)^2)/sum(w)) inside the mask."""
    atlas, mask, scalar = _pseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(
            tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0
        ).outputs.out_file
    )

    inside = mask > 0
    d = scalar[inside]
    for row in (0, 1):
        w = atlas[..., row][inside]
        mu = float((w * d).sum() / w.sum())
        sd = float(np.sqrt((w * (d - mu) ** 2).sum() / w.sum()))
        assert df['mean'][row] == pytest.approx(mu, rel=1e-5)
        assert df['standard_deviation'][row] == pytest.approx(sd, rel=1e-5)


def test_pseg_binarize_matches_the_plain_masked_statistics(tmp_path, monkeypatch):
    """With binarize=True the weights are 0/1, so both statistics reduce to the
    plain mean and population SD over (parcel n mask)."""
    atlas, mask, scalar = _pseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(
        _run(
            tmp_path,
            statistics=['mean', 'standard_deviation'],
            binarize=True,
            min_coverage=0.0,
        ).outputs.out_file
    )

    inside = mask > 0
    for row in (0, 1):
        sel = (atlas[..., row] > 0) & inside
        vals = scalar[sel]
        assert df['mean'][row] == pytest.approx(float(vals.mean()), rel=1e-5)
        assert df['standard_deviation'][row] == pytest.approx(float(vals.std()), rel=1e-5)


def test_unsupported_statistic_is_rejected(tmp_path, monkeypatch):
    _dseg_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='mode'):
        _run(tmp_path, statistics=['mode'])


def test_label_count_must_match_a_4d_atlas(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    atlas = np.ones(shape + (2,), 'float32')
    mask = np.ones(shape, 'uint8')
    scalar = np.ones(shape, 'float32')
    _write(tmp_path, atlas, mask, scalar, ['only_one'])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='2 volumes.*1 label'):
        _run(tmp_path)


def test_parcels_lost_from_the_atlas_image_are_reported_as_missing(tmp_path, monkeypatch):
    """The sidecar describes the *original* atlas; the warped image may lack parcels.

    nilearn returns one column per label actually present in ``labels_img``, so a
    naive implementation misaligns (or, with three labels and two parcels, raises
    ``All arrays must be of the same length``).  The dropped parcel must land in the
    right row -- here the *middle* one, so a simple truncation cannot pass.
    """
    shape = (6, 6, 6)
    atlas = np.zeros(shape, 'int16')
    atlas[:3] = 1
    atlas[3:] = 3  # label 2 exists in the sidecar but not in the image
    mask = np.ones(shape, 'uint8')
    rng = np.random.default_rng(3)
    scalar = rng.random(shape).astype('float32')
    _write(tmp_path, atlas, mask, scalar, ['A', 'GONE', 'C'])
    monkeypatch.chdir(tmp_path)

    result = _run(tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0)
    df = pd.read_table(result.outputs.out_file)

    assert df['node'].tolist() == ['A', 'GONE', 'C']
    assert np.isnan(df['mean'][1])
    assert np.isnan(df['standard_deviation'][1])
    assert df['mean'][0] == pytest.approx(float(scalar[atlas == 1].mean()), rel=1e-5)
    assert df['mean'][2] == pytest.approx(float(scalar[atlas == 3].mean()), rel=1e-5)

    coverage = pd.read_table(result.outputs.coverage)
    assert coverage['coverage'].tolist() == [1.0, 0.0, 1.0]


def test_an_atlas_with_no_surviving_parcels_is_an_error(tmp_path, monkeypatch):
    shape = (6, 6, 6)
    _write(
        tmp_path,
        np.zeros(shape, 'int16'),
        np.ones(shape, 'uint8'),
        np.ones(shape, 'float32'),
        ['A', 'B'],
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match='None of the 2 parcels'):
        _run(tmp_path, min_coverage=0.0)


def test_zero_coverage_is_missing_even_at_min_coverage_zero(tmp_path, monkeypatch):
    """A parcel wholly outside the brain mask has no data, so it is n/a -- not 0.0.

    ``min_coverage=0.0`` would otherwise let ``coverage >= min_coverage`` through and
    publish nilearn's fabricated zero as a real mean.  ProbSegParcellate and the CIFTI
    path both exclude it; the 3D path must agree.
    """
    shape = (6, 6, 6)
    atlas = np.zeros(shape, 'int16')
    atlas[:3] = 1
    atlas[3:] = 2
    mask = np.zeros(shape, 'uint8')
    mask[:3] = 1  # parcel B is entirely outside the mask
    rng = np.random.default_rng(4)
    scalar = rng.random(shape).astype('float32')
    _write(tmp_path, atlas, mask, scalar, ['A', 'B'])
    monkeypatch.chdir(tmp_path)

    df = pd.read_table(
        _run(
            tmp_path, statistics=['mean', 'standard_deviation'], min_coverage=0.0
        ).outputs.out_file
    )

    assert df['mean'][0] == pytest.approx(float(scalar[:3].mean()), rel=1e-5)
    assert np.isnan(df['mean'][1])
    assert np.isnan(df['standard_deviation'][1])
