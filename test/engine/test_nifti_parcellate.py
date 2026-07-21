# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the vendored XCP-D NiftiParcellate."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bdt.interfaces.connectivity import NiftiParcellate  # noqa: E402


def _fixture(tmp_path):
    """Two parcels of 256 voxels; the mask covers 192 of parcel 1 and 64 of parcel 2.

    Expected coverage is therefore 192/256 = 0.75 and 64/256 = 0.25.  NaNs are
    placed in the *data* to prove coverage does not depend on it.
    """
    aff = np.eye(4)
    shape = (8, 8, 8)
    atlas = np.zeros(shape, 'int16')
    atlas[0:4, :, :] = 1
    atlas[4:8, :, :] = 2
    mask = np.zeros(shape, 'uint8')
    mask[0:3, :, :] = 1
    mask[4:8, 0:2, :] = 1
    rng = np.random.default_rng(0)
    data = (rng.random(shape + (4,)) * 10).astype('float32')
    data[3, :, :, :] = np.nan

    nb.Nifti1Image(atlas, aff).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(mask, aff).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(data, aff).to_filename(tmp_path / 'data.nii.gz')
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )
    return tmp_path


@pytest.mark.filterwarnings(
    'ignore:Non-finite values detected\\. :UserWarning'
)
def test_coverage_is_mask_over_atlas(tmp_path, monkeypatch):
    """Coverage is |parcel n mask| / |parcel|, computed from the atlas alone.

    Regression: the binarized atlas was built as uint8, for which nilearn 0.14.0's
    ``strategy='sum'`` returns 0 -- the denominator was zero, coverage came out inf,
    and ``inf < min_coverage`` is False so no parcel was ever NaN-masked.

    The fixture deliberately injects NaN into the data volume (not the atlas) to
    prove that coverage is computed independently of the data; nilearn's expected
    "Non-finite values detected" UserWarning is the harmless consequence of that.
    """
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = NiftiParcellate(
        filtered_file=str(tmp_path / 'data.nii.gz'),
        mask=str(tmp_path / 'mask.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        min_coverage=0.5,
    ).run()

    coverage = pd.read_table(res.outputs.coverage, index_col='Node')['coverage']
    assert np.isfinite(coverage).all(), f'non-finite coverage: {coverage.to_dict()}'
    assert coverage['A'] == pytest.approx(0.75)
    assert coverage['B'] == pytest.approx(0.25)


@pytest.mark.filterwarnings(
    'ignore:Non-finite values detected\\. :UserWarning'
)
def test_parcels_below_min_coverage_are_nan(tmp_path, monkeypatch):
    """Parcel B (0.25 coverage) is NaN-masked at min_coverage=0.5; A (0.75) is kept.

    The fixture deliberately injects NaN into the data volume, so nilearn's
    "Non-finite values detected" UserWarning is an expected, harmless side effect.
    """
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = NiftiParcellate(
        filtered_file=str(tmp_path / 'data.nii.gz'),
        mask=str(tmp_path / 'mask.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        min_coverage=0.5,
    ).run()

    ts = pd.read_table(res.outputs.timeseries)
    assert list(ts.columns) == ['A', 'B']
    assert ts['B'].isna().all(), 'B is below min_coverage and must be NaN'
    assert ts['A'].notna().all(), 'A is above min_coverage and must be retained'
