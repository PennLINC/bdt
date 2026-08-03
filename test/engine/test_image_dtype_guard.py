# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Integer-typed inputs must not silently truncate (or wrap) parcel statistics."""

import pytest

pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.utils.images import as_float_img  # noqa: E402

STATISTICS = ('mean', 'median', 'sum', 'minimum', 'maximum', 'standard_deviation', 'variance')


def test_float_images_are_returned_untouched():
    """No copy for the common case -- derivatives are already float32."""
    img = nb.Nifti1Image(np.ones((2, 2, 2), 'float32'), np.eye(4))
    assert as_float_img(img) is img


def test_integer_images_are_cast():
    img = nb.Nifti1Image(np.ones((2, 2, 2), 'int16'), np.eye(4))
    assert np.issubdtype(as_float_img(img).get_data_dtype(), np.floating)


def test_a_path_is_loaded(tmp_path):
    path = tmp_path / 'x.nii.gz'
    nb.Nifti1Image(np.ones((2, 2, 2), 'int16'), np.eye(4)).to_filename(path)
    assert np.issubdtype(as_float_img(str(path)).get_data_dtype(), np.floating)


@pytest.mark.parametrize('dtype', ['int16', 'uint8', 'float32'])
def test_every_statistic_survives_the_storage_dtype(tmp_path, monkeypatch, dtype):
    """The same values stored as int16/uint8/float32 must give the same answers.

    Without the cast nilearn reduces in the input dtype: `sum` WRAPS (int16 goes
    negative, uint8 hits 0) and mean/median/SD/variance truncate toward zero.  The
    parcel is deliberately 512 voxels of 0..200 values -- large enough that the
    uint8 sum (102400) overflows many times over.
    """
    from bdt.interfaces.parcellate_stats import ParcellateScalarStatistics

    shape = (8, 8, 8)
    rng = np.random.default_rng(0)
    values = rng.integers(0, 200, size=shape)

    nb.Nifti1Image(np.ones(shape, 'int16'), np.eye(4)).to_filename(tmp_path / 'atlas.nii.gz')
    nb.Nifti1Image(np.ones(shape, 'uint8'), np.eye(4)).to_filename(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(values.astype(dtype), np.eye(4)).to_filename(tmp_path / 'scalar.nii.gz')
    pd.DataFrame({'index': [1], 'name': ['A']}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )
    monkeypatch.chdir(tmp_path)

    result = ParcellateScalarStatistics(
        scalar=str(tmp_path / 'scalar.nii.gz'),
        atlas=str(tmp_path / 'atlas.nii.gz'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        mask=str(tmp_path / 'mask.nii.gz'),
        statistics=list(STATISTICS),
        min_coverage=0.0,
    ).run()
    got = pd.read_table(result.outputs.out_file).iloc[0]

    # brute force, in float, independent of the implementation
    truth = {
        'mean': values.mean(),
        'median': np.median(values),
        'sum': values.sum(),
        'minimum': values.min(),
        'maximum': values.max(),
        'standard_deviation': values.std(),
        'variance': values.var(),
    }
    for stat, expected in truth.items():
        assert got[stat] == pytest.approx(float(expected), rel=1e-5), (dtype, stat)
    # The sum must be far past a uint8 accumulator's range, or the uint8 case would
    # pass without the guard and this test would prove nothing.
    assert truth['sum'] > 10 * 255
