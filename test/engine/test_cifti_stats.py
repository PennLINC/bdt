# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Merging per-statistic parcellated CIFTIs into one tidy table."""

import pytest

pytest.importorskip('nibabel')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.cifti_stats import PscalarsToTidyTsv  # noqa: E402


def _pscalar(path, values, names=('P1', 'P2', 'P3')):
    """A minimal parcellated CIFTI: one map over ``names`` parcels."""
    from nibabel.cifti2 import cifti2_axes as cax

    parcels = cax.ParcelsAxis(
        name=list(names),
        voxels=[np.array([[0, 0, 0]]) for _ in names],
        vertices=[{} for _ in names],
        affine=np.eye(4),
        volume_shape=(2, 2, 2),
        nvertices={},
    )
    scalars = cax.ScalarAxis(name=['x'])
    hdr = nb.cifti2.Cifti2Header.from_axes((scalars, parcels))
    nb.Cifti2Image(np.asarray(values, dtype=float)[None, :], hdr).to_filename(str(path))
    return str(path)


def test_columns_are_node_plus_each_statistic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, 2.0, 3.0])
    b = _pscalar(tmp_path / 'sd.pscalar.nii', [0.1, 0.2, 0.3])

    res = PscalarsToTidyTsv(
        in_files=[a, b], statistics=['mean', 'standard_deviation']
    ).run()
    df = pd.read_table(res.outputs.out_file)

    assert list(df.columns) == ['node', 'mean', 'standard_deviation']
    assert df['node'].tolist() == ['P1', 'P2', 'P3']
    assert df['mean'].tolist() == [1.0, 2.0, 3.0]
    assert df['standard_deviation'].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_single_statistic_gives_one_value_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [4.0, 5.0, 6.0])
    res = PscalarsToTidyTsv(in_files=[a], statistics=['mean']).run()
    df = pd.read_table(res.outputs.out_file)
    assert list(df.columns) == ['node', 'mean']


def test_nan_parcels_round_trip_as_na(tmp_path, monkeypatch):
    """Low-coverage parcels are NaN upstream and must stay missing, not become 0."""
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, np.nan, 3.0])
    res = PscalarsToTidyTsv(in_files=[a], statistics=['mean']).run()
    df = pd.read_table(res.outputs.out_file)
    assert np.isnan(df['mean'][1])
    assert 'n/a' in (tmp_path / 'parcellated.tsv').read_text()


def test_file_and_statistic_counts_must_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='1 file.*2 statistic'):
        PscalarsToTidyTsv(
            in_files=[a], statistics=['mean', 'standard_deviation']
        ).run()


def test_mismatched_parcel_names_are_rejected(tmp_path, monkeypatch):
    """Every statistic must describe the same parcels, in the same order."""
    monkeypatch.chdir(tmp_path)
    a = _pscalar(tmp_path / 'mean.pscalar.nii', [1.0, 2.0, 3.0])
    b = _pscalar(tmp_path / 'sd.pscalar.nii', [1.0, 2.0], names=('P1', 'P2'))
    with pytest.raises(ValueError, match='parcel'):
        PscalarsToTidyTsv(
            in_files=[a, b], statistics=['mean', 'standard_deviation']
        ).run()
