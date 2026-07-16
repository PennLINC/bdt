# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Tests for the pure-Python parcellation / connectivity I/O helpers."""

import pytest

pytest.importorskip('nibabel')
pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.actions._io import (  # noqa: E402
    cifti_to_tsv,
    is_cifti,
    nifti_parcellate_to_tsv,
    tsv_correlation,
)


def test_is_cifti():
    assert is_cifti('x.dtseries.nii')
    assert is_cifti('x.dlabel.nii')
    assert not is_cifti('x.nii.gz')
    assert not is_cifti('x.tsv')


def _synthetic_ptseries(path, names=('P1', 'P2'), ntime=3):
    from nibabel.cifti2 import cifti2_axes as cax

    parcels = cax.ParcelsAxis(
        name=list(names),
        voxels=[np.array([[0, 0, 0]]) for _ in names],
        vertices=[{} for _ in names],
        affine=np.eye(4),
        volume_shape=(2, 2, 2),
        nvertices={},
    )
    series = cax.SeriesAxis(start=0.0, step=1.0, size=ntime)
    hdr = nb.cifti2.Cifti2Header.from_axes((series, parcels))
    data = np.arange(ntime * len(names), dtype=float).reshape(ntime, len(names))
    nb.Cifti2Image(data, hdr).to_filename(str(path))


def test_cifti_to_tsv(tmp_path):
    pt = tmp_path / 'x.ptseries.nii'
    _synthetic_ptseries(pt, names=('P1', 'P2', 'P3'), ntime=4)
    out = cifti_to_tsv(pt, tmp_path / 'x.tsv')
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['P1', 'P2', 'P3']
    assert df.shape == (4, 3)


def test_nifti_parcellate_to_tsv(tmp_path):
    # 4x4x4 grid, 5 timepoints; atlas with 2 labelled regions.
    bold = np.random.default_rng(0).random((4, 4, 4, 5))
    atlas = np.zeros((4, 4, 4), dtype=np.int16)
    atlas[:2] = 1
    atlas[2:] = 2
    affine = np.eye(4)
    bold_p = tmp_path / 'bold.nii.gz'
    atlas_p = tmp_path / 'atlas.nii.gz'
    nb.Nifti1Image(bold, affine).to_filename(str(bold_p))
    nb.Nifti1Image(atlas, affine).to_filename(str(atlas_p))

    out = nifti_parcellate_to_tsv(bold_p, atlas_p, tmp_path / 'parc.tsv')
    df = pd.read_csv(out, sep='\t')
    assert df.shape == (5, 2)  # 5 timepoints x 2 regions
    assert set(df.columns) == {'1', '2'}


def test_tsv_correlation(tmp_path):
    ts = tmp_path / 'ts.tsv'
    rng = np.random.default_rng(1)
    pd.DataFrame(rng.random((20, 3)), columns=['a', 'b', 'c']).to_csv(ts, sep='\t', index=False)
    out = tsv_correlation(ts, tmp_path / 'relmat.tsv')
    mat = pd.read_csv(out, sep='\t')
    assert mat.shape == (3, 3)
    # diagonal correlations are 1
    assert np.allclose(np.diag(mat.to_numpy()), 1.0)
