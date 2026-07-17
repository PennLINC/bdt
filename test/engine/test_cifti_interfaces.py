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
"""Unit tests for the CIFTI coverage interfaces + write_ndata (synthetic CIFTI)."""

import pytest

pytest.importorskip('nibabel')
pytest.importorskip('nipype')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
from nibabel.cifti2 import cifti2_axes as cax  # noqa: E402


def _dense_dtseries(path, data):
    """A dense .dtseries.nii with a left-cortex BrainModelAxis. data is (T, V)."""
    ntime, nvert = data.shape
    bm = cax.BrainModelAxis.from_surface(
        np.arange(nvert), nvert, name='CIFTI_STRUCTURE_CORTEX_LEFT'
    )
    series = cax.SeriesAxis(start=0.0, step=1.0, size=ntime)
    hdr = nb.cifti2.Cifti2Header.from_axes((series, bm))
    nb.Cifti2Image(np.asarray(data, dtype=float), hdr).to_filename(str(path))


def _ptseries(path, data, names):
    """A .ptseries.nii (or pscalar via a 1-row data) over parcels ``names``."""
    parcels = cax.ParcelsAxis(
        name=list(names),
        voxels=[np.array([[0, 0, 0]]) for _ in names],
        vertices=[{} for _ in names],
        affine=np.eye(4),
        volume_shape=(2, 2, 2),
        nvertices={},
    )
    series = cax.SeriesAxis(start=0.0, step=1.0, size=data.shape[0])
    hdr = nb.cifti2.Cifti2Header.from_axes((series, parcels))
    nb.Cifti2Image(np.asarray(data, dtype=float), hdr).to_filename(str(path))


def test_write_ndata_roundtrip_dscalar(tmp_path):
    from bdt.utils.write_save import write_ndata

    template = tmp_path / 'template.dtseries.nii'
    _dense_dtseries(template, np.ones((4, 6)))
    values = np.arange(6, dtype=float)  # one value per grayordinate
    out = tmp_path / 'scalar.dscalar.nii'
    write_ndata(values, template=str(template), filename=str(out))

    img = nb.load(str(out))
    assert img.shape == (1, 6)
    assert np.allclose(img.get_fdata()[0], values)


def test_cifti_vertex_mask_flags_uncovered(tmp_path):
    from bdt.interfaces.cifti import CiftiVertexMask

    # 6 vertices, 4 timepoints; vertices 1 and 4 are all-zero (uncovered).
    data = np.ones((4, 6))
    data[:, 1] = 0
    data[:, 4] = 0
    dts = tmp_path / 'bold.dtseries.nii'
    _dense_dtseries(dts, data)

    res = CiftiVertexMask(in_file=str(dts)).run()
    mask = nb.load(res.outputs.mask_file).get_fdata()[0]
    assert list(mask) == [1, 0, 1, 1, 0, 1]


def test_cifti_mask_sets_subthreshold_to_nan(tmp_path):
    from bdt.interfaces.cifti import CiftiMask

    pt = tmp_path / 'parc.ptseries.nii'
    _ptseries(pt, np.arange(9, dtype=float).reshape(3, 3), names=('A', 'B', 'C'))
    mask = tmp_path / 'mask.pscalar.nii'
    _ptseries(mask, np.array([[1.0, 0.0, 1.0]]), names=('A', 'B', 'C'))  # drop parcel B

    res = CiftiMask(in_file=str(pt), mask=str(mask)).run()
    out = nb.load(res.outputs.out_file).get_fdata()
    assert np.all(np.isnan(out[:, 1]))  # masked parcel -> NaN
    assert not np.any(np.isnan(out[:, [0, 2]]))  # kept parcels intact
