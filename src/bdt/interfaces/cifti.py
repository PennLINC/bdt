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
"""CIFTI coverage/masking interfaces (numeric, not ``wb_command``).

These reproduce XCP-D's parcel-coverage handling, which
:func:`bdt.engine.factories.init_parcellate_timeseries_wf` composes with
``wb_command -cifti-parcellate``:

* :class:`CiftiVertexMask` builds a vertex-wise 0/1 coverage map (a vertex is
  *uncovered* if its series is all-zero or all-NaN).  Feeding it to
  ``-cifti-parcellate -cifti-weights`` makes the parcel mean average only the
  covered vertices, instead of diluting it with the zero-filled ones.
* :class:`CiftiMask` replaces sub-threshold parcels with NaN.

Ported from ``xcp_d.interfaces.connectivity``.
"""

from __future__ import annotations

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
)

from bdt.utils.filemanip import fname_presuffix
from bdt.utils.write_save import write_ndata


class _CiftiVertexMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='CIFTI dense series to mask.')


class _CiftiVertexMaskOutputSpec(TraitedSpec):
    mask_file = File(exists=True, desc='Vertex-wise 0/1 coverage CIFTI (dscalar).')


class CiftiVertexMask(SimpleInterface):
    """Build a vertex-wise coverage mask (1 = has data, 0 = all-zero/all-NaN)."""

    input_spec = _CiftiVertexMaskInputSpec
    output_spec = _CiftiVertexMaskOutputSpec

    def _run_interface(self, runtime):
        data_file = self.inputs.in_file
        data_arr = nb.load(data_file).get_fdata()

        # Vertices whose series is entirely zero or NaN are "uncovered".
        bad = np.where(np.all(np.logical_or(data_arr == 0, np.isnan(data_arr)), axis=0))[0]
        data_arr[:, bad] = np.nan
        vertex_weights = np.all(~np.isnan(data_arr), axis=0).astype(int)

        self._results['mask_file'] = fname_presuffix(
            data_file, suffix='.dscalar.nii', newpath=runtime.cwd, use_ext=False
        )
        write_ndata(vertex_weights, template=data_file, filename=self._results['mask_file'])
        return runtime


class _CiftiMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Parcellated CIFTI to mask.')
    mask = File(exists=True, mandatory=True, desc='0/1 mask (pscalar/dscalar) to apply.')


class _CiftiMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='CIFTI with masked-out elements set to NaN.')


class CiftiMask(SimpleInterface):
    """Replace elements where ``mask`` is 0 with NaN (parcel coverage masking)."""

    input_spec = _CiftiMaskInputSpec
    output_spec = _CiftiMaskOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        mask = self.inputs.mask

        in_supported = ('.ptseries.nii', '.pscalar.nii', '.dtseries.nii', '.dscalar.nii')
        if not in_file.endswith(in_supported):
            raise ValueError(f"Unsupported CIFTI extension for 'in_file': {in_file}")
        if not mask.endswith(('.pscalar.nii', '.dscalar.nii')):
            raise ValueError(f"Unsupported CIFTI extension for 'mask': {mask}")

        in_img = nb.load(in_file)
        mask_img = nb.load(mask)
        if in_img.shape[1] != mask_img.shape[1]:
            raise ValueError(
                f'CIFTI files have different element counts: {in_file} ({in_img.shape}) '
                f'vs {mask} ({mask_img.shape})'
            )

        in_data = in_img.get_fdata()
        keep = mask_img.get_fdata()[0, :].astype(bool)
        in_data[:, ~keep] = np.nan

        self._results['out_file'] = fname_presuffix(
            in_file, prefix='masked_', newpath=runtime.cwd, use_ext=True
        )
        write_ndata(in_data.T, template=in_file, filename=self._results['out_file'])
        return runtime
