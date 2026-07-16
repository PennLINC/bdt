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
"""CIFTI intent helpers + a CIFTI array writer.

``wb_command`` occasionally writes a CIFTI with the wrong NIfTI intent code (a
known ``-cifti-smooth`` quirk); :class:`bdt.interfaces.workbench.FixCiftiIntent`
uses this table to set the correct intent for a given CIFTI extension.  The
names are nibabel's canonical CIFTI-2 intent labels.

:func:`write_ndata` writes a numpy array back to a CIFTI, taking its
brainordinate/parcel axis from a template CIFTI — the numeric counterpart of the
``wb_command`` interfaces, used by the coverage/mask steps in
``init_parcellate_timeseries_wf``.  (Ported from xcp_d; CIFTI-only — the NIfTI
branch of the original is intentionally omitted.)
"""

from __future__ import annotations

# CIFTI extension -> nibabel NIfTI intent name.
CIFTI_INTENTS: dict[str, str] = {
    '.dtseries.nii': 'ConnDenseSeries',
    '.dscalar.nii': 'ConnDenseScalar',
    '.dlabel.nii': 'ConnDenseLabel',
    '.ptseries.nii': 'ConnParcelSries',
    '.pscalar.nii': 'ConnParcelScalr',
    '.pconn.nii': 'ConnParcels',
    '.dconn.nii': 'ConnDense',
    '.pdconn.nii': 'ConnParcelDense',
    '.dpconn.nii': 'ConnDenseParcel',
    '.pconnseries.nii': 'ConnPPSr',
    '.pconnscalar.nii': 'ConnPPSc',
}


def get_cifti_intents() -> dict[str, str]:
    """Return the CIFTI-extension -> NIfTI-intent-name mapping."""
    return dict(CIFTI_INTENTS)


def write_ndata(data_matrix, template: str, filename: str, TR: float = 1):
    """Write a ``(S, T)`` array to a CIFTI file, borrowing axes from ``template``.

    ``data_matrix`` is grayordinates/parcels × timepoints (a 1-D array is treated
    as a single map).  ``template`` supplies the second (brain-model / parcels)
    axis and, for a same-length series, the full header.  Supports ``.dscalar.nii``
    / ``.pscalar.nii`` (scalar axis) and ``.dtseries.nii`` / ``.ptseries.nii``
    (series axis).  Returns ``filename``.
    """
    import nibabel as nb
    import numpy as np

    from bdt.utils.filemanip import split_filename

    if data_matrix.ndim not in (1, 2):
        raise ValueError(f'Input data must be a 1-2D array, not {data_matrix.ndim}.')

    cifti_intents = get_cifti_intents()
    _, _, template_extension = split_filename(template)
    if template_extension not in cifti_intents:
        raise ValueError(f'write_ndata only supports CIFTI templates, got {template!r}')

    # transpose from (S, T) to (T, S)
    data_matrix = np.asarray(data_matrix).T
    template_img = nb.load(template)
    if data_matrix.ndim == 1:
        data_matrix = data_matrix[None, :]
    n_volumes = data_matrix.shape[0]
    _, _, out_extension = split_filename(filename)

    if filename.endswith(('.dscalar.nii', '.pscalar.nii')):
        # (ScalarAxis, BrainModelAxis|ParcelsAxis)
        ax_0 = nb.cifti2.cifti2_axes.ScalarAxis(name=[f'#{i + 1}' for i in range(n_volumes)])
        ax_1 = template_img.header.get_axis(1)
        img = nb.Cifti2Image(data_matrix, nb.Cifti2Header.from_axes((ax_0, ax_1)))
    elif filename.endswith(('.dtseries.nii', '.ptseries.nii')):
        # (SeriesAxis, BrainModelAxis|ParcelsAxis)
        if n_volumes == template_img.shape[0]:
            img = nb.Cifti2Image(
                dataobj=data_matrix,
                header=template_img.header,
                file_map=template_img.file_map,
                nifti_header=template_img.nifti_header,
            )
        else:
            ax_1 = template_img.header.get_axis(1)
            ax_0 = nb.cifti2.SeriesAxis(start=0, step=TR, size=n_volumes)
            img = nb.Cifti2Image(data_matrix, nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1)))
    else:
        raise ValueError(f"Unsupported CIFTI extension '{out_extension}'")

    img.nifti_header.set_intent(cifti_intents[out_extension])
    img.to_filename(filename)
    return filename
