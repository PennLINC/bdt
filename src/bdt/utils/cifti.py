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
"""Pure-Python CIFTI / NIfTI tabular helpers.

The numeric bits that are *not* a Connectome-Workbench call: reading a
parcellated CIFTI into a TSV, and parcellating a NIfTI with nilearn.
Dependency-light (nibabel / nilearn / pandas) and unit-testable with small
synthetic inputs.  The nipype action factories wrap these in ``Function`` nodes
/ ``SimpleInterface``\\ s.

Correlating a parcel-timeseries TSV lives in XCP-D's vendored
:class:`~bdt.interfaces.connectivity.TSVConnect`, not here.
"""

from __future__ import annotations

from pathlib import Path

_CIFTI_EXTENSIONS = (
    '.dtseries.nii',
    '.dscalar.nii',
    '.dlabel.nii',
    '.ptseries.nii',
    '.pscalar.nii',
    '.pconn.nii',
    '.dconn.nii',
)


def is_cifti(path: str | Path) -> bool:
    """Whether a path is a CIFTI file (by its compound extension)."""
    return str(path).endswith(_CIFTI_EXTENSIONS)


def cifti_to_tsv(cifti_path: str | Path, out_path: str | Path) -> str:
    """Write a parcellated CIFTI (ptseries / pscalar / pconn) to a TSV.

    Columns are the parcel names taken from the CIFTI's ``ParcelsAxis``; rows are
    the other axis (timepoints, maps, or parcels for a pconn).
    """
    import nibabel as nb
    import numpy as np
    import pandas as pd

    img = nb.load(str(cifti_path))
    data = np.asarray(img.get_fdata())
    axes = [img.header.get_axis(i) for i in range(data.ndim)]
    parc_idx = next(
        (i for i, ax in enumerate(axes) if isinstance(ax, nb.cifti2.ParcelsAxis)), None
    )
    if parc_idx is None:
        raise ValueError(f'{cifti_path} is not a parcellated CIFTI (no ParcelsAxis).')
    names = list(axes[parc_idx].name)
    if parc_idx == 0:
        data = data.T  # put parcels on the columns
    pd.DataFrame(data, columns=names).to_csv(out_path, sep='\t', index=False)
    return str(out_path)


def nifti_parcellate_to_tsv(
    data_path: str | Path, atlas_path: str | Path, out_path: str | Path
) -> str:
    """Parcellate a (3D scalar or 4D timeseries) NIfTI with a label NIfTI atlas.

    Uses nilearn's ``NiftiLabelsMasker`` to extract the mean signal per region;
    assumes the atlas and data already share a grid/space.  Columns are the atlas
    region ids.
    """
    import pandas as pd
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(labels_img=str(atlas_path), strategy='mean')
    timeseries = masker.fit_transform(str(data_path))  # (n_rows, n_regions)
    columns = [str(label) for label in masker.labels_]
    pd.DataFrame(timeseries, columns=columns).to_csv(out_path, sep='\t', index=False)
    return str(out_path)
