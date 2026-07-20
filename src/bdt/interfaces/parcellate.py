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
"""Coverage-aware volumetric (NIfTI) parcellation of a scalar over a label atlas.

Handles three atlas forms: a 4D dseg (per-region binary masks), a 4D pseg
(per-region probabilistic weights), and a 3D dseg (integer labels).  The 4D
forms share one voxel-value-**weighted** mean per volume (a binary dseg reduces
to the plain in-mask mean); the 3D form is the per-label mean.  Per region, a
coverage fraction (region weight over voxels with valid data, out of the region's
total weight) NaN-masks regions below ``min_coverage`` — mirroring the CIFTI
parcellation path (:func:`bdt.engine.factories._init_parcellate_cifti_wf`).
"""

from __future__ import annotations

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


def _region_row(index, weight, scalar, valid, min_coverage, labels):
    """One output row for a region given its voxel weight map."""
    import numpy as np

    total = float(weight.sum())
    covered = float((weight * valid).sum())
    coverage = covered / total if total > 0 else 0.0
    if covered > 0:
        scalar0 = np.where(valid, scalar, 0.0)  # zero out non-valid (avoids NaN*0)
        mean = float((weight * scalar0).sum() / covered)
    else:
        mean = float('nan')
    if coverage < min_coverage:
        mean = float('nan')
    name = str(index)
    if labels is not None and index in labels:
        name = labels[index]
    return {'index': int(index), 'name': name, 'mean': mean, 'coverage': coverage}


def _parcellate_volumetric(scalar_path, atlas_path, out_path, min_coverage=0.5, labels=None):
    """Parcellate ``scalar_path`` over ``atlas_path`` into a tidy per-region TSV.

    ``labels`` is an optional ``{index: name}`` mapping; region names default to
    the string of the region index when absent.  Columns: ``index``, ``name``,
    ``mean``, ``coverage`` (one row per region).  A 4D atlas is weighted per
    volume (regions 1-based, matching ``EntitiesToSegTSV``); a 3D atlas is a
    per-integer-label mean.
    """
    import nibabel as nb
    import numpy as np
    import pandas as pd

    scalar = np.asarray(nb.load(scalar_path).dataobj, dtype='float64')
    atlas_img = nb.load(atlas_path)
    atlas = np.asarray(atlas_img.dataobj, dtype='float64')
    valid = np.isfinite(scalar) & (scalar != 0)

    rows = []
    if atlas_img.ndim > 3:
        for i in range(atlas.shape[3]):
            rows.append(
                _region_row(i + 1, atlas[..., i], scalar, valid, min_coverage, labels)
            )
    else:
        for value in np.unique(atlas):
            if value == 0:
                continue
            weight = (atlas == value).astype('float64')
            rows.append(_region_row(int(value), weight, scalar, valid, min_coverage, labels))

    pd.DataFrame(rows, columns=['index', 'name', 'mean', 'coverage']).to_csv(
        out_path, sep='\t', index=False
    )
    return str(out_path)


class _ParcellateVolumetricInputSpec(BaseInterfaceInputSpec):
    scalar = File(exists=True, mandatory=True, desc='scalar NIfTI in the atlas grid/space')
    atlas = File(exists=True, mandatory=True, desc='3D dseg or 4D dseg/pseg label atlas')
    atlas_labels = File(
        exists=True, desc='optional BIDS dseg.tsv (index/name) naming the regions'
    )
    min_coverage = traits.Float(0.5, usedefault=True, desc='NaN-mask regions below this')
    out_file = traits.Str('parcellated.tsv', usedefault=True, desc='output TSV filename')


class _ParcellateVolumetricOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy per-region TSV: index, name, mean, coverage')


class ParcellateVolumetric(SimpleInterface):
    """Coverage-aware volumetric parcellation of a scalar over a label atlas."""

    input_spec = _ParcellateVolumetricInputSpec
    output_spec = _ParcellateVolumetricOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        labels = None
        if self.inputs.atlas_labels:
            table = pd.read_csv(self.inputs.atlas_labels, sep='\t')
            labels = {int(i): str(n) for i, n in zip(table['index'], table['name'])}

        out_file = self.inputs.out_file
        if not os.path.isabs(out_file):
            out_file = os.path.join(runtime.cwd, out_file)
        _parcellate_volumetric(
            self.inputs.scalar, self.inputs.atlas, out_file,
            min_coverage=self.inputs.min_coverage, labels=labels,
        )
        self._results['out_file'] = out_file
        return runtime
