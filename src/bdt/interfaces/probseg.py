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
"""Parcellation of 4D (probabilistic or binarized) volumetric atlases.

XCP-D's :class:`~bdt.interfaces.connectivity.NiftiParcellate` covers 3D integer
label atlases via ``NiftiLabelsMasker``, which cannot represent the *overlapping*
parcels a 4D atlas encodes (one volume per region).  This module is the 4D
counterpart, sharing XCP-D's coverage definition and output format.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from bdt.utils.statistics import WEIGHTED_STATISTICS

LOGGER = logging.getLogger('nipype.interface')


class _ProbSegParcellateInputSpec(BaseInterfaceInputSpec):
    data = File(
        exists=True,
        mandatory=True,
        desc='3D scalar or 4D timeseries NIfTI, already on the atlas grid',
    )
    atlas = File(
        exists=True, mandatory=True, desc='4D atlas: one volume per region (pseg or dseg)'
    )
    atlas_labels = File(exists=True, mandatory=True, desc='BIDS dseg.tsv with index/name columns')
    mask = File(exists=True, mandatory=True, desc='brain mask on the data grid')
    binarize = traits.Bool(
        False,
        usedefault=True,
        desc=(
            'Binarize each atlas volume before averaging.  Set for a thresholded '
            '(dseg) 4D atlas, where the weighted mean reduces to the plain mean '
            'over the parcel -- identical to a per-volume NiftiLabelsMasker.'
        ),
    )
    min_coverage = traits.Float(
        0.5,
        usedefault=True,
        desc='Parcels with coverage below this are replaced with NaN.',
    )
    statistics = traits.List(
        traits.Str,
        value=['mean'],
        usedefault=True,
        desc=(
            'Weighted statistics to compute, in request order; one wide table each. '
            'Only the weighted definitions apply -- see WEIGHTED_STATISTICS.'
        ),
    )


class _ProbSegParcellateOutputSpec(TraitedSpec):
    out_files = traits.List(File(exists=True), desc='one table per statistic, in request order')
    coverage = File(exists=True, desc='Parcel-wise coverage file.')


class ProbSegParcellate(SimpleInterface):
    """Mask-restricted weighted statistics of a 4D atlas' regions.

    The mean is ``sum(w * d) / sum(w)`` over voxels inside the brain mask, where
    ``w`` is the region's probability map, and the standard deviation the matching
    weighted *population* SD ``sqrt(sum(w * (d - mu)^2) / sum(w))`` -- the same
    definitions :class:`~bdt.interfaces.cifti_probseg.CiftiProbSegParcellate` uses
    for grayordinates, so the two modalities agree.  ``coverage = sum(w * mask) / sum(w)``, taken
    from the atlas and mask alone -- the data never enters it, on the assumption
    (see the design doc) that the brain mask already excludes NaN and
    zero-variance voxels.

    ``NiftiMapsMasker`` is deliberately not used: its extraction is a least-squares
    unmixing that returns ``sum(w*d)/sum(w**2)`` even one map at a time, which is a
    scaling coefficient rather than a mean.
    """

    input_spec = _ProbSegParcellateInputSpec
    output_spec = _ProbSegParcellateOutputSpec

    def _run_interface(self, runtime):
        atlas_img = nb.load(self.inputs.atlas)
        if atlas_img.ndim != 4:
            raise ValueError(
                f'ProbSegParcellate expects a 4D atlas, got {atlas_img.ndim}D: {self.inputs.atlas}'
            )
        n_parcels = atlas_img.shape[3]

        labels_df = pd.read_table(self.inputs.atlas_labels).sort_values(by='index')
        names = labels_df['name'].astype(str).tolist()
        if len(names) != n_parcels:
            raise ValueError(
                f'Atlas {self.inputs.atlas} has {n_parcels} volumes but '
                f'{self.inputs.atlas_labels} has {len(names)} labels; they must agree '
                '(a 4D atlas volume maps to the 1-based index in the labels table).'
            )

        # standardize=None: standardize=False raises a FutureWarning on nilearn 0.14.
        masker = NiftiMasker(mask_img=self.inputs.mask, standardize=None)
        weights = np.atleast_2d(masker.fit_transform(self.inputs.atlas))  # (n_parcels, n_vox)
        data = np.atleast_2d(masker.transform(self.inputs.data))  # (n_t, n_vox)

        total = np.asarray(atlas_img.dataobj, dtype='float64')
        total = total.reshape(-1, n_parcels)
        if self.inputs.binarize:
            weights = (weights > 0).astype('float64')
            total = (total > 0).astype('float64')
        total_weight = total.sum(axis=0)

        covered = weights.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(total_weight > 0, covered / total_weight, 0.0)

        statistics = list(self.inputs.statistics)
        undefined = [s for s in statistics if s not in WEIGHTED_STATISTICS]
        if undefined:
            raise ValueError(
                f'Atlas {self.inputs.atlas} is 4D (probabilistic), where only '
                f'{", ".join(WEIGHTED_STATISTICS)} have a weighted definition, but '
                f'{", ".join(undefined)} was requested. Either request only the '
                f'weighted statistics or supply a 3D label atlas.'
            )

        n_t = data.shape[0]
        values = {s: np.full((n_t, n_parcels), np.nan, dtype='float64') for s in statistics}
        usable = (covered > 0) & (coverage >= self.inputs.min_coverage)
        if usable.any():
            w = weights[usable]
            # (n_parcels, n_vox) @ (n_vox, n_t) -> (n_parcels, n_t), then transpose
            means = (w @ data.T) / covered[usable][:, None]
            if 'mean' in values:
                values['mean'][:, usable] = means.T
            if 'standard_deviation' in values:
                # deviations formed explicitly rather than via E[d^2] - mu^2, which
                # cancels catastrophically on data with a large offset (BOLD ~10000);
                # one parcel at a time keeps the (n_parcels, n_t, n_vox) array off the heap
                sd = np.empty_like(means)
                for row in range(w.shape[0]):
                    deviation = data - means[row][:, None]
                    sd[row] = np.sqrt((w[row] * deviation**2).sum(axis=1) / covered[usable][row])
                values['standard_deviation'][:, usable] = sd.T

        n_dropped = int((~usable).sum())
        if n_dropped:
            LOGGER.warning(
                '%d/%d parcels fall below min_coverage=%.2f and are set to NaN.',
                n_dropped,
                n_parcels,
                self.inputs.min_coverage,
            )

        self._results['out_files'] = []
        for stat in statistics:
            path = os.path.join(runtime.cwd, f'parcellated_{stat}.tsv')
            pd.DataFrame(values[stat], columns=names).to_csv(
                path, sep='\t', na_rep='n/a', index=False
            )
            self._results['out_files'].append(path)

        self._results['coverage'] = os.path.join(runtime.cwd, 'coverage.tsv')
        pd.DataFrame(coverage.astype(np.float32), index=names, columns=['coverage']).to_csv(
            self._results['coverage'], sep='\t', na_rep='n/a', index_label='Node'
        )

        return runtime
