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
"""Multi-statistic parcellation of a volumetric *scalar* into a tidy table.

``parcellate_timeseries`` keeps XCP-D's wide (timepoints x parcels) layout via
:class:`~bdt.interfaces.connectivity.NiftiParcellate` and
:class:`~bdt.interfaces.probseg.ProbSegParcellate`.  A parcellated *scalar* has no
time axis, so it is reported tidily instead — a row per parcel, a column per
requested statistic — matching the along-tract profile tables.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from bdt.utils.images import as_float_img
from bdt.utils.statistics import SUPPORTED_STATISTICS, WEIGHTED_STATISTICS

LOGGER = logging.getLogger('nipype.interface')


class _ParcellateScalarStatisticsInputSpec(BaseInterfaceInputSpec):
    scalar = File(exists=True, mandatory=True, desc='3D scalar on the atlas grid')
    atlas = File(exists=True, mandatory=True, desc='3D label or 4D per-region atlas')
    atlas_labels = File(exists=True, mandatory=True, desc='BIDS dseg.tsv (index/name)')
    mask = File(exists=True, mandatory=True, desc='brain mask on the scalar grid')
    statistics = traits.List(
        traits.Str,
        value=['mean'],
        usedefault=True,
        desc='per-parcel statistics to compute, in output column order',
    )
    binarize = traits.Bool(False, usedefault=True, desc='binarize each volume of a 4D atlas first')
    min_coverage = traits.Float(
        0.5, usedefault=True, desc='parcels below this coverage are NaN in every column'
    )


class _ParcellateScalarStatisticsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy TSV: node + one column per statistic')
    coverage = File(exists=True, desc='Parcel-wise coverage file.')


class ParcellateScalarStatistics(SimpleInterface):
    """Per-parcel statistics of a scalar, as a tidy table.

    A **3D** atlas delegates to ``NiftiLabelsMasker(strategy=...)``, once per
    statistic.  A **4D** atlas (one volume per region, possibly overlapping) is
    weighted: the mean is ``sum(w*d)/sum(w)`` and the standard deviation the
    matching weighted population SD ``sqrt(sum(w*(d-mu)^2)/sum(w))``, both over
    voxels inside the brain mask.

    Coverage is ``|parcel n mask| / |parcel|`` in both cases — from the atlas and
    the mask alone, never from the data.
    """

    input_spec = _ParcellateScalarStatisticsInputSpec
    output_spec = _ParcellateScalarStatisticsOutputSpec

    def _run_interface(self, runtime):
        statistics = list(self.inputs.statistics)
        unsupported = [s for s in statistics if s not in SUPPORTED_STATISTICS]
        if unsupported:
            raise ValueError(
                f'Unsupported statistic(s) {", ".join(unsupported)}. '
                f'Supported: {", ".join(SUPPORTED_STATISTICS)}.'
            )

        labels_df = pd.read_table(self.inputs.atlas_labels).sort_values(by='index')
        atlas_img = nb.load(self.inputs.atlas)

        if atlas_img.ndim == 3:
            names, values, coverage = self._label_atlas(labels_df, atlas_img, statistics)
        else:
            names, values, coverage = self._weighted_atlas(labels_df, atlas_img, statistics)

        # A parcel with no covered voxels has no data at all, so it is missing rather
        # than low -- otherwise ``min_coverage: 0`` would report a fabricated 0.0 mean
        # for it.  Matches ProbSegParcellate and the CIFTI path, where a zero-coverage
        # parcel is likewise excluded.
        usable = (coverage > 0) & (coverage >= self.inputs.min_coverage)
        n_dropped = int((~usable).sum())
        if n_dropped:
            LOGGER.warning(
                '%d/%d parcels fall below min_coverage=%.2f and are set to NaN.',
                n_dropped,
                len(names),
                self.inputs.min_coverage,
            )

        table = {'node': names}
        for stat in statistics:
            column = np.asarray(values[stat], dtype='float64')
            table[stat] = np.where(usable, column, np.nan)

        self._results['out_file'] = os.path.join(runtime.cwd, 'parcellated.tsv')
        pd.DataFrame(table).to_csv(self._results['out_file'], sep='\t', na_rep='n/a', index=False)

        self._results['coverage'] = os.path.join(runtime.cwd, 'coverage.tsv')
        pd.DataFrame(coverage.astype(np.float32), index=names, columns=['coverage']).to_csv(
            self._results['coverage'], sep='\t', na_rep='n/a', index_label='Node'
        )
        return runtime

    def _label_atlas(self, labels_df, atlas_img, statistics):
        """3D integer-label atlas -> nilearn, one masker per statistic."""
        full = labels_df[['index', 'name']].reset_index(drop=True)
        names = full['name'].astype(str).tolist()
        n_nodes = len(names)

        # The sidecar describes the *original* atlas, but the image handed to this
        # interface has been warped (nearest-neighbour) into the scalar's space, which
        # routinely loses small parcels.  nilearn silently returns one column per label
        # actually present, so the lut has to be restricted first and the results
        # scattered back into the full node list -- as NiftiParcellate does.
        atlas_values = np.unique(np.asarray(atlas_img.dataobj))
        atlas_values = atlas_values[atlas_values != 0].astype(int)
        keep = full['index'].isin(atlas_values).to_numpy()
        positions = np.flatnonzero(keep)
        if not positions.size:
            raise ValueError(
                f'None of the {n_nodes} parcels in {self.inputs.atlas_labels} are '
                f'present in {self.inputs.atlas}; the atlas is empty in this space.'
            )
        if positions.size != n_nodes:
            LOGGER.warning(
                '%d/%d parcels are absent from the atlas image (lost when the atlas '
                'was resampled) and are reported as missing.',
                n_nodes - positions.size,
                n_nodes,
            )
        lut = full.loc[keep].reset_index(drop=True)

        # Coverage from the binarized atlas alone: masked voxel count over total.
        # float32, never uint8: `strategy='sum'` accumulates in the input dtype, so a
        # uint8 image of ones wraps modulo 256 (256 voxels -> 0, 300 -> 44), silently
        # corrupting these counts.  Parcels of <=255 voxels are unaffected, so a small
        # fixture will not reproduce it -- real parcels are far larger.
        binary = nb.Nifti1Image((atlas_img.get_fdata() > 0).astype(np.float32), atlas_img.affine)
        counts = {}
        for key, mask_img in (('covered', self.inputs.mask), ('total', None)):
            masker = NiftiLabelsMasker(
                labels_img=atlas_img,
                lut=lut,
                background_label=0,
                mask_img=mask_img,
                strategy='sum',
                resampling_target=None,
                keep_masked_labels=True,
                standardize=None,
            )
            counts[key] = np.atleast_1d(np.squeeze(masker.fit_transform(binary)))
        with np.errstate(invalid='ignore', divide='ignore'):
            found = np.where(counts['total'] > 0, counts['covered'] / counts['total'], 0.0)
        coverage = np.zeros(n_nodes, dtype='float64')
        coverage[positions] = found

        values = {}
        for stat in statistics:
            masker = NiftiLabelsMasker(
                labels_img=atlas_img,
                lut=lut,
                background_label=0,
                mask_img=self.inputs.mask,
                strategy=stat,  # SUPPORTED_STATISTICS is nilearn's own vocabulary
                resampling_target=None,
                keep_masked_labels=True,
                standardize=None,
            )
            column = np.full(n_nodes, np.nan, dtype='float64')
            column[positions] = np.atleast_1d(
                # as_float_img: nilearn reduces in the input dtype, so an integer
                # scalar would truncate every statistic and wrap ``sum`` outright.
                np.squeeze(masker.fit_transform(as_float_img(self.inputs.scalar)))
            )
            values[stat] = column
        return names, values, coverage

    def _weighted_atlas(self, labels_df, atlas_img, statistics):
        """4D per-region atlas -> mask-restricted weighted statistics."""
        undefined = [s for s in statistics if s not in WEIGHTED_STATISTICS]
        if undefined:
            raise ValueError(
                f'Atlas {self.inputs.atlas} is 4D (probabilistic), where only '
                f'{", ".join(WEIGHTED_STATISTICS)} have a weighted definition, but '
                f'{", ".join(undefined)} was requested. Either request only the '
                f'weighted statistics or supply a 3D label atlas.'
            )

        n_parcels = atlas_img.shape[3]
        names = labels_df['name'].astype(str).tolist()
        if len(names) != n_parcels:
            raise ValueError(
                f'Atlas {self.inputs.atlas} has {n_parcels} volumes but '
                f'{self.inputs.atlas_labels} has {len(names)} labels; they must agree.'
            )

        masker = NiftiMasker(mask_img=self.inputs.mask, standardize=None)
        weights = np.atleast_2d(masker.fit_transform(self.inputs.atlas))
        data = np.atleast_2d(masker.transform(self.inputs.scalar))[0]

        total = np.asarray(atlas_img.dataobj, dtype='float64').reshape(-1, n_parcels)
        if self.inputs.binarize:
            weights = (weights > 0).astype('float64')
            total = (total > 0).astype('float64')
        total_weight = total.sum(axis=0)
        covered = weights.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(total_weight > 0, covered / total_weight, 0.0)

        safe = np.where(covered > 0, covered, np.nan)
        means = (weights @ data) / safe
        values = {}
        if 'mean' in statistics:
            values['mean'] = means
        if 'standard_deviation' in statistics:
            deviation = data[np.newaxis, :] - means[:, np.newaxis]
            variance = (weights * deviation**2).sum(axis=1) / safe
            values['standard_deviation'] = np.sqrt(variance)
        return names, values, coverage
