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
"""Parcellation of grayordinate data by a *probabilistic* (dscalar) atlas.

Workbench cannot do this at all: ``wb_command -cifti-parcellate`` takes a dlabel
and rejects a dscalar with "input cifti label file has the wrong mapping types".
So the weighted statistics are computed here, on the grayordinate arrays --  the
grayordinate counterpart of :class:`~bdt.interfaces.probseg.ProbSegParcellate`,
with the CIFTI vertex mask playing the brain mask's role.

Because a probabilistic parcellation has no crisp per-parcel membership, these
nodes write **tables only** -- there is no honest ``ParcelsAxis`` to hang a
ptseries/pscalar on.  See :func:`bdt.outputs.plan.build_sink_plan`.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
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


class _CiftiProbSegParcellateInputSpec(BaseInterfaceInputSpec):
    data = File(exists=True, mandatory=True, desc='dtseries or dscalar to parcellate')
    atlas = File(exists=True, mandatory=True, desc='dscalar probseg: one map per region')
    atlas_labels = File(exists=True, mandatory=True, desc='BIDS tsv with index/name')
    vertex_mask = File(exists=True, mandatory=True, desc='dscalar, 1 where the data has data')
    statistics = traits.List(
        traits.Str, value=['mean'], usedefault=True, desc='weighted statistics to compute'
    )
    tidy = traits.Bool(
        False,
        usedefault=True,
        desc=(
            'Emit ONE tidy table (a row per parcel, a column per statistic) instead '
            'of one wide table per statistic. Set for a scalar, whose single row per '
            'parcel leaves room for statistics as columns; a series cannot, its table '
            'already being timepoints x parcels.'
        ),
    )
    min_coverage = traits.Float(
        0.5, usedefault=True, desc='parcels below this coverage are NaN'
    )


class _CiftiProbSegParcellateOutputSpec(TraitedSpec):
    out_files = traits.List(File(exists=True), desc='one table per statistic, in request order')
    tsv = File(desc='the single tidy table, when `tidy` is set')
    coverage = File(exists=True, desc='parcel-wise coverage table')


class CiftiProbSegParcellate(SimpleInterface):
    """Weighted per-parcel statistics of grayordinate data.

    For parcel *p* with weights ``w`` over the grayordinates the vertex mask marks
    as covered, the mean is ``sum(w*d)/sum(w)`` and the standard deviation the
    matching weighted *population* SD ``sqrt(sum(w*(d-mu)^2)/sum(w))`` -- the same
    definitions :class:`~bdt.interfaces.probseg.ProbSegParcellate` uses for volumes,
    so the two modalities agree.

    Coverage is ``sum(w over covered grayordinates) / sum(w over all)``: the share
    of a parcel's probability mass that the data actually covers.  Note this is
    *data-derived*, like the rest of the CIFTI path and unlike the volumetric one
    (which uses the brain mask) -- see :func:`_init_parcellate_cifti_wf`.
    """

    input_spec = _CiftiProbSegParcellateInputSpec
    output_spec = _CiftiProbSegParcellateOutputSpec

    def _run_interface(self, runtime):
        statistics = list(self.inputs.statistics)
        undefined = [s for s in statistics if s not in WEIGHTED_STATISTICS]
        if undefined:
            raise ValueError(
                f'Atlas {self.inputs.atlas} is probabilistic, where only '
                f'{", ".join(WEIGHTED_STATISTICS)} have a weighted definition, but '
                f'{", ".join(undefined)} was requested. Either request only the '
                f'weighted statistics or supply a dlabel atlas.'
            )

        atlas_img = nb.load(self.inputs.atlas)
        data_img = nb.load(self.inputs.data)
        mask_img = nb.load(self.inputs.vertex_mask)

        weights = np.asarray(atlas_img.get_fdata(), dtype='float64')  # (n_parcels, n_gray)
        data = np.atleast_2d(np.asarray(data_img.get_fdata(), dtype='float64'))  # (n_t, n_gray)
        covered = np.asarray(mask_img.get_fdata(), dtype='float64')[0] > 0  # (n_gray,)

        n_parcels, n_gray = weights.shape
        if data.shape[1] != n_gray or covered.size != n_gray:
            raise ValueError(
                f'Grayordinate counts disagree: atlas {self.inputs.atlas} has {n_gray}, '
                f'data {self.inputs.data} has {data.shape[1]}, vertex mask '
                f'{self.inputs.vertex_mask} has {covered.size}. They must be the same '
                'dense space (same template and density).'
            )

        names = self._names(n_parcels)

        total_weight = weights.sum(axis=1)
        inside = weights[:, covered]
        weight = inside.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(total_weight > 0, weight / total_weight, 0.0)

        usable = (weight > 0) & (coverage >= self.inputs.min_coverage)
        n_dropped = int((~usable).sum())
        if n_dropped:
            LOGGER.warning(
                '%d/%d parcels fall below min_coverage=%.2f and are set to NaN.',
                n_dropped, n_parcels, self.inputs.min_coverage,
            )

        values = self._statistics(statistics, inside, data[:, covered], weight, usable)

        self._results['coverage'] = os.path.join(runtime.cwd, 'coverage.tsv')
        pd.DataFrame(coverage.astype(np.float32), index=names, columns=['coverage']).to_csv(
            self._results['coverage'], sep='\t', na_rep='n/a', index_label='Node'
        )

        if self.inputs.tidy:
            table = {'node': names}
            table.update({stat: values[stat][0] for stat in statistics})
            self._results['tsv'] = os.path.join(runtime.cwd, 'parcellated.tsv')
            pd.DataFrame(table).to_csv(
                self._results['tsv'], sep='\t', na_rep='n/a', index=False
            )
            self._results['out_files'] = []
            return runtime

        self._results['out_files'] = []
        for stat in statistics:
            path = os.path.join(runtime.cwd, f'parcellated_{stat}.tsv')
            pd.DataFrame(values[stat], columns=names).to_csv(
                path, sep='\t', na_rep='n/a', index=False
            )
            self._results['out_files'].append(path)
        return runtime

    def _names(self, n_parcels):
        labels_df = pd.read_table(self.inputs.atlas_labels).sort_values(by='index')
        names = labels_df['name'].astype(str).tolist()
        if len(names) != n_parcels:
            raise ValueError(
                f'Atlas {self.inputs.atlas} has {n_parcels} maps but '
                f'{self.inputs.atlas_labels} has {len(names)} labels; they must agree '
                '(a dscalar map maps to the 1-based index in the labels table).'
            )
        return names

    def _statistics(self, statistics, weights, data, weight, usable):
        """``{stat: (n_t, n_parcels)}``, NaN wherever ``usable`` is False."""
        n_t = data.shape[0]
        values = {s: np.full((n_t, len(weight)), np.nan, dtype='float64') for s in statistics}
        if not usable.any():
            return values

        w = weights[usable]  # (n_usable, n_gray_covered)
        # (n_usable, n_gray) @ (n_gray, n_t) -> (n_usable, n_t)
        means = (w @ data.T) / weight[usable][:, None]
        if 'mean' in values:
            values['mean'][:, usable] = means.T
        if 'standard_deviation' in values:
            # E[d^2] - mu^2 would be cheaper but cancels catastrophically on data
            # with a large offset (BOLD is ~10000), so the deviations are formed
            # explicitly, one parcel at a time to keep the (n_parcels, n_t, n_gray)
            # intermediate off the heap.
            sd = np.empty_like(means)
            for row in range(w.shape[0]):
                deviation = data - means[row][:, None]  # (n_t, n_gray)
                sd[row] = np.sqrt((w[row] * deviation**2).sum(axis=1) / weight[usable][row])
            values['standard_deviation'][:, usable] = sd.T
        return values
