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
"""Fold per-statistic parcellated CIFTIs into one tidy table.

A CIFTI scalar map holds exactly one value per parcel, so several statistics need
several files.  The *table*, though, should look the same as the volumetric one —
a row per parcel, a column per statistic — so the two modalities are readable side
by side.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _PscalarsToTidyTsvInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='parcellated CIFTIs, one per statistic, ordered like ``statistics``',
    )
    statistics = traits.List(
        traits.Str, mandatory=True, desc='column name for each file, in order'
    )
    out_file = File('parcellated.tsv', usedefault=True, desc='output tidy TSV')


class _PscalarsToTidyTsvOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy TSV: node + one column per statistic')


class PscalarsToTidyTsv(SimpleInterface):
    """Merge parcellated CIFTIs into a ``node`` + one-column-per-statistic table."""

    input_spec = _PscalarsToTidyTsvInputSpec
    output_spec = _PscalarsToTidyTsvOutputSpec

    def _run_interface(self, runtime):
        in_files = list(self.inputs.in_files)
        statistics = list(self.inputs.statistics)
        if len(in_files) != len(statistics):
            raise ValueError(
                f'Got {len(in_files)} file(s) for {len(statistics)} statistic(s); '
                'each statistic needs exactly one parcellated CIFTI.'
            )

        table = {}
        names = None
        for path, stat in zip(in_files, statistics, strict=True):
            parcels, values = _read_pscalar(path)
            if names is None:
                names = parcels
            elif parcels != names:
                raise ValueError(
                    f'{path} describes different parcels than the first input; every '
                    'statistic must cover the same parcels in the same order.'
                )
            table[stat] = values

        out_file = os.path.abspath(self.inputs.out_file)
        pd.DataFrame({'node': names, **table}).to_csv(
            out_file, sep='\t', na_rep='n/a', index=False
        )
        self._results['out_file'] = out_file
        return runtime


def _read_pscalar(path):
    """``(parcel names, values)`` from a parcellated CIFTI."""
    img = nb.load(str(path))
    data = np.asarray(img.get_fdata())
    axes = [img.header.get_axis(i) for i in range(data.ndim)]
    parc_idx = next(
        (i for i, ax in enumerate(axes) if isinstance(ax, nb.cifti2.ParcelsAxis)), None
    )
    if parc_idx is None:
        raise ValueError(f'{path} is not a parcellated CIFTI (no ParcelsAxis).')
    names = list(axes[parc_idx].name)
    if parc_idx == 0:
        data = data.T
    return names, np.squeeze(data)
