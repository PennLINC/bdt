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
"""The per-parcel statistic vocabulary, shared by the plan and the factories.

Statistic names are nilearn's ``strategy`` vocabulary verbatim, so the volumetric
path passes them straight to ``NiftiLabelsMasker`` and only the grayordinate path
needs a lookup table.  :data:`SUPPORTED_STATISTICS` is exactly nilearn's set of
seven; every one of them has a Workbench ``-method`` equivalent, so both backends
support the whole vocabulary.

Probabilistic (4D) atlases are the exception — see
:data:`WEIGHTED_STATISTICS`.
"""

from __future__ import annotations

import re

#: Statistics a spec may request, in nilearn's ``strategy`` vocabulary.  This is
#: exactly the set ``NiftiLabelsMasker`` accepts (verified against nilearn 0.14.0,
#: which rejects anything else with "'strategy' must be one of …").
#:
#: ``standard_deviation`` and ``variance`` are *population* moments (ddof=0) —
#: what ``numpy.std()``/``numpy.var()``, nilearn and Workbench's ``STDEV``/
#: ``VARIANCE`` all return.  Workbench's ``SAMPSTDEV`` (ddof=1) has no nilearn
#: counterpart and is deliberately not offered, so the two backends agree.
SUPPORTED_STATISTICS = (
    'mean',
    'median',
    'sum',
    'minimum',
    'maximum',
    'standard_deviation',
    'variance',
)

#: statistic -> ``wb_command -cifti-parcellate -method`` for the grayordinate path.
#: Every supported statistic maps; the names simply differ from nilearn's.
WORKBENCH_METHOD = {
    'mean': 'MEAN',
    'median': 'MEDIAN',
    'sum': 'SUM',
    'minimum': 'MIN',
    'maximum': 'MAX',
    'standard_deviation': 'STDEV',
    'variance': 'VARIANCE',
}

#: Statistics whose Workbench method refuses ``-cifti-weights``::
#:
#:     ERROR: weighted reduction not supported for 'MIN' method
#:
#: Measured against wb_command by trying every method in :data:`WORKBENCH_METHOD`;
#: MEAN/MEDIAN/SUM/STDEV/VARIANCE all accept weights, MIN and MAX do not (they are
#: selections, so there is nothing for a weight to scale).  The grayordinate path
#: therefore NaN-masks the data for these two and relies on ``-only-numeric``
#: instead — see :func:`~bdt.engine.factories._init_parcellate_cifti_wf`.
WORKBENCH_UNWEIGHTED = ('minimum', 'maximum')

#: What a *probabilistic* (4D, non-binarized) atlas can express.  Weighting is
#: intrinsic to such an atlas, and only the weighted moments have an agreed
#: definition — a "weighted median"/"weighted minimum" would be an invention, and
#: a weighted sum is just an unnormalized mean.  Requesting anything else with a
#: 4D atlas is an error rather than a silent substitution.
WEIGHTED_STATISTICS = ('mean', 'standard_deviation')

_ENTITY_ILLEGAL = re.compile(r'[^A-Za-z0-9+]')


def parse_statistics(parameters: dict) -> list[str]:
    """The statistics a node requests, in the order requested.

    Defaults to ``['mean']``.  A bare string is accepted for the common
    single-statistic case.  Order is preserved because it sets the column order of
    the tidy table.
    """
    requested = (parameters or {}).get('statistics', ['mean'])
    if isinstance(requested, str):
        requested = [requested]
    # A bare `statistics:` key in YAML parses to None, which .get()'s default cannot
    # catch; treat it as the empty request it looks like rather than a TypeError.
    requested = list(requested or [])

    if not requested:
        raise ValueError(
            'statistics must name at least one statistic; supported: '
            f'{", ".join(SUPPORTED_STATISTICS)}.'
        )
    unsupported = [s for s in requested if s not in SUPPORTED_STATISTICS]
    if unsupported:
        raise ValueError(
            f'Unsupported statistic(s) {", ".join(map(str, unsupported))}. '
            f'Supported: {", ".join(SUPPORTED_STATISTICS)}.'
        )
    if len(set(requested)) != len(requested):
        raise ValueError(
            f'statistics contains duplicate entries ({", ".join(requested)}); each '
            'statistic names one output, so repeats would collide.'
        )
    return requested


def normalize_statistic(name: str) -> str:
    """A statistic name as a BIDS entity value.

    Entity values are alphanumeric, plus ``+`` which BIDS allows and which we use
    as the composition separator — so ``standard_deviation`` becomes
    ``standarddeviation``.  The readable spelling is kept for TSV column headers.
    """
    return _ENTITY_ILLEGAL.sub('', str(name))


def compose_statistic_entity(source: str | None, statistic: str) -> str:
    """The ``statistic`` entity for a product, source statistic first.

    A parcellated ALFF map is both an ALFF map and a mean, and the filename should
    say so: ``stat-alff`` + ``mean`` -> ``stat-alff+mean``.  With no source
    statistic (e.g. CBF) the result is just the parcellation statistic.
    """
    statistic = normalize_statistic(statistic)
    if not source:
        return statistic
    return f'{normalize_statistic(source)}+{statistic}'
