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

Statistic names are chosen to match nilearn's ``strategy`` vocabulary so the
volumetric path can pass them straight to ``NiftiLabelsMasker``.  ``mean`` and
``standard_deviation`` are the only ones implemented; nilearn and Workbench both
offer more, and adding one is a matter of extending
:data:`SUPPORTED_STATISTICS` plus the two backend lookup tables.
"""

from __future__ import annotations

import re

#: Statistics a spec may request.  ``standard_deviation`` is the *population* SD
#: (ddof=0) across a parcel's voxels/vertices — what ``numpy.std()``,
#: ``NiftiLabelsMasker(strategy='standard_deviation')`` and Workbench's
#: ``-method STDEV`` all return.
SUPPORTED_STATISTICS = ('mean', 'standard_deviation')

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
