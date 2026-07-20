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
"""A :class:`~bdt.engine.selection.DataProvider` backed by pybids.

Indexes each ``--datasets`` root as a (possibly derivative / atlas) BIDS dataset
and answers selection queries.  BDT ships a small pybids entity config
(``data/bdt_entities.json``) registering the BEP/atlas entities pybids does not
know out of the box (``atlas``, ``stat``, ``param``, ``scale``, ``meas``,
``tract``, ``track``, ``tpl``, ...); without it those filters would silently
match nothing.  Matched files are returned with short-name BDT entities so they
compose cleanly into output filenames.
"""

from __future__ import annotations

import os
from pathlib import Path

from bdt.engine.selection import Match, _matches, _query_name

_ENTITY_CONFIG = str(Path(__file__).resolve().parent.parent / 'data' / 'bdt_entities.json')


def _as_query(value):
    """Turn a serialized ``'Query.<NAME>'`` string into the pybids ``Query`` enum so
    ``layout.get`` resolves presence filters; pass any other value through unchanged."""
    name = _query_name(value)
    if name is None or type(value).__name__ == 'Query':
        return value
    from bids.layout import Query

    return getattr(Query, name)


class BIDSDataProvider:
    """Resolve selection queries against pybids layouts of the ``--datasets`` roots."""

    def __init__(self, datasets: dict[str, str | Path], database_dir: str | Path | None = None):
        self.roots = {k: Path(v) for k, v in datasets.items()}
        self._database_dir = Path(database_dir) if database_dir else None
        self._layouts: dict[str, object] = {}

    def _layout(self, dataset: str):
        if dataset not in self._layouts:
            if dataset not in self.roots:
                raise KeyError(f'Unknown dataset key {dataset!r}; known: {sorted(self.roots)}')
            from bids import BIDSLayout

            db = None
            if self._database_dir is not None:
                db = self._database_dir / dataset
            self._layouts[dataset] = BIDSLayout(
                str(self.roots[dataset]),
                validate=False,
                config=['bids', 'derivatives', _ENTITY_CONFIG],
                database_path=str(db) if db else None,
            )
        return self._layouts[dataset]

    def select(
        self, dataset: str, filters: dict, exclude: list | None = None, subject: str | None = None
    ) -> list[Match]:
        layout = self._layout(dataset)

        # pybids resolves ``Query`` presence sentinels natively, but only as the enum
        # — a serialized ``'Query.ANY'`` string (from spec YAML) would match nothing.
        prefilter = {k: _as_query(v) for k, v in filters.items()}

        # Only narrow by subject when the dataset is actually subject-indexed;
        # standard-space atlas datasets have no subjects and must ignore it.
        if subject is not None and layout.get_subjects():
            prefilter['subject'] = subject

        matches: list[Match] = []
        want_json = str(filters.get('extension', '')).endswith('json')
        for bidsfile in layout.get(return_type='object', **prefilter):
            entities = bidsfile.get_entities()
            if not want_json and entities.get('extension') == '.json':
                continue
            if _matches(entities, filters) and not any(
                _matches(entities, clause) for clause in (exclude or [])
            ):
                matches.append(Match(path=bidsfile.path, entities=entities))
        return matches

    def relpath(self, dataset: str, path: str) -> str:
        return os.path.relpath(path, self.roots[dataset])

    def subjects(self, dataset: str) -> list[str]:
        """Subjects present in a dataset (for driving the participant loop)."""
        return list(self._layout(dataset).get_subjects())
