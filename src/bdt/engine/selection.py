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
"""Selection nodes (``select_data`` / ``select_atlases``) and the data provider.

File matching is abstracted behind a :class:`DataProvider` so the executor is
testable without pybids; the production provider (a pybids ``BIDSLayout`` query
over the ``--datasets`` roots) implements the same protocol.  A selection with
zero matches on a required role is an error; multiple matches are normal and fan
out downstream (spec Q1 / section 1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Match:
    """One file matched by a selection node."""

    path: str
    entities: dict = field(default_factory=dict)  # short-name BIDS entities


class DataProvider(Protocol):
    """Resolves a selection node's ``dataset`` + ``filters`` to matched files.

    ``subject`` narrows a participant-scope query to one subject; providers ignore
    it for datasets that are not subject-indexed (e.g. a standard-space atlas
    dataset), so a ``select_atlases`` still resolves under a participant node.
    """

    def select(
        self,
        dataset: str,
        filters: dict,
        exclude: list | None = None,
        subject: str | None = None,
    ) -> list[Match]: ...

    def relpath(self, dataset: str, path: str) -> str: ...


class DictDataProvider:
    """In-memory :class:`DataProvider` for tests and small fixtures.

    Built from ``{dataset_key: [Match, ...]}``; ``select`` returns matches whose
    entities are a superset of ``filters`` and disjoint from every ``exclude``
    clause.  Filter values may be scalars or lists (any-of); ``relpath`` returns
    the match path unchanged.
    """

    def __init__(self, data: dict[str, list[Match]]):
        self.data = data

    def select(
        self,
        dataset: str,
        filters: dict,
        exclude: list | None = None,
        subject: str | None = None,
    ) -> list[Match]:
        matches = self.data.get(dataset, [])
        out = []
        for m in matches:
            # subject-independent matches (no ``sub`` entity, e.g. atlases) always pass;
            # subject-scoped matches must belong to the requested subject.
            if subject is not None and m.entities.get('sub', subject) != subject:
                continue
            if _matches(m.entities, filters) and not any(
                _matches(m.entities, clause) for clause in (exclude or [])
            ):
                out.append(m)
        return out

    def relpath(self, dataset: str, path: str) -> str:
        return path


def _matches(entities: dict, query: dict) -> bool:
    for key, want in (query or {}).items():
        have = entities.get(key)
        if isinstance(want, (list, tuple, set)):
            if have not in {str(w) for w in want} and have not in want:
                return False
        elif str(have) != str(want):
            return False
    return True


class SelectionError(RuntimeError):
    """A selection node produced no matches for a required input."""
