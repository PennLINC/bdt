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
"""Value objects passed between engine nodes and to action builders."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeResult:
    """One concrete result produced by a node (a node may produce several via fan-out).

    ``files`` is the produced/selected file(s); for a selection match or a
    single-file transform it has one element, for a grouped input (e.g. a whole
    ``surfaces`` set) it may have several.  ``entities`` are short-name BIDS
    entities describing the result; ``sources`` are ``bids:`` URIs for provenance.
    """

    node: str
    action: str
    fmt: str
    files: list[str]
    entities: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    scope: str = 'participant'
    space: str | None = None
    dataset: str | None = None  # selection nodes only: the --datasets key read from


@dataclass
class RoleValue:
    """A single resolved value for one input role of a processing node.

    A fan-out role contributes one :class:`RoleValue` per upstream result; a group
    role collapses all its upstream results into a single :class:`RoleValue` whose
    ``files`` concatenates the group.  ``members`` keeps each file paired with its
    own entities, so a group consumer (e.g. routing L/R white/pial/midthickness
    ``surfaces``) can still tell the grouped files apart.
    """

    files: list[str]
    entities: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    space: str | None = None
    members: list[tuple[str, dict]] = field(default_factory=list)  # (file, its entities)

    @classmethod
    def from_result(cls, result: NodeResult) -> RoleValue:
        return cls(
            files=list(result.files),
            entities=dict(result.entities),
            sources=list(result.sources),
            space=result.space,
            members=[(f, dict(result.entities)) for f in result.files],
        )

    @classmethod
    def group(cls, results: list[NodeResult]) -> RoleValue:
        files: list[str] = []
        sources: list[str] = []
        members: list[tuple[str, dict]] = []
        for r in results:
            files.extend(r.files)
            sources.extend(r.sources)
            members.extend((f, dict(r.entities)) for f in r.files)
        first = results[0]
        return cls(
            files=files,
            entities=dict(first.entities),
            sources=sources,
            space=first.space,
            members=members,
        )


@dataclass
class BuildContext:
    """Ambient services an action builder may need.

    ``transform_graph`` is a :class:`bdt.transforms.graph.TransformGraph` (or
    ``None`` in pure-spec tests); ``work_dir`` is where intermediates go.
    """

    transform_graph: object = None
    work_dir: str | None = None
