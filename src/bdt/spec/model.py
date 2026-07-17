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
"""Dataclasses for a parsed BDT spec, plus shape-level parsing.

A spec is a mapping with up to two node lists — ``dataset:`` (run once) and
``nodes:`` (per participant).  This module turns raw dicts into typed
:class:`Node` / :class:`Spec` objects and raises :class:`SpecError` on *shape*
problems (wrong types, missing ``name``/``action``).  All *semantic* rules
(reference resolution, acyclicity, role/format contracts, scope lineage) live in
:mod:`bdt.spec.validate`, which operates on the objects built here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bdt.spec.actions import ACTIONS, PROCESSING, SELECTION, ActionSpec

PARTICIPANT = 'participant'
DATASET = 'dataset'


class SpecError(ValueError):
    """Raised for structurally malformed specs (bad shapes, missing keys)."""


@dataclass
class Node:
    """One node in a BDT spec (selection or processing).

    ``scope`` records which list the node came from (``'participant'`` for
    ``nodes:``, ``'dataset'`` for ``dataset:``); it is *placement*, distinct from
    the subject-independence of the node's lineage, which the validator checks.
    ``inputs`` is normalised so every role maps to a *list* of upstream node
    names (a single ``str`` in the YAML becomes a one-element list), which keeps
    fan-out uniform downstream.
    """

    name: str
    action: str
    scope: str = PARTICIPANT
    desc: str | None = None
    write_outputs: bool = False
    # selection-only
    dataset: str | None = None
    filters: dict = field(default_factory=dict)
    exclude: list = field(default_factory=list)
    # processing-only
    inputs: dict[str, list[str]] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    @property
    def action_spec(self) -> ActionSpec | None:
        """The declared contract for this node's action, or ``None`` if unknown."""
        return ACTIONS.get(self.action)

    @property
    def is_selection(self) -> bool:
        spec = self.action_spec
        if spec is not None:
            return spec.kind == SELECTION
        # Unknown action: fall back to key shape so callers can still reason.
        return self.dataset is not None and not self.inputs

    @property
    def is_processing(self) -> bool:
        spec = self.action_spec
        if spec is not None:
            return spec.kind == PROCESSING
        return bool(self.inputs) and self.dataset is None

    @property
    def input_nodes(self) -> list[str]:
        """Flat list of upstream node names this node references (dedup-preserving)."""
        seen: dict[str, None] = {}
        for names in self.inputs.values():
            for n in names:
                seen.setdefault(n, None)
        return list(seen)


@dataclass
class Spec:
    """A parsed spec: an ordered dataset-scope list and a participant-scope list."""

    dataset: list[Node] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)

    @property
    def all_nodes(self) -> list[Node]:
        return [*self.dataset, *self.nodes]

    def by_name(self) -> dict[str, Node]:
        return {n.name: n for n in self.all_nodes}


def _require_str(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f'{what} must be a non-empty string, got {value!r}')
    return value


def _normalize_inputs(raw, node_name: str) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SpecError(
            f"Node {node_name!r}: 'inputs' must be a mapping, got {type(raw).__name__}"
        )
    out: dict[str, list[str]] = {}
    for role, value in raw.items():
        if isinstance(value, str):
            names = [value]
        elif isinstance(value, (list, tuple)):
            names = list(value)
        else:
            raise SpecError(
                f'Node {node_name!r}: input role {role!r} must be a node name or list of '
                f'node names, got {type(value).__name__}'
            )
        for n in names:
            _require_str(n, f'Node {node_name!r} input role {role!r} value')
        out[str(role)] = names
    return out


def parse_node(raw: dict, scope: str) -> Node:
    """Build a :class:`Node` from one raw spec entry (shape checks only)."""
    if not isinstance(raw, dict):
        raise SpecError(f'Each node must be a mapping, got {type(raw).__name__}: {raw!r}')
    name = _require_str(raw.get('name'), "Node 'name'")
    action = _require_str(raw.get('action'), f"Node {name!r} 'action'")

    filters = raw.get('filters') or {}
    if not isinstance(filters, dict):
        raise SpecError(f"Node {name!r}: 'filters' must be a mapping")
    exclude = raw.get('exclude') or []
    if not isinstance(exclude, list):
        raise SpecError(f"Node {name!r}: 'exclude' must be a list")
    parameters = raw.get('parameters') or {}
    if not isinstance(parameters, dict):
        raise SpecError(f"Node {name!r}: 'parameters' must be a mapping")
    desc = raw.get('desc')
    if desc is not None:
        _require_str(desc, f"Node {name!r} 'desc'")

    return Node(
        name=name,
        action=action,
        scope=scope,
        desc=desc,
        write_outputs=bool(raw.get('write_outputs', False)),
        dataset=raw.get('dataset'),
        filters=filters,
        exclude=exclude,
        inputs=_normalize_inputs(raw.get('inputs'), name),
        parameters=parameters,
    )


def parse_spec(doc: dict) -> Spec:
    """Build a :class:`Spec` from a raw mapping (as loaded from YAML/JSON)."""
    if not isinstance(doc, dict):
        raise SpecError(f'A spec must be a mapping with dataset:/nodes:, got {type(doc).__name__}')
    unknown = set(doc) - {'dataset', 'nodes'}
    if unknown:
        raise SpecError(
            f'Unknown top-level spec keys: {sorted(unknown)} (allowed: dataset, nodes)'
        )
    dataset_raw = doc.get('dataset') or []
    nodes_raw = doc.get('nodes') or []
    for what, block in (('dataset', dataset_raw), ('nodes', nodes_raw)):
        if not isinstance(block, list):
            raise SpecError(f"Top-level '{what}' must be a list of nodes")
    spec = Spec(
        dataset=[parse_node(n, DATASET) for n in dataset_raw],
        nodes=[parse_node(n, PARTICIPANT) for n in nodes_raw],
    )
    if not spec.all_nodes:
        raise SpecError('Spec is empty: at least one of dataset:/nodes: must contain a node')
    return spec
