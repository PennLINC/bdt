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
"""Per-action builders and the shared input-resolution / fan-out logic.

An action builder takes ``(BuildContext, node, resolved_inputs)`` — where
``resolved_inputs`` maps each wired role to the list of upstream
:class:`~bdt.engine.result.NodeResult` — and returns the node's own list of
results.  A generic :func:`passthrough_builder` implements the fan-out/group and
entity/``desc`` composition rules so the entire graph runs end-to-end before any
real (NiWrap) tool builders exist; real builders register with
:func:`register_builder` and reuse :func:`fan_combinations` /
:func:`compose_output_entities`.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

from bdt.engine.result import BuildContext, NodeResult, RoleValue
from bdt.outputs.sink import compose_desc
from bdt.spec.model import Node

Builder = Callable[[BuildContext, Node, dict], list[NodeResult]]

BUILDERS: dict[str, Builder] = {}


def register_builder(action: str) -> Callable[[Builder], Builder]:
    """Decorator registering a builder for ``action`` (overrides the passthrough)."""

    def deco(fn: Builder) -> Builder:
        BUILDERS[action] = fn
        return fn

    return deco


def resolve_role_values(
    node: Node, resolved: dict[str, list[NodeResult]]
) -> dict[str, list[RoleValue]]:
    """Turn resolved upstream results into per-role fan/group :class:`RoleValue` lists."""
    spec = node.action_spec
    values: dict[str, list[RoleValue]] = {}
    for role in spec.roles:
        ups = resolved.get(role.name)
        if not ups:
            continue
        if role.fan_out:
            values[role.name] = [RoleValue.from_result(r) for r in ups]
        else:
            values[role.name] = [RoleValue.group(ups)]
    return values


def fan_combinations(node: Node, resolved: dict[str, list[NodeResult]]):
    """Yield ``dict[role -> RoleValue]`` for each fan-out combination of a node."""
    values = resolve_role_values(node, resolved)
    role_names = [r.name for r in node.action_spec.roles if r.name in values]
    for combo in product(*(values[name] for name in role_names)):
        yield dict(zip(role_names, combo, strict=True))


def compose_output_entities(node: Node, by_role: dict[str, RoleValue]) -> tuple[dict, RoleValue]:
    """Compose a result's output entities from one fan combination.

    Seeds from the action's ``primary_role`` (the data being transformed), overlays
    the action's fixed entities (``stat``/``den``/...), adds the ``atlas`` label
    when an ``atlas`` role is present, and prepend-composes ``desc`` as
    ``primary_desc -> action_desc -> node_desc``.  Returns ``(entities, primary)``.
    """
    spec = node.action_spec
    out = spec.out
    primary_name = out.primary_role if out and out.primary_role in by_role else next(iter(by_role))
    primary = by_role[primary_name]

    entities = {k: v for k, v in primary.entities.items() if k != 'desc'}
    fixed = dict(out.entities) if out else {}
    action_desc = fixed.pop('desc', None)
    entities.update(fixed)

    if 'atlas' in by_role:
        atlas_label = by_role['atlas'].entities.get('atlas')
        if atlas_label:
            entities['atlas'] = atlas_label

    desc = compose_desc(primary.entities.get('desc'), action_desc)
    desc = compose_desc(desc, node.desc)
    if desc:
        entities['desc'] = desc
    return entities, primary


def passthrough_builder(
    ctx: BuildContext, node: Node, resolved: dict[str, list[NodeResult]]
) -> list[NodeResult]:
    """Default builder: no real computation, but correct fan-out, naming, provenance.

    Carries the primary input's file(s) through unchanged and tags the result with
    the action's declared ``produces`` format and composed output entities.  This
    is what lets ``select -> parcellate -> connectivity -> write`` run end-to-end
    while the real tool builders are still being written.
    """
    spec = node.action_spec
    results: list[NodeResult] = []
    for by_role in fan_combinations(node, resolved):
        entities, primary = compose_output_entities(node, by_role)
        sources: list[str] = []
        for rv in by_role.values():
            sources.extend(rv.sources)
        results.append(
            NodeResult(
                node=node.name,
                action=node.action,
                fmt=spec.produces,
                files=list(primary.files),
                entities=entities,
                sources=list(dict.fromkeys(sources)),
                scope=node.scope,
                space=primary.space,
            )
        )
    return results
