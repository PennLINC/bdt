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
"""Static validation of a BDT spec — reject bad specs *before* execution.

Implements the checks from the user-stories spec, section 1.8: unique names;
every ``inputs:`` value resolves to a defined node; every ``dataset:`` resolves
to a ``--datasets`` key; the graph is acyclic; each node's roles match its
action's declared contract (required/optional roles + accepted input formats);
and a ``dataset:`` node's lineage is fully subject-independent.  All problems are
collected and reported together, so a user fixes a spec in one pass rather than
one error at a time.

Output *path* collisions between two ``write_outputs`` nodes are enforced at
write time by the outputs layer, since they depend on runtime-resolved entities.
"""

from __future__ import annotations

from bdt.spec.actions import PROCESSING, SELECTION, infer_selection_format
from bdt.spec.model import DATASET, Node, Spec


class SpecValidationError(ValueError):
    """Raised when a spec fails static validation; carries every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        joined = '\n'.join(f'  - {e}' for e in self.errors)
        super().__init__(f'Spec failed validation ({len(self.errors)} problem(s)):\n{joined}')


def _format_of(node: Node, by_name: dict[str, Node]) -> str:
    """The data format a node emits, for role/format checking.

    Processing nodes report their action's declared ``produces``; selection nodes
    are inferred from filters (``'unknown'`` when ambiguous).  ``'unknown'`` for
    any node whose action is unrecognised.
    """
    spec = node.action_spec
    if spec is None:
        return 'unknown'
    if spec.kind == SELECTION:
        if node.action == 'select_atlases':
            return 'atlas'
        return infer_selection_format(node.filters)
    return spec.produces


def _check_kind_keys(node: Node, errors: list[str]) -> None:
    """Selection/processing nodes may only use their own key families."""
    spec = node.action_spec
    if spec is None:
        return
    if spec.kind == SELECTION:
        if node.inputs:
            errors.append(f"Selection node {node.name!r} may not use 'inputs:'.")
        if node.dataset is None:
            errors.append(f"Selection node {node.name!r} must set 'dataset:'.")
    elif spec.kind == PROCESSING:
        if node.dataset is not None:
            errors.append(f"Processing node {node.name!r} may not use 'dataset:'.")
        if node.filters:
            errors.append(f"Processing node {node.name!r} may not use 'filters:'.")
        if node.exclude:
            errors.append(f"Processing node {node.name!r} may not use 'exclude:'.")
        if not node.inputs:
            errors.append(f'Processing node {node.name!r} must wire at least one input role.')


def _check_roles(node: Node, by_name: dict[str, Node], errors: list[str]) -> None:
    """Role names, required roles, list-valued roles, and accepted formats."""
    spec = node.action_spec
    if spec is None or spec.kind != PROCESSING:
        return
    wired = set(node.inputs)
    unknown = wired - spec.role_names
    for role in sorted(unknown):
        errors.append(
            f'Node {node.name!r} ({node.action}): unknown input role {role!r} '
            f'(allowed: {sorted(spec.role_names)}).'
        )
    for role in sorted(spec.required_roles - wired):
        errors.append(f'Node {node.name!r} ({node.action}): missing required role {role!r}.')

    for role_name, upstream in node.inputs.items():
        role = spec.role(role_name)
        if role is None:
            continue  # already reported as unknown
        if len(upstream) > 1 and not role.list_ok:
            errors.append(
                f'Node {node.name!r}: role {role_name!r} does not accept a list of nodes.'
            )
        for up_name in upstream:
            up = by_name.get(up_name)
            if up is None:
                continue  # reported by reference-resolution check
            fmt = _format_of(up, by_name)
            if fmt != 'unknown' and fmt not in role.accepts:
                errors.append(
                    f'Node {node.name!r}: role {role_name!r} accepts '
                    f'{sorted(role.accepts)} but upstream node {up_name!r} produces '
                    f'{fmt!r}.'
                )


def _check_references(node: Node, by_name: dict[str, Node], errors: list[str]) -> None:
    for up_name in node.input_nodes:
        if up_name not in by_name:
            errors.append(f'Node {node.name!r}: input references undefined node {up_name!r}.')
        elif up_name == node.name:
            errors.append(f'Node {node.name!r}: input references itself.')


def _check_acyclic(spec: Spec, by_name: dict[str, Node], errors: list[str]) -> None:
    """Kahn's algorithm over resolvable edges; any leftover nodes form a cycle."""
    indeg = {n.name: 0 for n in spec.all_nodes}
    adj: dict[str, list[str]] = {n.name: [] for n in spec.all_nodes}
    for node in spec.all_nodes:
        for up in node.input_nodes:
            if up in by_name and up != node.name:
                adj[up].append(node.name)
                indeg[node.name] += 1
    queue = [name for name, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        name = queue.pop()
        visited += 1
        for nxt in adj[name]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if visited != len(indeg):
        in_cycle = sorted(name for name, d in indeg.items() if d > 0)
        errors.append(f'Spec graph is not acyclic; nodes involved in a cycle: {in_cycle}.')


def _check_scope_lineage(node: Node, by_name: dict[str, Node], errors: list[str]) -> None:
    """A ``dataset:`` node may not reference a participant node (spec 1.2)."""
    if node.scope != DATASET:
        return
    for up_name in node.input_nodes:
        up = by_name.get(up_name)
        if up is not None and up.scope != DATASET:
            errors.append(
                f'Dataset-level node {node.name!r} references participant-level node '
                f'{up_name!r}; a dataset node must have a fully subject-independent lineage.'
            )


def validate_spec(spec: Spec, datasets: set[str] | None = None) -> None:
    """Validate ``spec`` and raise :class:`SpecValidationError` if it is invalid.

    Parameters
    ----------
    spec
        A parsed :class:`~bdt.spec.model.Spec`.
    datasets
        The set of ``--datasets`` keys available at runtime.  When provided,
        every selection node's ``dataset:`` must be one of these.  When ``None``
        (e.g. in unit tests), the key-existence check is skipped but a selection
        node must still declare a ``dataset:``.
    """
    errors: list[str] = []
    by_name = spec.by_name()

    # 1. unique names
    seen: set[str] = set()
    for node in spec.all_nodes:
        if node.name in seen:
            errors.append(f'Duplicate node name {node.name!r}.')
        seen.add(node.name)

    for node in spec.all_nodes:
        # 2. known action
        if node.action_spec is None:
            errors.append(f'Node {node.name!r}: unknown action {node.action!r}.')
            continue  # contract-dependent checks are meaningless without a spec
        # 3. selection/processing key usage
        _check_kind_keys(node, errors)
        # 4. dataset key resolves
        if node.is_selection and node.dataset is not None and datasets is not None:
            if node.dataset not in datasets:
                errors.append(
                    f'Selection node {node.name!r}: dataset {node.dataset!r} is not one of '
                    f'the provided --datasets keys {sorted(datasets)}.'
                )
        # 5. reference resolution
        _check_references(node, by_name, errors)
        # 6. roles + formats
        _check_roles(node, by_name, errors)
        # 7. scope lineage
        _check_scope_lineage(node, by_name, errors)

    # 8. acyclic (whole graph)
    _check_acyclic(spec, by_name, errors)

    if errors:
        raise SpecValidationError(errors)
