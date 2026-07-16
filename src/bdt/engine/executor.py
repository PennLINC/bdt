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
"""The node-graph DAG executor.

Validates the spec, then evaluates the ``dataset:`` scope once and the ``nodes:``
scope per subject, in dependency order.  Each node evaluates to a list of
:class:`~bdt.engine.result.NodeResult` (one per fan-out branch); nodes that set
``write_outputs`` are materialized through the :mod:`bdt.outputs` sink.  This is
the conceptual analogue of QSIRecon's ``init_dwi_recon_workflow`` wiring loop,
but resolves inputs by *declared role* and runs the graph directly rather than
assembling a nipype workflow.
"""

from __future__ import annotations

from bdt.engine.builders import passthrough_builder
from bdt.engine.result import BuildContext, NodeResult
from bdt.engine.selection import DataProvider, build_selection
from bdt.outputs.provenance import build_sidecar, generated_by
from bdt.outputs.sink import DerivativeSink
from bdt.spec.model import Node, Spec
from bdt.spec.validate import validate_spec


class Executor:
    """Runs a validated :class:`~bdt.spec.model.Spec` over a data provider + sink."""

    def __init__(
        self,
        spec: Spec,
        provider: DataProvider,
        sink: DerivativeSink,
        datasets: set[str] | None = None,
        context: BuildContext | None = None,
        builders: dict | None = None,
    ):
        validate_spec(spec, datasets)
        self.spec = spec
        self.provider = provider
        self.sink = sink
        self.ctx = context or BuildContext()
        # Per-action builders; any action without one falls back to the passthrough.
        # Defaults to all-passthrough so the engine runs without external tools.
        self.builders = dict(builders or {})

    def run(self, subjects: list[str] | None = None):
        """Run the ``dataset:`` scope once, then ``nodes:`` for each subject.

        With ``subjects=None`` the participant scope runs once with no subject
        filter (useful in tests / single-subject fixtures) and its result dict is
        returned directly; otherwise a ``{subject: results}`` mapping is returned.
        """
        dataset_results = self.run_dataset()
        if subjects is None:
            return self.run_participant(dataset_results, None)
        return {sub: self.run_participant(dataset_results, sub) for sub in subjects}

    def run_dataset(self) -> dict[str, list[NodeResult]]:
        """Evaluate and materialize the ``dataset:`` nodes (outputs -> ``tpl-``)."""
        return self._eval_scope(self.spec.dataset, {}, subject=None)

    def run_participant(
        self, dataset_results: dict[str, list[NodeResult]], subject: str | None
    ) -> dict[str, list[NodeResult]]:
        """Evaluate and materialize the ``nodes:`` scope for one subject."""
        return self._eval_scope(self.spec.nodes, dict(dataset_results), subject=subject)

    # -- internals --------------------------------------------------------

    def _eval_scope(
        self, nodes: list[Node], base: dict[str, list[NodeResult]], subject: str | None
    ) -> dict[str, list[NodeResult]]:
        results = dict(base)
        for node in _topo_order(nodes, set(base)):
            node_results = self._eval_node(node, results, subject)
            results[node.name] = node_results
            if node.write_outputs:
                self._materialize(node, node_results)
        return results

    def _eval_node(
        self, node: Node, results: dict[str, list[NodeResult]], subject: str | None
    ) -> list[NodeResult]:
        if node.is_selection:
            return build_selection(node, self.provider, subject)
        resolved = {
            role: [r for name in names for r in results[name]]
            for role, names in node.inputs.items()
        }
        builder = self.builders.get(node.action, passthrough_builder)
        return builder(self.ctx, node, resolved)

    def _materialize(self, node: Node, node_results: list[NodeResult]) -> None:
        out = node.action_spec.out
        if out is None:
            raise ValueError(
                f'Node {node.name!r} has write_outputs but action {node.action!r} '
                f'declares no output spec.'
            )
        gb = [generated_by(node.name, node.action, node.parameters)]
        for result in node_results:
            if not result.files:
                raise ValueError(f'Node {node.name!r} produced no file to materialize.')
            self.sink.write(
                node_name=node.name,
                in_file=result.files[0],
                entities=result.entities,
                suffix=out.suffix,
                extension=out.extension,
                datatype=out.datatype,
                scope=node.scope,
                sidecar=build_sidecar(result.sources, gb),
            )


def _topo_order(nodes: list[Node], available: set[str]) -> list[Node]:
    """Order ``nodes`` so intra-scope dependencies precede dependents.

    References to names already in ``available`` (e.g. dataset-scope results seen
    by participant nodes) count as satisfied.  The spec is validated acyclic
    before this runs, so a stall would be an internal error.
    """
    names = {n.name for n in nodes}
    placed = set(available)
    remaining = list(nodes)
    ordered: list[Node] = []
    while remaining:
        progressed = False
        for node in list(remaining):
            deps = [d for d in node.input_nodes if d in names]
            if all(d in placed for d in deps):
                ordered.append(node)
                placed.add(node.name)
                remaining.remove(node)
                progressed = True
        if not progressed:
            stuck = [n.name for n in remaining]
            raise RuntimeError(f'Internal error: could not order nodes (cycle?): {stuck}')
    return ordered
