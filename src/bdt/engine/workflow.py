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
"""Compile a validated BDT spec into a nipype ``Workflow``.

This is the nipype engine: selection nodes are resolved to files (by the pybids
provider) and injected as ``IdentityInterface`` source nodes; each processing
node becomes an ``init_<action>_wf`` sub-workflow (:mod:`bdt.engine.factories`);
and ``inputs[role]`` wires an upstream node's ``outputnode.out`` into this node's
``inputnode.<role>``.  Running the returned workflow with nipype's engine gives
File(exists) validation, content-hash caching, resume, and multiproc/cluster
plugins.

The wiring loop mirrors qsirecon's ``init_dwi_recon_workflow`` but resolves by
declared role rather than by field-name intersection.  Fan-out over multi-match
selections / list-valued roles (MapNode / iterables) is a follow-up; this handles
the single-match-per-role case.
"""

from __future__ import annotations

from bdt.spec.model import Node, Spec


def _topo_order(nodes: list[Node]) -> list[Node]:
    """Order nodes so each node's referenced inputs precede it."""
    names = {n.name for n in nodes}
    placed: set[str] = set()
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
            raise RuntimeError(f'Cannot order nodes (cycle?): {[n.name for n in remaining]}')
    return ordered


def init_bdt_wf(
    spec: Spec,
    selections: dict[str, str],
    name: str = 'bdt_wf',
    base_directory: str | None = None,
    sink_plan: dict | None = None,
):
    """Build a nipype ``Workflow`` for one scope of a validated spec.

    Parameters
    ----------
    spec
        A validated :class:`~bdt.spec.model.Spec`.
    selections
        ``{selection_node_name: file_path}`` resolved by the caller (the pybids
        provider).  Single match per selection for now.
    name
        Workflow name.
    base_directory
        Derivatives output root.  Required together with ``sink_plan`` to attach
        ``write_outputs`` sink nodes; omit both (the assembly-test path) to build
        the compute graph only.
    sink_plan
        ``{node_name: [OutputProduct, ...]}`` from
        :func:`bdt.outputs.plan.build_sink_plan`.
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe

    from bdt.engine.factories import WORKFLOW_FACTORIES

    wf = pe.Workflow(name=name)
    built: dict[str, tuple[str, object]] = {}

    for node in _topo_order(spec.all_nodes):
        if node.is_selection:
            source = pe.Node(niu.IdentityInterface(fields=['out']), name=node.name)
            if node.name in selections:
                source.inputs.out = selections[node.name]
            wf.add_nodes([source])
            built[node.name] = ('selection', source)
        else:
            factory = WORKFLOW_FACTORIES.get(node.action)
            if factory is None:
                raise NotImplementedError(
                    f'No nipype workflow factory for action {node.action!r} (node {node.name!r}).'
                )
            built[node.name] = ('processing', factory(node))

    for node in spec.all_nodes:
        if node.is_selection:
            continue
        _, downstream = built[node.name]
        for role, upstream_names in node.inputs.items():
            up_kind, up_obj = built[upstream_names[0]]  # single-match; fan-out is a follow-up
            src_field = 'out' if up_kind == 'selection' else 'outputnode.out'
            wf.connect(up_obj, src_field, downstream, f'inputnode.{role}')

    if sink_plan and base_directory is not None:
        _attach_sinks(wf, built, sink_plan, base_directory)

    return wf


def _attach_sinks(wf, built: dict, sink_plan: dict, base_directory: str) -> None:
    """Attach a sink node per planned :class:`~bdt.outputs.plan.OutputProduct`.

    A ``passthrough`` product wires the node's ``outputnode.out`` straight into a
    :class:`~bdt.interfaces.derivatives.BDTDerivativeSink`; a ``cifti_to_tsv``
    product routes it through a :class:`~bdt.interfaces.derivatives.CiftiToTsv`
    first.  Naming/scope/sidecar are fixed on each sink at build time.
    """
    from nipype.pipeline import engine as pe

    from bdt.interfaces.derivatives import BDTDerivativeSink, CiftiToTsv
    from bdt.outputs.plan import CIFTI_TO_TSV

    for node_name, products in sink_plan.items():
        entry = built.get(node_name)
        if entry is None:
            continue
        _, producer = entry
        for i, product in enumerate(products):
            sink = pe.Node(
                BDTDerivativeSink(
                    base_directory=base_directory,
                    entities=dict(product.entities),
                    suffix=product.suffix,
                    extension=product.extension,
                    datatype=product.datatype,
                    scope=product.scope,
                    node_name=node_name,
                    **({'sidecar': dict(product.sidecar)} if product.sidecar else {}),
                ),
                name=f'{node_name}_sink{i}',
                run_without_submitting=True,
            )
            src = f'outputnode.{product.source_field}'
            if product.derive == CIFTI_TO_TSV:
                convert = pe.Node(CiftiToTsv(), name=f'{node_name}_totsv{i}')
                wf.connect(producer, src, convert, 'in_file')
                wf.connect(convert, 'out_file', sink, 'in_file')
            else:
                wf.connect(producer, src, sink, 'in_file')
