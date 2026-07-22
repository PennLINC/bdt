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
"""Plan *what* each ``write_outputs`` node materializes, and *how it is named*.

Given a validated spec and the selection-node matches resolved by the provider,
this computes — with **no nipype dependency**, so it is unit-testable on its own:

* ``node_output_entities`` — the short-name BIDS entities each node's primary
  product carries, walked in topological order.  A processing node seeds its
  entities from its ``primary_role`` upstream (the "data" being transformed),
  injects ``atlas-<label>`` from any wired atlas role, layers the action's fixed
  entities (``stat-mean`` …), and prepend-composes ``desc`` (spec 1.7).
* ``build_sink_plan`` — the list of :class:`OutputProduct` s to write for each
  ``write_outputs`` node.  Per the 2026-07-16 decision, a CIFTI-valued node emits
  the native CIFTI (ptseries/pconn) *and* a flattened TSV; a NIfTI/volumetric
  node emits the TSV only.

The nipype compiler (:mod:`bdt.engine.workflow`) consumes this plan and wires a
sink node per product; it never re-derives a name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from bdt.engine.selection import Match
from bdt.outputs.provenance import bids_uri, build_sidecar, generated_by
from bdt.outputs.sink import compose_desc
from bdt.spec.model import Node, Spec
from bdt.utils.cifti import is_cifti
from bdt.utils.statistics import compose_statistic_entity, parse_statistics

# How a product's bytes are derived from a node's ``outputnode.out``.
PASSTHROUGH = 'passthrough'  # copy outputnode.out as-is (the native product)
CIFTI_TO_TSV = 'cifti_to_tsv'  # flatten a parcellated CIFTI to a TSV


@dataclass
class OutputProduct:
    """One materialized file for a ``write_outputs`` node.

    ``derive`` says how to get the bytes from the sub-workflow output
    (:data:`PASSTHROUGH` or :data:`CIFTI_TO_TSV`); ``source_field`` is the
    ``outputnode`` field it reads (``'out'`` for the primary product, e.g.
    ``'coverage'`` for a secondary one); ``entities`` is the mid-filename
    short-name entity dict (no leading token, no suffix/extension).
    """

    derive: str
    suffix: str
    extension: str
    datatype: str
    scope: str
    entities: dict = field(default_factory=dict)
    sidecar: dict | None = None
    source_field: str = 'out'


def _atlas_label(node: Node, entities_by_node: dict[str, dict]) -> str | None:
    """The ``atlas`` entity of any atlas-typed role wired into ``node``."""
    spec = node.action_spec
    if spec is None:
        return None
    for role_name, upstream in node.inputs.items():
        role = spec.role(role_name)
        if role is None or 'atlas' not in role.accepts:
            continue
        for up_name in upstream:
            up_ent = entities_by_node.get(up_name, {})
            if up_ent.get('atlas'):
                return up_ent['atlas']
    return None


def _own_atlas_label(node: Node, base: dict) -> str:
    """The ``atlas`` entity for a node that *builds* an atlas.

    An explicit ``atlas`` parameter always wins.  Failing that, the label is
    inferred from the source's **bundle-set** entity (``bundles-``), which exists
    only when every streamline came from one aggregate file — e.g.
    ``..._bundles-DSIStudio_streamlines.tck.gz`` gives ``atlas-DSIStudio``.

    When the source is one file *per bundle* (``bundle-<Name>``, the QSIRecon
    layout) there is no common set name to infer, so an explicit parameter is
    required rather than guessed: the label ends up in every downstream filename.
    """
    explicit = node.parameters.get('atlas')
    if explicit:
        return str(explicit)
    inferred = base.get('bundles')
    if inferred:
        return str(inferred)
    raise ValueError(
        f'Node {node.name!r} builds an atlas but its label cannot be inferred: the '
        "source carries no 'bundles' (bundle-set) entity, which happens when there "
        'is one file per bundle rather than a single aggregate file. Set it '
        f"explicitly, e.g.\n  - name: {node.name}\n    parameters: {{atlas: MyAtlas}}"
    )


def _primary_upstream(node: Node) -> str | None:
    """The upstream node name feeding this node's ``primary_role``."""
    spec = node.action_spec
    if spec is None or spec.out is None:
        return None
    role = spec.out.primary_role
    if role and node.inputs.get(role):
        return node.inputs[role][0]  # single-match; fan-out is a follow-up
    # Fall back to the first wired input so naming still resolves.
    for upstream in node.inputs.values():
        if upstream:
            return upstream[0]
    return None


def node_output_entities(spec: Spec, resolved: dict[str, Match]) -> dict[str, dict]:
    """Short-name entities for every node's primary product, in topological order.

    Selection nodes report their matched file's entities; processing nodes compose
    from their ``primary_role`` upstream + atlas injection + the action's fixed
    entities + ``desc`` prepend-compose.  ``suffix`` and ``datatype`` are carried
    (for propagation / ``preserve_source`` naming and to seed the output folder);
    ``extension`` is dropped (each product sets its own).  The returned dicts may
    therefore contain ``suffix``/``datatype`` — the sink takes those as explicit
    fields and :func:`bdt.outputs.sink.bids_name` ignores them in the mid-filename.
    """
    from bdt.engine.workflow import _topo_order

    entities: dict[str, dict] = {}

    for node in _topo_order(spec.all_nodes):
        if node.is_selection:
            match = resolved.get(node.name)
            entities[node.name] = dict(match.entities) if match is not None else {}
            continue

        aspec = node.action_spec
        out = aspec.out if aspec is not None else None
        primary = _primary_upstream(node)
        base = dict(entities.get(primary, {})) if primary else {}
        base.pop('extension', None)  # geometry changes; each product sets extension
        # A processing node's product is never per-hemi (surface actions combine
        # L+R into one dscalar; others are already whole-brain) — drop hemi.
        base.pop('hemi', None)

        atlas = _atlas_label(node, entities)
        if atlas is not None:
            base['atlas'] = atlas

        # A node that *produces* an atlas has none to inherit, so it must name its
        # own.  Otherwise every downstream parcellation is unlabelled and two atlases
        # in one spec collide.
        if aspec is not None and aspec.produces == 'atlas':
            base['atlas'] = _own_atlas_label(node, base)
            base.pop('bundle', None)  # a per-bundle label does not survive aggregation

        # The real product suffix may be threshold-dependent (dynamic_suffix), e.g.
        # tractogram_to_pseg -> dseg vs probseg.  Reflect it here so propagation and
        # entity-driven factory decisions (e.g. warp interpolation) see the true
        # suffix; build_sink_plan recomputes the same value for the sink name.
        eff_suffix = out.suffix if out is not None else None
        if out is not None and out.dynamic_suffix is not None:
            eff_suffix = out.dynamic_suffix(node.parameters)
        if out is not None and not out.preserve_source:
            # A new-derivative-type action fixes its own suffix/datatype + entities.
            base['suffix'] = eff_suffix
            base['datatype'] = out.datatype
            base.update(out.entities)
        elif out is not None:
            # Preserve the source's suffix/datatype (fall back to the spec's).
            base.setdefault('suffix', eff_suffix)
            base.setdefault('datatype', out.datatype)
            base.update(out.entities)

        composed = compose_desc(base.get('desc'), node.desc)
        if composed is None:
            base.pop('desc', None)
        else:
            base['desc'] = composed

        entities[node.name] = base

    return entities


def _produces_cifti(spec: Spec, resolved: dict[str, Match]) -> dict[str, bool]:
    """Whether each node's primary product is a CIFTI file.

    A selection is CIFTI by its matched file's extension; a processing node (of the
    CIFTI-preserving actions ported so far) is CIFTI iff its primary input is.
    """
    from bdt.engine.workflow import _topo_order

    flag: dict[str, bool] = {}
    for node in _topo_order(spec.all_nodes):
        if node.is_selection:
            match = resolved.get(node.name)
            flag[node.name] = bool(match and is_cifti(match.path))
        else:
            aspec = node.action_spec
            out = aspec.out if aspec is not None else None
            if out is not None and out.output_is_cifti:
                # dense CIFTI produced from a (per-hemi GIFTI) input regardless of format
                flag[node.name] = True
            else:
                flag[node.name] = flag.get(_primary_upstream(node), False)
    return flag


def _selection_leaves(node: Node, by_name: dict[str, Node]) -> list[str]:
    """Selection nodes transitively upstream of ``node`` (order-preserving)."""
    seen: dict[str, None] = {}
    stack = list(node.input_nodes)
    while stack:
        name = stack.pop(0)
        up = by_name.get(name)
        if up is None:
            continue
        if up.is_selection:
            seen.setdefault(name, None)
        else:
            stack.extend(up.input_nodes)
    return list(seen)


def _sources(
    node: Node, spec: Spec, resolved: dict[str, Match], roots: dict[str, str]
) -> list[str]:
    """``bids:``-style source URIs for a node's written output."""
    by_name = spec.by_name()
    uris: list[str] = []
    for leaf_name in _selection_leaves(node, by_name):
        leaf = by_name.get(leaf_name)
        match = resolved.get(leaf_name)
        if leaf is None or match is None:
            continue
        dataset = leaf.dataset or ''
        root = roots.get(dataset)
        rel = os.path.relpath(match.path, root) if root else os.path.basename(match.path)
        uris.append(bids_uri(dataset, rel))
    return uris


def build_sink_plan(
    spec: Spec,
    resolved: dict[str, Match],
    roots: dict[str, str] | None = None,
) -> dict[str, list[OutputProduct]]:
    """Products to materialize for each ``write_outputs`` node.

    Parameters
    ----------
    spec
        A validated :class:`~bdt.spec.model.Spec`.
    resolved
        ``{selection_node_name: Match}`` from the provider (single match per
        selection; fan-out is a follow-up).
    roots
        ``{dataset_key: root_path}`` so ``Sources`` URIs can be made relative.
    """
    roots = roots or {}
    entities_by_node = node_output_entities(spec, resolved)
    cifti_by_node = _produces_cifti(spec, resolved)
    plan: dict[str, list[OutputProduct]] = {}

    for node in spec.all_nodes:
        if node.is_selection or not node.write_outputs:
            continue
        aspec = node.action_spec
        if aspec is None or aspec.out is None:
            continue
        out = aspec.out
        node_ent = dict(entities_by_node.get(node.name, {}))
        # ``suffix``/``datatype`` were carried for naming; the sink takes them as
        # explicit fields, so pull them out of the mid-filename entity dict.
        tsv_suffix = node_ent.pop('suffix', out.suffix)
        datatype = node_ent.pop('datatype', out.datatype)
        # Threshold-aware (or otherwise parameter-driven) primary suffix.
        primary_suffix = tsv_suffix
        if out.dynamic_suffix is not None:
            primary_suffix = out.dynamic_suffix(node.parameters)
        mid = node_ent
        # For a preserved source, the native CIFTI keeps the same suffix as the TSV.
        cifti_suffix = tsv_suffix if out.preserve_source else out.cifti_suffix
        sidecar = build_sidecar(
            sources=_sources(node, spec, resolved, roots),
            generated_by_records=[generated_by(node.name, node.action, node.parameters)],
        )
        products: list[OutputProduct] = []
        common = {'datatype': datatype, 'scope': node.scope}

        if cifti_by_node.get(node.name) and cifti_suffix and out.cifti_extension:
            # One native CIFTI per statistic: a parcellated CIFTI holds a single
            # value per parcel, so several statistics need several files.  Actions
            # that do not accept ``statistics`` keep exactly one product.
            if 'statistics' in aspec.parameters:
                requested = parse_statistics(node.parameters)
            else:
                requested = []
            if requested:
                source_statistic = mid.get('statistic')
                for stat in requested:
                    stat_ent = dict(mid)
                    stat_ent['statistic'] = compose_statistic_entity(source_statistic, stat)
                    products.append(
                        OutputProduct(
                            derive=PASSTHROUGH,
                            suffix=cifti_suffix,
                            extension=out.cifti_extension,
                            entities=stat_ent,
                            sidecar=dict(sidecar),
                            source_field=f'out_{stat}',
                            **common,
                        )
                    )
            else:
                products.append(
                    OutputProduct(
                        derive=PASSTHROUGH,
                        suffix=cifti_suffix,
                        extension=out.cifti_extension,
                        entities=dict(mid),
                        sidecar=dict(sidecar),
                        **common,
                    )
                )
            # ... plus a flattened TSV for tabular (parcellated) outputs, but not for
            # a dense CIFTI (a resampled/mapped surface scalar has no table form).
            # The table holds every statistic at once, so it keeps the source's own
            # ``statistic`` entity and is read straight off the sub-workflow when the
            # action builds it there.
            if out.emit_tsv:
                products.append(
                    OutputProduct(
                        derive=PASSTHROUGH if out.tsv_source_field else CIFTI_TO_TSV,
                        suffix=tsv_suffix,
                        extension=out.extension,
                        entities=dict(mid),
                        sidecar=dict(sidecar),
                        source_field=out.tsv_source_field or 'out',
                        **common,
                    )
                )
        else:
            # Volumetric / already-tabular: the primary product only.
            products.append(
                OutputProduct(
                    derive=PASSTHROUGH,
                    suffix=primary_suffix,
                    extension=out.extension,
                    entities=dict(mid),
                    sidecar=dict(sidecar),
                    **common,
                )
            )

        # Secondary products (e.g. the parcel-coverage map) from other outputnode fields.
        for ep in out.extra:
            is_cifti_node = bool(cifti_by_node.get(node.name))
            if not is_cifti_node and ep.cifti_only and ep.volumetric_extension is None:
                continue
            ep_extension = (
                ep.extension if is_cifti_node else (ep.volumetric_extension or ep.extension)
            )
            ep_ent = dict(mid)
            if ep.stat is not None:
                ep_ent['statistic'] = ep.stat
            ep_suffix = primary_suffix if ep.match_primary_suffix else ep.suffix
            products.append(
                OutputProduct(
                    derive=PASSTHROUGH,
                    suffix=ep_suffix,
                    extension=ep_extension,
                    entities=ep_ent,
                    sidecar=dict(sidecar),
                    source_field=ep.source_field,
                    **common,
                )
            )

        plan[node.name] = products

    return plan
