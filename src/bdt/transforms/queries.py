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
"""The two typed transform queries over a single :class:`TransformGraph`.

The subtlety this module encodes (plan-review "direction-convention landmine"):

* **Image resample** (Strategy A, ANTs ``ApplyTransforms``, *pull* semantics).
  To resample a ``src``-space image onto a ``dst`` grid, apply the ``from-src_to-dst``
  files following the forward path ``src -> ... -> dst``, but hand them to ANTs in
  **reverse** path order (last hop first) — the canonical fMRIPrep ordering, e.g.
  ``-t from-T1w_to-MNI -t from-boldref_to-T1w`` for ``boldref -> T1w -> MNI``.  A
  missing forward hop may be satisfied by inverting a reverse *affine* edge.

* **Point warp** (Strategy B, ``trxrs`` / ``giftirs``).  To warp coordinates
  ``src -> dst`` you pass the **opposite-named** file ``from-dst_to-src`` for each
  hop, applied in forward path order.  These files must *physically exist* — a
  displacement-field warp cannot be synthesised by inversion — so the point-warp
  query traverses the graph's reversed edges and never sets an invert flag.

The two chains are therefore order/inversion mirrors of one another, which the
tests assert directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from bdt.transforms.graph import TransformGraph, Xfm


class NoTransformPathError(RuntimeError):
    """No usable transform chain exists between two spaces for a given query."""


@dataclass(frozen=True)
class XfmStep:
    """One entry in a resolved transform chain."""

    file: str
    frm: str  # the transform file's own ``from`` space
    to: str  # the transform file's own ``to`` space
    invert: bool = False  # ANTs ``[file, 1]``; only ever True for affine image hops


def _pick(files: list[Xfm], *, invertible_only: bool = False) -> Xfm:
    """Deterministically choose one transform file for an edge."""
    candidates = [f for f in files if f.invertible] if invertible_only else list(files)
    return sorted(candidates, key=lambda f: f.path)[0]


def _hops(path: list[str]) -> list[tuple[str, str]]:
    return list(zip(path[:-1], path[1:], strict=True))


def _shortest_path(usable: nx.DiGraph, src: str, dst: str, tg: TransformGraph) -> list[str]:
    try:
        return nx.shortest_path(usable, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise NoTransformPathError(
            f'No transform chain from {src!r} to {dst!r}. '
            f'Known spaces: {tg.spaces}. BDT applies only transforms already present '
            f'in the input derivatives; it does not compute new normalizations.'
        ) from exc


def chain_for_image_resample(tg: TransformGraph, src: str, dst: str) -> list[XfmStep]:
    """Ordered ANTs ``-t`` chain to resample a ``src``-space image onto a ``dst`` grid.

    Strategy A.  Returned in ANTs application order (reverse of the forward path).
    A missing forward edge may be covered by inverting a reverse *affine* edge.
    """
    if src == dst:
        return []

    usable = nx.DiGraph()
    usable.add_nodes_from(tg.g.nodes)
    for a, b, data in tg.g.edges(data=True):
        usable.add_edge(a, b)  # image a->b via the direct from-a_to-b file
        if any(f.invertible for f in data['files']):
            usable.add_edge(b, a)  # image b->a by inverting the affine from-a_to-b

    path = _shortest_path(usable, src, dst, tg)

    steps: list[XfmStep] = []
    for u, v in _hops(path):
        if tg.has_edge(u, v):
            xfm = _pick(tg.files(u, v))
            steps.append(XfmStep(file=xfm.path, frm=xfm.frm, to=xfm.to, invert=False))
        else:  # invert a reverse affine
            xfm = _pick(tg.files(v, u), invertible_only=True)
            steps.append(XfmStep(file=xfm.path, frm=xfm.frm, to=xfm.to, invert=True))

    steps.reverse()  # ANTs applies the first-listed transform last
    return steps


def chain_for_point_warp(tg: TransformGraph, src: str, dst: str) -> list[XfmStep]:
    """Ordered chain to warp coordinates ``src -> dst`` (streamlines / surfaces).

    Strategy B.  Each hop uses the opposite-named file ``from-dst_to-src``, which
    must physically exist; returned in forward path (application) order.
    """
    if src == dst:
        return []

    # Point warp uses the graph's *reversed* edges: an edge u->v is usable iff the
    # opposite-named file (a real transform v... actually from-v-to-u? see below)
    # exists.  Concretely: to move points u->v we need file ``from-v_to-u`` = a real
    # graph edge v->u.
    usable = nx.DiGraph()
    usable.add_nodes_from(tg.g.nodes)
    for u, v in tg.g.edges:
        # real edge u->v (file from-u_to-v) enables the point hop v->u.
        usable.add_edge(v, u)

    try:
        path = _shortest_path(usable, src, dst, tg)
    except NoTransformPathError:
        # Give a sharper message when the *image* direction exists but the
        # correctly-named point-warp file does not (the invertibility guard).
        image_usable = nx.DiGraph()
        image_usable.add_edges_from(tg.g.edges)
        if _has_path(image_usable, src, dst):
            raise NoTransformPathError(
                f'Cannot warp points {src!r} -> {dst!r}: the only transform present maps '
                f'the opposite direction and is not flag-invertible (displacement fields '
                f'cannot be inverted). A point warp requires the correctly-named '
                f"'from-{dst}_to-...'-style file to physically exist. Known spaces: {tg.spaces}."
            ) from None
        raise

    steps: list[XfmStep] = []
    for u, v in _hops(path):
        xfm = _pick(tg.files(v, u))  # the opposite-named file from-v_to-u
        steps.append(XfmStep(file=xfm.path, frm=xfm.frm, to=xfm.to, invert=False))
    return steps


def _has_path(graph: nx.DiGraph, src: str, dst: str) -> bool:
    try:
        return nx.has_path(graph, src, dst)
    except nx.NodeNotFound:
        return False
