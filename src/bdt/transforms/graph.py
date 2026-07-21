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
"""Discover transform files and build the space graph.

A BIDS transform file is named ``..._from-<X>_to-<Y>[_mode-image]_xfm.<ext>`` and
means, by the BIDS/ANTs convention, "resample an *image* in space X onto a grid
in space Y".  We index those files as directed edges ``X -> Y`` and record, per
edge, whether the transform is numerically *invertible* by a flag (affine only —
displacement-field warps and composite ``.h5`` are not).  ``mode-points`` files
are skipped: point warps are derived from the image-mode files via the
opposite-named rule (see :mod:`bdt.transforms.queries`), so indexing a
point-mode file as an image edge would double-count and mirror the geometry.

BDT never computes a normalization; :func:`build_transform_graph` is a pure scan
of files already present in the ``--datasets`` derivatives (plus any injected
standard->standard edges, e.g. from TemplateFlow — passed in so this module
stays free of a templateflow dependency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

# ``from-X_to-Y[_mode-image]_xfm.ext``.  ``mode`` is optional; spaces are BIDS
# alphanumeric labels.  Extensions cover ITK composite (.h5), affine (.mat/.txt),
# and displacement-field warps (.nii/.nii.gz).
# ``(?:^|_)`` rather than a bare ``_``: a transform BDT generates itself has no
# subject prefix, so its name *starts* with ``from-`` (e.g.
# ``from-ACPC_to-T1w_mode-image_xfm.mat``).  Requiring the leading underscore made
# every such file unparsable, and unparsable bridges are dropped -- which silently
# removed ACPC from the transform graph entirely.  pybids' own ``from`` entity
# pattern is anchored the same way.
_XFM_RE = re.compile(
    r'(?:^|_)from-(?P<frm>[A-Za-z0-9]+)_to-(?P<to>[A-Za-z0-9]+)'
    r'(?:_mode-(?P<mode>[A-Za-z0-9]+))?'
    r'_xfm\.(?P<ext>h5|mat|txt|nii\.gz|nii)$'
)


def _classify(ext: str) -> tuple[str, bool]:
    """Return ``(xfm_type, invertible)`` for a transform file extension."""
    if ext in ('mat', 'txt'):
        return 'affine', True
    if ext == 'h5':
        # A composite may embed a displacement field, which cannot be flag-inverted;
        # be conservative and treat composites as non-invertible.
        return 'composite', False
    return 'warp', False  # nii / nii.gz displacement field


@dataclass(frozen=True)
class Xfm:
    """One discovered transform file, as a directed ``frm -> to`` image edge."""

    path: str
    frm: str
    to: str
    xfm_type: str  # 'affine' | 'composite' | 'warp'
    invertible: bool
    mode: str | None = None


class TransformGraph:
    """A directed graph of spaces linked by discovered transform files.

    Edge ``u -> v`` carries a list of :class:`Xfm` (the ``from-u_to-v`` files).
    Query it with :func:`bdt.transforms.queries.chain_for_image_resample` and
    :func:`bdt.transforms.queries.chain_for_point_warp`.
    """

    def __init__(self, graph: nx.DiGraph | None = None):
        self.g = graph if graph is not None else nx.DiGraph()

    def add(self, xfm: Xfm) -> None:
        if self.g.has_edge(xfm.frm, xfm.to):
            self.g[xfm.frm][xfm.to]['files'].append(xfm)
        else:
            self.g.add_edge(xfm.frm, xfm.to, files=[xfm])

    def files(self, frm: str, to: str) -> list[Xfm]:
        """The transform files for the directed edge ``frm -> to`` (possibly empty)."""
        if self.g.has_edge(frm, to):
            return list(self.g[frm][to]['files'])
        return []

    def has_edge(self, frm: str, to: str) -> bool:
        return self.g.has_edge(frm, to)

    @property
    def spaces(self) -> list[str]:
        return sorted(self.g.nodes)

    def __repr__(self) -> str:
        return f'TransformGraph(spaces={self.spaces}, edges={self.g.number_of_edges()})'


def parse_xfm_filename(path: str | Path) -> Xfm | None:
    """Parse a BIDS transform filename into an :class:`Xfm`, or ``None``.

    Returns ``None`` for names that are not BIDS ``from-/to-`` transforms, and for
    ``mode-points`` files (which are handled implicitly by the point-warp query).
    """
    path = Path(path)
    m = _XFM_RE.search(path.name)
    if m is None:
        return None
    if m.group('mode') == 'points':
        return None
    xfm_type, invertible = _classify(m.group('ext'))
    return Xfm(
        path=str(path),
        frm=m.group('frm'),
        to=m.group('to'),
        xfm_type=xfm_type,
        invertible=invertible,
        mode=m.group('mode'),
    )


def build_transform_graph(search_paths, extra_edges: list[Xfm] | None = None) -> TransformGraph:
    """Scan ``search_paths`` for BIDS transform files and build the space graph.

    Parameters
    ----------
    search_paths
        One path or an iterable of paths (the ``--datasets`` roots) to scan
        recursively for ``*_xfm.{h5,mat,txt,nii,nii.gz}`` files.
    extra_edges
        Optional pre-built :class:`Xfm` edges to inject — e.g. standard->standard
        transforms resolved from TemplateFlow by the caller.  Injecting them keeps
        this module free of a templateflow import.
    """
    if isinstance(search_paths, (str, Path)):
        search_paths = [search_paths]

    tg = TransformGraph()
    for root in search_paths:
        root = Path(root)
        if not root.exists():
            continue
        for candidate in root.rglob('*_xfm.*'):
            xfm = parse_xfm_filename(candidate)
            if xfm is not None:
                tg.add(xfm)
    for xfm in extra_edges or []:
        tg.add(xfm)
    return tg
