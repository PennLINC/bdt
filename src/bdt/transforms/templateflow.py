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
"""TemplateFlow inter-template transforms as edges for the transform graph.

TemplateFlow names a cross-template transform ``tpl-<TO>_from-<FROM>_mode-image_xfm.<ext>``
-- the ``tpl-`` entity is the *target* space and there is no ``_to-`` token, so
:func:`bdt.transforms.graph.parse_xfm_filename` (which requires ``_from-..._to-...``)
does not match them.  This module parses them into :class:`Xfm` edges that the
caller injects via ``build_transform_graph(..., extra_edges=...)``, keeping
``graph.py`` free of a templateflow dependency.

Enumeration is metadata-only (``api.ls`` reads the manifest; nothing is
downloaded).  Only the transforms on a chosen chain are materialized, lazily,
by :func:`templateflow_fetch`.
"""

from __future__ import annotations

import re
from pathlib import Path

from bdt.transforms.graph import Xfm

# ``tpl-<TO>_from-<FROM>[_mode-image]_xfm.<ext>``.  Template labels may include
# a ``+`` (e.g. ``MNIInfant+2``).  ``.h5`` composites and displacement warps only.
_TF_XFM_RE = re.compile(
    r'tpl-(?P<to>[A-Za-z0-9+]+)_from-(?P<frm>[A-Za-z0-9+]+)'
    r'(?:_mode-(?P<mode>[A-Za-z0-9]+))?'
    r'_xfm\.(?P<ext>h5|nii\.gz|nii)$'
)


def parse_tf_xfm(name: str | Path) -> Xfm | None:
    """Parse a TemplateFlow cross-template xfm filename into an :class:`Xfm`, or ``None``.

    ``to`` is the ``tpl-`` (target) entity, ``frm`` is the ``from-`` (source)
    entity.  TF cross-template transforms are nonlinear -> ``invertible=False``.
    ``mode-points`` and non-matching names return ``None``.
    """
    m = _TF_XFM_RE.search(Path(name).name)
    if m is None or m.group('mode') == 'points':
        return None
    return Xfm(
        path=str(name),
        frm=m.group('frm'),
        to=m.group('to'),
        xfm_type='composite' if m.group('ext') == 'h5' else 'warp',
        invertible=False,
        mode=m.group('mode'),
    )


def templateflow_edges(templates_fn=None, ls_fn=None) -> list[Xfm]:
    """Enumerate TemplateFlow cross-template image xfms as graph edges (no download).

    Uses ``api.ls`` (manifest only) to list ``xfm`` files for every template and
    parses their names.  ``templates_fn``/``ls_fn`` are injectable for hermetic
    tests; they default to ``templateflow.api.templates``/``api.ls``.
    """
    if templates_fn is None or ls_fn is None:
        from templateflow import api

        templates_fn = templates_fn or api.templates
        ls_fn = ls_fn or api.ls

    edges: list[Xfm] = []
    for tpl in templates_fn():
        for ext in ('.h5', '.nii.gz'):
            for path in ls_fn(tpl, suffix='xfm', extension=ext):
                xfm = parse_tf_xfm(path)
                if xfm is not None:
                    edges.append(xfm)
    return edges


def templateflow_fetch(path: str | Path, get_fn=None) -> str:
    """Ensure a transform file is on disk, returning its local path.

    Local dataset files (and already-cached TF files) exist and are returned
    unchanged.  A TF file that is missing or an empty skeleton stub -- TemplateFlow
    keeps a 0-byte placeholder on disk for every known file until ``api.get``
    downloads it, so existence alone does not mean it is materialized -- is
    (re)materialized via ``api.get``, re-deriving its query from the
    ``tpl-``/``from-`` filename.  ``get_fn`` is injectable for tests.
    """
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return str(p)
    xfm = parse_tf_xfm(p.name)
    if xfm is None:
        raise FileNotFoundError(
            f'Transform file does not exist and is not a TemplateFlow xfm: {path}'
        )
    if get_fn is None:
        from templateflow import api

        get_fn = api.get
    extension = p.name[p.name.rindex('_xfm') + len('_xfm'):]  # '.h5' / '.nii.gz'
    query = {'from': xfm.frm}
    if xfm.mode is not None:
        query['mode'] = xfm.mode  # disambiguate image vs points at the same extension
    got = get_fn(xfm.to, suffix='xfm', extension=extension, **query)
    if isinstance(got, (list, tuple)):
        got = got[0]
    return str(got)
