# Transform Query + Chaining Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warp an image between arbitrary spaces by resolving the shortest chain of transforms (local derivative xfms + computed bridges + TemplateFlow inter-template xfms) and applying it with `nitransforms`, inside one nipype node.

**Architecture:** The graph + chain-query layer already exists and is tested in `src/bdt/transforms/` (`graph.py`, `queries.py`). This plan adds only the two missing pieces: (1) a TemplateFlow enumeration module that turns TF cross-template xfms into `Xfm` edges for the existing graph's `extra_edges` hook, and (2) a runtime `ResolveApplyTransforms` SimpleInterface that builds the graph from injected file lists, calls the existing `chain_for_image_resample`, loads each resolved step as a `nitransforms` transform, composes them into a `TransformChain`, and resamples the moving image onto the reference grid.

**Tech Stack:** Python 3.12, nipype (`SimpleInterface`), `nitransforms` 25.1.0 (composition + resampling engine — ANTs CLI is not installed), `templateflow` 25.1.2 (`api.templates`/`api.ls`/`api.get`), the existing `bdt.transforms` package (networkx-backed).

## Global Constraints

- **Environment:** run everything via `micromamba run -n bdtenv ...` (Python 3.12.13). Never install into base; do not create environments.
- **No git commits / edit-in-place:** the user manages version control. Do **not** run `git commit`. Each task ends by running its test file, not by committing.
- **Reuse, do not rewrite:** `src/bdt/transforms/graph.py` and `queries.py` are tested and stay as-is. Do not create `utils/transforms.py`, and do not change `Xfm`, `TransformGraph`, `XfmStep`, `parse_xfm_filename`, `build_transform_graph`, or `chain_for_image_resample`.
- **Application engine is `nitransforms`** (pure Python). No `antsApplyTransforms`, no network calls in unit tests.
- **TF naming fact (verified):** TemplateFlow cross-template xfms are named `tpl-<TO>_from-<FROM>_mode-image_xfm.<ext>` — the `tpl-` entity is the **target** and there is **no** `_to-` token. The existing `parse_xfm_filename` (which requires `_from-…_to-…`) therefore does **not** match them; TF needs its own parser.
- **nitransforms facts (verified):** `linear.load(path, fmt='itk')` → `Affine` (invert via `~`); `manip.load(path, fmt='ITK')` → `TransformChain` (for `.h5` composites); `nonlinear.load(path)` → displacement warp. `TransformChain(transforms=[t0, t1, …])` applies `t0` **first** to coordinates — the same grid-space-first order `chain_for_image_resample` already returns. Resample via `nitransforms.resampling.apply(transform, moving, reference=, order=, mode='constant', cval=0.0)` where `order=0` is nearest and `order=1` is linear. Pull semantics: a `+2` x-translation moves a marker at voxel x=3 to output voxel x=1.

---

### Task 1: TemplateFlow edge enumeration + fetch helper

**Files:**
- Create: `src/bdt/transforms/templateflow.py`
- Test: `test/spec/test_templateflow_edges.py`

**Interfaces:**
- Consumes: `Xfm` from `bdt.transforms.graph` (frozen dataclass: `path, frm, to, xfm_type, invertible, mode`).
- Produces:
  - `parse_tf_xfm(name) -> Xfm | None` — parse a TF cross-template xfm filename; `to` = the `tpl-` entity, `frm` = the `from-` entity, `xfm_type='composite'` for `.h5` else `'warp'`, `invertible=False`. Returns `None` for non-matching names and `mode-points`.
  - `templateflow_edges(templates_fn=None, ls_fn=None) -> list[Xfm]` — enumerate TF `xfm` files (metadata only, no download) as edges. `templates_fn`/`ls_fn` are injectable for hermetic tests; default to `templateflow.api.templates`/`api.ls`.
  - `templateflow_fetch(path, get_fn=None) -> str` — return `path` unchanged if it exists on disk; otherwise (a TF file listed but not downloaded) materialize it via `api.get` re-deriving the query from the filename. `get_fn` injectable.

- [ ] **Step 1: Write the failing tests**

Create `test/spec/test_templateflow_edges.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for TemplateFlow edge enumeration and fetch (bdt.transforms.templateflow)."""

import pytest

from bdt.transforms.templateflow import (
    parse_tf_xfm,
    templateflow_edges,
    templateflow_fetch,
)
from bdt.transforms.graph import build_transform_graph
from bdt.transforms.queries import chain_for_image_resample


def test_parse_tf_xfm_target_is_tpl_entity():
    # TF names the TARGET with tpl- and the SOURCE with from-; there is no _to-.
    xfm = parse_tf_xfm('tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5')
    assert (xfm.frm, xfm.to) == ('MNI152NLin2009cAsym', 'MNI152NLin6Asym')
    assert xfm.xfm_type == 'composite'
    assert xfm.invertible is False


def test_parse_tf_xfm_handles_plus_in_template_label():
    xfm = parse_tf_xfm('tpl-MNIInfant+2_from-MNI152NLin6Asym_mode-image_xfm.h5')
    assert (xfm.frm, xfm.to) == ('MNI152NLin6Asym', 'MNIInfant+2')


def test_parse_tf_xfm_ignores_non_tf_and_points():
    assert parse_tf_xfm('sub-01_desc-preproc_T1w.nii.gz') is None
    assert parse_tf_xfm('sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5') is None
    assert parse_tf_xfm('tpl-A_from-B_mode-points_xfm.h5') is None


def test_templateflow_edges_injected_and_feeds_graph():
    templates_fn = lambda: ['MNI152NLin6Asym']
    ls_fn = lambda tpl, suffix, extension: (
        ['tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5']
        if extension == '.h5'
        else []
    )
    edges = templateflow_edges(templates_fn=templates_fn, ls_fn=ls_fn)
    assert [(e.frm, e.to) for e in edges] == [('MNI152NLin2009cAsym', 'MNI152NLin6Asym')]

    tg = build_transform_graph([], extra_edges=edges)
    steps = chain_for_image_resample(tg, 'MNI152NLin2009cAsym', 'MNI152NLin6Asym')
    assert [(s.frm, s.to) for s in steps] == [('MNI152NLin2009cAsym', 'MNI152NLin6Asym')]


def test_templateflow_fetch_returns_existing_unchanged(tmp_path):
    p = tmp_path / 'sub-01_from-A_to-B_mode-image_xfm.mat'
    p.write_bytes(b'')
    assert templateflow_fetch(str(p)) == str(p)


def test_templateflow_fetch_materializes_missing_tf_file(tmp_path):
    calls = {}

    def fake_get(template, **kwargs):
        calls['template'] = template
        calls['kwargs'] = kwargs
        out = tmp_path / 'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5'
        out.write_bytes(b'')
        return str(out)

    got = templateflow_fetch(
        '/does/not/exist/tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        get_fn=fake_get,
    )
    assert got.endswith('_xfm.h5')
    assert calls['template'] == 'MNI152NLin6Asym'
    assert calls['kwargs']['from'] == 'MNI152NLin2009cAsym'
    assert calls['kwargs']['extension'] == '.h5'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/spec/test_templateflow_edges.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdt.transforms.templateflow'`.

- [ ] **Step 3: Write the implementation**

Create `src/bdt/transforms/templateflow.py`:

```python
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
    unchanged.  A TF file that ``api.ls`` listed but that is not downloaded yet is
    materialized via ``api.get``, re-deriving its query from the ``tpl-``/``from-``
    filename.  ``get_fn`` is injectable for tests.
    """
    p = Path(path)
    if p.exists():
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
    got = get_fn(xfm.to, suffix='xfm', extension=extension, **{'from': xfm.frm})
    if isinstance(got, (list, tuple)):
        got = got[0]
    return str(got)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/spec/test_templateflow_edges.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Verify (no commit — edit-in-place)**

Run: `micromamba run -n bdtenv python -m pytest test/spec/test_transforms.py test/spec/test_templateflow_edges.py -q`
Expected: PASS — the existing 11 graph/query tests still pass alongside the 6 new ones. Do **not** commit.

---

### Task 2: Transform loading + chain assembly helpers

**Files:**
- Create: `src/bdt/interfaces/transforms.py` (module-level helpers only in this task; the interface class is added in Task 3)
- Test: `test/engine/test_transforms_interface.py`

**Interfaces:**
- Consumes: `XfmStep` (`file, frm, to, invert`) from `bdt.transforms.queries`; `chain_for_image_resample`, `build_transform_graph`, `Xfm` from `bdt.transforms`; `templateflow_fetch` from `bdt.transforms.templateflow`; `nitransforms`.
- Produces:
  - `_load_transform(path, invert=False) -> TransformBase` — dispatch on extension (`.mat`/`.txt` → `linear.load(fmt='itk')`; `.h5` → `manip.load(fmt='ITK')`; `.nii`/`.nii.gz` → `nonlinear.load`); `invert=True` returns `~xf` (affine only).
  - `_build_chain(steps, fetch=templateflow_fetch) -> TransformChain` — fetch each step's file, load it (honoring `invert`), flatten any composite `TransformChain` in place, and return a single `TransformChain` in the order `steps` provides (grid-space hop first).

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_transforms_interface.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for bdt.interfaces.transforms (transform loading, chain, resolve/apply)."""

import numpy as np
import pytest

pytest.importorskip('nipype')
pytest.importorskip('nitransforms')


def _itk_affine(path, translate_x):
    """Write an ITK affine that translates +translate_x in world x."""
    from nitransforms.linear import Affine

    m = np.eye(4)
    m[0, 3] = translate_x
    Affine(m).to_filename(str(path), fmt='itk')
    return str(path)


def test_load_transform_affine_and_invert(tmp_path):
    from bdt.interfaces.transforms import _load_transform

    p = _itk_affine(tmp_path / 'sub-01_from-A_to-B_mode-image_xfm.mat', 2.0)
    fwd = _load_transform(p)
    inv = _load_transform(p, invert=True)
    assert np.asarray(fwd.matrix)[0, 3] == pytest.approx(2.0)
    assert np.asarray(inv.matrix)[0, 3] == pytest.approx(-2.0)


def test_build_chain_composes_in_grid_first_order(tmp_path):
    from bdt.interfaces.transforms import _build_chain
    from bdt.transforms import build_transform_graph, chain_for_image_resample

    # A -> B (+2), B -> C (+3). The resample chain maps a C-grid point to the A
    # sampling coordinate: steps are [from-B_to-C, from-A_to-B] and TransformChain
    # applies transforms[0] first, so map(0) = (0 + 3) + 2 = +5.
    _itk_affine(tmp_path / 'sub-01_from-A_to-B_mode-image_xfm.mat', 2.0)
    _itk_affine(tmp_path / 'sub-01_from-B_to-C_mode-image_xfm.mat', 3.0)
    tg = build_transform_graph(tmp_path)
    steps = chain_for_image_resample(tg, 'A', 'C')

    chain = _build_chain(steps)
    mapped = chain.map(np.array([[0.0, 0.0, 0.0]]))
    assert mapped[0, 0] == pytest.approx(5.0)


def test_build_chain_inverts_reverse_affine(tmp_path):
    from bdt.interfaces.transforms import _build_chain
    from bdt.transforms import build_transform_graph, chain_for_image_resample

    # Only from-B_to-A (+2) exists; resampling A->B inverts that affine, so the
    # chain maps a B-grid point via ~(+2) = -2 to the A sampling coordinate.
    _itk_affine(tmp_path / 'sub-01_from-B_to-A_mode-image_xfm.mat', 2.0)
    tg = build_transform_graph(tmp_path)
    steps = chain_for_image_resample(tg, 'A', 'B')
    assert steps[0].invert is True

    chain = _build_chain(steps)
    mapped = chain.map(np.array([[0.0, 0.0, 0.0]]))
    assert mapped[0, 0] == pytest.approx(-2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdt.interfaces.transforms'`.

- [ ] **Step 3: Write the helpers**

Create `src/bdt/interfaces/transforms.py`:

```python
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
"""Resolve and apply a transform chain between two spaces with ``nitransforms``.

The graph + chain-query layer lives in :mod:`bdt.transforms`; this module loads
the resolved :class:`~bdt.transforms.queries.XfmStep` chain into ``nitransforms``
objects and resamples an image.  ``nitransforms`` (not the ANTs CLI) is the
engine: ``TransformChain(transforms=[t0, t1, ...])`` applies ``t0`` first, which
matches the grid-space-first order ``chain_for_image_resample`` returns.
"""

from __future__ import annotations


def _load_transform(path: str, invert: bool = False):
    """Load one transform file as a ``nitransforms`` object.

    Dispatches on extension: ITK affine (``.mat``/``.txt``), ITK composite
    (``.h5``), or a displacement-field warp (``.nii``/``.nii.gz``).  ``invert``
    (only ever ``True`` for affine hops) returns the exact inverse via ``~``.
    """
    from nitransforms import linear, manip, nonlinear

    lower = path.lower()
    if lower.endswith(('.mat', '.txt')):
        xf = linear.load(path, fmt='itk')
    elif lower.endswith('.h5'):
        xf = manip.load(path, fmt='ITK')
    elif lower.endswith(('.nii', '.nii.gz')):
        xf = nonlinear.load(path)
    else:
        raise ValueError(f'Unsupported transform file extension: {path}')
    return ~xf if invert else xf


def _build_chain(steps, fetch=None):
    """Assemble one ordered ``TransformChain`` from resolved ``XfmStep``s.

    ``steps`` come from ``chain_for_image_resample`` (grid-space hop first), the
    same order ``TransformChain`` applies to coordinates.  Composite files load as
    their own ``TransformChain`` and are flattened in place so the overall order
    is preserved.
    """
    from nitransforms.manip import TransformChain

    if fetch is None:
        from bdt.transforms.templateflow import templateflow_fetch as fetch

    flat = []
    for step in steps:
        xf = _load_transform(fetch(step.file), invert=step.invert)
        if isinstance(xf, TransformChain):
            flat.extend(xf.transforms)
        else:
            flat.append(xf)
    return TransformChain(transforms=flat)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Verify (no commit — edit-in-place)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py test/spec/test_transforms.py -q`
Expected: PASS. Do **not** commit.

---

### Task 3: `ResolveApplyTransforms` SimpleInterface

**Files:**
- Modify: `src/bdt/interfaces/transforms.py` (append the interface class)
- Test: `test/engine/test_transforms_interface.py` (append node tests)

**Interfaces:**
- Consumes: `_build_chain` (Task 2); `parse_xfm_filename`, `build_transform_graph` from `bdt.transforms.graph`; `chain_for_image_resample` from `bdt.transforms.queries`; `templateflow_edges` from `bdt.transforms.templateflow`; `nitransforms.resampling.apply`.
- Produces: `ResolveApplyTransforms(SimpleInterface)`.
  - Inputs: `source` (Str, mandatory), `target` (Str, mandatory), `moving` (File exists, mandatory), `reference` (File exists, mandatory), `local_transforms` (List[File], default `[]`), `bridges` (List[File], default `[]`), `interpolation` (Enum `'linear'`/`'nearest'`, default `'linear'`), `out_file` (Str, default `'resampled.nii.gz'`).
  - Outputs: `out_file` (File), `out_transforms` (List[Str]), `out_inversions` (List[Bool]).

- [ ] **Step 1: Write the failing tests**

Append to `test/engine/test_transforms_interface.py`:

```python
def _label_img(path, marker_vox, shape=(6, 6, 6), label=5):
    import nibabel as nb

    data = np.zeros(shape, dtype=np.int16)
    data[marker_vox] = label
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


def test_resolve_apply_warps_and_preserves_labels(tmp_path, monkeypatch):
    import nibabel as nb

    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms

    # No TemplateFlow involvement in this hermetic test.
    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    xfm = _itk_affine(tmp_path / 'sub-01_from-ATLAS_to-GRID_mode-image_xfm.mat', 2.0)
    moving = _label_img(tmp_path / 'atlas.nii.gz', (3, 3, 3))
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    res = ResolveApplyTransforms(
        source='ATLAS',
        target='GRID',
        moving=moving,
        reference=reference,
        local_transforms=[xfm],
        interpolation='nearest',
        out_file=str(tmp_path / 'out.nii.gz'),
    ).run()

    out = nb.load(res.outputs.out_file)
    data = np.asarray(out.dataobj)
    # +2 world translation, pull semantics -> marker moves from x=3 to x=1.
    assert np.argwhere(data == 5).tolist() == [[1, 3, 3]]
    # nearest interpolation keeps integer labels intact.
    assert set(np.unique(data).tolist()) == {0, 5}
    assert res.outputs.out_transforms == [xfm]
    assert res.outputs.out_inversions == [False]


def test_resolve_apply_identity_resamples_onto_reference(tmp_path, monkeypatch):
    import nibabel as nb

    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms

    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    moving = _label_img(tmp_path / 'atlas.nii.gz', (3, 3, 3))
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    res = ResolveApplyTransforms(
        source='SAME',
        target='SAME',
        moving=moving,
        reference=reference,
        interpolation='nearest',
        out_file=str(tmp_path / 'out.nii.gz'),
    ).run()

    data = np.asarray(nb.load(res.outputs.out_file).dataobj)
    assert np.argwhere(data == 5).tolist() == [[3, 3, 3]]  # unchanged
    assert res.outputs.out_transforms == []
    assert res.outputs.out_inversions == []


def test_resolve_apply_raises_when_no_path(tmp_path, monkeypatch):
    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms
    from bdt.transforms import NoTransformPathError

    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    xfm = _itk_affine(tmp_path / 'sub-01_from-ATLAS_to-GRID_mode-image_xfm.mat', 2.0)
    moving = _label_img(tmp_path / 'atlas.nii.gz', (3, 3, 3))
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    with pytest.raises(NoTransformPathError):
        ResolveApplyTransforms(
            source='ATLAS',
            target='UNREACHABLE',
            moving=moving,
            reference=reference,
            local_transforms=[xfm],
            out_file=str(tmp_path / 'out.nii.gz'),
        ).run()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py -k resolve_apply -v`
Expected: FAIL — `ImportError: cannot import name 'ResolveApplyTransforms'`.

- [ ] **Step 3: Append the interface class**

Add to the top-level imports of `src/bdt/interfaces/transforms.py`:

```python
import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
```

Append at the end of `src/bdt/interfaces/transforms.py`:

```python
class _ResolveApplyInputSpec(BaseInterfaceInputSpec):
    source = traits.Str(mandatory=True, desc='space of the moving image')
    target = traits.Str(mandatory=True, desc='space of the reference grid')
    moving = File(exists=True, mandatory=True, desc='image to warp')
    reference = File(exists=True, mandatory=True, desc='image defining the output grid')
    local_transforms = traits.List(
        File(exists=True), value=[], usedefault=True,
        desc='discovered BIDS xfm files from the input derivatives',
    )
    bridges = traits.List(
        File(exists=True), value=[], usedefault=True,
        desc='computed transform files (e.g. the ACPC<->T1w bridge)',
    )
    interpolation = traits.Enum(
        'linear', 'nearest', usedefault=True,
        desc="resampling interpolation ('nearest' for label/dseg images)",
    )
    out_file = traits.Str('resampled.nii.gz', usedefault=True, desc='output filename')


class _ResolveApplyOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the warped image')
    out_transforms = traits.List(traits.Str, desc='resolved chain (file paths)')
    out_inversions = traits.List(traits.Bool, desc='per-step inversion flags')


class ResolveApplyTransforms(SimpleInterface):
    """Resolve the shortest transform chain ``source -> target`` and apply it.

    Builds the space graph from the injected transform file lists plus
    TemplateFlow inter-template edges, finds the image-resample chain via
    :func:`bdt.transforms.queries.chain_for_image_resample`, composes it with
    ``nitransforms``, and resamples ``moving`` onto ``reference``.
    """

    input_spec = _ResolveApplyInputSpec
    output_spec = _ResolveApplyOutputSpec

    def _run_interface(self, runtime):
        from nitransforms.linear import Affine
        from nitransforms.resampling import apply

        from bdt.transforms.graph import build_transform_graph, parse_xfm_filename
        from bdt.transforms.queries import chain_for_image_resample

        local = [
            parse_xfm_filename(p)
            for p in (list(self.inputs.local_transforms) + list(self.inputs.bridges))
        ]
        local = [x for x in local if x is not None]
        tf_edges = templateflow_edges()
        tg = build_transform_graph([], extra_edges=local + tf_edges)

        steps = chain_for_image_resample(tg, self.inputs.source, self.inputs.target)
        transform = _build_chain(steps) if steps else Affine()

        order = 0 if self.inputs.interpolation == 'nearest' else 1
        out_file = self.inputs.out_file
        if not os.path.isabs(out_file):
            out_file = os.path.join(runtime.cwd, out_file)

        resampled = apply(
            transform,
            self.inputs.moving,
            reference=self.inputs.reference,
            order=order,
            mode='constant',
            cval=0.0,
        )
        resampled.to_filename(out_file)

        self._results['out_file'] = out_file
        self._results['out_transforms'] = [s.file for s in steps]
        self._results['out_inversions'] = [bool(s.invert) for s in steps]
        return runtime
```

Also add a module-level import so `templateflow_edges` is patchable by the tests (near the other imports, at module top — not inside a function):

```python
from bdt.transforms.templateflow import templateflow_edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_transforms_interface.py -v`
Expected: PASS (6 passed — 3 from Task 2, 3 new).

- [ ] **Step 5: Verify the whole subsystem (no commit — edit-in-place)**

Run: `micromamba run -n bdtenv python -m pytest test/spec/test_transforms.py test/spec/test_templateflow_edges.py test/engine/test_transforms_interface.py -q`
Expected: PASS (11 + 6 + 6 = 23 passed). Do **not** commit.

---

## Notes for the executor

- **Do not modify** `src/bdt/transforms/graph.py` or `queries.py`. If a test seems to need a change there, stop and flag it — the existing behavior is intentional and covered by `test/spec/test_transforms.py`.
- `templateflow_edges()` in the node hits the TemplateFlow manifest at runtime; every unit test **monkeypatches it to `[]`** so tests stay offline and fast. Keep it a module-level name in `bdt.interfaces.transforms` so `monkeypatch.setattr` works.
- The ACPC↔T1w computed bridge is **not** wired here — the node merely accepts it via the `bridges` input. Wiring it into a real workflow is Spec 2 (volumetric `parcellate_scalar`).
```
