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
"""Small helpers shared by the real action builders."""

from __future__ import annotations

import tempfile
from pathlib import Path

_STEM_KEYS = ('sub', 'ses', 'task', 'model', 'param', 'atlas', 'stat', 'desc')


def ensure_workdir(ctx, node) -> Path:
    """A per-node scratch directory (under ``ctx.work_dir`` or a temp dir)."""
    base = Path(ctx.work_dir) if getattr(ctx, 'work_dir', None) else Path(tempfile.mkdtemp('bdt'))
    directory = base / node.name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stem_for(node, entities: dict) -> str:
    """A filename stem unique per fan-out combination (for intermediates)."""
    parts = [f'{k}-{entities[k]}' for k in _STEM_KEYS if k in entities]
    return '_'.join(parts) or node.name


def sources_of(by_role: dict) -> list[str]:
    """Deduplicated union of the ``bids:`` sources across a fan combination."""
    out: list[str] = []
    for role_value in by_role.values():
        out.extend(role_value.sources)
    return list(dict.fromkeys(out))
