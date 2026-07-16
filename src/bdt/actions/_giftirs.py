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
"""Thin wrapper over the ``giftirs`` CLI (from gifti-rs).

``giftirs transform`` warps ``.surf.gii`` vertex coordinates through an ANTs/ITK
``.h5``/``.mat``/``.txt`` transform (Strategy B — lossless point warp), handling
the FreeSurfer C_RAS offset automatically and preserving vertex order/topology.
BDT uses it to bring a subject's surfaces into a scalar's space before sampling
the volume onto them.  ``giftirs`` is a Rust binary shipped on ``PATH`` in the
container; here we shell out to it (NiWrap has no wrapper for our own binaries —
a Styx/Boutiques descriptor could route it through NiWrap's runner later).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def giftirs_transform_cargs(in_surf, out_surf, transform) -> list[str]:
    """The ``giftirs transform`` command line (factored out for testing)."""
    return [
        'giftirs',
        'transform',
        str(in_surf),
        str(out_surf),
        '--transform',
        str(transform),
        '--overwrite',
    ]


def run_giftirs_transform(in_surf, out_surf, transform) -> str:
    """Warp ``in_surf`` vertices by ``transform``, writing ``out_surf``.

    ``transform`` is the *opposite-named* file for the desired point-warp
    direction (see :func:`bdt.transforms.queries.chain_for_point_warp`).
    """
    Path(out_surf).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(giftirs_transform_cargs(in_surf, out_surf, transform), check=True)
    return str(out_surf)
