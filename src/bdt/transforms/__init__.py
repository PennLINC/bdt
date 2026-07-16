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
"""BDT transform engine: one graph, two typed queries.

Spatial transforms discovered in the input derivatives form a directed graph
(nodes = spaces, edges = ``from-X_to-Y`` transform files).  Two typed queries
traverse that single graph in *opposite* directions, because grid resampling
(ANTs ``ApplyTransforms``, pull semantics) and point warping (``trxrs`` /
``giftirs``, the "opposite-named h5" rule) consume transforms with mirror-image
ordering and inversion rules.  BDT only ever *applies* transforms it finds — it
never computes a normalization.
"""

from bdt.transforms.graph import TransformGraph, Xfm, build_transform_graph
from bdt.transforms.queries import (
    NoTransformPathError,
    XfmStep,
    chain_for_image_resample,
    chain_for_point_warp,
)

__all__ = (
    'NoTransformPathError',
    'TransformGraph',
    'Xfm',
    'XfmStep',
    'build_transform_graph',
    'chain_for_image_resample',
    'chain_for_point_warp',
)
