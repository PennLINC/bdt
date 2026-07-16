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
"""BDT output layer: BIDS-derivative naming, provenance sidecars, materialization.

Materialize the nodes that set ``write_outputs: true`` as BIDS derivatives with
composed entities, a ``desc`` that prepend-composes down the graph, and a JSON
provenance sidecar (``GeneratedBy`` + ``Sources``).  Dataset-level outputs go to
``tpl-<space>/`` rather than being duplicated per subject, and two nodes that
resolve to the same path are a hard error.
"""

from bdt.outputs.provenance import bids_uri, build_sidecar, generated_by
from bdt.outputs.sink import DerivativeSink, OutputCollisionError, bids_name, compose_desc

__all__ = (
    'DerivativeSink',
    'OutputCollisionError',
    'bids_name',
    'bids_uri',
    'build_sidecar',
    'compose_desc',
    'generated_by',
)
