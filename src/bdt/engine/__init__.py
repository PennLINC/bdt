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
"""BDT node-graph execution engine.

A custom DAG executor over a validated :class:`~bdt.spec.model.Spec`.  It
topologically orders each scope (``dataset:`` once, then ``nodes:`` per subject),
resolves each node's ``inputs:`` by declared role (fanning out over list-valued /
multi-match roles, grouping the rest), dispatches to a per-action builder, and
materializes ``write_outputs`` nodes through the :mod:`bdt.outputs` sink.

The builder layer is the only framework-specific seam: action builders receive
resolved inputs and return :class:`NodeResult` objects.  A generic passthrough
builder lets the whole graph run end-to-end before any real (NiWrap) tool
builders are wired in.
"""

from bdt.engine.builders import BUILDERS, register_builder
from bdt.engine.executor import Executor
from bdt.engine.pipeline import run_spec
from bdt.engine.result import BuildContext, NodeResult, RoleValue
from bdt.engine.selection import DataProvider, DictDataProvider, Match, build_selection

__all__ = (
    'BUILDERS',
    'BuildContext',
    'DataProvider',
    'DictDataProvider',
    'Executor',
    'Match',
    'NodeResult',
    'RoleValue',
    'build_selection',
    'register_builder',
    'run_spec',
)
