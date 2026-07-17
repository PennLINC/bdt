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

The engine compiles a validated :class:`~bdt.spec.model.Spec` into a nipype
``Workflow`` (:func:`bdt.engine.workflow.init_bdt_wf`): selection nodes resolve to
files via a :class:`~bdt.engine.selection.DataProvider`, and each processing node
becomes an ``init_<action>_wf`` sub-workflow (:mod:`bdt.engine.factories`) wired by
declared role.  Running that workflow with nipype gives File(exists) validation,
content-hash caching, resume, and multiproc/cluster plugins.

``bdt.engine.workflow`` / ``bdt.engine.factories`` are imported lazily (they pull
in nipype); this package's own imports stay light.
"""

from bdt.engine.pybids_provider import BIDSDataProvider
from bdt.engine.selection import DataProvider, DictDataProvider, Match, SelectionError

__all__ = (
    'BIDSDataProvider',
    'DataProvider',
    'DictDataProvider',
    'Match',
    'SelectionError',
)
