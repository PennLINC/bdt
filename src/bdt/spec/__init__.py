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
"""BDT node-graph spec: grammar model, loader, action registry, static validator.

This subpackage is deliberately dependency-light (standard library + PyYAML).
It defines the *framework-agnostic* half of BDT: what a spec looks like, what
actions exist and how they wire, and whether a given spec is valid — with no
dependency on nipype, NiWrap, templateflow, or the neuroimaging stack.  The
execution layer (:mod:`bdt.engine`) consumes these objects.
"""

from bdt.spec.actions import ACTIONS, ActionSpec, Role
from bdt.spec.load import load_spec, parse_spec
from bdt.spec.model import Node, Spec, SpecError
from bdt.spec.validate import SpecValidationError, validate_spec

__all__ = (
    'ACTIONS',
    'ActionSpec',
    'Node',
    'Role',
    'Spec',
    'SpecError',
    'SpecValidationError',
    'load_spec',
    'parse_spec',
    'validate_spec',
)
