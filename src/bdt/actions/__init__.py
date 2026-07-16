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
"""Real action builders (register with the engine on import).

Importing this package runs the ``@register_builder`` decorators, populating
:data:`bdt.engine.builders.BUILDERS`.  :func:`bdt.engine.run_spec` imports it when
``use_builtin_actions`` is true; structural/engine tests that want the passthrough
builder simply do not import it.
"""

from bdt.actions import (  # noqa: F401 - side-effect: registration
    cifti,
    connectivity,
    parcellate,
    surface,
)
from bdt.engine.builders import BUILDERS

__all__ = ('BUILDERS',)
