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
"""Tests for the NiWrap runner configuration (requires niwrap)."""

import pytest

niwrap = pytest.importorskip('niwrap')

from bdt.tools.niwrap_run import configure_runner  # noqa: E402


def test_configure_local_runner(tmp_path):
    runner = configure_runner(tmp_path / 'work', engine='local')
    assert runner is not None
    assert niwrap.get_global_runner() is runner


def test_unknown_engine_raises(tmp_path):
    with pytest.raises(ValueError, match='Unknown NiWrap engine'):
        configure_runner(tmp_path, engine='bogus')
