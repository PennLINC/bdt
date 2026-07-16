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
"""Verify the workbench wrappers build the correct ``wb_command`` invocations.

Uses NiWrap's dry runner, which records the command line without executing it, so
these run without ``wb_command`` installed.
"""

import pytest

niwrap = pytest.importorskip('niwrap')

from bdt.actions._workbench import run_cifti_correlation, run_cifti_parcellate  # noqa: E402


def test_cifti_parcellate_command():
    dry = niwrap.use_dry()
    out = run_cifti_parcellate(
        'in.dtseries.nii', 'atlas.dlabel.nii', 'out.ptseries.nii', direction='COLUMN'
    )
    assert str(out) == 'out.ptseries.nii'
    cargs = dry.last_cargs
    assert cargs[:2] == ['wb_command', '-cifti-parcellate']
    assert 'in.dtseries.nii' in cargs
    assert 'atlas.dlabel.nii' in cargs
    assert 'COLUMN' in cargs
    assert '-only-numeric' in cargs


def test_cifti_correlation_covariance_flag():
    dry = niwrap.use_dry()
    run_cifti_correlation('in.ptseries.nii', 'out.pconn.nii', covariance=True)
    cargs = dry.last_cargs
    assert cargs[:2] == ['wb_command', '-cifti-correlation']
    assert '-covariance' in cargs


def test_cifti_correlation_no_covariance():
    dry = niwrap.use_dry()
    run_cifti_correlation('in.ptseries.nii', 'out.pconn.nii', covariance=False)
    assert '-covariance' not in dry.last_cargs
