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
"""Tests for resample_subcortical (nilearn) and assemble_cifti (stub)."""

import pytest

pytest.importorskip('nibabel')
pytest.importorskip('nilearn')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402

from bdt.actions.cifti import build_assemble_cifti, build_resample_subcortical  # noqa: E402
from bdt.engine.result import BuildContext, NodeResult  # noqa: E402
from bdt.spec.model import Node  # noqa: E402


def _nifti(path, array, affine):
    nb.Nifti1Image(array, affine).to_filename(str(path))
    return str(path)


def _scalar_result(path, space='MNI152NLin6Asym'):
    return NodeResult(
        node='scalar',
        action='select_data',
        fmt='scalar',
        files=[path],
        entities={'sub': '01', 'param': 'icvf', 'space': space},
        sources=['bids:qsirecon:scalar.nii.gz'],
        space=space,
    )


def _structures_result(path, space='MNI152NLin6Asym'):
    return NodeResult(
        node='structures',
        action='select_atlases',
        fmt='structures',
        files=[path],
        entities={'atlas': 'HCPSubcortical', 'space': space},
        sources=['bids:atlases:hcpsubcort.nii.gz'],
        space=space,
    )


def _node():
    return Node(
        name='subcort',
        action='resample_subcortical',
        inputs={'scalar': ['scalar'], 'structures': ['structures']},
    )


def test_resample_subcortical(tmp_path):
    # scalar on a 4x4x4 grid; structures on a coarser 2x2x2 grid (same world space).
    scalar = _nifti(tmp_path / 's.nii.gz', np.random.default_rng(0).random((4, 4, 4)), np.eye(4))
    struct_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    structures = _nifti(tmp_path / 'st.nii.gz', np.ones((2, 2, 2), dtype=np.int16), struct_affine)

    ctx = BuildContext(work_dir=str(tmp_path / 'work'))
    results = build_resample_subcortical(
        ctx,
        _node(),
        {'scalar': [_scalar_result(scalar)], 'structures': [_structures_result(structures)]},
    )
    assert len(results) == 1
    res = results[0]
    assert res.fmt == 'subcortical_volume'
    out = nb.load(res.files[0])
    assert out.shape == (2, 2, 2)  # resampled onto the structures grid


def test_resample_subcortical_cross_space_raises(tmp_path):
    scalar = _nifti(tmp_path / 's.nii.gz', np.zeros((4, 4, 4)), np.eye(4))
    structures = _nifti(tmp_path / 'st.nii.gz', np.ones((2, 2, 2), dtype=np.int16), np.eye(4))
    ctx = BuildContext(work_dir=str(tmp_path / 'work'))
    resolved = {
        'scalar': [_scalar_result(scalar, space='ACPC')],
        'structures': [_structures_result(structures, space='MNI152NLin6Asym')],
    }
    with pytest.raises(NotImplementedError, match='ANTs warp'):
        build_resample_subcortical(ctx, _node(), resolved)


def test_assemble_cifti_raises():
    node = Node(
        name='cifti',
        action='assemble_cifti',
        inputs={'surface': ['s'], 'volume': ['v']},
    )
    with pytest.raises(NotImplementedError, match='assemble_cifti'):
        build_assemble_cifti(BuildContext(), node, {})
