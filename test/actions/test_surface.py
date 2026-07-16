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
"""Tests for the surface-mapping builders (hemi routing + wb command wiring)."""

import pytest

niwrap = pytest.importorskip('niwrap')

from bdt.actions._giftirs import giftirs_transform_cargs  # noqa: E402
from bdt.actions._workbench import (  # noqa: E402
    run_cifti_create_dense_scalar,
    run_metric_dilate,
    run_metric_resample,
    run_volume_to_surface_mapping,
)
from bdt.actions.surface import build_map_scalar_to_surface, group_surfaces  # noqa: E402
from bdt.engine.result import BuildContext, NodeResult, RoleValue  # noqa: E402
from bdt.spec.model import Node  # noqa: E402


def _surface_results(space='T1w'):
    # sMRIPrep native surfaces are in the T1w anatomical space (fsnative = mesh density).
    out = []
    for hemi in ('L', 'R'):
        for suffix in ('white', 'pial', 'midthickness'):
            out.append(
                NodeResult(
                    node='surfaces',
                    action='select_data',
                    fmt='surfaces',
                    files=[f'/data/sub-01_hemi-{hemi}_{suffix}.surf.gii'],
                    entities={'sub': '01', 'hemi': hemi, 'suffix': suffix},
                    sources=[f'bids:smriprep:sub-01_hemi-{hemi}_{suffix}.surf.gii'],
                    space=space,
                )
            )
    return out


def test_group_surfaces():
    grouped = group_surfaces(RoleValue.group(_surface_results()))
    assert set(grouped) == {'L', 'R'}
    assert set(grouped['L']) == {'white', 'pial', 'midthickness'}
    assert grouped['R']['pial'].endswith('hemi-R_pial.surf.gii')


def test_wb_surface_commands():
    dry = niwrap.use_dry()
    run_volume_to_surface_mapping(
        'sc.nii.gz', 'mid.surf.gii', 'wh.surf.gii', 'pi.surf.gii', 'o.func.gii'
    )
    cargs = dry.last_cargs
    assert cargs[:2] == ['wb_command', '-volume-to-surface-mapping']
    assert '-ribbon-constrained' in cargs
    assert 'wh.surf.gii' in cargs
    assert 'pi.surf.gii' in cargs

    run_metric_dilate('m.func.gii', 'mid.surf.gii', 'd.func.gii', distance=10.0)
    assert dry.last_cargs[:2] == ['wb_command', '-metric-dilate']
    assert '-nearest' in dry.last_cargs

    run_metric_resample(
        'm.func.gii', 'cs.surf.gii', 'ns.surf.gii', 'ca.surf.gii', 'na.surf.gii', 'r.func.gii'
    )
    cargs = dry.last_cargs
    assert cargs[:2] == ['wb_command', '-metric-resample']
    assert 'ADAP_BARY_AREA' in cargs
    assert '-area-surfs' in cargs

    run_cifti_create_dense_scalar(
        'dense.dscalar.nii', left_metric='L.func.gii', right_metric='R.func.gii'
    )
    cargs = dry.last_cargs
    assert cargs[:2] == ['wb_command', '-cifti-create-dense-scalar']
    assert '-left-metric' in cargs
    assert '-right-metric' in cargs


def _map_node():
    return Node(
        name='on_surf',
        action='map_scalar_to_surface',
        inputs={'scalar': ['load_scalar'], 'surfaces': ['surfaces']},
    )


def _scalar_result(space='T1w'):
    return NodeResult(
        node='load_scalar',
        action='select_data',
        fmt='scalar',
        files=['/data/sub-01_param-icvf_dwimap.nii.gz'],
        entities={'sub': '01', 'param': 'icvf', 'space': space},
        sources=['bids:qsirecon:sub-01_param-icvf_dwimap.nii.gz'],
        space=space,
    )


def test_giftirs_transform_cargs():
    cargs = giftirs_transform_cargs('in.surf.gii', 'out.surf.gii', 'from-ACPC_to-T1w_xfm.h5')
    assert cargs[:2] == ['giftirs', 'transform']
    assert cargs[2:5] == ['in.surf.gii', 'out.surf.gii', '--transform']
    assert 'from-ACPC_to-T1w_xfm.h5' in cargs


def test_map_scalar_to_surface_same_space(tmp_path):
    # surfaces and scalar both in T1w -> no surface warp needed
    dry = niwrap.use_dry()
    resolved = {'scalar': [_scalar_result('T1w')], 'surfaces': _surface_results('T1w')}
    ctx = BuildContext(work_dir=str(tmp_path))
    results = build_map_scalar_to_surface(ctx, _map_node(), resolved)
    assert len(results) == 1
    res = results[0]
    assert res.fmt == 'surface_scalar'
    assert res.entities['param'] == 'icvf'
    assert res.files[0].endswith('.dscalar.nii')
    # the pipeline reached the dense-scalar assembly with both hemispheres
    assert dry.last_cargs[:2] == ['wb_command', '-cifti-create-dense-scalar']
    assert '-left-metric' in dry.last_cargs
    assert '-right-metric' in dry.last_cargs


def test_map_scalar_to_surface_warps_surfaces_cross_space(tmp_path, monkeypatch):
    """Surfaces in T1w, scalar in ACPC -> warp surface vertices T1w->ACPC via giftirs."""
    import subprocess

    from bdt.transforms.graph import build_transform_graph

    dry = niwrap.use_dry()
    calls = []

    def fake_run(cargs, check=False, **kwargs):
        calls.append(cargs)
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(subprocess, 'run', fake_run)

    # a from-ACPC_to-T1w file makes the T1w->ACPC *point* warp resolvable (opposite-named)
    (tmp_path / 'sub-01_from-ACPC_to-T1w_mode-image_xfm.h5').write_bytes(b'')
    tg = build_transform_graph(tmp_path)

    resolved = {'scalar': [_scalar_result('ACPC')], 'surfaces': _surface_results('T1w')}
    ctx = BuildContext(work_dir=str(tmp_path / 'work'), transform_graph=tg)
    results = build_map_scalar_to_surface(ctx, _map_node(), resolved)

    giftirs_calls = [c for c in calls if c[:2] == ['giftirs', 'transform']]
    assert len(giftirs_calls) == 6  # white/pial/midthickness x L/R
    assert all(any('from-ACPC_to-T1w' in str(a) for a in c) for c in giftirs_calls)
    assert results[0].fmt == 'surface_scalar'
    assert dry.last_cargs[:2] == ['wb_command', '-cifti-create-dense-scalar']


def test_map_scalar_missing_hemisphere_raises(tmp_path):
    niwrap.use_dry()
    surfaces = [
        r
        for r in _surface_results()
        if not (r.entities['hemi'] == 'R' and r.entities['suffix'] == 'midthickness')
    ]
    resolved = {'scalar': [_scalar_result()], 'surfaces': surfaces}
    ctx = BuildContext(work_dir=str(tmp_path))
    with pytest.raises(ValueError, match='midthickness'):
        build_map_scalar_to_surface(ctx, _map_node(), resolved)
