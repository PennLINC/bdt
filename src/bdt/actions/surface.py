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
"""Surface-mapping builders (Strategy B / Connectome Workbench).

``map_scalar_to_surface`` samples a volumetric scalar onto each hemisphere's mid-
thickness with ribbon-constrained mapping (white..pial), dilates, and staples the
two hemispheres into a dense scalar CIFTI.

Strategy B — when the scalar and the surfaces are in different *coordinate* spaces
(e.g. an sMRIPrep surface in **T1w** and a QSIRecon scalar in **ACPC**), the
surface *vertices* are warped losslessly into the scalar's space with
``giftirs transform`` (preserving vertex order) and the volume is then sampled
there.  We do not resample the volume.  Note: ``fsnative`` is a mesh *density*,
not a coordinate space — the coordinate space comes from the surfaces' ``space``
entity (sMRIPrep native surfaces are in T1w), so the (loosely named) spec
``source_space`` parameter is *not* used as a coordinate frame here.

``resample_surface_scalar`` (native -> fsLR density) is not yet wired: it
additionally needs the subject's registration spheres and the TemplateFlow fsLR
target spheres / midthickness, plus CIFTI separate/combine plumbing.  It raises
rather than silently passing through.
"""

from __future__ import annotations

from bdt.actions._giftirs import run_giftirs_transform
from bdt.actions._util import ensure_workdir, sources_of, stem_for
from bdt.actions._workbench import (
    run_cifti_create_dense_scalar,
    run_metric_dilate,
    run_volume_to_surface_mapping,
)
from bdt.actions.parcellate import ctx_runner
from bdt.engine.builders import compose_output_entities, fan_combinations, register_builder
from bdt.engine.result import NodeResult, RoleValue

_RIBBON_SURFACES = ('white', 'pial', 'midthickness')

# sMRIPrep native surfaces carry no explicit coordinate ``space`` entity; they are
# in the subject anatomical (T1w) reference.
_DEFAULT_SURFACE_SPACE = 'T1w'


def _warp_surface_to_space(surf, from_space, to_space, transform_graph, work, tag) -> str:
    """Warp a ``.surf.gii`` from ``from_space`` into ``to_space`` (Strategy B).

    Returns ``surf`` unchanged when the spaces match.  Otherwise resolves the
    point-warp chain and applies ``giftirs transform`` once per hop (the
    opposite-named files the chain returns).
    """
    if not from_space or not to_space or from_space == to_space:
        return str(surf)
    if transform_graph is None:
        raise ValueError(
            f'Surfaces are in space {from_space!r} but the scalar is in {to_space!r}, and no '
            f'transform graph is available to warp the surface vertices.'
        )
    from bdt.transforms.queries import chain_for_point_warp

    current = str(surf)
    for i, step in enumerate(chain_for_point_warp(transform_graph, from_space, to_space)):
        out = work / f'{tag}_warp{i}.surf.gii'
        current = run_giftirs_transform(current, out, step.file)
    return current


def group_surfaces(surfaces: RoleValue) -> dict[str, dict[str, str]]:
    """Route a grouped ``surfaces`` role into ``{hemi: {suffix: path}}``.

    Uses each grouped file's own entities (``hemi`` + ``suffix``), so the L/R
    white/pial/midthickness set that arrived as one group can be told apart.
    """
    out: dict[str, dict[str, str]] = {}
    for path, entities in surfaces.members:
        hemi = entities.get('hemi')
        suffix = entities.get('suffix')
        if hemi and suffix:
            out.setdefault(hemi, {})[suffix] = path
    return out


@register_builder('map_scalar_to_surface')
def build_map_scalar_to_surface(ctx, node, resolved) -> list[NodeResult]:
    results: list[NodeResult] = []
    for by_role in fan_combinations(node, resolved):
        entities, primary = compose_output_entities(node, by_role)
        scalar = primary.files[0]
        surfaces = group_surfaces(by_role['surfaces'])
        work = ensure_workdir(ctx, node)
        stem = stem_for(node, entities)
        runner = ctx_runner(ctx)

        # Strategy B: bring the surface vertices into the scalar's coordinate space.
        scalar_space = primary.space
        surface_space = by_role['surfaces'].space or _DEFAULT_SURFACE_SPACE
        transform_graph = getattr(ctx, 'transform_graph', None)

        hemi_metrics: dict[str, str] = {}
        for hemi in ('L', 'R'):
            hs = surfaces.get(hemi, {})
            missing = [s for s in _RIBBON_SURFACES if s not in hs]
            if missing:
                raise ValueError(
                    f'Node {node.name!r}: surfaces group is missing {missing} for hemi-{hemi}; '
                    f'map_scalar_to_surface needs white, pial and midthickness per hemisphere.'
                )
            warped = {
                s: _warp_surface_to_space(
                    hs[s],
                    surface_space,
                    scalar_space,
                    transform_graph,
                    work,
                    f'{stem}_hemi-{hemi}_{s}',
                )
                for s in _RIBBON_SURFACES
            }
            mapped = run_volume_to_surface_mapping(
                scalar,
                warped['midthickness'],
                warped['white'],
                warped['pial'],
                work / f'{stem}_hemi-{hemi}.func.gii',
                runner=runner,
            )
            hemi_metrics[hemi] = run_metric_dilate(
                mapped,
                hs['midthickness'],
                work / f'{stem}_hemi-{hemi}_dil.func.gii',
                runner=runner,
            )

        dscalar = run_cifti_create_dense_scalar(
            work / f'{stem}.dscalar.nii',
            left_metric=hemi_metrics['L'],
            right_metric=hemi_metrics['R'],
            runner=runner,
        )
        results.append(
            NodeResult(
                node=node.name,
                action=node.action,
                fmt=node.action_spec.produces,
                files=[str(dscalar)],
                entities=entities,
                sources=sources_of(by_role),
                scope=node.scope,
                space=primary.space,
            )
        )
    return results


@register_builder('resample_surface_scalar')
def build_resample_surface_scalar(ctx, node, resolved) -> list[NodeResult]:
    raise NotImplementedError(
        f'Node {node.name!r}: resample_surface_scalar (fsnative -> fsLR) is not yet wired. '
        'It needs the subject registration spheres plus the TemplateFlow fsLR target '
        'spheres/midthickness and CIFTI separate/combine around wb_command -metric-resample. '
        'See the fsLR resample follow-up task.'
    )
