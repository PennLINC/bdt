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
"""``parcellate_timeseries`` / ``parcellate_scalar`` builders (Strategy A).

CIFTI inputs are parcellated with ``wb_command -cifti-parcellate`` and written to
a parcel TSV; NIfTI inputs that already share the atlas's space are parcellated
with nilearn.  Warping a volumetric atlas into a differently-spaced NIfTI's grid
(the full Strategy-A image-resample via ANTs) is a follow-up — for now that case
raises so it never silently mis-parcellates.
"""

from __future__ import annotations

from bdt.actions._io import cifti_to_tsv, is_cifti, nifti_parcellate_to_tsv
from bdt.actions._util import ensure_workdir, sources_of, stem_for
from bdt.actions._workbench import run_cifti_parcellate
from bdt.engine.builders import compose_output_entities, fan_combinations, register_builder
from bdt.engine.result import NodeResult


def _parcellate_one(ctx, node, by_role) -> NodeResult:
    spec = node.action_spec
    entities, primary = compose_output_entities(node, by_role)
    atlas = by_role['atlas']
    data_file = primary.files[0]
    atlas_file = atlas.files[0]
    work = ensure_workdir(ctx, node)
    stem = stem_for(node, entities)

    data_cifti, atlas_cifti = is_cifti(data_file), is_cifti(atlas_file)
    if data_cifti and atlas_cifti:
        ptseries = run_cifti_parcellate(
            data_file,
            atlas_file,
            work / f'{stem}.ptseries.nii',
            direction='COLUMN',
            only_numeric=True,
            runner=ctx_runner(ctx),
        )
        tsv = cifti_to_tsv(ptseries, work / f'{stem}.tsv')
    elif not data_cifti and not atlas_cifti:
        _require_same_space(node, primary, atlas)
        tsv = nifti_parcellate_to_tsv(data_file, atlas_file, work / f'{stem}.tsv')
    else:
        raise ValueError(
            f'Node {node.name!r}: cannot parcellate a '
            f'{"CIFTI" if data_cifti else "NIfTI"} scalar with a '
            f'{"CIFTI" if atlas_cifti else "NIfTI"} atlas; geometries must match.'
        )

    return NodeResult(
        node=node.name,
        action=node.action,
        fmt=spec.produces,
        files=[tsv],
        entities=entities,
        sources=sources_of(by_role),
        scope=node.scope,
        space=primary.space,
    )


def _require_same_space(node, primary, atlas) -> None:
    a, b = primary.space, atlas.space
    if a and b and a != b:
        raise NotImplementedError(
            f'Node {node.name!r}: parcellating a NIfTI in space {a!r} with an atlas in '
            f'space {b!r} needs a Strategy-A ANTs warp of the atlas into the data grid, '
            f'which is not yet wired. Provide the atlas in {a!r} or use the CIFTI path.'
        )


def ctx_runner(ctx):
    """The NiWrap runner to use (``None`` -> the configured global runner)."""
    return getattr(ctx, 'runner', None)


@register_builder('parcellate_timeseries')
def build_parcellate_timeseries(ctx, node, resolved) -> list[NodeResult]:
    return [_parcellate_one(ctx, node, by_role) for by_role in fan_combinations(node, resolved)]


@register_builder('parcellate_scalar')
def build_parcellate_scalar(ctx, node, resolved) -> list[NodeResult]:
    return [_parcellate_one(ctx, node, by_role) for by_role in fan_combinations(node, resolved)]
