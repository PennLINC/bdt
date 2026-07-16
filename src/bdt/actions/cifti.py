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
"""Subcortical resampling and dense-CIFTI assembly.

``resample_subcortical`` resamples a *continuous* volumetric scalar onto the
grayordinate subcortical grid (the ``structures`` atlas grid), with continuous
interpolation — never ``GenericLabel`` (that would quantize a continuous map).

``assemble_cifti`` (cortex surface + subcortex volume -> dense CIFTI) is not yet
wired: it needs the cortex dscalar separated into per-hemi metrics and the
subcortical structure-label volume; it raises rather than passing through.
"""

from __future__ import annotations

from bdt.actions._util import ensure_workdir, sources_of, stem_for
from bdt.engine.builders import compose_output_entities, fan_combinations, register_builder
from bdt.engine.result import NodeResult


def _resample_to_grid(scalar_path, target_path, out_path) -> str:
    """Resample ``scalar`` onto ``target``'s grid with continuous interpolation."""
    import nibabel as nb
    from nilearn.image import resample_to_img

    resampled = resample_to_img(
        str(scalar_path),
        str(target_path),
        interpolation='continuous',
        force_resample=True,
        copy_header=True,
    )
    nb.save(resampled, str(out_path))
    return str(out_path)


@register_builder('resample_subcortical')
def build_resample_subcortical(ctx, node, resolved) -> list[NodeResult]:
    results: list[NodeResult] = []
    for by_role in fan_combinations(node, resolved):
        entities, primary = compose_output_entities(node, by_role)
        structures = by_role['structures']
        s_space, t_space = primary.space, structures.space
        if s_space and t_space and s_space != t_space:
            raise NotImplementedError(
                f'Node {node.name!r}: resampling a scalar in space {s_space!r} onto '
                f'subcortical structures in space {t_space!r} needs a Strategy-A ANTs warp '
                f'(T1w->MNI) applied to the continuous scalar, which is not yet wired.'
            )
        work = ensure_workdir(ctx, node)
        stem = stem_for(node, entities)
        out = _resample_to_grid(primary.files[0], structures.files[0], work / f'{stem}.nii.gz')
        results.append(
            NodeResult(
                node=node.name,
                action=node.action,
                fmt=node.action_spec.produces,
                files=[out],
                entities=entities,
                sources=sources_of(by_role),
                scope=node.scope,
                space=t_space or s_space,
            )
        )
    return results


@register_builder('assemble_cifti')
def build_assemble_cifti(ctx, node, resolved) -> list[NodeResult]:
    raise NotImplementedError(
        f'Node {node.name!r}: assemble_cifti (cortex surface + subcortex volume -> dense '
        'CIFTI) is not yet wired. It needs the cortex dscalar separated into per-hemi '
        'metrics (wb_command -cifti-separate) and the subcortical structure-label volume '
        'to feed wb_command -cifti-create-dense-scalar -volume. See the dense-CIFTI '
        'assembly follow-up task.'
    )
