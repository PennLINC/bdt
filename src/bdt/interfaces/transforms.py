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
"""Resolve and apply a transform chain between two spaces with ``nitransforms``.

The graph + chain-query layer lives in :mod:`bdt.transforms`; this module loads
the resolved :class:`~bdt.transforms.queries.XfmStep` chain into ``nitransforms``
objects and resamples an image.  ``nitransforms`` (not the ANTs CLI) is the
engine: ``TransformChain(transforms=[t0, t1, ...])`` applies ``t0`` first, which
matches the grid-space-first order ``chain_for_image_resample`` returns.
"""

from __future__ import annotations

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from bdt.transforms.templateflow import templateflow_edges


def _load_transform(path: str, invert: bool = False):
    """Load one transform file as a ``nitransforms`` object.

    Dispatches on extension: ITK affine (``.mat``/``.txt``), ITK composite
    (``.h5``), or a displacement-field warp (``.nii``/``.nii.gz``).  ``invert``
    (only ever ``True`` for affine hops) returns the exact inverse via ``~``.
    """
    from nitransforms import linear, manip, nonlinear

    lower = path.lower()
    if lower.endswith(('.mat', '.txt')):
        xf = linear.load(path, fmt='itk')
    elif lower.endswith('.h5'):
        xf = manip.load(path, fmt='ITK')
    elif lower.endswith(('.nii', '.nii.gz')):
        xf = nonlinear.load(path, fmt='itk')
    else:
        raise ValueError(f'Unsupported transform file extension: {path}')
    return ~xf if invert else xf


def _build_chain(steps, fetch=None):
    """Assemble one ordered ``TransformChain`` from resolved ``XfmStep``s.

    ``steps`` come from ``chain_for_image_resample`` (grid-space hop first), the
    same order ``TransformChain`` applies to coordinates.  Composite files load as
    their own ``TransformChain`` and are flattened in place so the overall order
    is preserved.

    ``steps`` must be non-empty: an empty chain is the identity (same-space)
    resample, which the caller handles by returning the moving image untouched
    rather than assembling a chain.  Called with no steps this raises
    ``ValueError`` instead of surfacing a cryptic ``nitransforms`` ``IndexError``.
    """
    from nitransforms.manip import TransformChain

    if not steps:
        raise ValueError(
            '_build_chain requires at least one step; an empty (identity) chain '
            'means source == target and should be handled by the caller.'
        )

    if fetch is None:
        from bdt.transforms.templateflow import templateflow_fetch as fetch

    flat = []
    for step in steps:
        xf = _load_transform(fetch(step.file), invert=step.invert)
        if isinstance(xf, TransformChain):
            flat.extend(xf.transforms)
        else:
            flat.append(xf)
    return TransformChain(transforms=flat)


class _ResolveApplyInputSpec(BaseInterfaceInputSpec):
    source = traits.Str(mandatory=True, desc='space of the moving image')
    target = traits.Str(mandatory=True, desc='space of the reference grid')
    moving = File(exists=True, mandatory=True, desc='image to warp')
    reference = File(exists=True, mandatory=True, desc='image defining the output grid')
    local_transforms = traits.List(
        File(exists=True),
        value=[],
        usedefault=True,
        desc='discovered BIDS xfm files from the input derivatives',
    )
    bridges = traits.List(
        File(exists=True),
        value=[],
        usedefault=True,
        desc='computed transform files (e.g. the ACPC<->T1w bridge)',
    )
    interpolation = traits.Enum(
        'linear',
        'nearest',
        usedefault=True,
        desc="resampling interpolation ('nearest' for label/dseg images)",
    )
    out_file = traits.Str('resampled.nii.gz', usedefault=True, desc='output filename')


class _ResolveApplyOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the warped image')
    out_transforms = traits.List(traits.Str, desc='resolved chain (file paths)')
    out_inversions = traits.List(traits.Bool, desc='per-step inversion flags')


class ResolveApplyTransforms(SimpleInterface):
    """Resolve the shortest transform chain ``source -> target`` and apply it.

    Builds the space graph from the injected transform file lists plus
    TemplateFlow inter-template edges, finds the image-resample chain via
    :func:`bdt.transforms.queries.chain_for_image_resample`, composes it with
    ``nitransforms``, and resamples ``moving`` onto ``reference``.
    """

    input_spec = _ResolveApplyInputSpec
    output_spec = _ResolveApplyOutputSpec

    def _run_interface(self, runtime):
        from nipype import logging
        from nitransforms.linear import Affine
        from nitransforms.resampling import apply

        from bdt.transforms.graph import build_transform_graph, parse_xfm_filename
        from bdt.transforms.queries import chain_for_image_resample

        iflogger = logging.getLogger('nipype.interface')

        injected = []
        for p in list(self.inputs.local_transforms) + list(self.inputs.bridges):
            xfm = parse_xfm_filename(p)
            if xfm is None:
                iflogger.warning('Ignoring unparseable transform file: %s', p)
            else:
                injected.append(xfm)
        tf_edges = templateflow_edges()
        tg = build_transform_graph([], extra_edges=injected + tf_edges)

        steps = chain_for_image_resample(tg, self.inputs.source, self.inputs.target)
        transform = _build_chain(steps) if steps else Affine()

        order = 0 if self.inputs.interpolation == 'nearest' else 1
        out_file = self.inputs.out_file
        if not os.path.isabs(out_file):
            out_file = os.path.join(runtime.cwd, out_file)

        import nibabel as nb
        from nibabel.funcs import concat_images, four_to_three

        moving_img = nb.load(self.inputs.moving)
        if moving_img.ndim > 3:
            # nitransforms.apply rejects 4D; warp each 3D volume with the single
            # resolved chain, then restack onto the (3D) reference grid.
            warped = [
                apply(
                    transform,
                    vol,
                    reference=self.inputs.reference,
                    order=order,
                    mode='constant',
                    cval=0.0,
                )
                for vol in four_to_three(moving_img)
            ]
            resampled = concat_images(warped)
        else:
            resampled = apply(
                transform,
                self.inputs.moving,
                reference=self.inputs.reference,
                order=order,
                mode='constant',
                cval=0.0,
            )
        resampled.to_filename(out_file)

        self._results['out_file'] = out_file
        self._results['out_transforms'] = [s.file for s in steps]
        self._results['out_inversions'] = [bool(s.invert) for s in steps]
        return runtime
