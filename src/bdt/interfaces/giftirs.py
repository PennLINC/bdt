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
"""nipype ``CommandLine`` interface for the ``giftirs`` Rust CLI (surface point-warp).

``giftirs transform`` applies an ANTs/ITK spatial transform to a GIFTI surface's
*vertices* (Strategy B — the data's coordinates move, losslessly, preserving
vertex order), the surface counterpart of ``antsApplyTransformsToPoints`` /
``trxrs``.  It obeys the **opposite-named rule**: to warp a surface from space A
into space B, pass ``from-B_to-A_xfm`` — the same file you would hand
``antsApplyTransforms`` to resample a B image onto A's grid.  It also detects a
FreeSurfer ``VolGeomC_R/A/S`` offset, bakes it into the warped vertices, and
zeroes it in the output so downstream tools don't double-apply it.

BDT uses this for the one computed bridge — warping fMRIPrep/sMRIPrep ``T1w``
surfaces into QSIPrep ``ACPC`` space with a freshly computed rigid transform —
before a ribbon-constrained volume-to-surface mapping.
"""

from __future__ import annotations

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class _GiftiTransformInputSpec(CommandLineInputSpec):
    overwrite = traits.Bool(
        True,
        usedefault=True,
        argstr='--overwrite',
        position=0,
        desc='replace the output file if it exists (safe in a fresh work dir)',
    )
    invert = traits.Bool(
        False,
        usedefault=True,
        argstr='--invert',
        position=1,
        desc='invert the transform before applying (affine-only transforms)',
    )
    transform = File(
        exists=True,
        mandatory=True,
        argstr='--transform %s',
        position=2,
        desc="ANTs/ITK transform; opposite-named: to warp A->B pass 'from-B_to-A'",
    )
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=3,
        desc='input GIFTI surface (.surf.gii)',
    )
    out_file = File(
        argstr='%s',
        position=4,
        name_source=['in_file'],
        name_template='%s_warped.surf.gii',
        keep_extension=False,
        desc='output warped GIFTI surface (.surf.gii)',
    )


class _GiftiTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the warped GIFTI surface')


class GiftiTransform(CommandLine):
    """Warp a GIFTI surface's vertices with an ANTs/ITK transform (``giftirs transform``).

    Examples
    --------
    >>> from bdt.interfaces.giftirs import GiftiTransform
    >>> warp = GiftiTransform()
    >>> warp.inputs.in_file = 'sub-01_hemi-L_pial.surf.gii'          # in T1w space
    >>> warp.inputs.transform = 'sub-01_from-ACPC_to-T1w_xfm.mat'    # -> warp into ACPC
    >>> warp.inputs.out_file = 'sub-01_space-ACPC_hemi-L_pial.surf.gii'
    >>> warp.cmdline
    'giftirs transform --overwrite --transform sub-01_from-ACPC_to-T1w_xfm.mat \
sub-01_hemi-L_pial.surf.gii sub-01_space-ACPC_hemi-L_pial.surf.gii'
    """

    _cmd = 'giftirs transform'
    input_spec = _GiftiTransformInputSpec
    output_spec = _GiftiTransformOutputSpec
