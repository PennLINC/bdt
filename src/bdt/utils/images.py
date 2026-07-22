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
"""Small nibabel helpers shared by the parcellation interfaces."""

from __future__ import annotations

import nibabel as nb
import numpy as np


def as_float_img(image):
    """``image`` with a floating-point data array, loading it if it is a path.

    nilearn's ``NiftiLabelsMasker`` reduces each parcel **in the input array's own
    dtype**, so an integer-typed image silently gives integer-typed answers:

    ========================  ==========  ==========
    strategy                  true        int16 in
    ========================  ==========  ==========
    ``mean``                    155.773     155.0
    ``median``                  158.5       158.0
    ``standard_deviation``       88.802      88.0
    ``variance``               7885.867    7885.0
    ``sum``                   79756       14220        <- wrapped, not truncated
    ========================  ==========  ==========

    (measured on nilearn 0.14.0, one 512-voxel parcel of random 0..299 values.)
    ``minimum``/``maximum`` are exact, being selections rather than arithmetic.

    ``sum`` is the dangerous one — it accumulates in the input dtype, so it wraps:
    an ``int16`` overflows to a *negative* number and a ``uint8`` image of 100s
    sums to 0.  The others merely truncate, which is easy to mistake for rounding.

    Derivatives are normally float32, so none of this shows up on fMRIPrep or
    ASLPrep output — which is exactly why it needs guarding rather than testing
    into.  A float image is returned untouched (no copy), so the cost is paid only
    by the integer inputs that would otherwise be wrong.
    """
    img = nb.load(image) if isinstance(image, str | bytes) else image
    if np.issubdtype(img.get_data_dtype(), np.floating):
        return img
    # The header is copied for its geometry, but its data dtype has to be reset --
    # a header still saying int16 would make nilearn read the array back as int16
    # and undo the cast.
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    return nb.Nifti1Image(np.asanyarray(img.dataobj, dtype='float32'), img.affine, header)
