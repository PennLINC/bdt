# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""nipype interfaces for building a bundle-wise segmentation from tractograms."""

import os
import re

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _ConcatenateNiftisInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='3D maps to stack (one volume per bundle)'
    )
    normalize = traits.Bool(
        True, usedefault=True, desc='peak-normalize each map to [0, 1] before stacking'
    )
    out_file = File('concatenated.nii.gz', usedefault=True, desc='output 4D file name')


class _ConcatenateNiftisOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='4D NIfTI, one input volume per 4th-dim index')


class ConcatenateNiftis(SimpleInterface):
    """Stack 3D maps into a 4D volume, optionally peak-normalizing each to [0, 1]."""

    input_spec = _ConcatenateNiftisInputSpec
    output_spec = _ConcatenateNiftisOutputSpec

    def _run_interface(self, runtime):
        first = nb.load(self.inputs.in_files[0])
        vols = []
        for path in self.inputs.in_files:
            img = nb.load(path)
            data = np.asarray(img.dataobj, dtype=np.float32)
            if data.shape != first.shape:
                raise ValueError(
                    f'Shape mismatch: {path} has {data.shape}, expected {first.shape}.'
                )
            if self.inputs.normalize:
                peak = float(data.max())
                if peak > 0:
                    data = data / peak
            vols.append(data)
        stacked = np.stack(vols, axis=-1).astype(np.float32)
        out_file = os.path.abspath(self.inputs.out_file)
        out_img = nb.Nifti1Image(stacked, first.affine, first.header)
        out_img.header.set_data_dtype(np.float32)
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


class _ThresholdNiftiInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='image to threshold')
    threshold = traits.Float(mandatory=True, desc='keep values strictly greater than this')
    binarize = traits.Bool(
        True, usedefault=True, desc='emit a 0/1 mask instead of the masked values'
    )
    out_file = File('thresholded.nii.gz', usedefault=True, desc='output file name')


class _ThresholdNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='thresholded image')


class ThresholdNifti(SimpleInterface):
    """Threshold an image at ``value > threshold``; optionally binarize to a 0/1 mask."""

    input_spec = _ThresholdNiftiInputSpec
    output_spec = _ThresholdNiftiOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = np.asarray(img.dataobj, dtype=np.float32)
        mask = data > self.inputs.threshold
        if self.inputs.binarize:
            out_data = mask.astype(np.uint8)
            dtype = np.uint8
        else:
            out_data = np.where(mask, data, 0).astype(np.float32)
            dtype = np.float32
        out_file = os.path.abspath(self.inputs.out_file)
        out_img = nb.Nifti1Image(out_data, img.affine, img.header)
        out_img.header.set_data_dtype(dtype)
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def _resample_streamline(points: np.ndarray, n_nodes: int) -> np.ndarray:
    """Resample one ``(N, 3)`` streamline to ``n_nodes`` equally arc-spaced points."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total == 0.0:  # degenerate (single point / zero length)
        return np.repeat(points[:1], n_nodes, axis=0)
    targets = np.linspace(0.0, total, n_nodes)
    return np.column_stack([np.interp(targets, arc, points[:, d]) for d in range(3)])


def _orient_like(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip a streamline so its endpoints run the same way as ``reference``.

    A simplified stand-in for pyAFQ/DIPY orientation: streamlines are reconstructed
    in arbitrary direction, so align each to a reference before averaging per node.
    """
    same = np.linalg.norm(points[0] - reference[0]) + np.linalg.norm(points[-1] - reference[-1])
    flip = np.linalg.norm(points[0] - reference[-1]) + np.linalg.norm(points[-1] - reference[0])
    return points[::-1] if flip < same else points


class _SampleTractProfilesInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='per-bundle .tck tractograms (bundle space)'
    )
    scalar = File(
        exists=True, mandatory=True, desc='scalar volume to sample, in the bundles\' space'
    )
    n_nodes = traits.Int(100, usedefault=True, desc='number of along-tract nodes')
    entity = traits.Str('bundle', usedefault=True, desc='BIDS entity key naming each bundle')
    out_file = File('tractprofile.tsv', usedefault=True, desc='output along-tract profile TSV')


class _SampleTractProfilesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tidy TSV: bundle, node, mean, std')


class SampleTractProfiles(SimpleInterface):
    """Sample a scalar volume along bundles into a per-node along-tract profile.

    For each bundle, every streamline is resampled to ``n_nodes`` equally arc-spaced
    points, oriented consistently, and the scalar is trilinearly sampled at those
    points (world→voxel via the image affine); the per-node ``mean``/``std`` across
    streamlines form the profile.  Emits a tidy TSV (one row per bundle × node).

    This is a self-contained profiler (nibabel + scipy) — a simplified,
    unweighted stand-in for pyAFQ/DIPY tractometry (which are not installed here);
    it assumes the scalar already lives in the bundles' space.
    """

    input_spec = _SampleTractProfilesInputSpec
    output_spec = _SampleTractProfilesOutputSpec

    def _run_interface(self, runtime):
        import nibabel.streamlines
        from scipy.ndimage import map_coordinates

        img = nb.load(self.inputs.scalar)
        data = np.asarray(img.dataobj, dtype=np.float32)
        inv_affine = np.linalg.inv(img.affine)
        n_nodes = self.inputs.n_nodes
        key = self.inputs.entity
        pattern = re.compile(rf'(?:^|[_/]){re.escape(key)}-([a-zA-Z0-9]+)')

        rows = []
        for path in self.inputs.in_files:
            name = os.path.basename(path)
            match = pattern.search(name)
            if match is None:
                raise ValueError(f'No {key!r} entity found in filename {name!r}.')
            bundle = match.group(1)

            streamlines = nibabel.streamlines.load(path).streamlines
            if len(streamlines) == 0:
                continue
            reference = _resample_streamline(np.asarray(streamlines[0]), n_nodes)
            samples = np.empty((len(streamlines), n_nodes), dtype=np.float32)
            for i, sl in enumerate(streamlines):
                pts = _orient_like(_resample_streamline(np.asarray(sl), n_nodes), reference)
                vox = nb.affines.apply_affine(inv_affine, pts)
                samples[i] = map_coordinates(
                    data, vox.T, order=1, mode='constant', cval=np.nan
                )

            mean = np.nanmean(samples, axis=0)
            std = np.nanstd(samples, axis=0)
            for node in range(n_nodes):
                rows.append(
                    {'bundle': bundle, 'node': node + 1, 'mean': mean[node], 'std': std[node]}
                )

        out_file = os.path.abspath(self.inputs.out_file)
        pd.DataFrame(rows, columns=['bundle', 'node', 'mean', 'std']).to_csv(
            out_file, sep='\t', index=False
        )
        self._results['out_file'] = out_file
        return runtime


class _EntitiesToSegTSVInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='files whose <entity>-<value> key names each segment (one per volume)',
    )
    entity = traits.Str('bundle', usedefault=True, desc='BIDS entity key to read from each name')
    out_file = File('dseg.tsv', usedefault=True, desc='output BIDS label TSV')


class _EntitiesToSegTSVOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='BIDS label TSV with columns index, name')


class EntitiesToSegTSV(SimpleInterface):
    """Build a BIDS ``index``/``name`` label TSV from a BIDS entity in each filename.

    Row order follows ``in_files`` order, so index ``i`` (1-based) names the ``i``-th
    volume of the matching 4D segmentation.
    """

    input_spec = _EntitiesToSegTSVInputSpec
    output_spec = _EntitiesToSegTSVOutputSpec

    def _run_interface(self, runtime):
        key = self.inputs.entity
        pattern = re.compile(rf'(?:^|[_/]){re.escape(key)}-([a-zA-Z0-9]+)')
        rows = []
        for i, path in enumerate(self.inputs.in_files, start=1):
            name = os.path.basename(path)
            match = pattern.search(name)
            if match is None:
                raise ValueError(f'No {key!r} entity found in filename {name!r}.')
            rows.append({'index': i, 'name': match.group(1)})
        out_file = os.path.abspath(self.inputs.out_file)
        pd.DataFrame(rows, columns=['index', 'name']).to_csv(out_file, sep='\t', index=False)
        self._results['out_file'] = out_file
        return runtime
