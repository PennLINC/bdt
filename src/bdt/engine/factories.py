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
"""nipype sub-workflow factories, one per action.

Each factory turns a processing :class:`~bdt.spec.model.Node` into a nipype
``Workflow`` with a standard boundary: an ``inputnode`` whose fields are the
action's declared roles, and an ``outputnode`` with a single ``out`` field
carrying the node's primary product.  The compiler (:mod:`bdt.engine.workflow`)
wires ``inputs[role]`` from an upstream node's ``outputnode.out`` into this
node's ``inputnode.<role>``.

This is the qsirecon ``init_*_wf`` pattern, keyed by *role* rather than by
field-name intersection.  Factories reuse the vendored nipype interfaces
(``bdt.interfaces.workbench`` etc.), so nipype gives us File(exists) validation,
content-hash caching, and resumability for free.
"""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from bdt.interfaces.cifti import CiftiMask, CiftiVertexMask
from bdt.interfaces.workbench import CiftiCorrelation, CiftiMath, CiftiParcellateWorkbench

WORKFLOW_FACTORIES: dict[str, callable] = {}


def workflow_factory(action: str):
    """Register a nipype sub-workflow factory for ``action``."""

    def deco(fn):
        WORKFLOW_FACTORIES[action] = fn
        return fn

    return deco


def _io_nodes(input_fields: list[str]) -> tuple[pe.Node, pe.Node]:
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')
    return inputnode, outputnode


def _init_parcellate_cifti_wf(node, name, in_role: str, out_file: str) -> pe.Workflow:
    """Shared coverage-aware CIFTI parcellation (XCP-D ``init_parcellate_cifti_wf``).

    A vertex-wise coverage mask (1 where a vertex has data) is used both as
    ``-cifti-weights`` for the parcel mean — so uncovered (zero) vertices don't
    dilute it — and, parcellated itself, as the per-parcel coverage fraction.
    Parcels below ``min_coverage`` are set to NaN.  ``inputnode`` takes the data on
    ``in_role`` (+ ``atlas``); ``outputnode`` exposes the masked result (``out``,
    written to ``out_file``: ptseries for a series, pscalar for a scalar) and the
    parcel coverage map (``coverage``).
    """
    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))

    inputnode = pe.Node(niu.IdentityInterface(fields=[in_role, 'atlas']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out', 'coverage']), name='outputnode')

    # 1. vertex-wise coverage (1 = has data, 0 = all-zero/NaN across the map/series)
    vertex_mask = pe.Node(CiftiVertexMask(), name='vertex_mask')

    # 2. per-parcel coverage fraction = mean of the binary vertex mask over the parcel
    parcellate_coverage = pe.Node(
        CiftiParcellateWorkbench(
            direction='COLUMN', only_numeric=True, out_file='coverage.pscalar.nii'
        ),
        name='parcellate_coverage',
    )

    # 3. coverage-weighted parcel mean of the data (uncovered vertices get weight 0)
    parcellate_data = pe.Node(
        CiftiParcellateWorkbench(direction='COLUMN', only_numeric=True, out_file=out_file),
        name='parcellate_data',
    )

    # 4. threshold the coverage -> a 0/1 parcel mask
    threshold = pe.Node(
        CiftiMath(expression=f'data > {min_coverage}', out_file='coverage_mask.pscalar.nii'),
        name='threshold',
    )

    # 5. NaN-out parcels below the coverage threshold
    mask = pe.Node(CiftiMask(), name='mask')

    wf.connect([
        (inputnode, vertex_mask, [(in_role, 'in_file')]),
        (inputnode, parcellate_coverage, [('atlas', 'atlas_label')]),
        (vertex_mask, parcellate_coverage, [('mask_file', 'in_file')]),
        (inputnode, parcellate_data, [(in_role, 'in_file'), ('atlas', 'atlas_label')]),
        (vertex_mask, parcellate_data, [('mask_file', 'cifti_weights')]),
        (parcellate_coverage, threshold, [('out_file', 'data')]),
        (parcellate_data, mask, [('out_file', 'in_file')]),
        (threshold, mask, [('out_file', 'mask')]),
        (mask, outputnode, [('out_file', 'out')]),
        (parcellate_coverage, outputnode, [('out_file', 'coverage')]),
    ])  # fmt:skip
    return wf


@workflow_factory('parcellate_timeseries')
def init_parcellate_timeseries_wf(node, name=None) -> pe.Workflow:
    """Parcellate a dense CIFTI series with a dlabel atlas, coverage-aware -> ptseries."""
    return _init_parcellate_cifti_wf(node, name, 'timeseries', 'parcellated.ptseries.nii')


@workflow_factory('parcellate_scalar')
def init_parcellate_scalar_wf(node, name=None) -> pe.Workflow:
    """Parcellate a dense CIFTI scalar (dscalar) with a dlabel atlas -> pscalar.

    Same coverage-aware machinery as :func:`init_parcellate_timeseries_wf`.  CIFTI
    inputs only for now (the volumetric/NIfTI Strategy-A path is a follow-up).
    """
    return _init_parcellate_cifti_wf(node, name, 'scalar', 'parcellated.pscalar.nii')


@workflow_factory('functional_connectivity')
def init_functional_connectivity_wf(node, name=None) -> pe.Workflow:
    """Correlate a parcellated series (ptseries) -> pconn relmat."""
    wf = pe.Workflow(name=name or node.name)
    inputnode, outputnode = _io_nodes(['timeseries'])
    correlate = pe.Node(CiftiCorrelation(out_file='correlations.pconn.nii'), name='correlate')
    wf.connect([
        (inputnode, correlate, [('timeseries', 'in_file')]),
        (correlate, outputnode, [('out_file', 'out')]),
    ])  # fmt:skip
    return wf
