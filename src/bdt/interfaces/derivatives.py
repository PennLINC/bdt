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
"""nipype interfaces for materializing BDT derivatives.

Two small :class:`~nipype.interfaces.base.SimpleInterface` wrappers used at the
``write_outputs`` boundary of the compiled workflow:

* :class:`BDTDerivativeSink` copies a produced file to its BIDS-derivative path
  and writes the JSON sidecar, reusing the pure-Python naming in
  :mod:`bdt.outputs.sink` (so the naming stays independently unit-testable).
* :class:`CiftiToTsv` flattens a parcellated CIFTI (ptseries/pconn) to a TSV via
  :func:`bdt.utils.cifti.cifti_to_tsv`, for the tabular product.
"""

from __future__ import annotations

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _BDTDerivativeSinkInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='file to materialize')
    base_directory = traits.Directory(mandatory=True, desc='derivatives output root')
    entities = traits.Dict(mandatory=True, desc='short-name mid-filename entities')
    suffix = traits.Str(mandatory=True, desc='BIDS suffix')
    extension = traits.Str(mandatory=True, desc='file extension, incl. leading dot')
    datatype = traits.Str(mandatory=True, desc='BIDS datatype folder (func/anat/dwi)')
    scope = traits.Enum(
        'participant', 'dataset', usedefault=True, desc='participant- or dataset-scope naming'
    )
    node_name = traits.Str('bdt', usedefault=True, desc='producing node (for messages)')
    sidecar = traits.Dict(desc='JSON sidecar contents (GeneratedBy/Sources)')


class _BDTDerivativeSinkOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the written derivative')


class BDTDerivativeSink(SimpleInterface):
    """Copy ``in_file`` to its BIDS-derivative path and write its JSON sidecar.

    Thin nipype wrapper over :class:`bdt.outputs.sink.DerivativeSink`; the name is
    composed from ``entities`` + ``suffix`` + ``extension`` in canonical order.
    """

    input_spec = _BDTDerivativeSinkInputSpec
    output_spec = _BDTDerivativeSinkOutputSpec

    def _run_interface(self, runtime):
        from bdt.outputs.sink import DerivativeSink

        sink = DerivativeSink(self.inputs.base_directory)
        sidecar = self.inputs.sidecar if isdefined(self.inputs.sidecar) else None
        dest = sink.write(
            node_name=self.inputs.node_name,
            in_file=self.inputs.in_file,
            entities=dict(self.inputs.entities),
            suffix=self.inputs.suffix,
            extension=self.inputs.extension,
            datatype=self.inputs.datatype,
            scope=self.inputs.scope,
            sidecar=sidecar,
        )
        self._results['out_file'] = str(dest)
        return runtime


class _CiftiToTsvInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='parcellated CIFTI (ptseries/pscalar/pconn)')
    out_file = File(desc='output TSV name (optional)')


class _CiftiToTsvOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='flattened TSV')


class CiftiToTsv(SimpleInterface):
    """Flatten a parcellated CIFTI to a TSV (parcel columns)."""

    input_spec = _CiftiToTsvInputSpec
    output_spec = _CiftiToTsvOutputSpec

    def _run_interface(self, runtime):
        from bdt.utils.cifti import cifti_to_tsv

        if isdefined(self.inputs.out_file):
            out_file = os.path.abspath(self.inputs.out_file)
        else:
            stem = os.path.basename(self.inputs.in_file)
            for ext in ('.ptseries.nii', '.pscalar.nii', '.pconn.nii', '.dscalar.nii', '.nii'):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            out_file = os.path.join(runtime.cwd, f'{stem}.tsv')
        cifti_to_tsv(self.inputs.in_file, out_file)
        self._results['out_file'] = out_file
        return runtime
