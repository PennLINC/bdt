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
"""``functional_connectivity`` builder.

Consumes a *parcellated* series (enforced by the spec validator) and produces a
region x region correlation matrix.  A parcel-timeseries TSV is correlated in
Python (Pearson); a parcellated CIFTI (``.ptseries.nii``) goes through
``wb_command -cifti-correlation`` and is then written to a relmat TSV.
"""

from __future__ import annotations

from bdt.actions._io import cifti_to_tsv, is_cifti, tsv_correlation
from bdt.actions._util import ensure_workdir, sources_of, stem_for
from bdt.actions._workbench import run_cifti_correlation
from bdt.actions.parcellate import ctx_runner
from bdt.engine.builders import compose_output_entities, fan_combinations, register_builder
from bdt.engine.result import NodeResult


@register_builder('functional_connectivity')
def build_functional_connectivity(ctx, node, resolved) -> list[NodeResult]:
    results: list[NodeResult] = []
    for by_role in fan_combinations(node, resolved):
        entities, primary = compose_output_entities(node, by_role)
        work = ensure_workdir(ctx, node)
        stem = stem_for(node, entities)
        series = primary.files[0]

        if is_cifti(series):
            pconn = run_cifti_correlation(
                series,
                work / f'{stem}.pconn.nii',
                covariance=bool(node.parameters.get('xdf_covariance', False)),
                runner=ctx_runner(ctx),
            )
            relmat = cifti_to_tsv(pconn, work / f'{stem}_relmat.tsv')
        else:
            relmat = tsv_correlation(series, work / f'{stem}_relmat.tsv')

        results.append(
            NodeResult(
                node=node.name,
                action=node.action,
                fmt=node.action_spec.produces,
                files=[relmat],
                entities=entities,
                sources=sources_of(by_role),
                scope=node.scope,
                space=primary.space,
            )
        )
    return results
