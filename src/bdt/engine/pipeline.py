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
"""Pipeline entry: resolve selections, compile, and run a BDT spec.

:func:`run_spec` ties the pieces together — load + statically validate a spec,
resolve each selection node against the ``--datasets`` roots with a
:class:`~bdt.engine.selection.DataProvider`, plan the outputs
(:func:`bdt.outputs.plan.build_sink_plan`), compile to a nipype ``Workflow``
(:func:`bdt.engine.workflow.init_bdt_wf`), and run it.

A multi-match selection *fans out*: the driver runs one workflow per combination
of selection matches (the Cartesian product), which covers the "all tasks / all
params" case without the in-graph ``MapNode``/``iterables`` fan-out that is the
richer follow-up.  The provider is injectable so the driver is testable without
pybids.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

from bdt.engine.selection import DataProvider, Match, SelectionError
from bdt.spec.model import DATASET, Spec, parse_spec
from bdt.spec.validate import validate_spec


@dataclass
class RunResult:
    """One executed scope: a subject × selection-combination and what it wrote."""

    subject: str | None
    selections: dict[str, str]
    outputs: list[str] = field(default_factory=list)


def _as_spec(spec) -> Spec:
    if isinstance(spec, Spec):
        return spec
    if isinstance(spec, dict):
        return parse_spec(spec)
    from bdt.spec.load import load_spec

    return load_spec(spec)


def _discover_subjects(spec: Spec, provider: DataProvider) -> list[str]:
    """Subjects to iterate: the union across participant-scope selection datasets."""
    datasets = {
        n.dataset
        for n in spec.nodes
        if n.is_selection and n.dataset is not None and n.scope != DATASET
    }
    subjects: dict[str, None] = {}
    for dataset in sorted(datasets):
        subjects_fn = getattr(provider, 'subjects', None)
        if subjects_fn is None:
            continue
        for sub in subjects_fn(dataset):
            subjects.setdefault(sub, None)
    return list(subjects)


def _resolve_selections(
    spec: Spec, provider: DataProvider, subject: str | None
) -> dict[str, list[Match]]:
    """Resolve every selection node to its matches for one subject."""
    resolved: dict[str, list[Match]] = {}
    for node in spec.all_nodes:
        if not node.is_selection:
            continue
        scope_subject = None if node.scope == DATASET else subject
        matches = provider.select(node.dataset, node.filters, node.exclude, subject=scope_subject)
        if not matches:
            raise SelectionError(
                f'Selection node {node.name!r} (dataset {node.dataset!r}, filters '
                f'{node.filters}) matched no files'
                + (f' for subject {subject!r}.' if scope_subject else '.')
            )
        resolved[node.name] = matches
    return resolved


def _classify_selections(spec: Spec) -> dict[str, bool]:
    """Map each selection node name to whether it is *grouped* (vs. *fanned*).

    A selection is **grouped** iff every processing role that consumes it declares
    ``fan_out=False`` — its matches are then passed as a single *list* to that role
    (e.g. the L/R ``surfaces`` set, or the L/R native ``surface_scalar`` combined
    into one dscalar).  Otherwise it **fans out**: the driver takes the Cartesian
    product of its matches (one downstream branch per file — the story-3.1 case of
    one output per task/atlas).  A selection with no consumer fans (degenerate).
    """
    selection_names = {n.name for n in spec.all_nodes if n.is_selection}
    consumer_flags: dict[str, list[bool]] = {name: [] for name in selection_names}
    for node in spec.all_nodes:
        if node.is_selection:
            continue
        aspec = node.action_spec
        for role_name, upstream in node.inputs.items():
            role = aspec.role(role_name) if aspec is not None else None
            fans = True if role is None else role.fan_out
            for up in upstream:
                if up in consumer_flags:
                    consumer_flags[up].append(fans)
    # grouped iff it has ≥1 consumer and none of them fan
    return {name: (bool(flags) and not any(flags)) for name, flags in consumer_flags.items()}


def _combinations(resolved: dict[str, list[Match]], grouped: dict[str, bool] | None = None):
    """One dict per scope: Cartesian product over *fanned* selections.

    A fanned selection contributes a single :class:`Match` to each combination; a
    *grouped* selection (per :func:`_classify_selections`) contributes its whole
    match *list* to every combination unchanged.  ``grouped`` defaults to empty, so
    the historical "fan everything" behaviour is preserved when it is omitted.
    """
    grouped = grouped or {}
    fanned = [n for n in resolved if not grouped.get(n, False)]
    group = [n for n in resolved if grouped.get(n, False)]
    for combo in itertools.product(*(resolved[name] for name in fanned)):
        out: dict[str, Match | list[Match]] = dict(zip(fanned, combo, strict=True))
        for name in group:
            out[name] = resolved[name]  # the full list of matches, grouped
        yield out


def _check_collisions(spec: Spec, plan: dict, base_directory: str) -> None:
    """Raise if two planned products resolve to the same derivative path."""
    from bdt.outputs.sink import DerivativeSink

    sink = DerivativeSink(base_directory)
    for node_name, products in plan.items():
        for product in products:
            # relpath() records nothing; use the sink's own bookkeeping via a probe.
            rel = sink.relpath(
                dict(product.entities),
                product.suffix,
                product.extension,
                product.datatype,
                product.scope,
            )
            if rel in sink._written:
                raise ValueError(
                    f'Output collision: node {node_name!r} product resolves to {rel!r}, '
                    f'already produced by node {sink._written[rel]!r}. Add a disambiguating desc:.'
                )
            sink._written[rel] = node_name


def run_spec(
    spec,
    datasets: dict[str, str],
    output_dir: str | Path,
    work_dir: str | Path,
    subjects: list[str] | None = None,
    provider: DataProvider | None = None,
    plugin: str = 'Linear',
    plugin_args: dict | None = None,
    validate: bool = True,
) -> list[RunResult]:
    """Resolve, compile, and run ``spec``; return a :class:`RunResult` per scope.

    Parameters
    ----------
    spec
        A :class:`~bdt.spec.model.Spec`, a raw dict, or a path / pre-packaged name.
    datasets
        ``{key: root_path}`` for every ``--datasets`` reference in the spec.
    output_dir, work_dir
        Derivatives root and the nipype working directory.
    subjects
        Participant labels to run; discovered from the datasets when ``None``.
    provider
        A :class:`~bdt.engine.selection.DataProvider`; a pybids-backed one over
        ``datasets`` is built when ``None``.
    plugin, plugin_args
        Passed to ``Workflow.run`` (default the single-process ``Linear``).
    """
    from bdt.engine.factories import FactoryContext
    from bdt.engine.workflow import init_bdt_wf
    from bdt.outputs.plan import build_sink_plan

    spec = _as_spec(spec)
    if validate:
        validate_spec(spec, datasets=set(datasets))

    output_dir = Path(output_dir)
    work_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if provider is None:
        from bdt.engine.pybids_provider import BIDSDataProvider

        provider = BIDSDataProvider(datasets, database_dir=work_dir / 'bids_db')

    roots = {k: str(v) for k, v in datasets.items()}

    run_subjects: list[str | None]
    if subjects is not None:
        run_subjects = list(subjects)
    else:
        run_subjects = _discover_subjects(spec, provider) or [None]

    grouped = _classify_selections(spec)
    results: list[RunResult] = []
    for subject in run_subjects:
        resolved_all = _resolve_selections(spec, provider, subject)
        for combo_idx, combo in enumerate(_combinations(resolved_all, grouped)):
            # grouped selection -> list of paths for its role; fanned -> single path
            sel_paths = {
                name: ([m.path for m in val] if isinstance(val, list) else val.path)
                for name, val in combo.items()
            }
            # one representative Match per selection seeds output naming (grouped
            # matches share their non-hemi entities; hemi is dropped downstream)
            combo_repr = {
                name: (val[0] if isinstance(val, list) else val) for name, val in combo.items()
            }
            plan = build_sink_plan(spec, combo_repr, roots)
            _check_collisions(spec, plan, str(output_dir))

            wf_name = f'bdt_{subject or "dataset"}_{combo_idx}'
            context = FactoryContext(
                provider=provider,
                subject=subject,
                spec=spec,
                datasets=list(datasets),
                resolved=combo_repr,
            )
            wf = init_bdt_wf(
                spec,
                sel_paths,
                name=wf_name,
                base_directory=str(output_dir),
                sink_plan=plan,
                context=context,
            )
            wf.base_dir = str(work_dir)
            run_kwargs = {'plugin': plugin}
            if plugin_args:
                run_kwargs['plugin_args'] = plugin_args
            wf.run(**run_kwargs)

            outputs = _planned_outputs(plan, str(output_dir))
            results.append(RunResult(subject=subject, selections=sel_paths, outputs=outputs))
    return results


def _planned_outputs(plan: dict, base_directory: str) -> list[str]:
    """Absolute paths the plan should have written (for verification/return)."""
    from bdt.outputs.sink import DerivativeSink

    sink = DerivativeSink(base_directory)
    paths: list[str] = []
    for products in plan.values():
        for product in products:
            rel = sink.relpath(
                dict(product.entities),
                product.suffix,
                product.extension,
                product.datatype,
                product.scope,
            )
            paths.append(str(Path(base_directory) / rel))
    return paths
