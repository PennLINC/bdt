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
"""Top-level orchestration: spec -> transform graph -> provider -> executor -> sink.

:func:`run_spec` is the single entry point the CLI calls (and that tests drive
with an injected provider).  It wires the four framework-agnostic pieces together
and runs the graph for each requested subject.  Dependencies are injectable so
the whole assembly is unit-testable without pybids / a real dataset.
"""

from __future__ import annotations

from pathlib import Path

from bdt.engine.executor import Executor
from bdt.engine.result import BuildContext
from bdt.engine.selection import DataProvider
from bdt.outputs.sink import DerivativeSink
from bdt.spec.load import load_spec
from bdt.spec.model import Spec


def run_spec(
    spec: str | Path | Spec,
    datasets: dict[str, str | Path],
    output_dir: str | Path,
    *,
    work_dir: str | Path | None = None,
    participant_labels: list[str] | None = None,
    provider: DataProvider | None = None,
    transform_graph=None,
    configure_niwrap: bool = False,
    builders: dict | None = None,
    use_builtin_actions: bool = True,
) -> dict:
    """Load, validate, and run a BDT spec.

    Parameters
    ----------
    spec
        A :class:`~bdt.spec.model.Spec`, or a path / pre-packaged name to load.
    datasets
        ``{key: path}`` for the ``--datasets`` roots (keys are what selection
        nodes reference).
    output_dir
        Derivatives output root.
    work_dir
        Scratch directory for tool intermediates (and NiWrap's data dir).
    participant_labels
        Subjects to run; when ``None`` they are discovered from the datasets.
    provider
        Injected :class:`DataProvider`; defaults to a pybids-backed provider.
    transform_graph
        Injected transform graph; defaults to scanning the dataset roots.
    configure_niwrap
        When true, configure NiWrap's local runner before executing.
    builders
        Explicit ``{action: builder}`` registry.  Overrides ``use_builtin_actions``.
    use_builtin_actions
        When true (and ``builders`` is not given), use the real registered action
        builders (:mod:`bdt.actions`); when false, every node uses the passthrough
        builder (useful for structural tests that have no external tools).
    """
    if not isinstance(spec, Spec):
        spec = load_spec(spec)

    if builders is None and use_builtin_actions:
        import bdt.actions  # noqa: F401 - populates the builder registry on import
        from bdt.engine.builders import BUILDERS

        builders = dict(BUILDERS)

    if provider is None:
        from bdt.engine.pybids_provider import BIDSDataProvider

        provider = BIDSDataProvider(datasets)

    if transform_graph is None:
        from bdt.transforms.graph import build_transform_graph

        transform_graph = build_transform_graph(list(datasets.values()))

    if configure_niwrap:
        from bdt.tools.niwrap_run import configure_runner

        configure_runner(work_dir)

    subjects = participant_labels or _discover_subjects(provider, datasets)

    sink = DerivativeSink(output_dir)
    ctx = BuildContext(
        transform_graph=transform_graph,
        work_dir=str(work_dir) if work_dir else None,
    )
    executor = Executor(
        spec, provider, sink, datasets=set(datasets), context=ctx, builders=builders
    )
    return executor.run(subjects)


def _discover_subjects(provider: DataProvider, datasets: dict) -> list[str]:
    """Union of subjects across all datasets that expose a ``subjects()`` method."""
    subjects: list[str] = []
    getter = getattr(provider, 'subjects', None)
    if getter is None:
        return subjects
    for key in datasets:
        try:
            found = list(getter(key))
        except Exception:  # noqa: BLE001, S112 - a dataset may be unindexable / atlas-only
            continue
        for sub in found:
            if sub not in subjects:
                subjects.append(sub)
    return sorted(subjects)
