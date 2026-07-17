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
"""BDT command-line entry point.

Parses the BIDS-App-style invocation
``bdt <bids_dir> <output_dir> participant --datasets k=path … --spec spec.yaml``
and runs the node-graph pipeline (:func:`bdt.engine.pipeline.run_spec`): load +
statically validate the spec, resolve each selection against the ``--datasets``
roots, compile to a nipype ``Workflow``, and execute it per participant.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a BDT spec; return a process exit code."""
    from bdt.cli.parser import _build_parser
    from bdt.engine.pipeline import run_spec
    from bdt.engine.selection import SelectionError
    from bdt.spec.model import SpecError

    # The parser restricts ``analysis_level`` to ``participant`` (the only scope the
    # node-graph engine runs today), so an invalid level exits here via argparse.
    opts = _build_parser().parse_args(argv)

    if not opts.datasets:
        print('bdt: at least one --datasets KEY=PATH is required.', file=sys.stderr)
        return 2

    datasets = {k: str(v) for k, v in opts.datasets.items()}
    work_dir = opts.work_dir / 'bdt_wf'
    # A single worker runs the fast, single-process Linear plugin; >1 fans the
    # nipype graph across processes with MultiProc.
    nprocs = opts.nprocs or 1
    plugin = 'MultiProc' if nprocs > 1 else 'Linear'
    plugin_args = {'n_procs': nprocs} if plugin == 'MultiProc' else None

    print(f'bdt: running spec {opts.spec!r}')
    print(f'bdt: datasets = {datasets}')
    print(f'bdt: output   = {opts.output_dir}')
    print(f'bdt: subjects = {opts.participant_label or "(auto-discover)"}')
    print(f'bdt: plugin   = {plugin} (n_procs={nprocs})')

    try:
        results = run_spec(
            opts.spec,
            datasets,
            opts.output_dir,
            work_dir,
            subjects=opts.participant_label or None,
            plugin=plugin,
            plugin_args=plugin_args,
        )
    except (SpecError, SelectionError, KeyError) as exc:
        # spec/validation and resolution errors are user errors -> a clean message
        print(f'bdt: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

    n_outputs = sum(len(r.outputs) for r in results)
    print(f'bdt: finished — {len(results)} scope(s), {n_outputs} derivative file(s) written.')
    for r in results:
        label = r.subject or 'dataset'
        for path in r.outputs:
            print(f'  [{label}] {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
