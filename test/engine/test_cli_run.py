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
"""Tool-free tests for the ``bdt`` CLI entry point (arg parsing -> run_spec)."""

import pytest

pytest.importorskip('templateflow')  # cli.parser -> config imports templateflow


def test_cli_maps_args_to_run_spec(tmp_path, monkeypatch, capsys):
    """`bdt <bids> <out> participant --datasets … --spec …` calls run_spec correctly."""
    import bdt.engine.pipeline as pipeline
    from bdt.cli.run import main
    from bdt.engine.pipeline import RunResult

    bids = tmp_path / 'bids'
    ds = tmp_path / 'anat'
    out = tmp_path / 'out'
    for d in (bids, ds, out):
        d.mkdir()
    spec = tmp_path / 'spec.yaml'
    spec.write_text('nodes: []\n')

    captured = {}

    def fake_run_spec(
        spec_arg, datasets, output_dir, work_dir, subjects=None, plugin='Linear', plugin_args=None
    ):
        captured.update(
            spec=spec_arg,
            datasets=datasets,
            output_dir=str(output_dir),
            work_dir=str(work_dir),
            subjects=subjects,
            plugin=plugin,
        )
        return [RunResult(subject='125511', selections={}, outputs=[str(output_dir / 'x.tsv')])]

    monkeypatch.setattr(pipeline, 'run_spec', fake_run_spec)

    rc = main(
        [
            str(bids),
            str(out),
            'participant',
            '--datasets',
            f'anat={ds}',
            '--spec',
            str(spec),
            '--participant-label',
            '125511',
            '--work-dir',
            str(tmp_path / 'work'),
        ]
    )

    assert rc == 0
    assert captured['spec'] == str(spec)
    assert captured['datasets'] == {'anat': str(ds)}  # ToDict + stringified
    assert captured['subjects'] == ['125511']
    assert captured['plugin'] == 'Linear'  # nprocs defaults to single-process
    assert 'work' in captured['work_dir']
    printed = capsys.readouterr().out
    assert 'finished' in printed
    assert 'x.tsv' in printed


def test_cli_multiproc_when_nprocs_gt_1(tmp_path, monkeypatch):
    """--nprocs > 1 selects the MultiProc plugin with n_procs."""
    import bdt.engine.pipeline as pipeline
    from bdt.cli.run import main
    from bdt.engine.pipeline import RunResult

    bids = tmp_path / 'bids'
    ds = tmp_path / 'anat'
    for d in (bids, ds):
        d.mkdir()
    spec = tmp_path / 'spec.yaml'
    spec.write_text('nodes: []\n')

    captured = {}

    def fake_run_spec(
        spec_arg, datasets, output_dir, work_dir, subjects=None, plugin='Linear', plugin_args=None
    ):
        captured.update(plugin=plugin, plugin_args=plugin_args)
        return [RunResult(subject=None, selections={}, outputs=[])]

    monkeypatch.setattr(pipeline, 'run_spec', fake_run_spec)

    rc = main(
        [
            str(bids),
            str(tmp_path / 'out'),
            'participant',
            '--datasets',
            f'anat={ds}',
            '--spec',
            str(spec),
            '--nprocs',
            '4',
        ]
    )
    assert rc == 0
    assert captured['plugin'] == 'MultiProc'
    assert captured['plugin_args'] == {'n_procs': 4}


def test_cli_rejects_unsupported_analysis_level(tmp_path):
    """Only participant-level is a valid choice; argparse rejects anything else."""
    from bdt.cli.run import main

    bids = tmp_path / 'bids'
    ds = tmp_path / 'anat'
    for d in (bids, ds):
        d.mkdir()
    spec = tmp_path / 'spec.yaml'
    spec.write_text('nodes: []\n')

    with pytest.raises(SystemExit) as exc:  # argparse invalid-choice exit
        main(
            [
                str(bids),
                str(tmp_path / 'out'),
                'group',
                '--datasets',
                f'anat={ds}',
                '--spec',
                str(spec),
            ]
        )
    assert exc.value.code == 2


def test_cli_errors_without_datasets(tmp_path):
    """A run with no --datasets is a clean user error (exit 2)."""
    from bdt.cli.run import main

    bids = tmp_path / 'bids'
    bids.mkdir()
    spec = tmp_path / 'spec.yaml'
    spec.write_text('nodes: []\n')

    rc = main([str(bids), str(tmp_path / 'out'), 'participant', '--spec', str(spec)])
    assert rc == 2
