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
"""Tests for the BIDS output naming + provenance sink."""

import json
import os

import pytest

from bdt import __version__
from bdt.outputs import (
    DerivativeSink,
    OutputCollisionError,
    bids_name,
    build_sidecar,
    compose_desc,
    dataset_generated_by,
    generated_by,
    write_dataset_description,
)


def test_compose_desc():
    assert compose_desc(None, None) is None
    assert compose_desc(None, 'strict') == 'strict'
    assert compose_desc('geneexpression', None) == 'geneexpression'
    assert compose_desc('geneexpression', 'strict') == 'geneexpressionStrict'
    # compounding stays a single alphanumeric token
    assert compose_desc('geneexpressionStrict', 'smoothed') == 'geneexpressionStrictSmoothed'


def test_bids_name_ordering():
    name = bids_name(
        'sub-01',
        {'space': 'fsLR', 'atlas': 'HCPMMP1', 'statistic': 'mean', 'desc': 'x', 'den': '32k'},
        'map',
        '.tsv',
    )
    # space, then den, then atlas, then stat, then desc (2026-07-16 decision)
    assert name == 'sub-01_space-fsLR_den-32k_atlas-HCPMMP1_stat-mean_desc-x_map.tsv'


def test_write_participant(tmp_path):
    src = tmp_path / 'in.tsv'
    src.write_text('data')
    sink = DerivativeSink(tmp_path / 'out')
    gb = [generated_by('parc', 'parcellate_timeseries', {'min_coverage': 0.5})]
    dest = sink.write(
        node_name='parc',
        in_file=src,
        entities={'subject': '01', 'task': 'rest', 'space': 'fsLR', 'atlas': '4S456Parcels'},
        suffix='timeseries',
        extension='.tsv',
        datatype='func',
        sidecar=build_sidecar(['bids:xcpd:sub-01/func/x_bold.dtseries.nii'], gb),
    )
    rel = dest.relative_to(tmp_path / 'out').as_posix()
    assert rel == ('sub-01/func/sub-01_task-rest_space-fsLR_atlas-4S456Parcels_timeseries.tsv')
    sidecar = dest.parent / 'sub-01_task-rest_space-fsLR_atlas-4S456Parcels_timeseries.json'
    meta = json.loads(sidecar.read_text())
    assert meta['GeneratedBy'][0]['Action'] == 'parcellate_timeseries'
    assert meta['Sources'] == ['bids:xcpd:sub-01/func/x_bold.dtseries.nii']


def test_write_dataset_scope_tpl(tmp_path):
    src = tmp_path / 'in.tsv'
    src.write_text('data')
    sink = DerivativeSink(tmp_path / 'out')
    dest = sink.write(
        node_name='geneexpr_parc',
        in_file=src,
        entities={'space': 'fsLR', 'atlas': 'HCPMMP1', 'desc': 'geneexpression'},
        suffix='map',
        extension='.tsv',
        datatype='func',
        scope='dataset',
    )
    rel = dest.relative_to(tmp_path / 'out').as_posix()
    # dataset scope -> tpl-<space>/, space folded into the tpl- label
    assert rel == 'tpl-fsLR/func/tpl-fsLR_atlas-HCPMMP1_desc-geneexpression_map.tsv'


def test_collision_is_error(tmp_path):
    src = tmp_path / 'in.tsv'
    src.write_text('data')
    sink = DerivativeSink(tmp_path / 'out')
    kwargs = {
        'in_file': src,
        'entities': {'subject': '01', 'atlas': 'A'},
        'suffix': 'timeseries',
        'extension': '.tsv',
        'datatype': 'func',
    }
    sink.write(node_name='a', **kwargs)
    with pytest.raises(OutputCollisionError, match='disambiguating desc'):
        sink.write(node_name='b', **kwargs)


def _write_input_desc(root, generated_by):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'dataset_description.json').write_text(
        json.dumps({'Name': root.name, 'GeneratedBy': generated_by})
    )


def test_dataset_generated_by_basic(monkeypatch):
    monkeypatch.delenv('BDT_DOCKER_TAG', raising=False)
    monkeypatch.delenv('BDT_SINGULARITY_URL', raising=False)
    rec = dataset_generated_by()
    assert rec['Name'] == 'BDT'
    assert rec['Version'] == __version__
    assert rec['CodeURL'].endswith(f'{__version__}.tar.gz')
    assert 'Container' not in rec


def test_dataset_generated_by_docker_container(monkeypatch):
    monkeypatch.setenv('BDT_DOCKER_TAG', '1.2.3')
    monkeypatch.delenv('BDT_SINGULARITY_URL', raising=False)
    rec = dataset_generated_by()
    assert rec['Container'] == {'Type': 'docker', 'Tag': 'nipreps/bdt:1.2.3'}


def test_dataset_generated_by_singularity_container(monkeypatch):
    monkeypatch.delenv('BDT_DOCKER_TAG', raising=False)
    monkeypatch.setenv('BDT_SINGULARITY_URL', 'docker://nipreps/bdt:1.2.3')
    rec = dataset_generated_by()
    assert rec['Container'] == {'Type': 'singularity', 'URI': 'docker://nipreps/bdt:1.2.3'}


def test_write_dataset_description_aggregates_and_links(tmp_path, monkeypatch):
    monkeypatch.delenv('BDT_DOCKER_TAG', raising=False)
    monkeypatch.delenv('BDT_SINGULARITY_URL', raising=False)
    a = tmp_path / 'A'
    _write_input_desc(a, [{'Name': 'smriprep', 'Version': '0.1'}])
    b = tmp_path / 'B'  # exists but has no dataset_description.json
    b.mkdir()
    tf = tmp_path / 'tf'
    tf.mkdir()
    bids = tmp_path / 'rawbids'
    bids.mkdir()
    out = tmp_path / 'out'
    out.mkdir()

    dest = write_dataset_description(
        out,
        {'A': str(a), 'B': str(b), 'templateflow': str(tf)},
        bids_dir=str(bids),
    )
    assert dest == out / 'dataset_description.json'
    desc = json.loads(dest.read_text())
    assert desc['DatasetType'] == 'derivative'
    assert desc['BIDSVersion'] == '1.10.0'
    assert desc['Name'] == 'BDT derivatives'
    # BDT record first; inherited smriprep entry present
    assert desc['GeneratedBy'][0]['Name'] == 'BDT'
    assert {'Name': 'smriprep', 'Version': '0.1'} in desc['GeneratedBy']
    # DatasetLinks: absolute paths, templateflow URL, raw link
    assert desc['DatasetLinks']['A'] == os.path.abspath(str(a))
    assert desc['DatasetLinks']['B'] == os.path.abspath(str(b))
    assert desc['DatasetLinks']['templateflow'] == ('https://github.com/templateflow/templateflow')
    assert desc['DatasetLinks']['raw'] == os.path.abspath(str(bids))


def test_write_dataset_description_dedups_generated_by(tmp_path):
    entry = {'Name': 'smriprep', 'Version': '0.1'}
    a = tmp_path / 'A'
    _write_input_desc(a, [entry])
    b = tmp_path / 'B'
    _write_input_desc(b, [entry])  # same entry surfaced by two inputs
    out = tmp_path / 'out'
    out.mkdir()
    gb = json.loads(write_dataset_description(out, {'A': str(a), 'B': str(b)}).read_text())[
        'GeneratedBy'
    ]
    assert gb.count(entry) == 1
    assert gb[0]['Name'] == 'BDT'


def test_write_dataset_description_raw_key_collision(tmp_path):
    # a --datasets key literally named 'raw' wins; bids_dir link is skipped
    raw_ds = tmp_path / 'rawds'
    raw_ds.mkdir()
    bids = tmp_path / 'rawbids'
    bids.mkdir()
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.warns(UserWarning, match="A --datasets key named 'raw' shadows"):
        dest = write_dataset_description(out, {'raw': str(raw_ds)}, bids_dir=str(bids))
    links = json.loads(dest.read_text())['DatasetLinks']
    assert links['raw'] == os.path.abspath(str(raw_ds))


def test_write_dataset_description_ignores_malformed_input(tmp_path):
    a = tmp_path / 'A'
    a.mkdir()
    (a / 'dataset_description.json').write_text('{ not valid json')
    out = tmp_path / 'out'
    out.mkdir()
    gb = json.loads(write_dataset_description(out, {'A': str(a)}).read_text())['GeneratedBy']
    assert len(gb) == 1
    assert gb[0]['Name'] == 'BDT'


def test_write_dataset_description_ignores_non_object_input(tmp_path):
    a = tmp_path / 'A'
    a.mkdir()
    (a / 'dataset_description.json').write_text('[1, 2, 3]')  # valid JSON, not an object
    out = tmp_path / 'out'
    out.mkdir()
    gb = json.loads(write_dataset_description(out, {'A': str(a)}).read_text())['GeneratedBy']
    assert len(gb) == 1
    assert gb[0]['Name'] == 'BDT'
