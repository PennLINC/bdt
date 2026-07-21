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

import pytest

from bdt.outputs import (
    DerivativeSink,
    OutputCollisionError,
    bids_name,
    build_sidecar,
    compose_desc,
    generated_by,
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
