# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for TemplateFlow edge enumeration and fetch (bdt.transforms.templateflow)."""

from bdt.transforms.graph import build_transform_graph
from bdt.transforms.queries import chain_for_image_resample
from bdt.transforms.templateflow import (
    parse_tf_xfm,
    templateflow_edges,
    templateflow_fetch,
)


def test_parse_tf_xfm_target_is_tpl_entity():
    # TF names the TARGET with tpl- and the SOURCE with from-; there is no _to-.
    xfm = parse_tf_xfm('tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5')
    assert (xfm.frm, xfm.to) == ('MNI152NLin2009cAsym', 'MNI152NLin6Asym')
    assert xfm.xfm_type == 'composite'
    assert xfm.invertible is False


def test_parse_tf_xfm_handles_plus_in_template_label():
    xfm = parse_tf_xfm('tpl-MNIInfant+2_from-MNI152NLin6Asym_mode-image_xfm.h5')
    assert (xfm.frm, xfm.to) == ('MNI152NLin6Asym', 'MNIInfant+2')


def test_parse_tf_xfm_ignores_non_tf_and_points():
    assert parse_tf_xfm('sub-01_desc-preproc_T1w.nii.gz') is None
    assert parse_tf_xfm('sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5') is None
    assert parse_tf_xfm('tpl-A_from-B_mode-points_xfm.h5') is None


def test_templateflow_edges_injected_and_feeds_graph():
    def templates_fn():
        return ['MNI152NLin6Asym']

    def ls_fn(tpl, suffix, extension):
        if extension != '.h5':
            return []
        return ['tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5']

    edges = templateflow_edges(templates_fn=templates_fn, ls_fn=ls_fn)
    assert [(e.frm, e.to) for e in edges] == [('MNI152NLin2009cAsym', 'MNI152NLin6Asym')]

    tg = build_transform_graph([], extra_edges=edges)
    steps = chain_for_image_resample(tg, 'MNI152NLin2009cAsym', 'MNI152NLin6Asym')
    assert [(s.frm, s.to) for s in steps] == [('MNI152NLin2009cAsym', 'MNI152NLin6Asym')]


def test_templateflow_fetch_returns_existing_local_file(tmp_path):
    p = tmp_path / 'sub-01_from-A_to-B_mode-image_xfm.mat'
    p.write_bytes(b'#Insight Transform File V1.0\n')  # a real local file has bytes
    assert templateflow_fetch(str(p)) == str(p)


def test_templateflow_fetch_materializes_zero_byte_skeleton_stub(tmp_path):
    # TemplateFlow keeps a 0-byte placeholder for every known file until api.get()
    # downloads it; fetch must treat that as "not materialized" and fetch it.
    stub = tmp_path / 'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5'
    stub.write_bytes(b'')  # exists, but 0 bytes -> not yet downloaded
    calls = {}

    def fake_get(template, **kwargs):
        calls['template'] = template
        calls['kwargs'] = kwargs
        stub.write_bytes(b'\x89HDF\r\n')  # simulate download populating the file
        return str(stub)

    got = templateflow_fetch(str(stub), get_fn=fake_get)
    assert got == str(stub)
    assert calls['template'] == 'MNI152NLin6Asym'
    assert calls['kwargs']['from'] == 'MNI152NLin2009cAsym'


def test_templateflow_fetch_materializes_missing_tf_file(tmp_path):
    calls = {}

    def fake_get(template, **kwargs):
        calls['template'] = template
        calls['kwargs'] = kwargs
        out = tmp_path / 'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5'
        out.write_bytes(b'')
        return str(out)

    got = templateflow_fetch(
        '/does/not/exist/tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        get_fn=fake_get,
    )
    assert got.endswith('_xfm.h5')
    assert calls['template'] == 'MNI152NLin6Asym'
    assert calls['kwargs']['from'] == 'MNI152NLin2009cAsym'
    assert calls['kwargs']['extension'] == '.h5'
