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
"""Tests for the transform graph and the two typed queries."""

import re

import pytest

from bdt.transforms import (
    NoTransformPathError,
    build_transform_graph,
    chain_for_image_resample,
    chain_for_point_warp,
)
from bdt.transforms.graph import parse_xfm_filename


def _touch(directory, name):
    p = directory / name
    p.write_bytes(b'')
    return p


def _graph(tmp_path, names):
    for name in names:
        _touch(tmp_path, name)
    return build_transform_graph(tmp_path)


def _pairs(steps):
    return [(s.frm, s.to) for s in steps]


# ---------------------------------------------------------------------------
# Filename parsing / scanning
# ---------------------------------------------------------------------------


def test_parse_and_classify():
    aff = parse_xfm_filename('sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.mat')
    assert (aff.frm, aff.to, aff.xfm_type, aff.invertible) == (
        'T1w',
        'MNI152NLin6Asym',
        'affine',
        True,
    )
    warp = parse_xfm_filename('sub-01_from-T1w_to-MNI152NLin6Asym_xfm.nii.gz')
    assert (warp.xfm_type, warp.invertible) == ('warp', False)
    comp = parse_xfm_filename('sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5')
    assert (comp.xfm_type, comp.invertible) == ('composite', False)
    # mode-points is skipped; non-xfm files ignored
    assert parse_xfm_filename('sub-01_from-T1w_to-MNI_mode-points_xfm.h5') is None
    assert parse_xfm_filename('sub-01_desc-preproc_T1w.nii.gz') is None


def test_scan_ignores_non_xfm(tmp_path):
    _touch(tmp_path, 'sub-01_desc-preproc_T1w.nii.gz')
    _touch(tmp_path, 'sub-01_from-ACPC_to-MNI152NLin6Asym_mode-image_xfm.h5')
    _touch(tmp_path, 'sub-01_from-ACPC_to-MNI152NLin6Asym_mode-points_xfm.h5')  # skipped
    tg = build_transform_graph(tmp_path)
    assert tg.has_edge('ACPC', 'MNI152NLin6Asym')
    assert tg.g.number_of_edges() == 1


# ---------------------------------------------------------------------------
# Image resample (Strategy A)
# ---------------------------------------------------------------------------


def test_image_single_hop_direct(tmp_path):
    tg = _graph(tmp_path, ['sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5'])
    steps = chain_for_image_resample(tg, 'MNI152NLin6Asym', 'T1w')
    assert _pairs(steps) == [('MNI152NLin6Asym', 'T1w')]
    assert steps[0].invert is False


def test_image_multi_hop_order(tmp_path):
    # boldref -> T1w -> MNI ; ANTs order is reverse of the forward path.
    tg = _graph(
        tmp_path,
        [
            'sub-01_from-boldref_to-T1w_mode-image_xfm.txt',
            'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5',
        ],
    )
    steps = chain_for_image_resample(tg, 'boldref', 'MNI152NLin6Asym')
    assert _pairs(steps) == [('T1w', 'MNI152NLin6Asym'), ('boldref', 'T1w')]


def test_image_uses_reverse_affine_by_inversion(tmp_path):
    # Only the affine T1w->MNI exists; resampling MNI->T1w inverts it.
    tg = _graph(tmp_path, ['sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.mat'])
    steps = chain_for_image_resample(tg, 'MNI152NLin6Asym', 'T1w')
    assert _pairs(steps) == [('T1w', 'MNI152NLin6Asym')]
    assert steps[0].invert is True


def test_image_warp_not_invertible(tmp_path):
    # A displacement-field warp cannot be flag-inverted -> no MNI->ACPC image chain.
    tg = _graph(tmp_path, ['sub-01_from-ACPC_to-MNI152NLin6Asym_xfm.nii.gz'])
    assert _pairs(chain_for_image_resample(tg, 'ACPC', 'MNI152NLin6Asym')) == [
        ('ACPC', 'MNI152NLin6Asym')
    ]
    with pytest.raises(NoTransformPathError):
        chain_for_image_resample(tg, 'MNI152NLin6Asym', 'ACPC')


def test_image_identity(tmp_path):
    tg = _graph(tmp_path, ['sub-01_from-ACPC_to-MNI152NLin6Asym_mode-image_xfm.h5'])
    assert chain_for_image_resample(tg, 'T1w', 'T1w') == []


# ---------------------------------------------------------------------------
# Point warp (Strategy B) — opposite-named rule + invertibility guard
# ---------------------------------------------------------------------------


def test_point_uses_opposite_named_file(tmp_path):
    # To warp points ACPC -> MNI, pass the opposite-named from-MNI_to-ACPC file.
    tg = _graph(tmp_path, ['sub-01_from-MNI152NLin6Asym_to-ACPC_mode-image_xfm.h5'])
    steps = chain_for_point_warp(tg, 'ACPC', 'MNI152NLin6Asym')
    assert _pairs(steps) == [('MNI152NLin6Asym', 'ACPC')]
    assert steps[0].invert is False


def test_point_warp_invertibility_guard(tmp_path):
    # Only from-ACPC_to-MNI (a warp) exists. The image direction ACPC->MNI is fine,
    # but warping points ACPC->MNI needs from-MNI_to-ACPC, which does not exist and
    # cannot be synthesised by inverting a displacement field.
    tg = _graph(tmp_path, ['sub-01_from-ACPC_to-MNI152NLin6Asym_xfm.nii.gz'])
    with pytest.raises(NoTransformPathError, match='cannot be inverted|not flag-invertible'):
        chain_for_point_warp(tg, 'ACPC', 'MNI152NLin6Asym')


def test_point_multi_hop_forward_order(tmp_path):
    # Opposite-named files present for each hop of ACPC -> T1w -> fsnative.
    tg = _graph(
        tmp_path,
        [
            'sub-01_from-T1w_to-ACPC_mode-image_xfm.txt',
            'sub-01_from-fsnative_to-T1w_mode-image_xfm.txt',
        ],
    )
    steps = chain_for_point_warp(tg, 'ACPC', 'fsnative')
    # Applied in forward path order; each file is the opposite-named one.
    assert _pairs(steps) == [('T1w', 'ACPC'), ('fsnative', 'T1w')]


# ---------------------------------------------------------------------------
# The mirror property (plan-review §5): image and point chains are
# order-reversed, opposite-named duals when both directions exist.
# ---------------------------------------------------------------------------


def test_image_and_point_chains_are_mirrors(tmp_path):
    tg = _graph(
        tmp_path,
        [
            'sub-01_from-src_to-mid_mode-image_xfm.h5',
            'sub-01_from-mid_to-dst_mode-image_xfm.h5',
            'sub-01_from-mid_to-src_mode-image_xfm.h5',
            'sub-01_from-dst_to-mid_mode-image_xfm.h5',
        ],
    )
    image = chain_for_image_resample(tg, 'src', 'dst')
    point = chain_for_point_warp(tg, 'src', 'dst')
    # reversed(image) swapped == point (order-reversed, opposite-named)
    reversed_image_swapped = [(s.to, s.frm) for s in reversed(image)]
    assert reversed_image_swapped == _pairs(point)


def test_parse_xfm_filename_accepts_a_name_starting_with_from():
    """A transform BDT generates itself has no subject prefix.

    Regression: the pattern required a leading underscore before ``from-``, so
    ``from-ACPC_to-T1w_mode-image_xfm.mat`` -- the rigid bridge BDT computes and
    injects -- was unparsable. Unparsable bridges are silently dropped, which
    removed ACPC from the transform graph entirely and made every ACPC hop fail
    with ``NodeNotFound: Target ACPC is not in G``.
    """
    xfm = parse_xfm_filename('from-ACPC_to-T1w_mode-image_xfm.mat')
    assert xfm is not None
    assert (xfm.frm, xfm.to, xfm.mode) == ('ACPC', 'T1w', 'image')
    assert xfm.invertible  # an affine .mat; the chain needs T1w->ACPC by inversion

    # still anchored: a bare 'from-' inside another word must not match
    assert parse_xfm_filename('notfrom-A_to-B_mode-image_xfm.mat') is None


def test_generated_acpc_bridge_name_is_parseable():
    """The name _register_acpc_to_t1w writes must round-trip through the parser.

    These two live in different modules, so nothing else pins them together; if the
    filename drifts, bridges go back to being silently ignored.
    """
    import inspect

    from bdt.engine.factories import _register_acpc_to_t1w

    source = inspect.getsource(_register_acpc_to_t1w)
    written = re.search(r"os\.path\.abspath\('([^']+)'\)", source).group(1)

    xfm = parse_xfm_filename(written)
    assert xfm is not None, f'{written!r} does not parse as a BIDS transform'
    assert (xfm.frm, xfm.to, xfm.mode) == ('ACPC', 'T1w', 'image')


def test_transform_filenames_are_parsed_by_pybids_not_a_bespoke_regex():
    """Every filename shape BDT actually meets, including two that a regex missed.

    The hand-rolled pattern failed twice: it required an underscore before ``from-``
    (so a transform BDT generated itself, whose name *starts* with ``from-``, was
    unparsable -- silently removing ACPC from the graph), and it required ``_xfm``
    to follow ``mode-`` immediately (so every fMRIPrep transform qualified with
    ``desc-coreg``/``desc-hmc`` was dropped, including boldref<->T1w).
    """
    # no subject prefix: BDT's own generated ACPC bridge
    bridge = parse_xfm_filename('from-ACPC_to-T1w_mode-image_xfm.mat')
    assert (bridge.frm, bridge.to, bridge.invertible) == ('ACPC', 'T1w', True)

    # entities *after* mode-, as fMRIPrep writes them
    coreg = parse_xfm_filename(
        'sub-01_ses-1_task-rest_acq-mb_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
    )
    assert (coreg.frm, coreg.to, coreg.xfm_type) == ('boldref', 'T1w', 'affine')
    hmc = parse_xfm_filename('sub-01_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    assert (hmc.frm, hmc.to) == ('orig', 'boldref')

    # and the things that must still be refused
    assert parse_xfm_filename('sub-01_desc-preproc_bold.nii.gz') is None
    assert parse_xfm_filename('sub-01_from-X_to-Y_mode-image_xfm.json') is None
    assert parse_xfm_filename('sub-01_from-X_to-Y_mode-points_xfm.txt') is None


def test_native_space_transform_chain_resolves_by_inverting_the_coregistration(tmp_path):
    """T1w -> boldref: fMRIPrep ships only boldref -> T1w, which is affine.

    This is the chain a native-space parcellation needs to warp a T1w-space atlas
    onto the BOLD grid, and it exists only if the desc-coreg filename parses.
    """
    from bdt.transforms.graph import build_transform_graph
    from bdt.transforms.queries import chain_for_image_resample

    (tmp_path / 'sub-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt').touch()
    graph = build_transform_graph([tmp_path])

    assert graph.has_edge('boldref', 'T1w')
    chain = chain_for_image_resample(graph, 'T1w', 'boldref')
    assert len(chain) == 1
    assert chain[0].invert is True
