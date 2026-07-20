# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for bdt.interfaces.transforms (transform loading, chain, resolve/apply)."""

import numpy as np
import pytest

pytest.importorskip('nipype')
pytest.importorskip('nitransforms')


def _itk_affine(path, translate_x):
    """Write an ITK affine that translates +translate_x in world x."""
    from nitransforms.linear import Affine

    m = np.eye(4)
    m[0, 3] = translate_x
    Affine(m).to_filename(str(path), fmt='itk')
    return str(path)


def _itk_scale_x(path, scale_x):
    """Write an ITK affine that scales world x by scale_x (non-commuting w/ shift)."""
    from nitransforms.linear import Affine

    m = np.eye(4)
    m[0, 0] = scale_x
    Affine(m).to_filename(str(path), fmt='itk')
    return str(path)


def _itk_warp_x(path, disp_x, shape=(4, 4, 4)):
    """Write an ITK displacement-field warp (5D vector NIfTI) shifting world x."""
    import nibabel as nb

    disp = np.zeros((*shape, 1, 3), dtype='float32')
    disp[..., 0, 0] = disp_x
    img = nb.Nifti1Image(disp, np.eye(4))
    img.header.set_intent('vector')
    img.to_filename(str(path))
    return str(path)


def _affine_x(translate_x):
    """An in-memory nitransforms Affine translating +translate_x in world x."""
    from nitransforms.linear import Affine

    m = np.eye(4)
    m[0, 3] = translate_x
    return Affine(m)


def test_load_transform_affine_and_invert(tmp_path):
    from bdt.interfaces.transforms import _load_transform

    p = _itk_affine(tmp_path / 'sub-01_from-A_to-B_mode-image_xfm.mat', 2.0)
    fwd = _load_transform(p)
    inv = _load_transform(p, invert=True)
    assert np.asarray(fwd.matrix)[0, 3] == pytest.approx(2.0)
    assert np.asarray(inv.matrix)[0, 3] == pytest.approx(-2.0)


def test_load_transform_displacement_warp(tmp_path):
    # A displacement-field .nii.gz must load as an ITK warp (fmt='itk'); the X5
    # default would raise "Input file is not HDF5". ITK stores displacements in
    # LPS, so a +2 world-x shift maps world x=1 -> -1 after the LPS->RAS flip.
    from nitransforms.nonlinear import DenseFieldTransform

    from bdt.interfaces.transforms import _load_transform

    p = _itk_warp_x(tmp_path / 'sub-01_from-A_to-B_xfm.nii.gz', 2.0)
    xf = _load_transform(p)
    assert isinstance(xf, DenseFieldTransform)
    mapped = xf.map(np.array([[1.0, 1.0, 1.0]]))
    assert mapped[0, 0] == pytest.approx(-1.0)


# The two chain-building tests build reference-less coordinate-only chains, so
# nitransforms emits a benign ``UserWarning: Reference space not set`` at chain
# construction (a reference grid is only wired in at apply time, in Task 3).
# Filter it so the pure-math tests keep pristine output.
@pytest.mark.filterwarnings('ignore:Reference space not set')
def test_build_chain_composes_in_grid_first_order(tmp_path):
    from bdt.interfaces.transforms import _build_chain
    from bdt.transforms import build_transform_graph, chain_for_image_resample

    # A -> B (shift +3), B -> C (scale x2). chain_for_image_resample returns the
    # ANTs pull order [from-B_to-C, from-A_to-B]; TransformChain applies
    # transforms[0] first, so map(x) = shift(scale(x)). Scale and shift do NOT
    # commute, so the numeric result pins down the order: for x=1,
    #   correct   shift(scale(1)) = (1*2) + 3 = 5
    #   misorder  scale(shift(1)) = (1+3) * 2 = 8
    _itk_affine(tmp_path / 'sub-01_from-A_to-B_mode-image_xfm.mat', 3.0)
    _itk_scale_x(tmp_path / 'sub-01_from-B_to-C_mode-image_xfm.mat', 2.0)
    tg = build_transform_graph(tmp_path)
    steps = chain_for_image_resample(tg, 'A', 'C')

    chain = _build_chain(steps)
    mapped = chain.map(np.array([[1.0, 0.0, 0.0]]))
    assert mapped[0, 0] == pytest.approx(5.0)


@pytest.mark.filterwarnings('ignore:Reference space not set')
def test_build_chain_inverts_reverse_affine(tmp_path):
    from bdt.interfaces.transforms import _build_chain
    from bdt.transforms import build_transform_graph, chain_for_image_resample

    # Only from-B_to-A (+2) exists; resampling A->B inverts that affine, so the
    # chain maps a B-grid point via ~(+2) = -2 to the A sampling coordinate.
    _itk_affine(tmp_path / 'sub-01_from-B_to-A_mode-image_xfm.mat', 2.0)
    tg = build_transform_graph(tmp_path)
    steps = chain_for_image_resample(tg, 'A', 'B')
    assert steps[0].invert is True

    chain = _build_chain(steps)
    mapped = chain.map(np.array([[0.0, 0.0, 0.0]]))
    assert mapped[0, 0] == pytest.approx(-2.0)


@pytest.mark.filterwarnings('ignore:Reference space not set')
def test_build_chain_flattens_composite(monkeypatch):
    """A composite that loads as its own TransformChain is flattened in place."""
    import bdt.interfaces.transforms as ift
    from bdt.interfaces.transforms import _build_chain
    from bdt.transforms.queries import XfmStep
    from nitransforms.manip import TransformChain

    # The .h5 step loads as a 2-transform composite (+2, then +3); the .mat step
    # loads as a single affine (+4). _build_chain must extend (not nest) the
    # composite so the result is one flat 3-transform chain in step order.
    def fake_load(path, invert=False):
        if path.endswith('.h5'):
            return TransformChain(transforms=[_affine_x(2.0), _affine_x(3.0)])
        return _affine_x(4.0)

    monkeypatch.setattr(ift, '_load_transform', fake_load)
    steps = [
        XfmStep(file='comp.h5', frm='A', to='C', invert=False),
        XfmStep(file='aff.mat', frm='C', to='D', invert=False),
    ]
    chain = _build_chain(steps, fetch=lambda p: p)

    assert len(chain.transforms) == 3
    assert [np.asarray(t.matrix)[0, 3] for t in chain.transforms] == [2.0, 3.0, 4.0]


def test_build_chain_empty_steps_raises():
    """Identity (same-space) resolves to no steps; _build_chain rejects it clearly."""
    from bdt.interfaces.transforms import _build_chain

    with pytest.raises(ValueError, match='at least one step'):
        _build_chain([])


def _label_img(path, marker_vox, shape=(6, 6, 6), label=5):
    import nibabel as nb

    data = np.zeros(shape, dtype=np.int16)
    data[marker_vox] = label
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


# Building the single-affine chain (via _build_chain) triggers the same benign
# nitransforms ``UserWarning: Reference space not set`` filtered above -- the
# reference grid is only wired in later, at ``apply()`` time.
@pytest.mark.filterwarnings('ignore:Reference space not set')
def test_resolve_apply_warps_and_preserves_labels(tmp_path, monkeypatch):
    import nibabel as nb

    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms

    # No TemplateFlow involvement in this hermetic test.
    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    xfm = _itk_affine(tmp_path / 'sub-01_from-ATLAS_to-GRID_mode-image_xfm.mat', 2.0)
    moving = _label_img(tmp_path / 'atlas.nii.gz', (3, 3, 3))
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    res = ResolveApplyTransforms(
        source='ATLAS',
        target='GRID',
        moving=moving,
        reference=reference,
        local_transforms=[xfm],
        interpolation='nearest',
        out_file=str(tmp_path / 'out.nii.gz'),
    ).run()

    out = nb.load(res.outputs.out_file)
    data = np.asarray(out.dataobj)
    # +2 world translation, pull semantics -> marker moves from x=3 to x=1.
    assert np.argwhere(data == 5).tolist() == [[1, 3, 3]]
    # nearest interpolation keeps integer labels intact.
    assert set(np.unique(data).tolist()) == {0, 5}
    assert res.outputs.out_transforms == [xfm]
    assert res.outputs.out_inversions == [False]


def test_resolve_apply_identity_resamples_onto_reference(tmp_path, monkeypatch):
    import nibabel as nb

    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms

    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    moving = _label_img(tmp_path / 'atlas.nii.gz', (3, 3, 3))
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    res = ResolveApplyTransforms(
        source='SAME',
        target='SAME',
        moving=moving,
        reference=reference,
        interpolation='nearest',
        out_file=str(tmp_path / 'out.nii.gz'),
    ).run()

    data = np.asarray(nb.load(res.outputs.out_file).dataobj)
    assert np.argwhere(data == 5).tolist() == [[3, 3, 3]]  # unchanged
    assert res.outputs.out_transforms == []
    assert res.outputs.out_inversions == []


def test_resolve_apply_raises_when_no_path(tmp_path, monkeypatch):
    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms
    from bdt.transforms import NoTransformPathError

    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    xfm = _itk_affine(tmp_path / 'sub-01_from-ATLAS_to-GRID_mode-image_xfm.mat', 2.0)
    moving = _label_img(tmp_path / 'atlas.nii.gz', (3, 3, 3))
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    with pytest.raises(NoTransformPathError):
        ResolveApplyTransforms(
            source='ATLAS',
            target='UNREACHABLE',
            moving=moving,
            reference=reference,
            local_transforms=[xfm],
            out_file=str(tmp_path / 'out.nii.gz'),
        ).run()


@pytest.mark.filterwarnings('ignore:Reference space not set')
def test_resolve_apply_warps_4d_atlas_per_volume(tmp_path, monkeypatch):
    import nibabel as nb

    import bdt.interfaces.transforms as mod
    from bdt.interfaces.transforms import ResolveApplyTransforms

    monkeypatch.setattr(mod, 'templateflow_edges', lambda: [])

    xfm = _itk_affine(tmp_path / 'sub-01_from-ATLAS_to-GRID_mode-image_xfm.mat', 2.0)
    # 4D atlas: volume 0 marks voxel (3,3,3); volume 1 marks voxel (2,2,2).
    data = np.zeros((6, 6, 6, 2), dtype=np.int16)
    data[3, 3, 3, 0] = 1
    data[2, 2, 2, 1] = 1
    moving = str(tmp_path / 'atlas4d.nii.gz')
    nb.Nifti1Image(data, np.eye(4)).to_filename(moving)
    reference = _label_img(tmp_path / 'grid.nii.gz', (0, 0, 0), label=0)

    res = ResolveApplyTransforms(
        source='ATLAS',
        target='GRID',
        moving=moving,
        reference=reference,
        local_transforms=[xfm],
        interpolation='nearest',
        out_file=str(tmp_path / 'out.nii.gz'),
    ).run()

    out = np.asarray(nb.load(res.outputs.out_file).dataobj)
    assert out.shape == (6, 6, 6, 2)
    # +2 world-x pull: vol0 marker 3->1, vol1 marker 2->0.
    assert np.argwhere(out[..., 0] > 0).tolist() == [[1, 3, 3]]
    assert np.argwhere(out[..., 1] > 0).tolist() == [[0, 2, 2]]
