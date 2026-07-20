# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Unit tests for bdt.interfaces.tractography."""

import nibabel as nb
import numpy as np
import pytest

pytest.importorskip('nipype')


def _nii(tmp_path, name, data):
    img = nb.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))
    path = str(tmp_path / name)
    img.to_filename(path)
    return path


def test_concatenate_niftis_normalizes_and_stacks(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    a = np.zeros((2, 2, 2), dtype=np.float32)
    a[0, 0, 0] = 4.0  # peak 4 -> normalizes to 1.0
    b = np.zeros((2, 2, 2), dtype=np.float32)
    b[1, 1, 1] = 2.0  # peak 2 -> normalizes to 1.0

    res = ConcatenateNiftis(
        in_files=[_nii(tmp_path, 'a.nii.gz', a), _nii(tmp_path, 'b.nii.gz', b)],
        normalize=True,
    ).run()

    out = nb.load(res.outputs.out_file)
    data = out.get_fdata()
    assert data.shape == (2, 2, 2, 2)
    assert data[..., 0].max() == pytest.approx(1.0)
    assert data[..., 1].max() == pytest.approx(1.0)
    assert data[0, 0, 0, 0] == pytest.approx(1.0)
    assert data[1, 1, 1, 1] == pytest.approx(1.0)


def test_concatenate_niftis_no_normalize_keeps_counts(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    a = np.zeros((2, 2, 2), dtype=np.float32)
    a[0, 0, 0] = 4.0
    res = ConcatenateNiftis(in_files=[_nii(tmp_path, 'a.nii.gz', a)], normalize=False).run()
    assert nb.load(res.outputs.out_file).get_fdata()[0, 0, 0, 0] == pytest.approx(4.0)


def test_concatenate_niftis_all_zero_stays_zero(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    z = np.zeros((2, 2, 2), dtype=np.float32)
    res = ConcatenateNiftis(in_files=[_nii(tmp_path, 'z.nii.gz', z)], normalize=True).run()
    assert float(nb.load(res.outputs.out_file).get_fdata().max()) == 0.0


def test_concatenate_niftis_shape_mismatch_raises(tmp_path):
    from bdt.interfaces.tractography import ConcatenateNiftis

    a = np.zeros((2, 2, 2), dtype=np.float32)
    b = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='[Ss]hape'):
        ConcatenateNiftis(
            in_files=[_nii(tmp_path, 'a.nii.gz', a), _nii(tmp_path, 'b.nii.gz', b)],
        ).run()


def test_threshold_nifti_binarizes_4d(tmp_path):
    from bdt.interfaces.tractography import ThresholdNifti

    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    data[0, 0, 0, 0] = 0.8
    data[1, 1, 1, 1] = 0.2
    path = _nii(tmp_path, 'p.nii.gz', data)

    res = ThresholdNifti(in_file=path, threshold=0.5, binarize=True).run()
    out = nb.load(res.outputs.out_file)
    out_data = out.get_fdata()
    assert out_data.shape == (2, 2, 2, 2)
    assert out_data[0, 0, 0, 0] == 1
    assert out_data[1, 1, 1, 1] == 0
    assert out.get_data_dtype() == np.uint8


def test_threshold_nifti_zero_threshold_keeps_any_nonzero(tmp_path):
    from bdt.interfaces.tractography import ThresholdNifti

    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 0.01
    path = _nii(tmp_path, 'q.nii.gz', data)

    res = ThresholdNifti(in_file=path, threshold=0.0, binarize=True).run()
    assert nb.load(res.outputs.out_file).get_fdata()[0, 0, 0, 0] == 1


def test_threshold_nifti_no_binarize_keeps_values(tmp_path):
    from bdt.interfaces.tractography import ThresholdNifti

    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 0.8
    path = _nii(tmp_path, 'r.nii.gz', data)

    res = ThresholdNifti(in_file=path, threshold=0.5, binarize=False).run()
    assert nb.load(res.outputs.out_file).get_fdata()[0, 0, 0, 0] == pytest.approx(0.8)


def _touch(tmp_path, name):
    path = tmp_path / name
    path.touch()
    return str(path)


def test_entities_to_seg_tsv_orders_by_input(tmp_path):
    import pandas as pd

    from bdt.interfaces.tractography import EntitiesToSegTSV

    files = [
        _touch(tmp_path, 'sub-01_bundle-CST_space-ACPC_streamlines.tck.gz'),
        _touch(tmp_path, 'sub-01_bundle-AF_space-ACPC_streamlines.tck.gz'),
    ]
    res = EntitiesToSegTSV(in_files=files, entity='bundle').run()

    df = pd.read_table(res.outputs.out_file)
    assert list(df.columns) == ['index', 'name']
    assert df['index'].tolist() == [1, 2]
    assert df['name'].tolist() == ['CST', 'AF']


def test_entities_to_seg_tsv_missing_entity_raises(tmp_path):
    from bdt.interfaces.tractography import EntitiesToSegTSV

    files = [_touch(tmp_path, 'sub-01_space-ACPC_streamlines.tck.gz')]
    with pytest.raises(ValueError, match='bundle'):
        EntitiesToSegTSV(in_files=files, entity='bundle').run()


def _tck(tmp_path, name, streamlines):
    import nibabel.streamlines as nibs

    tractogram = nibs.Tractogram(
        streamlines=[np.asarray(s, dtype=np.float32) for s in streamlines],
        affine_to_rasmm=np.eye(4),
    )
    path = str(tmp_path / name)
    nibs.save(tractogram, path)
    return path


def test_sample_tract_profiles_follows_scalar_gradient(tmp_path):
    import pandas as pd

    from bdt.interfaces.tractography import SampleTractProfiles

    # scalar volume where value == x voxel index; identity affine => world == voxel
    ramp = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 10, 10))
    scalar = _nii(tmp_path, 'ramp.nii.gz', ramp)

    # two streamlines running along +x at y=z=5; the second is stored reversed
    fwd = [(x, 5.0, 5.0) for x in np.linspace(1.0, 8.0, 20)]
    tck = _tck(tmp_path, 'sub-01_bundle-CST_streamlines.tck', [fwd, fwd[::-1]])

    res = SampleTractProfiles(in_files=[tck], scalar=scalar, n_nodes=8).run()
    df = pd.read_table(res.outputs.out_file)

    assert list(df.columns) == ['bundle', 'node', 'mean', 'std']
    assert df['bundle'].unique().tolist() == ['CST']
    assert df['node'].tolist() == list(range(1, 9))
    means = df['mean'].to_numpy()
    # orientation-aligned: the reversed streamline is flipped, so the profile is
    # monotonic and tracks the ramp from ~1 to ~8 (not flattened to a constant ~4.5)
    assert np.all(np.diff(means) > 0)
    assert means[0] == pytest.approx(1.0, abs=0.3)
    assert means[-1] == pytest.approx(8.0, abs=0.3)


def test_sample_tract_profiles_multiple_bundles_and_missing_entity(tmp_path):
    import pandas as pd

    from bdt.interfaces.tractography import SampleTractProfiles

    ramp = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 10, 10))
    scalar = _nii(tmp_path, 'ramp.nii.gz', ramp)
    line = [(x, 5.0, 5.0) for x in np.linspace(1.0, 8.0, 15)]

    cst = _tck(tmp_path, 'sub-01_bundle-CST_streamlines.tck', [line])
    af = _tck(tmp_path, 'sub-01_bundle-AF_streamlines.tck', [line])
    res = SampleTractProfiles(in_files=[cst, af], scalar=scalar, n_nodes=5).run()
    df = pd.read_table(res.outputs.out_file)
    assert sorted(df['bundle'].unique()) == ['AF', 'CST']
    assert len(df) == 10  # 2 bundles x 5 nodes

    nameless = _tck(tmp_path, 'sub-01_streamlines.tck', [line])
    with pytest.raises(ValueError, match='bundle'):
        SampleTractProfiles(in_files=[nameless], scalar=scalar, n_nodes=5).run()
