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


def test_sample_tract_profiles_accepts_trailing_singleton_volume(tmp_path):
    """ASLPrep writes CBF as (x, y, z, 1) -- 4D holding a single volume.

    Regression: this reached scipy as 4D data with a 3D coordinate array and died
    with ``RuntimeError: invalid shape for coordinate array``. The profile must match
    the equivalent 3D image exactly.
    """
    import pandas as pd

    from bdt.interfaces.tractography import SampleTractProfiles

    ramp = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 10, 10))
    scalar_3d = _nii(tmp_path, 'ramp3d.nii.gz', np.asarray(ramp))
    scalar_4d = _nii(tmp_path, 'ramp4d.nii.gz', np.asarray(ramp)[..., None])

    fwd = [(x, 5.0, 5.0) for x in np.linspace(1.0, 8.0, 20)]
    tck = _tck(tmp_path, 'sub-01_bundle-CST_streamlines.tck', [fwd])

    got = pd.read_table(
        SampleTractProfiles(in_files=[tck], scalar=scalar_4d, n_nodes=8).run().outputs.out_file
    )
    expected = pd.read_table(
        SampleTractProfiles(in_files=[tck], scalar=scalar_3d, n_nodes=8).run().outputs.out_file
    )
    assert np.allclose(got['mean'].to_numpy(), expected['mean'].to_numpy())


def test_sample_tract_profiles_rejects_multi_volume_scalar(tmp_path):
    """A genuine timeseries cannot be profiled -- fail with an actionable message."""
    from bdt.interfaces.tractography import SampleTractProfiles

    series = np.zeros((10, 10, 10, 40), dtype=np.float32)
    scalar = _nii(tmp_path, 'series.nii.gz', series)
    fwd = [(x, 5.0, 5.0) for x in np.linspace(1.0, 8.0, 20)]
    tck = _tck(tmp_path, 'sub-01_bundle-CST_streamlines.tck', [fwd])

    with pytest.raises(ValueError, match='40 volumes'):
        SampleTractProfiles(in_files=[tck], scalar=scalar, n_nodes=8).run()


def test_sample_tract_profiles_keeps_singleton_spatial_axis(tmp_path):
    """A singleton *spatial* axis must survive -- a blanket squeeze would break it."""
    import pandas as pd

    from bdt.interfaces.tractography import SampleTractProfiles

    flat = np.zeros((10, 1, 10), dtype=np.float32)
    flat[:, 0, :] = np.arange(10, dtype=np.float32)[:, None]
    scalar = _nii(tmp_path, 'flat.nii.gz', flat)
    fwd = [(x, 0.0, 5.0) for x in np.linspace(1.0, 8.0, 20)]
    tck = _tck(tmp_path, 'sub-01_bundle-CST_streamlines.tck', [fwd])

    df = pd.read_table(
        SampleTractProfiles(in_files=[tck], scalar=scalar, n_nodes=8).run().outputs.out_file
    )
    assert np.all(np.diff(df['mean'].to_numpy()) > 0)


def _profile_node(name='cbf_profile'):
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        inputs={'scalar': ['load_cbf'], 'bundles': ['load_bundles']},
        parameters={'n_nodes': 100},
    )


def _profile_context(scalar_space, bundles_space, tmp_path=None, provider=None):
    from bdt.engine.factories import FactoryContext
    from bdt.engine.selection import Match

    return FactoryContext(
        provider=provider,
        subject='01',
        datasets=['aslprep', 'qsiprep'],
        resolved={
            'load_cbf': Match('cbf.nii.gz', {'space': scalar_space, 'suffix': 'cbf'}),
            'load_bundles': Match('b.tck.gz', {'space': bundles_space, 'suffix': 'streamlines'}),
        },
    )


def _anat_provider(tmp_path):
    """Native + ACPC anatomicals and brain masks, as the ACPC bridge needs them."""
    from bdt.engine.selection import DictDataProvider, Match

    def _w(name, entities):
        path = tmp_path / name
        nb.Nifti1Image(np.zeros((2, 2, 2), 'float32'), np.eye(4)).to_filename(path)
        return Match(str(path), entities)

    return DictDataProvider(
        {
            'aslprep': [
                _w(
                    'sub-01_desc-preproc_T1w.nii.gz',
                    {'desc': 'preproc', 'suffix': 'T1w', 'datatype': 'anat'},
                ),
                _w(
                    'sub-01_desc-brain_mask.nii.gz',
                    {'desc': 'brain', 'suffix': 'mask', 'datatype': 'anat'},
                ),
            ],
            'qsiprep': [
                _w(
                    'sub-01_space-ACPC_desc-preproc_T1w.nii.gz',
                    {'space': 'ACPC', 'desc': 'preproc', 'suffix': 'T1w', 'datatype': 'anat'},
                ),
                _w(
                    'sub-01_space-ACPC_desc-brain_mask.nii.gz',
                    {'space': 'ACPC', 'desc': 'brain', 'suffix': 'mask', 'datatype': 'anat'},
                ),
            ],
        }
    )


def test_tract_profile_same_space_samples_directly(tmp_path):
    """FA and bundles are both ACPC -- no warp node, scalar wired straight through."""
    from bdt.engine.factories import init_parcellate_scalar_as_tract_profile_wf

    wf = init_parcellate_scalar_as_tract_profile_wf(
        _profile_node('fa_profile'), context=_profile_context('ACPC', 'ACPC')
    )
    names = set(wf.list_node_names())
    assert 'profile' in names
    assert 'warp_scalar' not in names
    assert 'register_acpc' not in names


def test_tract_profile_warps_scalar_into_bundle_space(tmp_path):
    """An MNI CBF map with ACPC bundles is warped, not silently mis-sampled.

    CBF is never produced in ACPC -- that space is unique to QSIPrep/QSIRecon -- so
    cross-space warping is the only way this pairing can be correct.
    """
    from bdt.engine.factories import init_parcellate_scalar_as_tract_profile_wf

    wf = init_parcellate_scalar_as_tract_profile_wf(
        _profile_node(),
        context=_profile_context('MNI152NLin6Asym', 'ACPC', provider=_anat_provider(tmp_path)),
    )
    names = set(wf.list_node_names())
    assert 'warp_scalar' in names
    warp = wf.get_node('warp_scalar')
    assert warp.inputs.source == 'MNI152NLin6Asym'
    assert warp.inputs.target == 'ACPC'
    assert warp.inputs.interpolation == 'linear'
    # streamlines have no grid; the reference is the ACPC anatomical
    assert warp.inputs.reference.endswith('sub-01_space-ACPC_desc-preproc_T1w.nii.gz')
    # ACPC is an endpoint, so the rigid bridge is registered on the fly
    assert 'register_acpc' in names
    assert 'bridge_list' in names


def test_tract_profile_warp_feeds_the_sampler_not_the_raw_scalar(tmp_path):
    """The profiler must read the warped image; wiring the raw scalar would be the
    original silent-wrong-answer bug with extra steps."""
    from bdt.engine.factories import init_parcellate_scalar_as_tract_profile_wf

    wf = init_parcellate_scalar_as_tract_profile_wf(
        _profile_node(),
        context=_profile_context('MNI152NLin6Asym', 'ACPC', provider=_anat_provider(tmp_path)),
    )
    sources = {(u.name, d.name): data['connect'] for u, d, data in wf._graph.edges(data=True)}
    assert ('warp_scalar', 'profile') in sources
    assert [('out_file', 'scalar')] == sources[('warp_scalar', 'profile')]
    assert ('inputnode', 'profile') not in sources


def test_tract_profile_cross_space_without_provider_raises(tmp_path):
    """The ACPC bridge needs a provider to resolve its anatomical references."""
    from bdt.engine.factories import init_parcellate_scalar_as_tract_profile_wf

    with pytest.raises(ValueError, match='ACPC'):
        init_parcellate_scalar_as_tract_profile_wf(
            _profile_node(), context=_profile_context('MNI152NLin6Asym', 'ACPC')
        )
