# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Build-time routing tests for parcellate_scalar (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_parcellate_scalar_wf  # noqa: E402


def _atlas(tmp_path, ndim, name='tpl-MNI152NLin6Asym_atlas-Y_dseg.nii.gz'):
    """A real on-disk atlas + its BIDS labels sidecar (role_atlas_ndim/labels need both)."""
    import nibabel as nb
    import numpy as np

    shape = (4, 4, 4) if ndim == 3 else (4, 4, 4, 2)
    path = tmp_path / name
    nb.Nifti1Image(np.zeros(shape, 'float32'), np.eye(4)).to_filename(path)
    (tmp_path / name.replace('.nii.gz', '.tsv')).write_text('index\tname\n1\tA\n2\tB\n')
    return str(path)


def _mask(tmp_path, space):
    """A real on-disk brain mask, for the discovery `_discover_brain_mask` now runs."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / f'sub-01_space-{space}_desc-brain_mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), 'uint8'), np.eye(4)).to_filename(path)
    return str(path)


def _match(path, entities):
    from bdt.engine.selection import Match

    return Match(path, entities)


def _node(name, role_to_upstreams, parameters=None):
    return SimpleNamespace(name=name, inputs=role_to_upstreams, parameters=parameters or {})


def test_cifti_scalar_routes_to_cifti_wf():
    # a .dscalar.nii scalar -> the existing CIFTI path (has a vertex_mask node,
    # which the volumetric path never builds).
    node = _node('alff_parc', {'scalar': ['load_alff'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        resolved={'load_alff': _match('alff.dscalar.nii', {'extension': '.dscalar.nii'})}
    )
    names = set(init_parcellate_scalar_wf(node, context=ctx).list_node_names())
    assert 'vertex_mask' in names


def test_volumetric_same_space_has_no_warp_node(tmp_path):
    # FA scalar (ACPC) + atlas (ACPC): same space -> parcellate directly, no warp.
    from bdt.engine.selection import DictDataProvider

    node = _node('fa_roi', {'scalar': ['load_fa'], 'atlas': ['bundle_rois']})
    ctx = FactoryContext(
        provider=DictDataProvider(
            {
                'anat': [
                    _match(
                        _mask(tmp_path, 'ACPC'),
                        {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC'},
                    )
                ],
            }
        ),
        subject='01',
        datasets=['anat'],
        resolved={
            'load_fa': _match('fa.nii.gz', {'extension': '.nii.gz', 'space': 'ACPC'}),
            'bundle_rois': _match(
                _atlas(tmp_path, 3, 'rois_dseg.nii.gz'),
                {'extension': '.nii.gz', 'space': 'ACPC', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_scalar_wf(node, context=ctx).list_node_names())
    assert 'parcellate' in names
    assert 'warp_atlas' not in names


def test_volumetric_cross_space_non_acpc_builds_warp_without_bridge(tmp_path):
    # CBF (MNI152NLin2009cAsym) + atlas (MNI152NLin6Asym): cross-space but no ACPC,
    # so a ResolveApplyTransforms warp is inserted and NO registration node is built.
    from bdt.engine.selection import DictDataProvider

    node = _node('roi', {'scalar': ['load_cbf'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider(
            {
                'aslprep': [
                    _match(
                        _mask(tmp_path, 'MNI152NLin2009cAsym'),
                        {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin2009cAsym'},
                    )
                ],
            }
        ),
        subject='01',
        datasets=['aslprep'],
        resolved={
            'load_cbf': _match(
                'cbf.nii.gz', {'extension': '.nii.gz', 'space': 'MNI152NLin2009cAsym'}
            ),
            'load_atlas': _match(
                _atlas(tmp_path, 3, 'atlas_dseg.nii.gz'),
                {'extension': '.nii.gz', 'space': 'MNI152NLin6Asym', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_scalar_wf(node, context=ctx).list_node_names())
    assert 'warp_atlas' in names
    assert 'register_acpc' not in names


def test_volumetric_cross_space_acpc_bridge_scopes_native_t1w_by_session(tmp_path):
    # Multi-session subject: two session-level native T1w anatomicals exist. The
    # ACPC bridge's fixed-image find_reference must scope to the atlas's session
    # (ses-1) or it matches 2 T1ws and raises -- regression for the dropped session=
    # on the fixed T1w/mask lookups (moving already scoped; now fixed does too).
    from bdt.engine.selection import DictDataProvider, Match

    def touch(rel):
        (tmp_path / rel).touch()
        return str(tmp_path / rel)

    provider = DictDataProvider(
        {
            'smriprep': [
                Match(
                    touch('sub-01_ses-1_desc-preproc_T1w.nii.gz'),
                    {'suffix': 'T1w', 'desc': 'preproc', 'datatype': 'anat', 'session': '1'},
                ),
                Match(
                    touch('sub-01_ses-2_desc-preproc_T1w.nii.gz'),
                    {'suffix': 'T1w', 'desc': 'preproc', 'datatype': 'anat', 'session': '2'},
                ),
                Match(
                    touch('sub-01_ses-1_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'datatype': 'anat', 'session': '1'},
                ),
                Match(
                    touch('sub-01_ses-2_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'datatype': 'anat', 'session': '2'},
                ),
            ],
            'qsiprep': [
                Match(
                    touch('sub-01_ses-1_space-ACPC_desc-preproc_T1w.nii.gz'),
                    {
                        'suffix': 'T1w',
                        'desc': 'preproc',
                        'space': 'ACPC',
                        'datatype': 'anat',
                        'session': '1',
                    },
                ),
                Match(
                    touch('sub-01_ses-1_space-ACPC_desc-brain_mask.nii.gz'),
                    {
                        'suffix': 'mask',
                        'desc': 'brain',
                        'space': 'ACPC',
                        'datatype': 'anat',
                        'session': '1',
                    },
                ),
                Match(
                    touch('sub-01_from-ACPC_to-T1w_mode-image_xfm.mat'),
                    {'suffix': 'xfm', 'from': 'ACPC', 'to': 'T1w'},
                ),  # excluded by discovery
            ],
            'aslprep': [
                Match(
                    touch('sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5'),
                    {'suffix': 'xfm', 'from': 'T1w', 'to': 'MNI152NLin6Asym'},
                ),
                Match(
                    _mask(tmp_path, 'MNI152NLin6Asym'),
                    {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                ),
            ],
        }
    )
    node = _node('cbf_roi', {'scalar': ['load_cbf'], 'atlas': ['bundle_rois']})
    ctx = FactoryContext(
        provider=provider,
        subject='01',
        datasets=['smriprep', 'qsiprep', 'aslprep'],
        resolved={
            'load_cbf': _match('cbf.nii.gz', {'extension': '.nii.gz', 'space': 'MNI152NLin6Asym'}),
            'bundle_rois': _match(
                _atlas(tmp_path, 3, 'rois_dseg.nii.gz'),
                {'extension': '.nii.gz', 'space': 'ACPC', 'suffix': 'dseg', 'session': '1'},
            ),
        },
    )
    wf = init_parcellate_scalar_wf(node, context=ctx)  # must not raise (session-scoped)
    names = set(wf.list_node_names())
    for want in ('warp_atlas', 'register_acpc', 'bridge_list', 'parcellate'):
        assert want in names, f'missing {want}'
    reg = wf.get_node('register_acpc')
    assert reg.inputs.fixed_image.endswith('sub-01_ses-1_desc-preproc_T1w.nii.gz')
    assert reg.inputs.moving_image.endswith('sub-01_ses-1_space-ACPC_desc-preproc_T1w.nii.gz')
    assert wf.get_node('warp_atlas').inputs.interpolation == 'nearest'
    locals_ = wf.get_node('warp_atlas').inputs.local_transforms
    assert any('to-MNI152NLin6Asym' in p for p in locals_)
    assert not any('ACPC' in p for p in locals_)


def test_cross_space_warp_inserted_for_processing_node_atlas(tmp_path):
    # The real pipeline shape: the atlas (bundle_rois) is a PROCESSING node, absent
    # from context.resolved (which holds selections only). role_space must derive its
    # inherited space (ACPC, via node_output_entities) so the CBF (MNI) cross-space
    # warp + ACPC bridge are actually inserted -- regression for the Critical where a
    # processing-node atlas read as space=None and the warp was silently skipped.
    from bdt.engine.selection import DictDataProvider, Match
    from bdt.spec import load_spec

    def touch(rel):
        (tmp_path / rel).touch()
        return str(tmp_path / rel)

    spec = load_spec('scripts/tract_parcellate.yml')
    cbf_roi = spec.by_name()['cbf_roi']

    provider = DictDataProvider(
        {
            'anat': [
                Match(
                    touch('sub-01_desc-preproc_T1w.nii.gz'),
                    {'suffix': 'T1w', 'desc': 'preproc', 'datatype': 'anat'},
                ),
                Match(
                    touch('sub-01_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'datatype': 'anat'},
                ),
            ],
            'qsiprep': [
                Match(
                    touch('sub-01_space-ACPC_desc-preproc_T1w.nii.gz'),
                    {'suffix': 'T1w', 'desc': 'preproc', 'space': 'ACPC', 'datatype': 'anat'},
                ),
                Match(
                    touch('sub-01_space-ACPC_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'},
                ),
                Match(
                    touch('sub-01_from-ACPC_to-T1w_mode-image_xfm.mat'),
                    {'suffix': 'xfm', 'from': 'ACPC', 'to': 'T1w'},
                ),  # excluded by discovery
            ],
            'aslprep': [
                Match(
                    touch('sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5'),
                    {'suffix': 'xfm', 'from': 'T1w', 'to': 'MNI152NLin6Asym'},
                ),
                Match(
                    touch('sub-01_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'),
                    {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                ),
            ],
        }
    )
    # resolved holds SELECTIONS ONLY -- bundle_rois (processing) is deliberately absent.
    resolved = {
        'load_bundles': Match(
            '/q/bundles.tck.gz', {'space': 'ACPC', 'suffix': 'streamlines', 'extension': '.tck.gz'}
        ),
        'load_fa': Match(
            '/q/fa.nii.gz', {'space': 'ACPC', 'suffix': 'dwimap', 'extension': '.nii.gz'}
        ),
        'load_cbf': Match(
            '/a/cbf.nii.gz', {'space': 'MNI152NLin6Asym', 'suffix': 'cbf', 'extension': '.nii.gz'}
        ),
    }
    ctx = FactoryContext(
        provider=provider,
        subject='01',
        spec=spec,
        datasets=['anat', 'qsiprep', 'aslprep'],
        resolved=resolved,
    )

    # the fix: the processing-node atlas's inherited space now resolves.
    assert ctx.role_space(cbf_roi, 'atlas') == 'ACPC'
    assert ctx.role_space(cbf_roi, 'scalar') == 'MNI152NLin6Asym'
    # tractogram_to_pseg with threshold:0.0 -> dynamic suffix 'dseg' -> nearest interp.
    assert ctx.role_suffix(cbf_roi, 'atlas') == 'dseg'

    wf = init_parcellate_scalar_wf(cbf_roi, context=ctx)
    names = set(wf.list_node_names())
    for want in ('warp_atlas', 'register_acpc', 'bridge_list', 'parcellate'):
        assert want in names, f'missing {want}'
    assert wf.get_node('warp_atlas').inputs.interpolation == 'nearest'
