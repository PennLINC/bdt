# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for FactoryContext file-type + transform-discovery helpers."""

from types import SimpleNamespace

import pytest

from bdt.engine.factories import FactoryContext


class _StubMatch:
    def __init__(self, path, entities=None):
        self.path = path
        self.entities = entities or {}


class _StubProvider:
    """Returns preset matches per dataset, ignoring filters/session."""

    def __init__(self, by_dataset):
        self._by_dataset = by_dataset

    def select(self, dataset, filters, exclude, subject=None):
        return list(self._by_dataset.get(dataset, []))


def _node(role_to_upstreams):
    return SimpleNamespace(inputs=role_to_upstreams)


def test_role_extension_and_suffix_read_resolved_entities():
    ctx = FactoryContext(
        resolved={'load_cbf': _StubMatch('cbf.nii.gz', {'extension': '.nii.gz', 'suffix': 'cbf'})}
    )
    node = _node({'scalar': ['load_cbf']})
    assert ctx.role_extension(node, 'scalar') == '.nii.gz'
    assert ctx.role_suffix(node, 'scalar') == 'cbf'
    assert ctx.role_extension(node, 'atlas', default='') == ''  # role absent -> default


def test_role_space_falls_back_to_template_entity():
    """A BIDS-Atlas file names its space with ``tpl-`` and has no ``space`` entity.

    Entities are verbatim from a real ``BIDSDataProvider`` query against AtlasPack.
    Without the fallback ``role_space`` returns None, cross-space detection reports
    same-space, and the atlas is parcellated against a bold in a *different* template
    with no warp — a silent wrong answer.
    """
    ctx = FactoryContext(
        resolved={
            'atlas_4s456': _StubMatch(
                '/a/tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_res-01_dseg.nii.gz',
                {
                    'template': 'MNI152NLin2009cAsym',
                    'atlas': '4S456Parcels',
                    'suffix': 'dseg',
                    'extension': '.nii.gz',
                    'res': '01',
                },
            ),
            # ``space-`` wins when present; the two never co-occur in practice.
            'load_bold': _StubMatch('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
        }
    )
    node = _node({'atlas': ['atlas_4s456'], 'timeseries': ['load_bold']})
    assert ctx.role_space(node, 'atlas') == 'MNI152NLin2009cAsym'
    assert ctx.role_space(node, 'timeseries') == 'MNI152NLin6Asym'
    # neither entity present -> the caller's default still applies
    surf = _node({'surface': ['load_surf']})
    assert ctx.role_space(surf, 'surface', default='T1w') == 'T1w'


def test_discover_transforms_excludes_acpc_endpoints_and_sorts():
    provider = _StubProvider(
        {
            'aslprep': [_StubMatch('/d/sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5')],
            'qsiprep': [
                _StubMatch('/d/sub-01_from-ACPC_to-T1w_mode-image_xfm.mat'),  # excluded (ACPC)
                _StubMatch('/d/sub-01_from-T1w_to-ACPC_mode-image_xfm.mat'),  # excluded (ACPC)
            ],
        }
    )
    ctx = FactoryContext(provider=provider, subject='01', datasets=['aslprep', 'qsiprep'])
    got = ctx.discover_transforms()
    assert got == ['/d/sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5']


def test_discover_transforms_no_provider_returns_empty():
    assert FactoryContext(datasets=['aslprep']).discover_transforms() == []


def test_role_is_cifti_falls_back_to_resolved_path_without_spec():
    ctx = FactoryContext(
        resolved={
            'load_alff': _StubMatch('alff.dscalar.nii', {'extension': '.dscalar.nii'}),
            'load_cbf': _StubMatch('cbf.nii.gz', {'extension': '.nii.gz'}),
        }
    )
    dense = _node({'scalar': ['load_alff']})
    vol = _node({'scalar': ['load_cbf']})
    assert ctx.role_is_cifti(dense, 'scalar') is True
    assert ctx.role_is_cifti(vol, 'scalar') is False
    # no resolvable info -> defaults to CIFTI (the build-safe historical default,
    # so a no-context assembly of a CIFTI pipeline keeps the grayordinate path).
    assert ctx.role_is_cifti(vol, 'atlas') is True
    assert FactoryContext().role_is_cifti(vol, 'scalar') is True


def test_role_atlas_ndim_reads_header_for_selection_atlas(tmp_path):
    """A selected atlas exists on disk at build time, so read its header."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / 'tpl-X_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), 'int16'), np.eye(4)).to_filename(path)
    ctx = FactoryContext(resolved={'atlas_sel': _StubMatch(str(path), {'suffix': 'dseg'})})
    node = _node({'atlas': ['atlas_sel']})
    assert ctx.role_atlas_ndim(node, 'atlas') == 3


def test_role_atlas_ndim_is_four_for_processing_atlas():
    """A processing-node atlas does not exist yet at build time, but is 4D by
    construction: tractogram_to_pseg is the only atlas-producing action and it
    stacks bundles via ConcatenateNiftis."""
    ctx = FactoryContext(resolved={})
    node = _node({'atlas': ['bundle_rois']})
    assert ctx.role_atlas_ndim(node, 'atlas') == 4


def test_role_atlas_labels_finds_sibling_sidecar(tmp_path):
    """AtlasPack ships tpl-..._dseg.tsv beside tpl-..._dseg.nii.gz."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / 'tpl-X_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), 'int16'), np.eye(4)).to_filename(path)
    sidecar = tmp_path / 'tpl-X_atlas-Y_dseg.tsv'
    sidecar.write_text('index\tname\n1\tA\n')
    ctx = FactoryContext(resolved={'atlas_sel': _StubMatch(str(path), {})})
    node = _node({'atlas': ['atlas_sel']})
    assert ctx.role_atlas_labels(node, 'atlas') == str(sidecar)


def test_role_atlas_labels_errors_when_sidecar_missing(tmp_path):
    """A selected atlas with no sidecar is a hard error naming the expected path."""
    import nibabel as nb
    import numpy as np

    path = tmp_path / 'tpl-X_atlas-Y_dseg.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), 'int16'), np.eye(4)).to_filename(path)
    ctx = FactoryContext(resolved={'atlas_sel': _StubMatch(str(path), {})})
    node = _node({'atlas': ['atlas_sel']})
    with pytest.raises(ValueError, match='no labels sidecar'):
        ctx.role_atlas_labels(node, 'atlas')


def test_role_atlas_labels_none_for_processing_atlas():
    """A processing-node atlas gets its labels over the wire, not from disk."""
    ctx = FactoryContext(resolved={})
    node = _node({'atlas': ['bundle_rois']})
    assert ctx.role_atlas_labels(node, 'atlas') is None


def test_role_datatype_reads_resolved_entities():
    ctx = FactoryContext(resolved={'load_bold': _StubMatch('bold.nii.gz', {'datatype': 'func'})})
    node = _node({'timeseries': ['load_bold']})
    assert ctx.role_datatype(node, 'timeseries') == 'func'
    assert ctx.role_datatype(node, 'atlas') is None
