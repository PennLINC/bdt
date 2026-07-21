# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Build-time routing tests for parcellate_timeseries (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf  # noqa: E402


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


def test_cifti_timeseries_routes_to_cifti_wf():
    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(resolved={'load_bold': _match('bold.dtseries.nii', {})})
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'vertex_mask' in names  # CIFTI path node


def test_volumetric_timeseries_same_space_builds_parcellate_no_warp(tmp_path):
    from bdt.engine.selection import DictDataProvider

    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider(
            {
                'fmriprep': [
                    _match(
                        _mask(tmp_path, 'MNI152NLin6Asym'),
                        {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                    )
                ],
            }
        ),
        subject='01',
        datasets=['fmriprep'],
        resolved={
            'load_bold': _match('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
            'load_atlas': _match(
                _atlas(tmp_path, 3, 'atlas_dseg.nii.gz'),
                {'space': 'MNI152NLin6Asym', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'parcellate' in names
    assert 'warp_atlas' not in names


def test_volumetric_timeseries_cross_space_builds_warp(tmp_path):
    from bdt.engine.selection import DictDataProvider

    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider(
            {
                'atlases': [
                    _match(
                        _mask(tmp_path, 'MNI152NLin6Asym'),
                        {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                    )
                ],
            }
        ),
        subject='01',
        datasets=['atlases'],
        resolved={
            'load_bold': _match('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
            'load_atlas': _match(
                _atlas(tmp_path, 3, 'atlas_dseg.nii.gz'),
                {'space': 'MNI152NLin2009cAsym', 'suffix': 'dseg'},
            ),
        },
    )
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'warp_atlas' in names
    assert 'register_acpc' not in names  # no ACPC endpoint


def test_nifti_parcellate_yml_compiles_end_to_end(tmp_path):
    """Acceptance: scripts/nifti_parcellate.yml builds a full nipype graph through
    the volumetric parcellate_timeseries + functional_connectivity path.

    The bold (MNI152NLin6Asym) and atlas (MNI152NLin2009cAsym) are in different
    standard spaces, so ``warp_atlas`` must appear (cross-space) but
    ``register_acpc`` must not (neither endpoint is ACPC). Build-only: no tool is
    run, but ``role_atlas_ndim``/``role_atlas_labels`` now read the atlas's header
    and sidecar at *build* time, so (unlike the bold, still a bare path) the atlas
    must be a real on-disk file with a BIDS ``dseg.tsv`` beside it.
    """
    from bdt.engine.selection import DictDataProvider
    from bdt.engine.workflow import init_bdt_wf
    from bdt.spec import load_spec

    spec = load_spec('scripts/nifti_parcellate.yml')

    bold_path = '/data/sub-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    atlas_path = _atlas(
        tmp_path, 3, 'tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_res-01_dseg.nii.gz'
    )

    resolved = {
        'load_bold': _match(
            bold_path,
            {
                'space': 'MNI152NLin6Asym',
                'desc': 'preproc',
                'suffix': 'bold',
                'extension': '.nii.gz',
            },
        ),
        'atlas_4s456': _match(
            atlas_path,
            {
                # verbatim from a real BIDSDataProvider query against AtlasPack:
                # the file is ``tpl-``-named and carries NO ``space`` entity.
                'template': 'MNI152NLin2009cAsym',
                'atlas': '4S456Parcels',
                'suffix': 'dseg',
                'extension': '.nii.gz',
                'res': '01',
            },
        ),
    }
    # no local xfms and no ACPC anatomicals needed for a standard->standard warp
    provider = DictDataProvider(
        {
            'fmriprep': [
                _match(
                    _mask(tmp_path, 'MNI152NLin6Asym'),
                    {'suffix': 'mask', 'desc': 'brain', 'space': 'MNI152NLin6Asym'},
                )
            ],
            'atlases': [],
        }
    )
    context = FactoryContext(
        spec=spec,
        resolved=resolved,
        provider=provider,
        subject='01',
        datasets=['fmriprep', 'atlases'],
    )
    selections = {'load_bold': bold_path, 'atlas_4s456': atlas_path}

    wf = init_bdt_wf(spec, selections, context=context)

    names = set(wf.list_node_names())

    def has_suffix(suffix):
        return any(n.endswith(suffix) for n in names)

    # volumetric nodes present
    assert has_suffix('parcellate_bold.parcellate'), names
    assert has_suffix('parcellate_bold.warp_atlas'), names
    assert has_suffix('fc_bold.correlate'), names
    # CIFTI-path nodes absent: the volumetric branch also builds a node named
    # 'correlate' (XCP-D's TSVConnect), so matching on that name alone no longer
    # proves the CIFTI path was skipped. Assert the interface type instead.
    from bdt.interfaces.connectivity import NiftiParcellate, TSVConnect

    correlate_node = wf.get_node(next(n for n in names if n.endswith('fc_bold.correlate')))
    assert isinstance(correlate_node.interface, TSVConnect), correlate_node.interface
    # the 3D dseg atlas must route through the XCP-D masker port, not the deleted
    # hand-rolled parcellator.
    parcellate_node = wf.get_node(
        next(n for n in names if n.endswith('parcellate_bold.parcellate'))
    )
    assert isinstance(parcellate_node.interface, NiftiParcellate), parcellate_node.interface
    assert not has_suffix('parcellate_bold.vertex_mask'), names
    assert not has_suffix('parcellate_bold.restrict_atlas'), names
    # no ACPC bridge (neither space is ACPC)
    assert not has_suffix('parcellate_bold.register_acpc'), names
