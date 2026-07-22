"""Grayordinate parcellation by a probabilistic (dscalar) atlas."""

import pytest

pytest.importorskip('nibabel')

import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bdt.interfaces.cifti_probseg import CiftiProbSegParcellate  # noqa: E402

N_GRAY = 20
N_PARCELS = 3


def _brain_model(n=N_GRAY):
    return nb.cifti2.BrainModelAxis.from_mask(np.ones(n, dtype='int8'), name='thalamus_left')


def _dscalar(data, path, names=None):
    rows = data.shape[0]
    axis = nb.cifti2.ScalarAxis(names or [f'm{i}' for i in range(rows)])
    img = nb.Cifti2Image(data, header=(axis, _brain_model()))
    img.to_filename(str(path))
    return str(path)


def _fixture(tmp_path, n_t=4, seed=0, covered=None):
    rng = np.random.default_rng(seed)
    weights = rng.random((N_PARCELS, N_GRAY))
    data = (rng.random((n_t, N_GRAY)) * 10).astype('float64')
    if covered is None:
        covered = np.ones(N_GRAY)
        covered[:4] = 0  # some grayordinates carry no data
    _dscalar(weights, tmp_path / 'atlas.dscalar.nii')
    _dscalar(data, tmp_path / 'data.dscalar.nii')
    _dscalar(covered[None, :], tmp_path / 'mask.dscalar.nii')
    pd.DataFrame({'index': range(1, N_PARCELS + 1), 'name': ['a', 'b', 'c']}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )
    return weights, data, covered.astype(bool)


def _run(tmp_path, **kwargs):
    return CiftiProbSegParcellate(
        data=str(tmp_path / 'data.dscalar.nii'),
        atlas=str(tmp_path / 'atlas.dscalar.nii'),
        atlas_labels=str(tmp_path / 'labels.tsv'),
        vertex_mask=str(tmp_path / 'mask.dscalar.nii'),
        **{'min_coverage': 0.0, **kwargs},
    ).run()


def test_weighted_statistics_match_brute_force(tmp_path, monkeypatch):
    """Same definitions as the volumetric ProbSegParcellate, so the two agree."""
    weights, data, covered = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = _run(tmp_path, statistics=['mean', 'standard_deviation'])

    got = {s: pd.read_table(f) for s, f in zip(
        ['mean', 'standard_deviation'], res.outputs.out_files, strict=True
    )}
    d = data[:, covered]
    for parcel, name in enumerate(['a', 'b', 'c']):
        w = weights[parcel][covered]
        mu = (w * d).sum(axis=1) / w.sum()
        sd = np.sqrt((w * (d - mu[:, None]) ** 2).sum(axis=1) / w.sum())
        assert got['mean'][name].to_numpy() == pytest.approx(mu, rel=1e-9)
        assert got['standard_deviation'][name].to_numpy() == pytest.approx(sd, rel=1e-9)


def test_uncovered_grayordinates_are_excluded(tmp_path, monkeypatch):
    """The vertex mask is load-bearing: masked-out values must not contribute."""
    weights, data, covered = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    baseline = pd.read_table(_run(tmp_path).outputs.out_files[0])

    # rewrite the data with wild values *only* where uncovered; results must not move
    data[:, ~covered] = 1e6
    _dscalar(data, tmp_path / 'data.dscalar.nii')
    after = pd.read_table(_run(tmp_path).outputs.out_files[0])
    pd.testing.assert_frame_equal(baseline, after)


def test_coverage_is_the_share_of_probability_mass_covered(tmp_path, monkeypatch):
    weights, _, covered = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    got = pd.read_table(_run(tmp_path).outputs.coverage, index_col='Node')['coverage']
    expected = weights[:, covered].sum(axis=1) / weights.sum(axis=1)
    assert got.to_numpy() == pytest.approx(expected, rel=1e-6)


def test_low_coverage_parcels_are_nan(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = pd.read_table(_run(tmp_path, min_coverage=1.1).outputs.out_files[0])
    assert df.isna().all().all()


def test_tidy_mode_is_one_row_per_parcel(tmp_path, monkeypatch):
    """A scalar has room for statistics as columns; a series does not."""
    _fixture(tmp_path, n_t=1)
    monkeypatch.chdir(tmp_path)
    res = _run(tmp_path, statistics=['mean', 'standard_deviation'], tidy=True)
    df = pd.read_table(res.outputs.tsv)
    assert list(df.columns) == ['node', 'mean', 'standard_deviation']
    assert df['node'].tolist() == ['a', 'b', 'c']
    assert res.outputs.out_files == []


def test_unweighted_statistic_is_rejected(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='probabilistic'):
        _run(tmp_path, statistics=['median'])


def test_grayordinate_mismatch_is_an_error(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    axis = nb.cifti2.ScalarAxis(['m0'])
    nb.Cifti2Image(
        np.ones((1, N_GRAY - 1)), header=(axis, _brain_model(N_GRAY - 1))
    ).to_filename(str(tmp_path / 'data.dscalar.nii'))
    with pytest.raises(ValueError, match='Grayordinate counts disagree'):
        _run(tmp_path)


def test_label_count_must_match_the_atlas(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({'index': [1], 'name': ['only']}).to_csv(
        tmp_path / 'labels.tsv', sep='\t', index=False
    )
    with pytest.raises(ValueError, match='3 maps but'):
        _run(tmp_path)


# --- sidecar resolution and build-time routing -------------------------------


def test_labels_sidecar_drops_realization_entities(tmp_path):
    """AtlasPack ships one TSV per atlas but one image per density.

    ``tpl-fsLR_atlas-DiFuMo_scale-64dimensions_den-91k_probseg.dscalar.nii`` is
    described by ``tpl-fsLR_atlas-DiFuMo_scale-64dimensions_probseg.tsv`` -- the
    ``den-`` entity is dropped, BIDS-inheritance style.
    """
    from bdt.engine.factories import _atlas_labels_sidecar

    stem = 'tpl-fsLR_atlas-DiFuMo_scale-64dimensions'
    sidecar = tmp_path / f'{stem}_probseg.tsv'
    sidecar.write_text('index\tname\n1\ta\n')
    image = tmp_path / f'{stem}_den-91k_probseg.dscalar.nii'
    image.touch()

    assert _atlas_labels_sidecar(str(image)) == str(sidecar)


def test_exact_sidecar_wins_over_a_dropped_one(tmp_path):
    """A density-specific table, if present, must beat the generic one."""
    from bdt.engine.factories import _atlas_labels_sidecar

    stem = 'tpl-fsLR_atlas-X'
    (tmp_path / f'{stem}_probseg.tsv').write_text('index\tname\n1\tgeneric\n')
    exact = tmp_path / f'{stem}_den-91k_probseg.tsv'
    exact.write_text('index\tname\n1\texact\n')
    image = tmp_path / f'{stem}_den-91k_probseg.dscalar.nii'
    image.touch()

    assert _atlas_labels_sidecar(str(image)) == str(exact)


def test_missing_sidecar_names_every_path_tried(tmp_path):
    from bdt.engine.factories import _atlas_labels_sidecar

    image = tmp_path / 'tpl-fsLR_atlas-X_den-91k_res-2_probseg.dscalar.nii'
    image.touch()
    with pytest.raises(ValueError) as exc:
        _atlas_labels_sidecar(str(image))
    message = str(exc.value)
    assert 'den-91k_res-2_probseg.tsv' in message  # the exact stem
    assert 'tpl-fsLR_atlas-X_probseg.tsv' in message  # both entities dropped


def test_dscalar_atlas_routes_to_the_probseg_workflow(tmp_path):
    """A dscalar atlas must never reach wb_command, which rejects it outright."""
    from types import SimpleNamespace

    from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf
    from bdt.engine.selection import Match

    _fixture(tmp_path)
    atlas = tmp_path / 'atlas.dscalar.nii'
    (tmp_path / 'atlas.tsv').write_text('index\tname\n1\ta\n2\tb\n3\tc\n')
    data = tmp_path / 'data.dscalar.nii'

    node = SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load'], 'atlas': ['load_atlas']},
        parameters={'statistics': ['mean', 'standard_deviation']},
        desc=None,
    )
    ctx = FactoryContext(
        resolved={
            'load': Match(str(data), {'extension': '.dscalar.nii'}),
            'load_atlas': Match(str(atlas), {'extension': '.dscalar.nii'}),
        }
    )
    wf = init_parcellate_timeseries_wf(node, context=ctx)
    names = set(wf.list_node_names())

    assert 'parcellate' in names
    assert isinstance(wf.get_node('parcellate').interface, CiftiProbSegParcellate)
    # none of the Workbench machinery, which cannot parcellate by a dscalar
    assert 'restrict_atlas' not in names
    assert not any(n.startswith('parcellate_data') for n in names)
    assert {'pick_mean', 'pick_standard_deviation'} <= names
    # ...but the vertex mask is still built: it defines coverage
    assert 'vertex_mask' in names


def test_dlabel_atlas_still_routes_to_workbench(tmp_path):
    """Guard the other direction -- a real dlabel must not take the numpy path."""
    from types import SimpleNamespace

    from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf
    from bdt.engine.selection import Match

    label_axis = nb.cifti2.LabelAxis(['parcels'], [{0: ('??', (0, 0, 0, 0))}])
    atlas = tmp_path / 'atlas.dlabel.nii'
    nb.Cifti2Image(
        np.zeros((1, N_GRAY)), header=(label_axis, _brain_model())
    ).to_filename(str(atlas))

    node = SimpleNamespace(
        name='parc',
        inputs={'timeseries': ['load'], 'atlas': ['load_atlas']},
        parameters={},
        desc=None,
    )
    ctx = FactoryContext(
        resolved={
            'load': Match('b.dtseries.nii', {'extension': '.dtseries.nii'}),
            'load_atlas': Match(str(atlas), {'extension': '.dlabel.nii'}),
        }
    )
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'restrict_atlas' in names
    assert 'parcellate_data_mean' in names


def test_fc_downstream_of_a_probseg_parcellation_correlates_the_tsv(tmp_path):
    """The regression: FC routed on its *input* format, not its input's product.

    ``parcellate_timeseries`` with a probabilistic atlas consumes a dtseries and
    emits a TSV, so FC must use TSVConnect.  Routing on the dtseries handed a .tsv
    to ``wb_command -cifti-correlation``, which died with "is not a valid NIfTI
    file".
    """
    from bdt.engine.factories import FactoryContext, init_functional_connectivity_wf
    from bdt.engine.selection import Match
    from bdt.interfaces.connectivity import TSVConnect
    from bdt.spec import parse_spec

    _fixture(tmp_path)
    atlas = tmp_path / 'atlas.dscalar.nii'
    data = tmp_path / 'data.dscalar.nii'

    spec = parse_spec(
        {
            'nodes': [
                {'name': 'load', 'action': 'select_data', 'dataset': 'xcpd'},
                {'name': 'atlas', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 'parc',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'load', 'atlas': 'atlas'},
                },
                {
                    'name': 'fc',
                    'action': 'functional_connectivity',
                    'inputs': {'timeseries': 'parc'},
                },
            ]
        }
    )
    resolved = {
        'load': Match(str(data), {'extension': '.dscalar.nii', 'suffix': 'bold'}),
        'atlas': Match(str(atlas), {'extension': '.dscalar.nii', 'atlas': 'DiFuMo'}),
    }
    ctx = FactoryContext(spec=spec, resolved=resolved)
    wf = init_functional_connectivity_wf(spec.by_name()['fc'], context=ctx)
    assert isinstance(wf.get_node('correlate').interface, TSVConnect)


def test_fc_downstream_of_a_dlabel_parcellation_still_uses_workbench(tmp_path):
    """The other direction: a real dlabel keeps the pconn path."""
    from bdt.engine.factories import FactoryContext, init_functional_connectivity_wf
    from bdt.engine.selection import Match
    from bdt.interfaces.workbench import CiftiCorrelation
    from bdt.spec import parse_spec

    label_axis = nb.cifti2.LabelAxis(['parcels'], [{0: ('??', (0, 0, 0, 0))}])
    atlas = tmp_path / 'atlas.dlabel.nii'
    nb.Cifti2Image(
        np.zeros((1, N_GRAY)), header=(label_axis, _brain_model())
    ).to_filename(str(atlas))

    spec = parse_spec(
        {
            'nodes': [
                {'name': 'load', 'action': 'select_data', 'dataset': 'xcpd'},
                {'name': 'atlas', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 'parc',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'load', 'atlas': 'atlas'},
                },
                {
                    'name': 'fc',
                    'action': 'functional_connectivity',
                    'inputs': {'timeseries': 'parc'},
                },
            ]
        }
    )
    resolved = {
        'load': Match('/x/b.dtseries.nii', {'extension': '.dtseries.nii'}),
        'atlas': Match(str(atlas), {'extension': '.dlabel.nii'}),
    }
    ctx = FactoryContext(spec=spec, resolved=resolved)
    wf = init_functional_connectivity_wf(spec.by_name()['fc'], context=ctx)
    assert isinstance(wf.get_node('correlate').interface, CiftiCorrelation)


def test_fc_plans_a_tsv_relmat_after_a_probseg_parcellation(tmp_path):
    """Not only the graph -- the sink plan must not promise a .pconn.nii either."""
    from bdt.engine.selection import Match
    from bdt.outputs.plan import build_sink_plan
    from bdt.spec import parse_spec

    _fixture(tmp_path)
    spec = parse_spec(
        {
            'nodes': [
                {'name': 'load', 'action': 'select_data', 'dataset': 'xcpd'},
                {'name': 'atlas', 'action': 'select_atlases', 'dataset': 'atlases'},
                {
                    'name': 'parc',
                    'action': 'parcellate_timeseries',
                    'inputs': {'timeseries': 'load', 'atlas': 'atlas'},
                },
                {
                    'name': 'fc',
                    'action': 'functional_connectivity',
                    'inputs': {'timeseries': 'parc'},
                    'write_outputs': True,
                },
            ]
        }
    )
    resolved = {
        'load': Match(str(tmp_path / 'data.dscalar.nii'), {'extension': '.dscalar.nii'}),
        'atlas': Match(str(tmp_path / 'atlas.dscalar.nii'), {'extension': '.dscalar.nii'}),
    }
    plan = build_sink_plan(spec, resolved, roots={'xcpd': str(tmp_path)})
    assert [(p.suffix, p.extension) for p in plan['fc']] == [('relmat', '.tsv')]
