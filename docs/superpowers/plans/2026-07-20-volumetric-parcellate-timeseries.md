# Volumetric `parcellate_timeseries` + `functional_connectivity` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `parcellate_timeseries` and `functional_connectivity` route volumetrically (like `parcellate_scalar` already does) so a NIfTI bold + NIfTI atlas parcellate and correlate end to end (`scripts/nifti_parcellate.yml`).

**Architecture:** Route both actions on CIFTI-ness (`_produces_cifti`, which propagates through processing nodes) rather than extension. The volumetric `parcellate_timeseries` warps the atlas into the bold's space via Spec 1's `ResolveApplyTransforms` (standard→standard TemplateFlow edges here — no ACPC, no `wb_command`) and parcellates into a wide (timepoints × regions) TSV plus a per-region coverage TSV; volumetric `functional_connectivity` correlates that wide TSV via the existing `tsv_correlation`.

**Tech Stack:** Python 3.12, nipype, nitransforms 25.1.0, nibabel, numpy, pandas; micromamba env `bdtenv`.

## Global Constraints

- Environment: `bdtenv` micromamba env. Run tests via `micromamba run -n bdtenv python -m pytest`.
- **No git commits** — edit in place; the user manages version control. Every task ends with a "Verify (no commit)" step. Do **not** run any `git` command.
- **Reuse, don't rewrite.** Spec 1 (`bdt.transforms`, `bdt.interfaces.transforms`) and the volumetric-scalar work (`bdt.interfaces.parcellate.ParcellateVolumetric`, `_init_parcellate_volumetric_wf`, `FactoryContext._role_entity`/`_entities_by_node`/`discover_transforms`) are complete and tested. The CIFTI paths (`_init_parcellate_cifti_wf`, `CiftiCorrelation`) must stay unchanged and must still be reached for CIFTI-lineage data.
- **Routing signal:** `_produces_cifti(spec, resolved)` from `bdt.outputs.plan` — a processing node is CIFTI iff its primary input is. A role fed by a processing node has **no** `extension` entity, so an extension check cannot route `functional_connectivity` (its input is the parcellate node).
- **Verified facts:** `nitransforms.resampling.apply` accepts a **4D** image as `reference` and warps a 3D `moving` onto its 3D spatial grid (output is 3D). `is_cifti(path)` (`bdt.utils.cifti`) is a compound-extension check. `tsv_correlation(tsv_path, out_path)` reads a wide TSV and writes `df.corr(method='pearson')`.
- **Wide TSV orientation:** rows = timepoints, columns = region ids (as strings). A region whose `coverage < min_coverage` has its whole column set to NaN. `min_coverage` default `0.5`. Validity of a voxel = its **across-time mean** is finite and nonzero.
- **Region index convention:** 4D atlas → 1-based (matches `EntitiesToSegTSV`); 3D dseg → the integer label values (skip 0).

---

### Task 1: `FactoryContext.role_is_cifti` + route `parcellate_scalar` on it

**Files:**
- Modify: `src/bdt/engine/factories.py` (add `role_is_cifti` + a memoized `_cifti_by_node` to `FactoryContext`; change `init_parcellate_scalar_wf`'s routing check)
- Test: `test/engine/test_factory_context_transforms.py` (append)

**Interfaces:**
- Consumes: `bdt.outputs.plan._produces_cifti`; `bdt.utils.cifti.is_cifti`; existing `FactoryContext` (`spec`, `resolved`).
- Produces: `FactoryContext.role_is_cifti(node, role) -> bool` — whether the file feeding `role` is CIFTI, propagated through processing nodes; falls back to the resolved match's `is_cifti(path)` when no `spec` is set.

- [ ] **Step 1: Write the failing tests**

Append to `test/engine/test_factory_context_transforms.py`:

```python
def test_role_is_cifti_falls_back_to_resolved_path_without_spec():
    ctx = FactoryContext(resolved={
        'load_alff': _StubMatch('alff.dscalar.nii', {'extension': '.dscalar.nii'}),
        'load_cbf': _StubMatch('cbf.nii.gz', {'extension': '.nii.gz'}),
    })
    dense = _node({'scalar': ['load_alff']})
    vol = _node({'scalar': ['load_cbf']})
    assert ctx.role_is_cifti(dense, 'scalar') is True
    assert ctx.role_is_cifti(vol, 'scalar') is False
    assert ctx.role_is_cifti(vol, 'atlas') is False  # role absent -> False
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py::test_role_is_cifti_falls_back_to_resolved_path_without_spec -v`
Expected: FAIL — `AttributeError: 'FactoryContext' object has no attribute 'role_is_cifti'`.

- [ ] **Step 3: Add the methods**

In `src/bdt/engine/factories.py`, add to `FactoryContext` immediately after `_entities_by_node` (which ends by returning `cached`):

```python
    def _cifti_by_node(self):
        """Cached CIFTI-ness for every node (propagated through processing nodes)."""
        if self.spec is None or self.resolved is None:
            return None
        cached = getattr(self, '_ciftimap_cache', None)
        if cached is None:
            from bdt.outputs.plan import _produces_cifti

            cached = _produces_cifti(self.spec, self.resolved)
            self._ciftimap_cache = cached
        return cached

    def role_is_cifti(self, node, role: str) -> bool:
        """Whether the file feeding ``node``'s ``role`` is CIFTI.

        Uses ``_produces_cifti`` (which propagates through processing nodes) when a
        ``spec`` is available — necessary because a role fed by a processing node has
        no ``extension`` entity.  Falls back to the resolved match's ``is_cifti(path)``
        on the build-stub path (no spec).
        """
        cmap = self._cifti_by_node()
        if cmap is not None:
            for up in node.inputs.get(role, []):
                if up in cmap:
                    return cmap[up]
            return False
        from bdt.utils.cifti import is_cifti

        if self.resolved is None:
            return False
        for up in node.inputs.get(role, []):
            match = self.resolved.get(up)
            if match is not None:
                return is_cifti(match.path)
        return False
```

Then change `init_parcellate_scalar_wf` (currently branches on `extension in ('.nii', '.nii.gz')`) to:

```python
@workflow_factory('parcellate_scalar')
def init_parcellate_scalar_wf(node, name=None, context=None) -> pe.Workflow:
    """Parcellate a scalar with a label atlas, coverage-aware.

    Routes on the scalar's CIFTI-ness (propagated through processing nodes): a CIFTI
    dscalar uses the surface/grayordinate path (:func:`_init_parcellate_cifti_wf`,
    unchanged); a volumetric NIfTI uses the Strategy-A path
    (:func:`_init_parcellate_volumetric_wf`).
    """
    context = context or FactoryContext()
    if context.role_is_cifti(node, 'scalar'):
        return _init_parcellate_cifti_wf(node, name, 'scalar', 'parcellated.pscalar.nii')
    return _init_parcellate_volumetric_wf(node, name, context)
```

- [ ] **Step 4: Run to verify pass + no routing regressions**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py test/engine/test_parcellate_scalar_routing.py -q`
Expected: PASS (the existing `parcellate_scalar` routing tests still pass: `role_is_cifti` returns True for `.dscalar.nii` paths and False for `.nii.gz`, and for the processing-node atlas test `_produces_cifti` propagates from `load_cbf`).

- [ ] **Step 5: Verify (no commit)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_factory_context_transforms.py test/engine/test_parcellate_scalar_routing.py test/engine/test_parcellate_volumetric.py -q`
Expected: PASS. Do **not** commit.

---

### Task 2: `ParcellateVolumetricTimeseries` interface

**Files:**
- Modify: `src/bdt/interfaces/parcellate.py` (append the pure fn + interface; do not touch `_region_row`/`_parcellate_volumetric`/`ParcellateVolumetric`)
- Test: `test/engine/test_parcellate_volumetric.py` (append)

**Interfaces:**
- Consumes: `nibabel`, `numpy`, `pandas`; the module's existing nipype-base imports.
- Produces:
  - `_parcellate_volumetric_timeseries(data_path, atlas_path, out_path, coverage_path, min_coverage=0.5) -> str` — writes the wide (timepoints × regions) TSV and the per-region coverage TSV; returns `out_path`.
  - `ParcellateVolumetricTimeseries(SimpleInterface)` — inputs `timeseries` (File, mandatory), `atlas` (File, mandatory), `min_coverage` (Float, default 0.5), `out_file` (Str, default `'parcellated.tsv'`), `coverage_file` (Str, default `'coverage.tsv'`); outputs `out_file` (File), `coverage_file` (File).

- [ ] **Step 1: Write the failing tests**

Append to `test/engine/test_parcellate_volumetric.py`:

```python
def test_timeseries_wide_weighted_means(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric_timeseries

    # 2 voxels, 3 timepoints. voxel0 series [1,2,3], voxel1 series [3,4,5].
    data = np.zeros((2, 1, 1, 3), dtype=np.float32)
    data[0, 0, 0, :] = [1.0, 2.0, 3.0]
    data[1, 0, 0, :] = [3.0, 4.0, 5.0]
    ts = _nii(tmp_path / 'bold.nii.gz', data)
    # 4D pseg: one region weighting voxel0 by 0.25, voxel1 by 0.75.
    atlas = np.zeros((2, 1, 1, 1), dtype=np.float32)
    atlas[0, 0, 0, 0] = 0.25
    atlas[1, 0, 0, 0] = 0.75
    at = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric_timeseries(
        ts, at, str(tmp_path / 'p.tsv'), str(tmp_path / 'c.tsv'), min_coverage=0.0
    )
    df = pd.read_csv(out, sep='\t')
    assert df.shape == (3, 1)  # 3 timepoints, 1 region
    # weighted mean per t: (.25*v0 + .75*v1); t0=(.25*1+.75*3)=2.5, t1=3.5, t2=4.5
    assert df['1'].tolist() == [2.5, 3.5, 4.5]
    cov = pd.read_csv(str(tmp_path / 'c.tsv'), sep='\t')
    assert cov['index'].tolist() == [1]
    assert cov['coverage'].tolist() == [1.0]


def test_timeseries_low_coverage_column_is_nan(tmp_path):
    from bdt.interfaces.parcellate import _parcellate_volumetric_timeseries

    # region covers 4 voxels; only voxel0 has data (nonzero across-time mean).
    data = np.zeros((4, 1, 1, 2), dtype=np.float32)
    data[0, 0, 0, :] = [8.0, 8.0]
    ts = _nii(tmp_path / 'bold.nii.gz', data)
    atlas = np.ones((4, 1, 1, 1), dtype=np.int16)
    at = _nii(tmp_path / 'atlas.nii.gz', atlas)

    out = _parcellate_volumetric_timeseries(
        ts, at, str(tmp_path / 'p.tsv'), str(tmp_path / 'c.tsv'), min_coverage=0.5
    )
    df = pd.read_csv(out, sep='\t')
    assert bool(df['1'].isna().all())  # coverage 0.25 < 0.5 -> whole column NaN
    cov = pd.read_csv(str(tmp_path / 'c.tsv'), sep='\t')
    assert cov['coverage'].tolist() == [0.25]


def test_timeseries_interface_emits_both_tsvs(tmp_path):
    from bdt.interfaces.parcellate import ParcellateVolumetricTimeseries

    data = np.zeros((2, 1, 1, 2), dtype=np.float32)
    data[0, 0, 0, :] = [1.0, 3.0]
    data[1, 0, 0, :] = [3.0, 5.0]
    ts = _nii(tmp_path / 'bold.nii.gz', data)
    at = _nii(tmp_path / 'atlas.nii.gz', np.array([1, 2], dtype=np.int16).reshape(2, 1, 1))

    res = ParcellateVolumetricTimeseries(
        timeseries=ts, atlas=at, min_coverage=0.0,
        out_file=str(tmp_path / 'p.tsv'), coverage_file=str(tmp_path / 'c.tsv'),
    ).run()
    df = pd.read_csv(res.outputs.out_file, sep='\t')
    assert sorted(df.columns) == ['1', '2']
    assert df['1'].tolist() == [1.0, 3.0]  # region 1 = voxel0 series
    assert df['2'].tolist() == [3.0, 5.0]  # region 2 = voxel1 series
    assert pd.read_csv(res.outputs.coverage_file, sep='\t')['index'].tolist() == [1, 2]
```

- [ ] **Step 2: Run to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_volumetric.py -k timeseries -v`
Expected: FAIL — `ImportError: cannot import name '_parcellate_volumetric_timeseries'`.

- [ ] **Step 3: Append the implementation**

Append to `src/bdt/interfaces/parcellate.py`:

```python
def _parcellate_volumetric_timeseries(
    data_path, atlas_path, out_path, coverage_path, min_coverage=0.5
):
    """Parcellate a 4D NIfTI timeseries over a label atlas into a wide TSV.

    Writes ``out_path`` (rows = timepoints, columns = region indices; a region's
    whole column is NaN when its coverage is below ``min_coverage``) and
    ``coverage_path`` (``index``, ``coverage`` per region).  A voxel is valid when
    its across-time mean is finite and nonzero; the per-timepoint value is the
    coverage-weighted mean over valid voxels.  Same region-weight convention as
    :func:`_parcellate_volumetric` (4D atlas -> 1-based; 3D -> integer labels).
    """
    import nibabel as nb
    import numpy as np
    import pandas as pd

    data = np.asarray(nb.load(data_path).dataobj, dtype='float64')
    if data.ndim == 3:
        data = data[..., np.newaxis]
    n_t = data.shape[3]
    atlas_img = nb.load(atlas_path)
    atlas = np.asarray(atlas_img.dataobj, dtype='float64')

    valid = np.isfinite(data.mean(axis=3)) & (data.mean(axis=3) != 0)  # (X,Y,Z)
    data0 = np.where(valid[..., np.newaxis], data, 0.0)  # zero non-valid (avoid NaN*0)

    if atlas_img.ndim > 3:
        regions = [(i + 1, atlas[..., i]) for i in range(atlas.shape[3])]
    else:
        regions = [
            (int(v), (atlas == v).astype('float64')) for v in np.unique(atlas) if v != 0
        ]

    columns = {}
    cov_rows = []
    for index, weight in regions:
        total = float(weight.sum())
        covered = float((weight * valid).sum())
        coverage = covered / total if total > 0 else 0.0
        cov_rows.append({'index': int(index), 'coverage': coverage})
        if covered > 0 and coverage >= min_coverage:
            num = (weight[..., np.newaxis] * data0).sum(axis=(0, 1, 2))  # (n_t,)
            columns[str(index)] = num / covered
        else:
            columns[str(index)] = np.full(n_t, np.nan)

    pd.DataFrame(columns).to_csv(out_path, sep='\t', index=False)
    pd.DataFrame(cov_rows, columns=['index', 'coverage']).to_csv(
        coverage_path, sep='\t', index=False
    )
    return str(out_path)


class _ParcellateVolumetricTimeseriesInputSpec(BaseInterfaceInputSpec):
    timeseries = File(exists=True, mandatory=True, desc='4D NIfTI in the atlas grid/space')
    atlas = File(exists=True, mandatory=True, desc='3D dseg or 4D dseg/pseg label atlas')
    min_coverage = traits.Float(0.5, usedefault=True, desc='NaN-mask regions below this')
    out_file = traits.Str('parcellated.tsv', usedefault=True, desc='wide (time x regions) TSV')
    coverage_file = traits.Str('coverage.tsv', usedefault=True, desc='per-region coverage TSV')


class _ParcellateVolumetricTimeseriesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='wide (timepoints x regions) parcellated TSV')
    coverage_file = File(exists=True, desc='per-region coverage TSV (index, coverage)')


class ParcellateVolumetricTimeseries(SimpleInterface):
    """Coverage-aware volumetric parcellation of a 4D timeseries over a label atlas."""

    input_spec = _ParcellateVolumetricTimeseriesInputSpec
    output_spec = _ParcellateVolumetricTimeseriesOutputSpec

    def _run_interface(self, runtime):
        out_file = self.inputs.out_file
        cov_file = self.inputs.coverage_file
        if not os.path.isabs(out_file):
            out_file = os.path.join(runtime.cwd, out_file)
        if not os.path.isabs(cov_file):
            cov_file = os.path.join(runtime.cwd, cov_file)
        _parcellate_volumetric_timeseries(
            self.inputs.timeseries, self.inputs.atlas, out_file, cov_file,
            min_coverage=self.inputs.min_coverage,
        )
        self._results['out_file'] = out_file
        self._results['coverage_file'] = cov_file
        return runtime
```

- [ ] **Step 4: Run to verify pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_volumetric.py -q -W error::UserWarning`
Expected: PASS (all scalar tests + 3 new timeseries tests), pristine.

- [ ] **Step 5: Verify (no commit)** — same command as Step 4. Do **not** commit.

---

### Task 3: shared warp helper + volumetric-timeseries subworkflow + routing

**Files:**
- Modify: `src/bdt/engine/factories.py` (extract `_warp_atlas_field` from `_init_parcellate_volumetric_wf`; add `_init_parcellate_volumetric_timeseries_wf`; branch `init_parcellate_timeseries_wf`)
- Test: `test/engine/test_parcellate_timeseries_routing.py`

**Interfaces:**
- Consumes: `ParcellateVolumetricTimeseries` (Task 2); `ResolveApplyTransforms`; `FactoryContext.role_is_cifti`/`role_space`/`role_suffix`/`role_session`/`discover_transforms`/`find_reference`; `_register_acpc_to_t1w`; `_init_parcellate_cifti_wf`.
- Produces: `_warp_atlas_field(wf, node, context, inputnode, data_role) -> tuple[object, str]` (the atlas source field, warped or direct); `init_parcellate_timeseries_wf` routing + `_init_parcellate_volumetric_timeseries_wf`.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_parcellate_timeseries_routing.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Build-time routing tests for parcellate_timeseries (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_parcellate_timeseries_wf


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


def test_volumetric_timeseries_same_space_builds_parcellate_no_warp():
    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(resolved={
        'load_bold': _match('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
        'load_atlas': _match('atlas.nii.gz', {'space': 'MNI152NLin6Asym', 'suffix': 'dseg'}),
    })
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'parcellate' in names
    assert 'warp_atlas' not in names


def test_volumetric_timeseries_cross_space_builds_warp():
    from bdt.engine.selection import DictDataProvider

    node = _node('parc', {'timeseries': ['load_bold'], 'atlas': ['load_atlas']})
    ctx = FactoryContext(
        provider=DictDataProvider({'atlases': []}), subject='01', datasets=['atlases'],
        resolved={
            'load_bold': _match('bold.nii.gz', {'space': 'MNI152NLin6Asym'}),
            'load_atlas': _match(
                'atlas.nii.gz', {'space': 'MNI152NLin2009cAsym', 'suffix': 'dseg'}
            ),
        },
    )
    names = set(init_parcellate_timeseries_wf(node, context=ctx).list_node_names())
    assert 'warp_atlas' in names
    assert 'register_acpc' not in names  # no ACPC endpoint
```

- [ ] **Step 2: Run to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_timeseries_routing.py -v`
Expected: FAIL — the volumetric tests fail (current `init_parcellate_timeseries_wf` always calls the CIFTI path, so no `parcellate`/`warp_atlas` node).

- [ ] **Step 3: Extract the helper, add the subworkflow, branch the factory**

In `src/bdt/engine/factories.py`, refactor the cross-space block of `_init_parcellate_volumetric_wf` into a shared helper and reuse it. Replace the body of `_init_parcellate_volumetric_wf` from `atlas_space = context.role_space(node, 'atlas')` through the `return wf` at the end of the ACPC block with a call to the new helper:

```python
def _warp_atlas_field(wf, node, context, inputnode, data_role):
    """The atlas source field for the parcellator, warped into the data's space when
    they differ (else ``(inputnode, 'atlas')``).

    Inserts a :class:`~bdt.interfaces.transforms.ResolveApplyTransforms` ``warp_atlas``
    node (nearest for a dseg, linear for a pseg; reference = the data on ``data_role``)
    and, only when ``ACPC`` is an endpoint, the computed rigid ACPC<->T1w bridge
    (``register_acpc`` + ``bridge_list``), mirroring :func:`init_map_scalar_to_surface_wf`.
    """
    from bdt.interfaces.transforms import ResolveApplyTransforms

    atlas_space = context.role_space(node, 'atlas')
    data_space = context.role_space(node, data_role)
    cross_space = (
        atlas_space is not None and data_space is not None and atlas_space != data_space
    )
    if not cross_space:
        return (inputnode, 'atlas')

    atlas_suffix = context.role_suffix(node, 'atlas', default='dseg') or 'dseg'
    interpolation = 'linear' if atlas_suffix in ('probseg', 'pseg') else 'nearest'
    warp = pe.Node(
        ResolveApplyTransforms(
            source=atlas_space,
            target=data_space,
            interpolation=interpolation,
            local_transforms=context.discover_transforms(
                session=context.role_session(node, 'atlas')
            ),
            out_file='atlas_in_data_space.nii.gz',
        ),
        name='warp_atlas',
    )
    wf.connect([(inputnode, warp, [('atlas', 'moving'), (data_role, 'reference')])])

    if 'ACPC' in (atlas_space, data_space):
        if context.provider is None:
            raise ValueError(
                f'parcellate node {node.name!r} is cross-space through ACPC and needs a '
                'FactoryContext provider to resolve the bridge references.'
            )
        atlas_ses = context.role_session(node, 'atlas')
        fixed_img = context.find_reference(
            {'suffix': ['T1w', 'T2w'], 'desc': 'preproc', 'space': None, 'datatype': 'anat'},
            session=atlas_ses,
        )
        fixed_mask = context.find_reference(
            {'suffix': 'mask', 'desc': 'brain', 'space': None, 'datatype': 'anat'},
            session=atlas_ses,
        )
        moving_img = context.find_reference(
            {'suffix': ['T1w', 'T2w'], 'desc': 'preproc', 'space': 'ACPC', 'datatype': 'anat'},
            session=atlas_ses,
        )
        moving_mask = context.find_reference(
            {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'},
            session=atlas_ses,
        )
        register = pe.Node(
            niu.Function(
                function=_register_acpc_to_t1w,
                input_names=['fixed_image', 'fixed_mask', 'moving_image', 'moving_mask'],
                output_names=['out'],
            ),
            name='register_acpc',
            n_procs=4,
        )
        register.inputs.fixed_image = fixed_img
        register.inputs.fixed_mask = fixed_mask
        register.inputs.moving_image = moving_img
        register.inputs.moving_mask = moving_mask
        bridge_list = pe.Node(niu.Merge(1), name='bridge_list')
        wf.connect([
            (register, bridge_list, [('out', 'in1')]),
            (bridge_list, warp, [('out', 'bridges')]),
        ])  # fmt:skip

    return (warp, 'out_file')
```

Rewrite `_init_parcellate_volumetric_wf` to use it (behavior identical):

```python
def _init_parcellate_volumetric_wf(node, name, context) -> pe.Workflow:
    """Volumetric (NIfTI) coverage-aware scalar parcellation, Strategy A."""
    from bdt.interfaces.parcellate import ParcellateVolumetric

    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))
    inputnode = pe.Node(niu.IdentityInterface(fields=['scalar', 'atlas']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')
    parcellate = pe.Node(
        ParcellateVolumetric(min_coverage=min_coverage, out_file='parcellated.tsv'),
        name='parcellate',
    )
    atlas_node, atlas_field = _warp_atlas_field(wf, node, context, inputnode, 'scalar')
    wf.connect([
        (inputnode, parcellate, [('scalar', 'scalar')]),
        (atlas_node, parcellate, [(atlas_field, 'atlas')]),
        (parcellate, outputnode, [('out_file', 'out')]),
    ])  # fmt:skip
    return wf
```

Add the timeseries factory + subworkflow. Replace the existing `init_parcellate_timeseries_wf` (currently a 2-line body calling `_init_parcellate_cifti_wf`) with:

```python
@workflow_factory('parcellate_timeseries')
def init_parcellate_timeseries_wf(node, name=None, context=None) -> pe.Workflow:
    """Parcellate a dense series with a dlabel/label atlas, coverage-aware.

    Routes on the timeseries' CIFTI-ness: a dense CIFTI uses the grayordinate path
    (:func:`_init_parcellate_cifti_wf`, unchanged); a volumetric NIfTI uses the
    Strategy-A path (:func:`_init_parcellate_volumetric_timeseries_wf`).
    """
    context = context or FactoryContext()
    if context.role_is_cifti(node, 'timeseries'):
        return _init_parcellate_cifti_wf(node, name, 'timeseries', 'parcellated.ptseries.nii')
    return _init_parcellate_volumetric_timeseries_wf(node, name, context)


def _init_parcellate_volumetric_timeseries_wf(node, name, context) -> pe.Workflow:
    """Volumetric (NIfTI) coverage-aware timeseries parcellation, Strategy A.

    ``inputnode`` takes the ``timeseries`` and ``atlas``; the atlas is warped into
    the bold's space when they differ (:func:`_warp_atlas_field`).  ``outputnode.out``
    is the wide (timepoints x regions) TSV; ``outputnode.coverage`` is the per-region
    coverage TSV (feeding the action's coverage product).
    """
    from bdt.interfaces.parcellate import ParcellateVolumetricTimeseries

    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['timeseries', 'atlas']), name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out', 'coverage']), name='outputnode'
    )
    parcellate = pe.Node(
        ParcellateVolumetricTimeseries(
            min_coverage=min_coverage,
            out_file='parcellated.tsv',
            coverage_file='coverage.tsv',
        ),
        name='parcellate',
    )
    atlas_node, atlas_field = _warp_atlas_field(wf, node, context, inputnode, 'timeseries')
    wf.connect([
        (inputnode, parcellate, [('timeseries', 'timeseries')]),
        (atlas_node, parcellate, [(atlas_field, 'atlas')]),
        (parcellate, outputnode, [('out_file', 'out'), ('coverage_file', 'coverage')]),
    ])  # fmt:skip
    return wf
```

- [ ] **Step 4: Run to verify pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_timeseries_routing.py test/engine/test_parcellate_scalar_routing.py -q`
Expected: PASS (new timeseries routing + the scalar routing still green — the refactor is behavior-preserving).

- [ ] **Step 5: Verify (no commit)**

Run: `micromamba run -n bdtenv python -m pytest test/engine -q`
Expected: only the 5 known pre-existing failures (`test_outputs` ×2, `test_pipeline`, `test_pybids_provider` ×2), zero new. Do **not** commit.

---

### Task 4: format-aware coverage `ExtraProduct` (volumetric coverage TSV)

**Files:**
- Modify: `src/bdt/spec/actions.py` (`ExtraProduct` gains `volumetric_extension`; update the `parcellate_timeseries` coverage product)
- Modify: `src/bdt/outputs/plan.py` (`build_sink_plan` extra-product loop)
- Test: `test/engine/test_output_plan.py` (append)

**Interfaces:**
- Consumes: existing `ExtraProduct`, `build_sink_plan`, `_produces_cifti`.
- Produces: `ExtraProduct.volumetric_extension: str | None = None`; `build_sink_plan` emits a coverage product for a volumetric node using `volumetric_extension`.

- [ ] **Step 1: Write the failing test**

First read one existing `build_sink_plan` test in `test/engine/test_output_plan.py` to copy its exact spec-construction idiom (how it builds a `Spec` + `resolved` and calls `build_sink_plan`), then append a test in that established style asserting:
- a **volumetric** `parcellate_timeseries` node (NIfTI bold + NIfTI atlas, `write_outputs: true`) yields a primary product with extension `.tsv` **and** a coverage product with extension `.tsv` (source_field `coverage`);
- a **CIFTI** `parcellate_timeseries` node still yields a `.ptseries.nii` primary + a `.pscalar.nii` coverage product.

Model the two nodes on `test_volumetric_same_space...` (Task 3) for the volumetric case and on the existing CIFTI parcellate story for the dense case. Assert on the products' `extension` and `source_field` fields returned by `build_sink_plan`.

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_output_plan.py -k coverage -v`
Expected: FAIL — the volumetric node emits **no** coverage product (the coverage `ExtraProduct` is `cifti_only=True`, skipped for volumetric).

- [ ] **Step 3: Add `volumetric_extension` and use it**

In `src/bdt/spec/actions.py`, add the field to `ExtraProduct` (after `cifti_only`):

```python
    volumetric_extension: str | None = None  # extension for a volumetric (non-CIFTI) node
```

Change the `parcellate_timeseries` coverage product to:

```python
            extra=(
                ExtraProduct(
                    'coverage', 'boldmap', '.pscalar.nii',
                    volumetric_extension='.tsv', cifti_only=False, stat='coverage',
                ),
            ),
```

In `src/bdt/outputs/plan.py`, update the extra-product loop in `build_sink_plan`. Replace:

```python
        for ep in out.extra:
            if ep.cifti_only and not cifti_by_node.get(node.name):
                continue
            ep_ent = dict(mid)
            if ep.stat is not None:
                ep_ent['stat'] = ep.stat
            ep_suffix = primary_suffix if ep.match_primary_suffix else ep.suffix
            products.append(
                OutputProduct(
                    derive=PASSTHROUGH,
                    suffix=ep_suffix,
                    extension=ep.extension,
                    entities=ep_ent,
                    sidecar=dict(sidecar),
                    source_field=ep.source_field,
                    **common,
                )
            )
```

with:

```python
        for ep in out.extra:
            is_cifti_node = bool(cifti_by_node.get(node.name))
            if not is_cifti_node and ep.cifti_only and ep.volumetric_extension is None:
                continue
            ep_extension = ep.extension if is_cifti_node else (
                ep.volumetric_extension or ep.extension
            )
            ep_ent = dict(mid)
            if ep.stat is not None:
                ep_ent['stat'] = ep.stat
            ep_suffix = primary_suffix if ep.match_primary_suffix else ep.suffix
            products.append(
                OutputProduct(
                    derive=PASSTHROUGH,
                    suffix=ep_suffix,
                    extension=ep_extension,
                    entities=ep_ent,
                    sidecar=dict(sidecar),
                    source_field=ep.source_field,
                    **common,
                )
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_output_plan.py -q`
Expected: PASS (new coverage test + existing plan tests — the CIFTI coverage product is unchanged: `is_cifti_node` true → `ep.extension` = `.pscalar.nii`).

- [ ] **Step 5: Verify (no commit)**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_output_plan.py test/engine/test_outputs.py test/spec/test_spec.py -q`
Expected: only pre-existing failures (`test_outputs` ×2), zero new. Do **not** commit.

---

### Task 5: volumetric `functional_connectivity` routing

**Files:**
- Modify: `src/bdt/engine/factories.py` (branch `init_functional_connectivity_wf`)
- Test: `test/engine/test_functional_connectivity_routing.py`

**Interfaces:**
- Consumes: `FactoryContext.role_is_cifti`; `bdt.utils.cifti.tsv_correlation`; existing `CiftiCorrelation`.
- Produces: `init_functional_connectivity_wf` routes CIFTI → `CiftiCorrelation`; volumetric → a `Function` node running `tsv_correlation`.

- [ ] **Step 1: Write the failing tests**

Create `test/engine/test_functional_connectivity_routing.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Routing tests for functional_connectivity (CIFTI vs volumetric)."""

from types import SimpleNamespace

import pytest

pytest.importorskip('nipype')

from bdt.engine.factories import FactoryContext, init_functional_connectivity_wf


def _match(path, entities):
    from bdt.engine.selection import Match

    return Match(path, entities)


def _node(name, role_to_upstreams):
    return SimpleNamespace(name=name, inputs=role_to_upstreams, parameters={})


def test_fc_cifti_uses_cifti_correlation():
    node = _node('fc', {'timeseries': ['parc']})
    ctx = FactoryContext(resolved={'parc': _match('p.ptseries.nii', {})})
    names = set(init_functional_connectivity_wf(node, context=ctx).list_node_names())
    assert 'correlate' in names


def test_fc_volumetric_uses_tsv_correlation():
    node = _node('fc', {'timeseries': ['parc']})
    ctx = FactoryContext(resolved={'parc': _match('p.tsv', {})})
    wf = init_functional_connectivity_wf(node, context=ctx)
    names = set(wf.list_node_names())
    assert 'correlate_tsv' in names
    assert 'correlate' not in names  # not the CIFTI node
```

- [ ] **Step 2: Run to verify they fail**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_functional_connectivity_routing.py -v`
Expected: FAIL — `test_fc_volumetric...` fails (current factory always builds the CIFTI `correlate` node).

- [ ] **Step 3: Branch the factory**

In `src/bdt/engine/factories.py`, replace `init_functional_connectivity_wf` with:

```python
@workflow_factory('functional_connectivity')
def init_functional_connectivity_wf(node, name=None, context=None) -> pe.Workflow:
    """Correlate a parcellated series into a relmat.

    CIFTI ptseries -> :class:`CiftiCorrelation` -> pconn (unchanged); a volumetric
    parcellated TSV -> Pearson correlation of its region columns
    (:func:`bdt.utils.cifti.tsv_correlation`) -> a relmat TSV.
    """
    context = context or FactoryContext()
    wf = pe.Workflow(name=name or node.name)
    inputnode, outputnode = _io_nodes(['timeseries'])

    if context.role_is_cifti(node, 'timeseries'):
        correlate = pe.Node(CiftiCorrelation(out_file='correlations.pconn.nii'), name='correlate')
        wf.connect([
            (inputnode, correlate, [('timeseries', 'in_file')]),
            (correlate, outputnode, [('out_file', 'out')]),
        ])  # fmt:skip
        return wf

    from bdt.utils.cifti import tsv_correlation

    correlate = pe.Node(
        niu.Function(
            function=tsv_correlation,
            input_names=['tsv_path', 'out_path'],
            output_names=['out'],
        ),
        name='correlate_tsv',
    )
    correlate.inputs.out_path = 'correlations.tsv'
    wf.connect([
        (inputnode, correlate, [('timeseries', 'tsv_path')]),
        (correlate, outputnode, [('out', 'out')]),
    ])  # fmt:skip
    return wf
```

- [ ] **Step 4: Run to verify pass**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_functional_connectivity_routing.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify (no commit)**

Run: `micromamba run -n bdtenv python -m pytest test/engine -q`
Expected: only the 5 known pre-existing failures, zero new. Do **not** commit.

---

### Task 6: acceptance — `nifti_parcellate.yml` compiles end to end + ledger

**Files:**
- Test: `test/engine/test_parcellate_timeseries_routing.py` (append the compile test)
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: the full stack from Tasks 1-5; `bdt.engine.workflow.init_bdt_wf`; `bdt.spec.load_spec`; `bdt.engine.selection.DictDataProvider`/`Match`.

- [ ] **Step 1: Write the compile test**

Read `test/engine/test_nipype_workflow.py` for the exact `init_bdt_wf(spec, selections, context=...)` idiom (how `selections` paths and a `FactoryContext` with `spec`/`resolved`/`provider` are supplied). Then append to `test/engine/test_parcellate_timeseries_routing.py` a test that:
- `load_spec('scripts/nifti_parcellate.yml')`;
- builds a `FactoryContext` with `spec`, a `DictDataProvider` (the atlas as a `select_atlases` selection, plus TemplateFlow xfm files touched for `discover_transforms`; standard→standard needs no ACPC anatomicals), and `resolved` giving `load_bold` a `MNI152NLin6Asym` `.nii.gz` and `atlas_4s456` a `MNI152NLin2009cAsym` `.nii.gz` dseg;
- calls `init_bdt_wf(spec, selections, context=context)` and asserts the built graph contains the volumetric nodes for `parcellate_bold` (`parcellate`, `warp_atlas`) and `fc_bold` (`correlate_tsv`), and NOT `vertex_mask`/`correlate` (the CIFTI nodes).

Use `set(wf.list_node_names())` and substring/`endswith` checks (node names are nested under the per-node subworkflow, e.g. `parcellate_bold.warp_atlas`).

- [ ] **Step 2: Run the compile test**

Run: `micromamba run -n bdtenv python -m pytest test/engine/test_parcellate_timeseries_routing.py -q`
Expected: PASS.

- [ ] **Step 3: Full subsystem + regression run**

Run: `micromamba run -n bdtenv python -m pytest test/engine test/spec -q`
Expected: only the 5 known pre-existing failures, zero new.

- [ ] **Step 4: Update the ledger**

Append a `# Progress ledger: volumetric parcellate_timeseries + FC` section to `.superpowers/sdd/progress.md` summarizing: `role_is_cifti` routing (parcellate_scalar retrofitted); `ParcellateVolumetricTimeseries` (wide TSV + coverage TSV, NaN-masked low-coverage columns); shared `_warp_atlas_field` helper (dedups the scalar/timeseries cross-space warp + ACPC bridge); format-aware coverage `ExtraProduct` (`.tsv` volumetric); volumetric FC via `tsv_correlation`; and that `nifti_parcellate.yml` compiles end to end. Note the still-index-based region names. Do **not** commit.

- [ ] **Step 5: Verify (no commit)** — re-run Step 3's command; confirm zero new failures. Do **not** commit.

---

## Notes for the executor

- **Region names remain index-based** (the atlas `dseg.tsv` sidecar is not carried by role wiring); the wide TSV's columns are region indices. Same documented limitation as `parcellate_scalar`.
- **The refactor in Task 3 must be behavior-preserving** for `parcellate_scalar`: `_init_parcellate_volumetric_wf` keeps building `parcellate` + (when cross-space) `warp_atlas`/`register_acpc`/`bridge_list` with identical names, so the existing `test_parcellate_scalar_routing.py` (including the multi-session ACPC test and the real-spec processing-node test) must stay green.
- **Do not run the real ds008325 data** in tests (no in-repo fixtures, no `wb_command`); the acceptance test is a build/compile check only. A live run is the user's to do.
- **Do not touch** `_init_parcellate_cifti_wf`, `CiftiCorrelation`, Spec 1, or the scalar `ParcellateVolumetric`/`_parcellate_volumetric`.
