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
"""The BDT action registry — the single source of truth for the node grammar.

Every action declares its *kind* (selection vs. processing), its named input
*roles* (with required/optional/list-valued flags and the upstream data
*formats* each role accepts), the format it *produces*, and the analysis
*scopes* it may run at.  Both the static validator (:mod:`bdt.spec.validate`)
and the executor dispatch read this registry, so adding an action is a
single-place change.

This mirrors QSIRecon's recon-spec ``action`` dispatch
(``qsirecon/qsirecon/workflows/recon/build_workflow.py``) but replaces its
hard-coded ``if/elif`` ladder and field-name-intersection wiring with an
explicit, introspectable contract keyed by *declared role*.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Controlled vocabulary of data formats that flow between nodes.
#
# Kept deliberately coarse — just fine-grained enough to enforce the contracts
# the spec calls out (e.g. ``functional_connectivity`` must consume a *parcellated*
# series, not a dense one).  ``unknown`` is used for values whose format can only
# be resolved at runtime (a ``select_data`` match that could not be typed from its
# filters); the validator skips strict format checks against ``unknown``.
# ---------------------------------------------------------------------------
FORMATS = frozenset(
    {
        'timeseries',  # dense/volumetric BOLD-like time series (dtseries / 4D nii)
        'scalar',  # a scalar map (NIfTI volume, CIFTI dscalar, or GIFTI metric)
        'surface_scalar',  # scalar data on a surface mesh (func/shape.gii or dscalar)
        'surfaces',  # surface geometry (white/pial/midthickness .surf.gii)
        'atlas',  # a labelled atlas (dseg/dlabel) usable in an ``atlas`` role
        'structures',  # subcortical structure definition (grayordinate assembly)
        'parcellated_timeseries',  # parcel x time (ptseries / tsv)
        'parcellated_scalar',  # parcel means (pscalar / tsv)
        'relmat',  # a connectivity matrix (region x region / bundle x region)
        'subcortical_volume',  # continuous scalar on the grayordinate subcortical grid
        'dense_cifti',  # assembled cortex + subcortex dense CIFTI
        'streamlines',  # a tractogram (TRX)
        'profile',  # per-node along-tract profile (tsv)
        'roi_means',  # per-region / per-bundle summary (tsv)
        'unknown',  # runtime-determined format
    }
)

SCOPES = frozenset({'participant', 'dataset', 'both'})

SELECTION = 'selection'
PROCESSING = 'processing'


@dataclass(frozen=True)
class Role:
    """A named input slot on a processing action.

    Parameters
    ----------
    name
        The role key used under a processing node's ``inputs:`` mapping.
    accepts
        The upstream :data:`FORMATS` this role will accept.  Used by the static
        validator to reject bad wiring (e.g. a dense series into an FC node).
    required
        Whether the role must be wired for the node to be valid.
    list_ok
        Whether the role may be wired to a *list* of upstream nodes (fan-out).
    fan_out
        Whether a multi-result upstream should be *fanned over* (one downstream
        branch per result; e.g. one output per atlas, per scalar) or consumed as a
        single *group* (e.g. the L/R white/pial/midthickness ``surfaces`` set, or
        all bundles feeding ``tractogram_to_dseg``).  Group roles collapse their
        upstream results' files into one value.
    """

    name: str
    accepts: frozenset[str]
    required: bool = True
    list_ok: bool = True
    fan_out: bool = True

    def __post_init__(self):
        bad = set(self.accepts) - FORMATS
        if bad:
            raise ValueError(f'Role {self.name!r} accepts unknown formats: {sorted(bad)}')


@dataclass(frozen=True)
class ExtraProduct:
    """A secondary file a ``write_outputs`` node materializes beyond its primary.

    ``source_field`` is the sub-workflow ``outputnode`` field it comes from (e.g.
    ``'coverage'``); ``stat`` overrides the ``stat-`` entity for this product.
    ``cifti_only`` restricts it to CIFTI-valued nodes.  Used for e.g. the
    ``stat-coverage`` parcel-coverage map that parcellation also emits.
    """

    source_field: str
    suffix: str
    extension: str
    stat: str | None = None
    cifti_only: bool = True
    volumetric_extension: str | None = None  # extension for a volumetric (non-CIFTI) node
    match_primary_suffix: bool = False  # label TSVs follow the resolved primary suffix


@dataclass(frozen=True)
class OutputSpec:
    """How an action's materialized output is named (BIDS suffix/entities/datatype).

    ``primary_role`` names the input role whose entities seed the output filename
    (the "data" being transformed); ``entities`` are fixed entities the action
    always adds (e.g. ``{'stat': 'mean'}`` for parcellated means).  These values
    are refined per action as each action's real workflow lands; they are enough
    for the engine to compose BIDS-legal output paths today.

    ``suffix``/``extension`` name the canonical (TSV) product.  ``cifti_suffix``/
    ``cifti_extension``, when set, name the *native CIFTI* product an action also
    emits when its input is CIFTI (e.g. ``timeseries``/``.ptseries.nii`` for
    parcellation, ``boldmap``/``.pconn.nii`` for connectivity) — per the
    2026-07-16 "CIFTI in → CIFTI + TSV out" decision.

    ``preserve_source`` marks a *data-identity-preserving* transform (surface
    mapping / resampling / scalar parcellation): the output keeps the source's
    ``suffix``, ``datatype``, and identity entities (``model``/``param``/``stat``/
    ``desc``), only adding ``atlas-`` and changing geometry — so a parcellated
    ALFF stays ``stat-alff_boldmap`` and a resampled NODDI stays
    ``model-noddi_param-*_dwimap``.  ``suffix``/``datatype`` then act as fallbacks
    when the source lacks them.

    ``emit_tsv`` controls whether a CIFTI-valued node also flattens to a TSV: true
    for parcellations (parcel×time / parcel means), false for a *dense* CIFTI
    output (a resampled/mapped surface scalar is a dscalar, not a table).
    ``output_is_cifti`` forces the node's primary product to be treated as CIFTI
    regardless of its input format — for actions that emit a dense CIFTI from a
    per-hemi GIFTI input (surface resampling/mapping/assembly).
    """

    suffix: str
    extension: str
    datatype: str
    primary_role: str | None = None
    entities: dict = None  # fixed extra entities; defaults to {} via __post_init__
    cifti_suffix: str | None = None
    cifti_extension: str | None = None
    extra: tuple[ExtraProduct, ...] = ()  # secondary products (e.g. coverage)
    preserve_source: bool = False
    emit_tsv: bool = True  # also flatten a CIFTI product to TSV (false for dense CIFTI)
    output_is_cifti: bool = False  # primary product is CIFTI regardless of input
    dynamic_suffix: object = None  # optional Callable[[dict params], str] overriding suffix

    def __post_init__(self):
        if self.entities is None:
            object.__setattr__(self, 'entities', {})


@dataclass(frozen=True)
class ActionSpec:
    """The declared contract for one action."""

    name: str
    kind: str  # SELECTION | PROCESSING
    produces: str  # a FORMATS member (selection nodes may resolve to a runtime format)
    scope: str = 'both'  # a SCOPES member
    roles: tuple[Role, ...] = ()  # processing only
    parameters: frozenset[str] = frozenset()  # accepted parameter keys (informational)
    out: OutputSpec | None = None  # how a materialized output is named (processing only)
    builder: str | None = None  # executor builder key; wired in the execution phases

    def __post_init__(self):
        if self.kind not in (SELECTION, PROCESSING):
            raise ValueError(f'{self.name}: unknown kind {self.kind!r}')
        if self.produces not in FORMATS:
            raise ValueError(f'{self.name}: unknown produced format {self.produces!r}')
        if self.scope not in SCOPES:
            raise ValueError(f'{self.name}: unknown scope {self.scope!r}')
        if self.kind == SELECTION and self.roles:
            raise ValueError(f'{self.name}: selection actions take no input roles')
        seen = set()
        for role in self.roles:
            if role.name in seen:
                raise ValueError(f'{self.name}: duplicate role {role.name!r}')
            seen.add(role.name)

    def role(self, name: str) -> Role | None:
        return next((r for r in self.roles if r.name == name), None)

    @property
    def role_names(self) -> frozenset[str]:
        return frozenset(r.name for r in self.roles)

    @property
    def required_roles(self) -> frozenset[str]:
        return frozenset(r.name for r in self.roles if r.required)


def _r(
    name: str, *accepts: str, required: bool = True, list_ok: bool = True, fan_out: bool = True
) -> Role:
    return Role(
        name=name,
        accepts=frozenset(accepts),
        required=required,
        list_ok=list_ok,
        fan_out=fan_out,
    )


def _o(
    suffix,
    extension,
    datatype,
    primary_role=None,
    cifti_suffix=None,
    cifti_extension=None,
    extra=(),
    preserve_source=False,
    emit_tsv=True,
    output_is_cifti=False,
    dynamic_suffix=None,
    **entities,
) -> OutputSpec:
    return OutputSpec(
        suffix=suffix,
        extension=extension,
        datatype=datatype,
        primary_role=primary_role,
        entities=entities,
        cifti_suffix=cifti_suffix,
        cifti_extension=cifti_extension,
        extra=tuple(extra),
        preserve_source=preserve_source,
        emit_tsv=emit_tsv,
        output_is_cifti=output_is_cifti,
        dynamic_suffix=dynamic_suffix,
    )


# ---------------------------------------------------------------------------
# The registry.  Role names below MUST match the ``inputs:`` keys used in the
# user-stories spec (docs/2026-07-15-bdt-user-stories-and-spec.md, section 2).
# ---------------------------------------------------------------------------
_ACTIONS: tuple[ActionSpec, ...] = (
    # -- selection ---------------------------------------------------------
    ActionSpec('select_data', SELECTION, 'unknown'),
    ActionSpec('select_atlases', SELECTION, 'atlas'),
    # -- grid / image parcellation + connectivity (Strategy A) -------------
    ActionSpec(
        'parcellate_timeseries',
        PROCESSING,
        'parcellated_timeseries',
        roles=(_r('timeseries', 'timeseries'), _r('atlas', 'atlas')),
        parameters=frozenset({'min_coverage'}),
        out=_o(
            'timeseries',
            '.tsv',
            'func',
            primary_role='timeseries',
            cifti_suffix='timeseries',
            cifti_extension='.ptseries.nii',
            extra=(
                ExtraProduct(
                    'coverage', 'boldmap', '.pscalar.nii',
                    volumetric_extension='.tsv', cifti_only=False, stat='coverage',
                ),
            ),
            statistic='mean',
        ),
    ),
    ActionSpec(
        'parcellate_scalar',
        PROCESSING,
        'parcellated_scalar',
        roles=(
            _r('scalar', 'scalar', 'surface_scalar', 'subcortical_volume', 'dense_cifti'),
            _r('atlas', 'atlas'),
        ),
        parameters=frozenset({'min_coverage'}),
        # Data-identity-preserving: keeps the source suffix/datatype/stat/desc,
        # just adds atlas- and parcellates (dscalar -> pscalar + tsv).
        out=_o(
            'map',
            '.tsv',
            'func',
            primary_role='scalar',
            cifti_extension='.pscalar.nii',
            preserve_source=True,
            extra=(
                ExtraProduct(
                    'coverage', 'map', '.pscalar.nii',
                    volumetric_extension='.tsv', cifti_only=False, stat='coverage',
                ),
            ),
        ),
    ),
    ActionSpec(
        'functional_connectivity',
        PROCESSING,
        'relmat',
        # Enforces the spec rule: FC consumes a *parcellated* series, not a dense one.
        roles=(_r('timeseries', 'parcellated_timeseries'),),
        parameters=frozenset({'xdf_covariance'}),
        out=_o(
            'relmat',
            '.tsv',
            'func',
            primary_role='timeseries',
            cifti_suffix='boldmap',
            cifti_extension='.pconn.nii',
            statistic='pearsoncorrelation',
        ),
    ),
    # -- surface mapping + depth profiles (Strategy B / wb_command) --------
    ActionSpec(
        'map_scalar_to_surface',
        PROCESSING,
        'surface_scalar',
        roles=(_r('scalar', 'scalar'), _r('surfaces', 'surfaces', fan_out=False)),
        parameters=frozenset({'source_space'}),
        # Ribbon-samples a volume onto the native surface, keeping the source
        # identity (model/param/dwimap); output is a per-hemi native-mesh metric
        # list, typically an intermediate feeding resample_surface_scalar.
        out=_o('map', '.func.gii', 'func', primary_role='scalar', preserve_source=True),
    ),
    ActionSpec(
        'resample_surface_scalar',
        PROCESSING,
        'surface_scalar',
        roles=(
            # Both roles are per-hemi sets grouped (L+R) into one dscalar, not fanned.
            _r('surface_scalar', 'surface_scalar', fan_out=False),
            _r('surfaces', 'surfaces', fan_out=False),
        ),
        parameters=frozenset({'target_space', 'target_density'}),
        # Data-identity-preserving: keeps the source suffix/datatype (thickness in
        # anat), drops hemi, adds space-fsLR + den-91k (grayordinate cortex, medial
        # wall removed).  A dense dscalar -> native CIFTI only, no TSV flatten.
        out=_o(
            'map',
            '.dscalar.nii',
            'anat',
            primary_role='surface_scalar',
            cifti_extension='.dscalar.nii',
            preserve_source=True,
            emit_tsv=False,
            output_is_cifti=True,
            space='fsLR',
            den='91k',
        ),
    ),
    ActionSpec(
        'cortical_depth_profile',
        PROCESSING,
        'surface_scalar',
        roles=(_r('scalar', 'scalar'), _r('surfaces', 'surfaces', fan_out=False)),
        parameters=frozenset({'n_surfaces', 'include_pial', 'include_white'}),
        out=_o('map', '.dscalar.nii', 'func', primary_role='scalar', desc='depth'),
    ),
    ActionSpec(
        'wm_depth_profile',
        PROCESSING,
        'surface_scalar',
        roles=(_r('scalar', 'scalar'), _r('surfaces', 'surfaces', fan_out=False)),
        parameters=frozenset({'origin', 'direction', 'distances_mm'}),
        out=_o('map', '.dscalar.nii', 'func', primary_role='scalar', desc='wmdepth'),
    ),
    ActionSpec(
        'resample_subcortical',
        PROCESSING,
        'subcortical_volume',
        roles=(_r('scalar', 'scalar'), _r('structures', 'structures', 'atlas', fan_out=False)),
        parameters=frozenset({'target_space', 'resolution'}),
        out=_o('map', '.nii.gz', 'func', primary_role='scalar'),
    ),
    ActionSpec(
        'assemble_cifti',
        PROCESSING,
        'dense_cifti',
        roles=(_r('surface', 'surface_scalar'), _r('volume', 'subcortical_volume')),
        out=_o('map', '.dscalar.nii', 'func', primary_role='surface', den='91k'),
    ),
    # -- streamlines + tract actions (Strategy B / trxrs) ------------------
    ActionSpec(
        'tractogram_to_pseg',
        PROCESSING,
        'atlas',
        roles=(
            _r('tractograms', 'streamlines', fan_out=False),
            # a single ACPC-space image defining the output voxel grid for tckmap
            _r(
                'reference',
                'scalar',
                'atlas',
                'timeseries',
                'subcortical_volume',
                list_ok=False,
            ),
        ),
        parameters=frozenset({'threshold'}),
        out=_o(
            'probseg',
            '.nii.gz',
            'dwi',
            primary_role='tractograms',
            dynamic_suffix=lambda params: (
                'dseg' if params.get('threshold') is not None else 'probseg'
            ),
            extra=(
                ExtraProduct(
                    'tsv',
                    'probseg',
                    '.tsv',
                    cifti_only=False,
                    match_primary_suffix=True,
                ),
            ),
        ),
    ),
    ActionSpec(
        'map_scalar_to_streamlines',
        PROCESSING,
        'streamlines',
        roles=(_r('scalar', 'scalar'), _r('streamlines', 'streamlines', fan_out=False)),
        parameters=frozenset({'name', 'per_vertex', 'per_streamline'}),
        out=_o('streamlines', '.trx', 'dwi', primary_role='streamlines'),
    ),
    ActionSpec(
        'parcellate_scalar_as_tract_profile',
        PROCESSING,
        'profile',
        roles=(_r('scalar', 'scalar'), _r('bundles', 'streamlines', fan_out=False)),
        parameters=frozenset({'n_nodes'}),
        out=_o('tractprofile', '.tsv', 'dwi', primary_role='scalar'),
    ),
    ActionSpec(
        'tract2region',
        PROCESSING,
        'relmat',
        roles=(_r('bundles', 'streamlines', fan_out=False), _r('atlas', 'atlas')),
        parameters=frozenset({'connectivity_type', 'connectivity_value'}),
        out=_o('relmat', '.tsv', 'dwi', primary_role='bundles'),
    ),
    ActionSpec(
        'region2region',
        PROCESSING,
        'relmat',
        roles=(_r('streamlines', 'streamlines', fan_out=False), _r('atlas', 'atlas')),
        parameters=frozenset({'search_radius', 'edges'}),
        out=_o('relmat', '.tsv', 'dwi', primary_role='streamlines'),
    ),
    # -- BAT atlas algebra (dataset level; same grammar) -------------------
    ActionSpec(
        'atlas_union',
        PROCESSING,
        'atlas',
        roles=(_r('a', 'atlas', fan_out=False), _r('b', 'atlas', fan_out=False)),
        parameters=frozenset({'output_atlas', 'precedence'}),
        out=_o('dseg', '.nii.gz', 'anat', primary_role='a'),
    ),
    ActionSpec(
        'atlas_intersect',
        PROCESSING,
        'atlas',
        roles=(_r('a', 'atlas', fan_out=False), _r('b', 'atlas', fan_out=False)),
        parameters=frozenset({'output_atlas'}),
        out=_o('dseg', '.nii.gz', 'anat', primary_role='a'),
    ),
    ActionSpec(
        'atlas_outer_product',
        PROCESSING,
        'atlas',
        roles=(_r('a', 'atlas', fan_out=False), _r('b', 'atlas', fan_out=False)),
        parameters=frozenset({'output_atlas'}),
        out=_o('dseg', '.nii.gz', 'anat', primary_role='a'),
    ),
)

ACTIONS: dict[str, ActionSpec] = {a.name: a for a in _ACTIONS}


# ---------------------------------------------------------------------------
# Best-effort format inference for ``select_data`` nodes.
#
# A selection node's true format is only known once files are matched at runtime,
# but its filters (extension / suffix) usually make the geometry obvious.  We use
# this to catch clear wiring mistakes statically (e.g. a surface geometry fed into
# a ``scalar`` role) while honouring explicit-over-heuristics: when inference is
# ambiguous we return ``'unknown'`` and the validator skips the strict check.
# ---------------------------------------------------------------------------
_SUFFIX_FORMAT = {
    'bold': 'timeseries',
    'streamlines': 'streamlines',
    'tractogram': 'streamlines',
    'pial': 'surfaces',
    'white': 'surfaces',
    'midthickness': 'surfaces',
    'inflated': 'surfaces',
    'sphere': 'surfaces',
    'dseg': 'atlas',
    'dlabel': 'atlas',
    'morph': 'surface_scalar',
    'dwimap': 'scalar',
    'boldmap': 'scalar',
    'cbf': 'scalar',
}
_EXT_FORMAT = {
    '.trx': 'streamlines',
    '.tck': 'streamlines',
    '.trk': 'streamlines',
    '.surf.gii': 'surfaces',
    '.dlabel.nii': 'atlas',
    '.func.gii': 'surface_scalar',
    '.shape.gii': 'surface_scalar',
    '.dscalar.nii': 'surface_scalar',
    '.dtseries.nii': 'timeseries',
}


def _as_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def infer_selection_format(filters: dict) -> str:
    """Best-effort format for a ``select_data`` node from its filters.

    Returns a :data:`FORMATS` member, or ``'unknown'`` when the filters do not
    unambiguously determine the geometry.  ``select_atlases`` never calls this —
    it always produces ``'atlas'``.
    """
    filters = filters or {}
    # Extension is the strongest signal.
    for ext in _as_set(filters.get('extension')):
        fmt = _EXT_FORMAT.get(ext if ext.startswith('.') else f'.{ext}')
        if fmt:
            return fmt
    # Otherwise fall back to suffix, but only if *every* listed suffix agrees.
    suffixes = _as_set(filters.get('suffix'))
    if suffixes:
        mapped = {_SUFFIX_FORMAT.get(s) for s in suffixes}
        if len(mapped) == 1 and None not in mapped:
            fmt = mapped.pop()
            # A ``map`` on a surface (fsLR + den) is a surface scalar, else volumetric.
            return fmt
    if suffixes == {'map'}:
        if filters.get('den') or filters.get('space') == 'fsLR':
            return 'surface_scalar'
        return 'scalar'
    return 'unknown'
