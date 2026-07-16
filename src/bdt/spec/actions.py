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
class OutputSpec:
    """How an action's materialized output is named (BIDS suffix/entities/datatype).

    ``primary_role`` names the input role whose entities seed the output filename
    (the "data" being transformed); ``entities`` are fixed entities the action
    always adds (e.g. ``{'stat': 'mean'}`` for parcellated means).  These values
    are refined per action as each action's real workflow lands; they are enough
    for the engine to compose BIDS-legal output paths today.
    """

    suffix: str
    extension: str
    datatype: str
    primary_role: str | None = None
    entities: dict = None  # fixed extra entities; defaults to {} via __post_init__

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


def _o(suffix, extension, datatype, primary_role=None, **entities) -> OutputSpec:
    return OutputSpec(
        suffix=suffix,
        extension=extension,
        datatype=datatype,
        primary_role=primary_role,
        entities=entities,
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
        out=_o('timeseries', '.tsv', 'func', primary_role='timeseries', stat='mean'),
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
        out=_o('map', '.tsv', 'func', primary_role='scalar', stat='mean'),
    ),
    ActionSpec(
        'functional_connectivity',
        PROCESSING,
        'relmat',
        # Enforces the spec rule: FC consumes a *parcellated* series, not a dense one.
        roles=(_r('timeseries', 'parcellated_timeseries'),),
        parameters=frozenset({'xdf_covariance'}),
        out=_o('relmat', '.tsv', 'func', primary_role='timeseries', stat='pearsoncorrelation'),
    ),
    # -- surface mapping + depth profiles (Strategy B / wb_command) --------
    ActionSpec(
        'map_scalar_to_surface',
        PROCESSING,
        'surface_scalar',
        roles=(_r('scalar', 'scalar'), _r('surfaces', 'surfaces', fan_out=False)),
        parameters=frozenset({'source_space'}),
        out=_o('map', '.dscalar.nii', 'func', primary_role='scalar'),
    ),
    ActionSpec(
        'resample_surface_scalar',
        PROCESSING,
        'surface_scalar',
        roles=(
            _r('surface_scalar', 'surface_scalar'),
            _r('surfaces', 'surfaces', fan_out=False),
        ),
        parameters=frozenset({'target_space', 'target_density'}),
        out=_o('map', '.dscalar.nii', 'func', primary_role='surface_scalar'),
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
        'tractogram_to_dseg',
        PROCESSING,
        'atlas',
        roles=(_r('tractograms', 'streamlines', fan_out=False),),
        parameters=frozenset({'threshold'}),
        out=_o('dseg', '.nii.gz', 'dwi', primary_role='tractograms'),
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
        'parcellate_scalar_as_roi',
        PROCESSING,
        'roi_means',
        roles=(_r('scalar', 'scalar'), _r('atlas', 'atlas')),
        out=_o('bundlemap', '.tsv', 'dwi', primary_role='scalar', stat='mean'),
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
