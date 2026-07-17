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
"""nipype sub-workflow factories, one per action.

Each factory turns a processing :class:`~bdt.spec.model.Node` into a nipype
``Workflow`` with a standard boundary: an ``inputnode`` whose fields are the
action's declared roles, and an ``outputnode`` with a single ``out`` field
carrying the node's primary product.  The compiler (:mod:`bdt.engine.workflow`)
wires ``inputs[role]`` from an upstream node's ``outputnode.out`` into this
node's ``inputnode.<role>``.

This is the qsirecon ``init_*_wf`` pattern, keyed by *role* rather than by
field-name intersection.  Factories reuse the vendored nipype interfaces
(``bdt.interfaces.workbench`` etc.), so nipype gives us File(exists) validation,
content-hash caching, and resumability for free.
"""

from __future__ import annotations

from dataclasses import dataclass

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from bdt.interfaces.cifti import CiftiMask, CiftiVertexMask
from bdt.interfaces.workbench import (
    CiftiCorrelation,
    CiftiCreateDenseFromTemplate,
    CiftiMath,
    CiftiParcellateWorkbench,
)

WORKFLOW_FACTORIES: dict[str, callable] = {}


@dataclass
class FactoryContext:
    """Auxiliary-input resolver handed to factories that need files beyond their
    wired roles — surface registration spheres, templateflow standard meshes.

    Most factories ignore it (their inputs all arrive by role); the surface
    factories use it to resolve, for the current subject, files the story spec does
    not wire: the ``desc-msmsulc`` fsnative→fsLR sphere and the subject's fsLR
    midthickness (from the ``surfaces`` role's own dataset), plus subject-
    independent templateflow meshes.  It is injectable so assembly tests can build
    the graph with a stub provider (nipype checks ``File(exists)`` only at run
    time, so build-time paths need not exist).
    """

    provider: object = None  # bdt.engine.selection.DataProvider
    subject: str | None = None
    spec: object = None  # bdt.spec.model.Spec — resolves a role's upstream dataset
    templateflow_get: object = None  # callable; defaults to templateflow.api.get
    datasets: list | None = None  # all --datasets keys (for cross-dataset references)
    resolved: dict | None = None  # {selection_name: Match} for this scope (entity lookup)

    def dataset_of_role(self, node, role: str) -> str | None:
        """The ``--datasets`` key feeding ``node``'s ``role`` (via its selection)."""
        if self.spec is None:
            return None
        by_name = self.spec.by_name()
        for up in node.inputs.get(role, []):
            sel = by_name.get(up)
            if sel is not None and sel.dataset is not None:
                return sel.dataset
        return None

    def role_space(self, node, role: str, default: str | None = None) -> str | None:
        """The ``space`` entity of the file feeding ``node``'s ``role``.

        Read from the resolved selection match's entities (surfaces default to their
        mesh's anatomical space when the ``space`` entity is absent — ``T1w``).
        """
        return self._role_entity(node, role, 'space', default)

    def role_session(self, node, role: str) -> str | None:
        """The ``ses`` entity of the file feeding ``node``'s ``role`` (``None`` if
        the data has no session, e.g. a subject-level anatomical)."""
        return self._role_entity(node, role, 'ses', None)

    def _role_entity(self, node, role, key, default):
        if self.resolved is None:
            return default
        for up in node.inputs.get(role, []):
            match = self.resolved.get(up)
            if match is not None:
                return match.entities.get(key, default)
        return default

    def _select_scoped(self, dataset, filters, exclude, session):
        """Select from ``dataset``, honouring the subject/session *anat level*.

        A BIDS anatomical may sit at **session level** (``sub-X/ses-Y/anat``) or
        **subject level** (``sub-X/anat``, shared across sessions).  When a session
        is known we take the session-matched file if one exists, else fall back to a
        *session-less* (subject-level) file — never a *different* session's file.
        """
        if session is not None:
            hits = self.provider.select(
                dataset, {**filters, 'ses': session}, exclude, subject=self.subject
            )
            if hits:
                return hits
            # subject-level fallback: only files that carry no session entity
            return self.provider.select(
                dataset, {**filters, 'ses': None}, exclude, subject=self.subject
            )
        return self.provider.select(dataset, filters, exclude, subject=self.subject)

    def aux_file(
        self, dataset: str, filters: dict, exclude: list | None = None, session: str | None = None
    ) -> str:
        """Resolve exactly one auxiliary file from ``dataset`` for the subject/session."""
        matches = self._select_scoped(dataset, filters, exclude, session)
        if len(matches) != 1:
            raise ValueError(
                f'Auxiliary selection on dataset {dataset!r} with {filters} (session '
                f'{session!r}) matched {len(matches)} files (expected exactly 1).'
            )
        return matches[0].path

    def find_reference(
        self, filters: dict, exclude: list | None = None, session: str | None = None
    ) -> str:
        """Find exactly one file matching ``filters`` across *all* datasets.

        For registration references that are not wired into the node graph and may
        live in a different dataset than either input — e.g. the QSIPrep
        ``space-ACPC`` anatomical used to compute the T1w↔ACPC bridge, which is in
        the ``qsiprep`` dataset while the scalar comes from ``qsirecon``.  Honours
        the subject/session anat level like :meth:`aux_file`.
        """
        hits: list[tuple[str, str]] = []
        for ds in self.datasets or []:
            for m in self._select_scoped(ds, filters, exclude, session):
                hits.append((ds, m.path))
        if len(hits) != 1:
            raise ValueError(
                f'Reference selection {filters} (session {session!r}) matched {len(hits)} files '
                f'across datasets {list(self.datasets or [])} (expected exactly 1): '
                f'{[h[1] for h in hits]}'
            )
        return hits[0][1]

    def tf_get(self, **kwargs) -> str:
        """Fetch a single templateflow file path (never a list)."""
        get = self.templateflow_get
        if get is None:
            from templateflow import api as tf

            get = tf.get
        res = get(**kwargs)
        if isinstance(res, (list, tuple)):
            if len(res) != 1:
                raise ValueError(
                    f'templateflow.get({kwargs}) returned {len(res)} files (expected 1).'
                )
            res = res[0]
        return str(res)


def workflow_factory(action: str):
    """Register a nipype sub-workflow factory for ``action``."""

    def deco(fn):
        WORKFLOW_FACTORIES[action] = fn
        return fn

    return deco


def _io_nodes(input_fields: list[str]) -> tuple[pe.Node, pe.Node]:
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')
    return inputnode, outputnode


def _init_parcellate_cifti_wf(node, name, in_role: str, out_file: str) -> pe.Workflow:
    """Shared coverage-aware CIFTI parcellation (XCP-D ``init_parcellate_cifti_wf``).

    A vertex-wise coverage mask (1 where a vertex has data) is used both as
    ``-cifti-weights`` for the parcel mean — so uncovered (zero) vertices don't
    dilute it — and, parcellated itself, as the per-parcel coverage fraction.
    Parcels below ``min_coverage`` are set to NaN.  ``inputnode`` takes the data on
    ``in_role`` (+ ``atlas``); ``outputnode`` exposes the masked result (``out``,
    written to ``out_file``: ptseries for a series, pscalar for a scalar) and the
    parcel coverage map (``coverage``).
    """
    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))

    inputnode = pe.Node(niu.IdentityInterface(fields=[in_role, 'atlas']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out', 'coverage']), name='outputnode')

    # 0. restrict the atlas to the data's brainordinates.  A cortex-only surface
    # scalar (59412 grayordinates) can't be parcellated by a whole-brain dlabel
    # (91282) — cifti-parcellate errors on the missing subcortical voxels — so drop
    # the atlas structures the data lacks.  Byte-identical no-op when they match, so
    # the dense-CIFTI (story 3.1 / ALFF) bit-for-bit results are unchanged.
    restrict_atlas = pe.Node(
        CiftiCreateDenseFromTemplate(out_file='atlas_restricted.dlabel.nii'),
        name='restrict_atlas',
    )

    # 1. vertex-wise coverage (1 = has data, 0 = all-zero/NaN across the map/series)
    vertex_mask = pe.Node(CiftiVertexMask(), name='vertex_mask')

    # 2. per-parcel coverage fraction = mean of the binary vertex mask over the parcel
    parcellate_coverage = pe.Node(
        CiftiParcellateWorkbench(
            direction='COLUMN', only_numeric=True, out_file='coverage.pscalar.nii'
        ),
        name='parcellate_coverage',
    )

    # 3. coverage-weighted parcel mean of the data (uncovered vertices get weight 0)
    parcellate_data = pe.Node(
        CiftiParcellateWorkbench(direction='COLUMN', only_numeric=True, out_file=out_file),
        name='parcellate_data',
    )

    # 4. threshold the coverage -> a 0/1 parcel mask
    threshold = pe.Node(
        CiftiMath(expression=f'data > {min_coverage}', out_file='coverage_mask.pscalar.nii'),
        name='threshold',
    )

    # 5. NaN-out parcels below the coverage threshold
    mask = pe.Node(CiftiMask(), name='mask')

    wf.connect([
        (inputnode, restrict_atlas, [(in_role, 'template_cifti'), ('atlas', 'label')]),
        (inputnode, vertex_mask, [(in_role, 'in_file')]),
        (restrict_atlas, parcellate_coverage, [('out_file', 'atlas_label')]),
        (vertex_mask, parcellate_coverage, [('mask_file', 'in_file')]),
        (inputnode, parcellate_data, [(in_role, 'in_file')]),
        (restrict_atlas, parcellate_data, [('out_file', 'atlas_label')]),
        (vertex_mask, parcellate_data, [('mask_file', 'cifti_weights')]),
        (parcellate_coverage, threshold, [('out_file', 'data')]),
        (parcellate_data, mask, [('out_file', 'in_file')]),
        (threshold, mask, [('out_file', 'mask')]),
        (mask, outputnode, [('out_file', 'out')]),
        (parcellate_coverage, outputnode, [('out_file', 'coverage')]),
    ])  # fmt:skip
    return wf


@workflow_factory('parcellate_timeseries')
def init_parcellate_timeseries_wf(node, name=None, context=None) -> pe.Workflow:
    """Parcellate a dense CIFTI series with a dlabel atlas, coverage-aware -> ptseries."""
    return _init_parcellate_cifti_wf(node, name, 'timeseries', 'parcellated.ptseries.nii')


@workflow_factory('parcellate_scalar')
def init_parcellate_scalar_wf(node, name=None, context=None) -> pe.Workflow:
    """Parcellate a dense CIFTI scalar (dscalar) with a dlabel atlas -> pscalar.

    Same coverage-aware machinery as :func:`init_parcellate_timeseries_wf`.  CIFTI
    inputs only for now (the volumetric/NIfTI Strategy-A path is a follow-up).
    """
    return _init_parcellate_cifti_wf(node, name, 'scalar', 'parcellated.pscalar.nii')


@workflow_factory('functional_connectivity')
def init_functional_connectivity_wf(node, name=None, context=None) -> pe.Workflow:
    """Correlate a parcellated series (ptseries) -> pconn relmat."""
    wf = pe.Workflow(name=name or node.name)
    inputnode, outputnode = _io_nodes(['timeseries'])
    correlate = pe.Node(CiftiCorrelation(out_file='correlations.pconn.nii'), name='correlate')
    wf.connect([
        (inputnode, correlate, [('timeseries', 'in_file')]),
        (correlate, outputnode, [('out_file', 'out')]),
    ])  # fmt:skip
    return wf


def _pick_surface_file(in_files, hemi, suffix=''):
    """Return the single per-hemisphere file from a grouped L/R role's file list.

    ``in_files`` is a grouped role's list (e.g. the L/R native ``surface_scalar``,
    or the L/R white/pial/midthickness ``surfaces`` set); pick the one for ``hemi``
    (``'L'``/``'R'``), optionally also matching ``suffix`` (e.g. ``midthickness``).
    """
    import os

    cands = [f for f in in_files if f'hemi-{hemi}' in os.path.basename(f)]
    if suffix:
        cands = [f for f in cands if suffix in os.path.basename(f)]
    if len(cands) != 1:
        raise ValueError(
            f'Expected exactly one hemi-{hemi} file (suffix={suffix!r}) in {in_files}, got {cands}'
        )
    return cands[0]


@workflow_factory('resample_surface_scalar')
def init_resample_surface_scalar_wf(node, name=None, context=None) -> pe.Workflow:
    """Resample a per-hemi native surface scalar (thickness/sulc) to fsLR grayordinates.

    Reproduces fmriprep/sMRIPrep morphometric grayordinate resampling bit-for-bit.
    Per hemisphere: a source medial-wall ROI (non-zero data — FreeSurfer
    morphometrics zero the medial wall) feeds ``-current-roi`` on
    ``wb -metric-resample ADAP_BARY_AREA``; the current sphere is the subject's
    ``desc-msmsulc`` fsnative→fsLR sphere, the new sphere the templateflow fsLR
    32k sphere, and the area surfaces go native-midthickness → the subject's own
    fsLR-32k midthickness.  The L/R metrics are then stapled with
    ``wb -cifti-create-dense-scalar`` + the templateflow ``nomedialwall`` ROIs into
    a den-91k (medial-wall-removed, 59412-grayordinate) cortex dscalar.

    ``surface_scalar`` and ``surfaces`` are *grouped* roles (L+R passed as lists);
    the ``desc-msmsulc`` spheres and fsLR midthickness are auxiliary inputs the
    story spec does not wire, resolved for the subject via ``context``.
    """
    from niworkflows.interfaces.workbench import MetricResample

    from bdt.interfaces.workbench import CiftiCreateDenseScalar, MetricMath

    context = context or FactoryContext()
    if context.provider is None:
        raise ValueError(
            f'resample_surface_scalar node {node.name!r} needs a FactoryContext with a '
            'provider to resolve the msmsulc spheres / fsLR midthickness.'
        )
    density = str(node.parameters.get('target_density', '32k'))
    surfaces_dataset = context.dataset_of_role(node, 'surfaces')
    surf_ses = context.role_session(node, 'surfaces')  # subject- vs session-level anat

    wf = pe.Workflow(name=name or node.name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['surface_scalar', 'surfaces']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')
    create = pe.Node(CiftiCreateDenseScalar(out_file='resampled.dscalar.nii'), name='create_dense')

    for hemi in ('L', 'R'):
        pick_scalar = pe.Node(
            niu.Function(
                function=_pick_surface_file,
                input_names=['in_files', 'hemi', 'suffix'],
                output_names=['out'],
            ),
            name=f'pick_scalar_{hemi}',
        )
        pick_scalar.inputs.hemi = hemi
        pick_midthick = pe.Node(
            niu.Function(
                function=_pick_surface_file,
                input_names=['in_files', 'hemi', 'suffix'],
                output_names=['out'],
            ),
            name=f'pick_midthick_{hemi}',
        )
        pick_midthick.inputs.hemi = hemi
        pick_midthick.inputs.suffix = 'midthickness'

        # source medial-wall ROI (FreeSurfer morphometrics are zero on the mwall)
        srcroi = pe.Node(
            MetricMath(expression='x != 0', out_file=f'srcroi_{hemi}.shape.gii'),
            name=f'srcroi_{hemi}',
        )

        # auxiliary inputs the spec does not wire (per subject/session / templateflow)
        msmsulc_sphere = context.aux_file(
            surfaces_dataset,
            {'suffix': 'sphere', 'space': 'fsLR', 'desc': 'msmsulc', 'hemi': hemi},
            session=surf_ses,
        )
        sub_fslr_midthick = context.aux_file(
            surfaces_dataset,
            {'suffix': 'midthickness', 'space': 'fsLR', 'den': density, 'hemi': hemi},
            session=surf_ses,
        )
        tpl_sphere = context.tf_get(
            template='fsLR',
            density=density,
            hemi=hemi,
            space=None,
            suffix='sphere',
            extension='.surf.gii',
        )
        tpl_roi = context.tf_get(
            template='fsLR',
            density=density,
            hemi=hemi,
            desc='nomedialwall',
            suffix='dparc',
            extension='.label.gii',
        )

        resample = pe.Node(
            MetricResample(
                method='ADAP_BARY_AREA',
                area_surfs=True,
                current_sphere=msmsulc_sphere,
                new_sphere=tpl_sphere,
                new_area=sub_fslr_midthick,
                out_file=f'resampled_{hemi}.shape.gii',
            ),
            name=f'resample_{hemi}',
        )

        wf.connect([
            (inputnode, pick_scalar, [('surface_scalar', 'in_files')]),
            (inputnode, pick_midthick, [('surfaces', 'in_files')]),
            (pick_scalar, srcroi, [('out', 'var_x')]),
            (pick_scalar, resample, [('out', 'in_file')]),
            (pick_midthick, resample, [('out', 'current_area')]),
            (srcroi, resample, [('out_file', 'roi_metric')]),
        ])  # fmt:skip
        if hemi == 'L':
            create.inputs.roi_left = tpl_roi
            wf.connect([(resample, create, [('out_file', 'left_metric')])])
        else:
            create.inputs.roi_right = tpl_roi
            wf.connect([(resample, create, [('out_file', 'right_metric')])])

    wf.connect([(create, outputnode, [('out_file', 'out')])])
    return wf


def _register_acpc_to_t1w(fixed_image, fixed_mask, moving_image, moving_mask):
    """Rigid, brain-masked ANTsPy registration -> the ``from-ACPC_to-T1w`` transform.

    ``fixed`` is the surfaces' T1w anatomical, ``moving`` the ``space-ACPC`` anatomical,
    each restricted to its brain mask.  ANTsPy's forward transform (moving→fixed for
    images) is, applied to *points*, the T1w→ACPC warp ``giftirs`` needs; it is an ITK
    GenericAffine ``.mat`` — the same format the ``antsRegistration`` CLI wrote — copied
    into the node's work dir so nipype tracks it.
    """
    import os
    import shutil

    import ants

    reg = ants.registration(
        fixed=ants.image_read(fixed_image),
        moving=ants.image_read(moving_image),
        type_of_transform='Rigid',
        mask=ants.image_read(fixed_mask),
        moving_mask=ants.image_read(moving_mask),
    )
    out = os.path.abspath('from-ACPC_to-T1w_xfm.mat')
    shutil.copyfile(reg['fwdtransforms'][0], out)
    return out


@workflow_factory('map_scalar_to_surface')
def init_map_scalar_to_surface_wf(node, name=None, context=None) -> pe.Workflow:
    """Map a volumetric scalar onto the cortical surface (ribbon-constrained).

    Per hemisphere, ``wb -volume-to-surface-mapping -ribbon-constrained`` samples the
    volume between the white and pial surfaces onto the midthickness mesh, then
    ``-metric-dilate`` fills small holes.  ``outputnode.out`` is the per-hemi list
    ``[L, R]`` of native-mesh metrics (so a downstream ``resample_surface_scalar``
    consumes it exactly like a grouped L/R selection).

    **Cross-space (Strategy B).** When the scalar's space differs from the surfaces'
    (the story case: QSIRecon NODDI in ``ACPC`` vs sMRIPrep surfaces in ``T1w``), the
    surface *vertices* are warped into the scalar's space before mapping.  Per the
    locked design decision, the T1w↔ACPC bridge is the one transform BDT *computes*:
    a rigid, brain-masked registration (**ANTsPy**, fixed = the surfaces' T1w
    anatomical, moving = the ``space-ACPC`` anatomical) whose ``from-ACPC_to-T1w``
    GenericAffine is applied to each surface with ``giftirs transform`` (lossless,
    vertex-order preserving).  QSIPrep's own stored ACPC↔anat transforms are
    deliberately *not* used.  Any other cross-space pairing raises
    ``NotImplementedError``.
    """
    from niworkflows.interfaces.workbench import MetricDilate, VolumeToSurfaceMapping

    from bdt.interfaces.giftirs import GiftiTransform

    context = context or FactoryContext()
    surface_space = context.role_space(node, 'surfaces', default='T1w')
    scalar_space = context.role_space(node, 'scalar', default=None)
    cross_space = scalar_space is not None and scalar_space != surface_space

    wf = pe.Workflow(name=name or node.name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['scalar', 'surfaces']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')
    merge = pe.Node(niu.Merge(2), name='merge_hemis')  # [L, R] -> outputnode.out

    xfm_source = None  # a node field emitting the from-ACPC_to-T1w transform, if cross
    if cross_space:
        if {surface_space, scalar_space} != {'T1w', 'ACPC'}:
            raise NotImplementedError(
                f'map_scalar_to_surface node {node.name!r}: only the T1w<->ACPC bridge is '
                f'computed (rigid registration); got surfaces in {surface_space!r}, scalar in '
                f'{scalar_space!r}. Other cross-space mappings are not implemented.'
            )
        if context.provider is None:
            raise ValueError(
                f'map_scalar_to_surface node {node.name!r} is cross-space and needs a '
                'FactoryContext provider to resolve the registration references.'
            )
        surfaces_dataset = context.dataset_of_role(node, 'surfaces')
        # match each reference's anat level (subject- vs session-level) to its data's
        # session: fixed to the surfaces' session, moving to the scalar's session.
        surf_ses = context.role_session(node, 'surfaces')
        scalar_ses = context.role_session(node, 'scalar')
        # fixed = the surfaces' own T1w anatomical (+ brain mask), native (no space)
        fixed_img = context.aux_file(
            surfaces_dataset, {'suffix': 'T1w', 'desc': 'preproc', 'space': None}, session=surf_ses
        )
        fixed_mask = context.aux_file(
            surfaces_dataset, {'suffix': 'mask', 'desc': 'brain', 'space': None}, session=surf_ses
        )
        # moving = the space-ACPC anatomical (+ brain mask), found across datasets.
        # QSIPrep's --anat-modality may make this a T2w rather than a T1w; the rigid
        # MI registration is contrast-agnostic, so accept either (fixed stays the
        # surfaces' T1w space and MI handles any cross-contrast fit).
        moving_img = context.find_reference(
            {'suffix': ['T1w', 'T2w'], 'desc': 'preproc', 'space': 'ACPC', 'datatype': 'anat'},
            session=scalar_ses,
        )
        moving_mask = context.find_reference(
            {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'},
            session=scalar_ses,
        )

        # one ANTsPy node: rigid, brain-masked reg -> the from-ACPC_to-T1w .mat.
        # (ANTsPy takes the masks natively, so no separate brain-extraction nodes.)
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
        xfm_source = (register, 'out')

    for i, hemi in enumerate(('L', 'R'), start=1):
        picks = {}
        for which in ('white', 'pial', 'midthickness'):
            pick = pe.Node(
                niu.Function(
                    function=_pick_surface_file,
                    input_names=['in_files', 'hemi', 'suffix'],
                    output_names=['out'],
                ),
                name=f'pick_{which}_{hemi}',
            )
            pick.inputs.hemi = hemi
            pick.inputs.suffix = which
            wf.connect([(inputnode, pick, [('surfaces', 'in_files')])])
            picks[which] = pick

        # surfaces used for mapping: warped into the scalar's space when cross-space
        surf_field = {}
        if cross_space:
            for which in ('white', 'pial', 'midthickness'):
                warp = pe.Node(
                    GiftiTransform(out_file=f'warped_hemi-{hemi}_{which}.surf.gii'),
                    name=f'warp_{which}_{hemi}',
                )
                wf.connect([
                    (picks[which], warp, [('out', 'in_file')]),
                    (xfm_source[0], warp, [(xfm_source[1], 'transform')]),
                ])  # fmt:skip
                surf_field[which] = (warp, 'out_file')
        else:
            for which in ('white', 'pial', 'midthickness'):
                surf_field[which] = (picks[which], 'out')

        mid_n, mid_f = surf_field['midthickness']
        white_n, white_f = surf_field['white']
        pial_n, pial_f = surf_field['pial']

        vol2surf = pe.Node(
            VolumeToSurfaceMapping(
                method='ribbon-constrained', out_file=f'mapped_hemi-{hemi}.func.gii'
            ),
            name=f'vol2surf_{hemi}',
        )
        dilate = pe.Node(
            MetricDilate(distance=10.0, nearest=True, out_file=f'mapped_dil_hemi-{hemi}.func.gii'),
            name=f'dilate_{hemi}',
        )
        wf.connect([
            (inputnode, vol2surf, [('scalar', 'volume_file')]),
            (mid_n, vol2surf, [(mid_f, 'surface_file')]),
            (white_n, vol2surf, [(white_f, 'inner_surface')]),
            (pial_n, vol2surf, [(pial_f, 'outer_surface')]),
            (vol2surf, dilate, [('out_file', 'in_file')]),
            (mid_n, dilate, [(mid_f, 'surf_file')]),
            (dilate, merge, [('out_file', f'in{i}')]),
        ])  # fmt:skip

    wf.connect([(merge, outputnode, [('out', 'out')])])
    return wf


@workflow_factory('tractogram_to_pseg')
def init_tractogram_to_pseg_wf(node, name=None, context=None) -> pe.Workflow:
    """Convert a list of tractogram files (tck.gz format) to a NIfTI pseg.

    First, use tckmap from mrtrix3 to convert each bundle-wise tck.gz file to a TDI NIfTI file.
    This requires the tck.gz file in ACPC space and a reference NIfTI in the same space.
    Then concatenate the TDI NIfTI files over time to create a 4D probabilistic segmentation NIfTI file.
    Optionally apply a threshold to binarize the pseg file.

    The bundle names will also have to be inferred from the input files and passed into a TSV
    indicating which volume in the NIfTI corresponds to which bundle.
    This should be in BIDS dseg/pseg format (i.e., two required columns: index and name).
    If threshold is not None/Undefined and write_outputs is True, the output file should have
    probseg suffix. If write_outputs is True and threshold is a number, the output file should
    have dseg suffix.
    """
    context = context or FactoryContext()
    threshold = context.role_space(node, 'threshold', default=None)

    wf = pe.Workflow(name=name or node.name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['tractograms', 'reference_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['seg', 'tsv']), name='outputnode')

    # TODO: Implement and validate workflow. Code below is non-working pseudocode.
    # Step 1: Use a MapNode to convert tck.gz files to TDI NIfTI files.
    tck_to_nii = pe.MapNode(
        Tckmap(),
        iterfield=['in_file'],
        name='tck_to_nii',
    )
    wf.connect([
        (inputnode, tck_to_nii, [
            ('tractograms', 'in_file'),
            ('reference_file', 'reference'),
        ]),
    ])  # fmt:skip

    # Step 2: Concatenate NIfTIs in fourth dimension
    concatenate_niis = pe.Node(
        ConcatenateNiftis(),
        name='concatenate_niis',
    )
    wf.connect([(tck_to_nii, concatenate_niis, [('out_file', 'in_files')])])

    # Step 3: Binarize 4D NIfTI if threshold is provided
    seg_buffer = pe.Node(niu.IdentityInterface(fields=['seg_file'], name='seg_buffer'))
    if threshold:
        threshold_nii = pe.Node(
            ThresholdNifti(threshold=threshold, binarize=True),
            name='threshold_nii',
        )
        wf.connect([
            (concatenate_niis, threshold_nii, [('out_file', 'in_file')]),
            (threshold_nii, seg_buffer, [('out_file', 'seg_file')]),
        ])  # fmt:skip
    else:
        wf.connect([
            (concatenate_niis, seg_buffer, [('out_file', 'seg_file')]),
        ])  # fmt:skip

    # Keeping seg_buffer for now in case we need to add a DataSink before the outputnode
    wf.connect([(seg_buffer, outputnode, [('seg_file', 'seg')])])

    # Step 4: Extract bundle names and compile in TSV
    bundles_to_tsv = pe.Node(
        EntitiesToSegTSV(entity='bundle'),
        name='bundles_to_tsv',
    )
    wf.connect([
        (inputnode, bundles_to_tsv, [('tractograms', 'in_files')]),
        (bundles_to_tsv, outputnode, [('out_file', 'tsv')]),
    ])  # fmt:skip

    return wf
