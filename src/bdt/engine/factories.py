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
        """The space of the file feeding ``node``'s ``role``.

        Read from the resolved selection match's entities (surfaces default to their
        mesh's anatomical space when the ``space`` entity is absent — ``T1w``).

        A file that *lives in* a template space names it with ``template`` (``tpl-``)
        rather than ``space-`` — that is how BIDS-Atlas names atlases, e.g.
        ``tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_dseg.nii.gz``.  The two entities
        never co-occur on one file (``space-`` marks a file *resampled into* a space,
        ``tpl-`` a file native to it), so falling back to ``template`` is unambiguous
        and is what makes cross-space detection see an atlas's real space.
        """
        space = self._role_entity(node, role, 'space', None)
        if space is None:
            space = self._role_entity(node, role, 'template', None)
        return default if space is None else space

    def role_session(self, node, role: str) -> str | None:
        """The session of the file feeding ``node``'s ``role`` (``None`` if the data
        has no session, e.g. a subject-level anatomical).

        pybids names this entity ``session``; ``ses`` is only the *filename* key.
        Reading ``ses`` alone silently returned ``None`` for every real query, which
        disabled session scoping everywhere it is used.
        """
        return self._role_entity(node, role, 'session', None)

    def role_extension(self, node, role: str, default: str | None = None) -> str | None:
        """The ``extension`` entity of the file feeding ``node``'s ``role``."""
        return self._role_entity(node, role, 'extension', default)

    def role_suffix(self, node, role: str, default: str | None = None) -> str | None:
        """The ``suffix`` entity of the file feeding ``node``'s ``role``."""
        return self._role_entity(node, role, 'suffix', default)

    def role_datatype(self, node, role: str, default: str | None = None) -> str | None:
        """The ``datatype`` entity of the file feeding ``node``'s ``role``."""
        return self._role_entity(node, role, 'datatype', default)

    def role_atlas_ndim(self, node, role: str = 'atlas') -> int | None:
        """Dimensionality of the atlas feeding ``role``, resolved at *build* time.

        A **selected** atlas exists on disk while the graph is built, so its header
        is read (a header read, not a data read).  A **processing** atlas does not
        exist yet, but is 4D by construction: ``tractogram_to_pseg`` is the only
        atlas-producing action and it stacks bundles via ``ConcatenateNiftis``.
        Warping preserves dimensionality, so a reading taken from the original
        selection stays valid for the warped atlas that reaches the masker.
        """
        import nibabel as nb

        for up in node.inputs.get(role, []):
            match = (self.resolved or {}).get(up)
            if match is None:
                return 4
            try:
                return int(nb.load(match.path).ndim)
            except Exception as exc:  # unreadable header -> name it, never guess
                raise ValueError(
                    f'Could not read the atlas header for role {role!r} of node '
                    f'{getattr(node, "name", node)!r}: {match.path}'
                ) from exc
        return None

    def role_atlas_labels(self, node, role: str = 'atlas') -> str | None:
        """Path to the BIDS ``dseg.tsv`` describing the atlas feeding ``role``.

        For a **selected** atlas this is the sibling sidecar (AtlasPack ships
        ``tpl-..._dseg.tsv`` beside ``tpl-..._dseg.nii.gz``).  It must be resolved
        from the *original selection*: the warped atlas in the node cwd has no
        sidecar beside it.  For a **processing** atlas this returns ``None`` --
        those labels arrive over a wired ``tsv`` edge instead.
        """
        import os

        for up in node.inputs.get(role, []):
            match = (self.resolved or {}).get(up)
            if match is None:
                return None
            path = match.path
            for ext in ('.nii.gz', '.nii'):
                if path.endswith(ext):
                    sidecar = path[: -len(ext)] + '.tsv'
                    if os.path.exists(sidecar):
                        return sidecar
                    raise ValueError(
                        f'Atlas {path} has no labels sidecar at {sidecar}. '
                        'A volumetric atlas needs a BIDS dseg.tsv (index/name) to name '
                        'its parcels and to detect parcels lost when the atlas is warped.'
                    )
            raise ValueError(f'Unrecognized atlas extension (expected .nii/.nii.gz): {path}')
        return None

    def discover_transforms(self, session: str | None = None) -> list[str]:
        """Subject transform files across all datasets, for Spec 1's ``local_transforms``.

        Enumerates BIDS ``_xfm`` files (via the provider) and **excludes** any whose
        endpoints include ``ACPC``: the ACPC↔T1w hop is always the transform BDT
        computes itself (the rigid bridge), never QSIPrep's stored one — matching the
        locked decision in :func:`init_map_scalar_to_surface_wf`.  Returns ``[]`` when
        no provider is configured (the build-time stub path).
        """
        from bdt.transforms.graph import parse_xfm_filename

        if self.provider is None:
            return []
        paths: list[str] = []
        for dataset in self.datasets or []:
            for match in self._select_scoped(dataset, {'suffix': 'xfm'}, None, session):
                xfm = parse_xfm_filename(match.path)
                if xfm is None or 'ACPC' in (xfm.frm, xfm.to):
                    continue
                paths.append(match.path)
        return sorted(paths)

    def _entities_by_node(self):
        """Cached output entities for *every* node, selection and processing alike.

        ``resolved`` holds only selection matches, so a role fed by a **processing**
        node (e.g. an ``atlas`` wired from ``tractogram_to_pseg``) is invisible to a
        plain ``resolved`` lookup — its ``space``/``suffix`` would read as absent and
        a cross-space warp would be silently skipped.  When a ``spec`` is available we
        propagate entities through the whole graph (``node_output_entities``), which
        composes each processing node's product from its ``primary_role`` upstream, so
        the atlas's inherited space/suffix resolve correctly.  Returns ``None`` when no
        ``spec`` is set (the build-stub path) so callers fall back to ``resolved``.
        """
        if self.spec is None or self.resolved is None:
            return None
        cached = getattr(self, '_entmap_cache', None)
        if cached is None:
            from bdt.outputs.plan import node_output_entities

            cached = node_output_entities(self.spec, self.resolved)
            self._entmap_cache = cached
        return cached

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
        no ``extension`` entity.  Otherwise uses the resolved match's ``is_cifti(path)``.
        When neither is available (a bare, context-less build-stub call), defaults to
        ``True`` (CIFTI): the historical build-safe default, so a no-context assembly
        of a CIFTI pipeline still takes the grayordinate path rather than misrouting to
        the volumetric one.
        """
        cmap = self._cifti_by_node()
        if cmap is not None:
            for up in node.inputs.get(role, []):
                if up in cmap:
                    return cmap[up]
            return True
        from bdt.utils.cifti import is_cifti

        if self.resolved is not None:
            for up in node.inputs.get(role, []):
                match = self.resolved.get(up)
                if match is not None:
                    return is_cifti(match.path)
        return True

    def _role_entity(self, node, role, key, default):
        entmap = self._entities_by_node()
        if entmap is not None:
            for up in node.inputs.get(role, []):
                entities = entmap.get(up)
                if entities:
                    return entities.get(key, default)
            return default
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
            # ``session`` is the entity name pybids knows; ``ses`` is only the
            # filename key and raises "'ses' is not a recognized entity" on a real
            # BIDSLayout query.
            hits = self.provider.select(
                dataset, {**filters, 'session': session}, exclude, subject=self.subject
            )
            if hits:
                return hits
            # subject-level fallback: only files that carry no session entity
            return self.provider.select(
                dataset, {**filters, 'session': None}, exclude, subject=self.subject
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
        hits = self.find_references(filters, exclude, session)
        if len(hits) != 1:
            raise ValueError(
                f'Reference selection {filters} (session {session!r}) matched {len(hits)} files '
                f'across datasets {list(self.datasets or [])} (expected exactly 1): '
                f'{hits}'
            )
        return hits[0]

    def find_references(
        self, filters: dict, exclude: list | None = None, session: str | None = None
    ) -> list[str]:
        """Every file matching ``filters`` across all datasets, without demanding one.

        The non-raising form of :meth:`find_reference`, for callers that need to try
        several candidate queries and pick the one that resolves unambiguously.
        Returns ``[]`` when no provider is configured (the build-stub path), matching
        :meth:`discover_transforms`, so callers surface their own domain error rather
        than an ``AttributeError`` from deep inside selection.
        """
        if self.provider is None:
            return []
        hits: list[str] = []
        for ds in self.datasets or []:
            for m in self._select_scoped(ds, filters, exclude, session):
                hits.append(m.path)
        return hits

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

    .. warning::

       **CIFTI and volumetric coverage do not mean the same thing**, even though
       both are controlled by ``min_coverage`` and both are written as a
       ``stat-coverage`` derivative.

       * Here (CIFTI), coverage is *data-derived*: :class:`CiftiVertexMask` marks a
         vertex uncovered when it is zero or NaN across the whole map/series, so
         coverage is "the fraction of this parcel's vertices that carry data".
       * The volumetric paths (:func:`_init_parcellate_volumetric_wf`) follow XCP-D
         and compute coverage from the **brain mask** alone — ``|parcel n mask| /
         |parcel|`` — never consulting the data.  That definition assumes the brain
         mask already excludes NaN and zero-variance voxels.

       The threshold direction also differs: this path drops a parcel at exactly
       ``min_coverage`` (``CiftiMath('data > …')``), while the volumetric paths keep
       it (``>=``).  Aligning the two would change existing CIFTI outputs, so it has
       not been done; treat coverage values as comparable only within a modality.
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
    """Parcellate a dense series with a dlabel/label atlas, coverage-aware.

    Routes on the timeseries' CIFTI-ness: a dense CIFTI uses the grayordinate path
    (:func:`_init_parcellate_cifti_wf`, unchanged); a volumetric NIfTI uses the
    Strategy-A path (:func:`_init_parcellate_volumetric_wf`).
    """
    context = context or FactoryContext()
    if context.role_is_cifti(node, 'timeseries'):
        return _init_parcellate_cifti_wf(node, name, 'timeseries', 'parcellated.ptseries.nii')
    return _init_parcellate_volumetric_wf(node, name, context, 'timeseries')


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
    return _init_parcellate_volumetric_wf(node, name, context, 'scalar')


# Anatomical reference preference: T1w is the registration target in every NiPreps
# derivative (its transforms are all ``to-T1w``), so T2w is a fallback for T2w-only
# subjects, never an equal alternative.
_ANATOMICAL_SUFFIXES = ('T1w', 'T2w')


def _find_anatomical(context, space: str | None, session: str | None) -> str:
    """The subject's preprocessed anatomical in ``space`` (``None`` = native).

    Tries each of :data:`_ANATOMICAL_SUFFIXES` **in order** and takes the first that
    resolves to exactly one file.  A single any-of query (``suffix: ['T1w','T2w']``)
    cannot express that: on a subject having both -- the normal case for a dataset
    with a T2w -- it matches two files and fails ``find_reference``'s exactly-one
    rule, taking the whole build down.

    Several files of the *same* suffix is a genuine ambiguity (e.g. two
    reconstructions of the T1w), so that raises rather than falling through to T2w.
    """
    base = {'desc': 'preproc', 'space': space, 'datatype': 'anat'}
    tried = []
    for suffix in _ANATOMICAL_SUFFIXES:
        filters = {**base, 'suffix': suffix}
        hits = context.find_references(filters, session=session)
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(
                f'Anatomical reference {filters} (session {session!r}) matched '
                f'{len(hits)} files, expected exactly 1: {hits}. Narrow the dataset '
                'so one preprocessed anatomical per subject/session is discoverable.'
            )
        tried.append(filters)
    raise ValueError(
        f'No preprocessed anatomical found in space {space!r} (session {session!r}) '
        f'across datasets {list(context.datasets or [])}. Tried, in preference order:\n'
        + '\n'.join(f'  {f} -> 0 matches' for f in tried)
    )


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
    cross_space = atlas_space is not None and data_space is not None and atlas_space != data_space
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
        _attach_acpc_bridge(wf, node, context, warp, context.role_session(node, 'atlas'))

    return (warp, 'out_file')


def _attach_acpc_bridge(wf, node, context, warp, session) -> None:
    """Feed ``warp.bridges`` the rigid ACPC<->T1w transform BDT computes itself.

    ``ACPC`` is QSIPrep/QSIRecon's own output space and no stored transform connects
    it to the anatomical space the other derivatives share, so the hop is registered
    on the fly (:func:`_register_acpc_to_t1w`) and injected as a bridge into the
    transform chain.  ``niu.Merge(1)`` is only there to present the single result as
    the list ``bridges`` expects.
    """
    if context.provider is None:
        raise ValueError(
            f'node {node.name!r} is cross-space through ACPC and needs a '
            'FactoryContext provider to resolve the bridge references.'
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
    register.inputs.fixed_image = _find_anatomical(context, None, session)
    register.inputs.fixed_mask = context.find_reference(
        {'suffix': 'mask', 'desc': 'brain', 'space': None, 'datatype': 'anat'},
        session=session,
    )
    register.inputs.moving_image = _find_anatomical(context, 'ACPC', session)
    register.inputs.moving_mask = context.find_reference(
        {'suffix': 'mask', 'desc': 'brain', 'space': 'ACPC', 'datatype': 'anat'},
        session=session,
    )
    bridge_list = pe.Node(niu.Merge(1), name='bridge_list')
    wf.connect([
        (register, bridge_list, [('out', 'in1')]),
        (bridge_list, warp, [('out', 'bridges')]),
    ])  # fmt:skip


# Entities that identify *which acquisition* a file belongs to.  A derivatives
# dataset holds one brain mask per BOLD run, so the mask query must carry these
# across from the data or it matches every run at once.
#
# These are **pybids entity names**, which are not always the BIDS key: ``acq-`` is
# ``acquisition``, ``rec-`` is ``reconstruction``, ``dir-`` is ``direction``.  Using
# the short forms silently matches nothing, which is how the multi-run bug survived
# its first fix.  ``session`` is deliberately absent: ``find_reference`` takes it
# separately, via the anat-level-aware ``session=`` argument.
_ACQUISITION_ENTITIES = (
    'task',
    'run',
    'acquisition',
    'reconstruction',
    'direction',
    'ceagent',
    'echo',
    'part',
    'res',
)


def _discover_brain_mask(context, node, data_role: str) -> str:
    """The brain mask belonging to the same acquisition as the data on ``data_role``.

    fMRIPrep writes one ``desc-brain_mask`` per BOLD run plus an anatomical one, so
    space alone is ambiguous on any ordinary multi-run subject.  We therefore narrow
    *progressively*: first ask for a mask carrying every acquisition entity the data
    itself has (:data:`_ACQUISITION_ENTITIES`), and only if that does not resolve to
    exactly one file fall back to the base space/session/datatype query.

    Both directions matter.  Too few entities and several runs' masks match at once;
    too many and we over-constrain -- a derivative may legitimately omit an entity the
    raw data carries (a ``part-mag`` BOLD whose mask has no ``part``), which would
    match zero files.  Trying the specific query first and the general one second
    handles both without guessing.

    Deliberately has no data-derived fallback: synthesizing a mask from the data's
    finite/non-zero support would make coverage depend on the data, contradicting
    XCP-D's definition (|parcel n mask| / |parcel|).  A space with no usable mask is
    a spec problem to fix, so this raises with both attempted queries named.
    """
    base = {'suffix': 'mask', 'desc': 'brain', 'space': context.role_space(node, data_role)}
    datatype = context.role_datatype(node, data_role)
    if datatype:
        base['datatype'] = datatype

    scoped = dict(base)
    for key in _ACQUISITION_ENTITIES:
        value = context._role_entity(node, data_role, key, None)
        if value is not None:
            scoped[key] = value

    session = context.role_session(node, data_role)
    attempts = []
    for filters in [scoped, base] if scoped != base else [base]:
        hits = context.find_references(filters, session=session)
        if len(hits) == 1:
            return hits[0]
        attempts.append((filters, hits))

    raise ValueError(
        f'Could not resolve exactly one brain mask for role {data_role!r} of node '
        f'{getattr(node, "name", node)!r} (session {session!r}) across datasets '
        f'{list(context.datasets or [])}. Tried, most specific first:\n'
        + '\n'.join(f'  {f} -> {len(h)} match(es): {h}' for f, h in attempts)
    )


def _init_parcellate_volumetric_wf(node, name, context, data_role: str) -> pe.Workflow:
    """Volumetric parcellation, dispatched on atlas form.

    3D integer-label atlas -> XCP-D's :class:`NiftiParcellate` (``NiftiLabelsMasker``).
    4D atlas (one volume per region, possibly overlapping) ->
    :class:`~bdt.interfaces.probseg.ProbSegParcellate`, binarized when the atlas is a
    thresholded ``dseg``.  ``outputnode`` exposes ``out`` (the wide TSV; a scalar is
    the one-row case) and ``coverage``.
    """
    from bdt.interfaces.connectivity import NiftiParcellate
    from bdt.interfaces.probseg import ProbSegParcellate

    wf = pe.Workflow(name=name or node.name)
    min_coverage = float(node.parameters.get('min_coverage', 0.5))

    fields = [data_role, 'atlas']
    labels_path = context.role_atlas_labels(node, 'atlas')
    if labels_path is None:
        # processing-node atlas: labels arrive over the secondary edge (workflow.py)
        fields.append('atlas_labels')
    inputnode = pe.Node(niu.IdentityInterface(fields=fields), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out', 'coverage']), name='outputnode')

    ndim = context.role_atlas_ndim(node, 'atlas')
    mask = _discover_brain_mask(context, node, data_role)

    if ndim == 3:
        parcellate = pe.Node(
            NiftiParcellate(mask=mask, min_coverage=min_coverage), name='parcellate'
        )
        data_field = 'filtered_file'
    else:
        atlas_suffix = context.role_suffix(node, 'atlas', default='probseg')
        parcellate = pe.Node(
            ProbSegParcellate(
                mask=mask,
                min_coverage=min_coverage,
                binarize=atlas_suffix == 'dseg',
            ),
            name='parcellate',
        )
        data_field = 'data'

    if labels_path is not None:
        parcellate.inputs.atlas_labels = labels_path
    else:
        wf.connect([(inputnode, parcellate, [('atlas_labels', 'atlas_labels')])])

    atlas_node, atlas_field = _warp_atlas_field(wf, node, context, inputnode, data_role)
    wf.connect([
        (inputnode, parcellate, [(data_role, data_field)]),
        (atlas_node, parcellate, [(atlas_field, 'atlas')]),
        (parcellate, outputnode, [('timeseries', 'out'), ('coverage', 'coverage')]),
    ])  # fmt:skip
    return wf


@workflow_factory('functional_connectivity')
def init_functional_connectivity_wf(node, name=None, context=None) -> pe.Workflow:
    """Correlate a parcellated series into a relmat.

    CIFTI ptseries -> :class:`CiftiCorrelation` -> pconn; a volumetric parcellated
    TSV -> XCP-D's :class:`~bdt.interfaces.connectivity.TSVConnect` -> a relmat TSV
    with ``Node`` row labels.
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

    from bdt.interfaces.connectivity import TSVConnect

    correlate = pe.Node(TSVConnect(), name='correlate')
    wf.connect([
        (inputnode, correlate, [('timeseries', 'timeseries')]),
        (correlate, outputnode, [('correlations', 'out')]),
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
    # The ``mode-image`` entity is required for this to be usable as a *bridge*: the
    # transform graph parses injected bridges by filename, and a name without it is
    # still parseable but under-specified.  Keep the full BIDS entity set.
    out = os.path.abspath('from-ACPC_to-T1w_mode-image_xfm.mat')
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
        # MI registration is contrast-agnostic, so either works (fixed stays the
        # surfaces' T1w space and MI handles any cross-contrast fit).  Preference
        # order, not any-of: a subject with both would otherwise match two files.
        moving_img = _find_anatomical(context, 'ACPC', scalar_ses)
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
    """Build a 4D bundle segmentation (probseg / dseg) from bundle-wise tractograms.

    ``inputnode.tractograms`` is the grouped list of per-bundle ``.tck.gz`` files (all
    in ACPC space).  Each is decompressed (``Gunzip``) and turned into a track-density
    image on a shared reference grid (``tckmap`` via nipype ``ComputeTDI``); the
    per-bundle maps are peak-normalized to ``[0, 1]`` and stacked into a 4D
    ``probseg`` (:class:`~bdt.interfaces.tractography.ConcatenateNiftis`).  When a
    ``threshold`` parameter is given, the stack is binarized (``value > threshold``)
    into a 4D ``dseg``.  A BIDS ``index``/``name`` label TSV (one row per volume, in
    input order) is emitted on ``outputnode.tsv``; the segmentation is on
    ``outputnode.out``.

    ``inputnode.reference`` is a single ACPC-space image (wired via the ``reference``
    role, e.g. a ``dwiref`` or any ACPC ``dwimap``) whose voxel grid defines the
    output segmentation grid.
    """
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces.mrtrix3 import ComputeTDI

    from bdt.interfaces.tractography import (
        ConcatenateNiftis,
        EntitiesToSegTSV,
        ThresholdNifti,
    )

    threshold = node.parameters.get('threshold')

    wf = pe.Workflow(name=name or node.name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['tractograms', 'reference']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['out', 'tsv']), name='outputnode')

    # per-bundle: .tck.gz -> .tck -> track-density image on the reference grid
    gunzip = pe.MapNode(Gunzip(), iterfield=['in_file'], name='gunzip')
    tck_to_tdi = pe.MapNode(
        ComputeTDI(out_file='tdi.nii.gz'),
        iterfield=['in_file'],
        name='tck_to_tdi',
    )
    # stack + peak-normalize -> 4D probseg
    concatenate = pe.Node(
        ConcatenateNiftis(normalize=True, out_file='pseg.nii.gz'), name='concatenate'
    )
    # label table (volume index -> bundle name), same order as inputnode.tractograms
    bundles_to_tsv = pe.Node(
        EntitiesToSegTSV(entity='bundle', out_file='dseg.tsv'), name='bundles_to_tsv'
    )

    wf.connect([
        (inputnode, gunzip, [('tractograms', 'in_file')]),
        (inputnode, tck_to_tdi, [('reference', 'reference')]),
        (gunzip, tck_to_tdi, [('out_file', 'in_file')]),
        (tck_to_tdi, concatenate, [('out_file', 'in_files')]),
        (inputnode, bundles_to_tsv, [('tractograms', 'in_files')]),
        (bundles_to_tsv, outputnode, [('out_file', 'tsv')]),
    ])  # fmt:skip

    if threshold is not None:
        binarize = pe.Node(
            ThresholdNifti(threshold=float(threshold), binarize=True, out_file='dseg.nii.gz'),
            name='binarize',
        )
        wf.connect([
            (concatenate, binarize, [('out_file', 'in_file')]),
            (binarize, outputnode, [('out_file', 'out')]),
        ])  # fmt:skip
    else:
        wf.connect([(concatenate, outputnode, [('out_file', 'out')])])

    return wf


@workflow_factory('parcellate_scalar_as_tract_profile')
def init_parcellate_scalar_as_tract_profile_wf(node, name=None, context=None) -> pe.Workflow:
    """Sample a scalar along bundles into an along-tract profile TSV.

    ``inputnode.scalar`` is a single volume (e.g. FA / CBF) and
    ``inputnode.bundles`` the grouped list of per-bundle ``.tck.gz`` streamlines.
    Each bundle is decompressed (``Gunzip``) and the scalar is sampled along it at
    ``n_nodes`` nodes (:class:`~bdt.interfaces.tractography.SampleTractProfiles`),
    producing a tidy ``bundle``/``node``/``mean``/``std`` TSV on ``outputnode.out``.

    The sampler reads the scalar at the streamlines' own **world** coordinates, so
    the two must share a space.  A mismatch would not fail -- it would silently read
    whatever anatomy sits at those coordinates and return plausible-looking numbers
    -- so when the spaces differ the scalar is warped into the bundles' space first
    (``warp_scalar``), and only an unresolvable mismatch raises.

    Streamlines have no voxel grid of their own, so the warp needs an explicit
    reference image: the subject's preprocessed anatomical in the bundles' space
    (typically QSIPrep's ``space-ACPC`` T1w).  ``ACPC`` is QSIPrep/QSIRecon's own
    output space with no stored transform reaching it, so the rigid ACPC<->T1w hop is
    registered on the fly and injected as a bridge, exactly as the parcellate paths do.
    """
    from nipype.algorithms.misc import Gunzip

    from bdt.interfaces.tractography import SampleTractProfiles
    from bdt.interfaces.transforms import ResolveApplyTransforms

    context = context or FactoryContext()
    n_nodes = int(node.parameters.get('n_nodes', 100))
    scalar_space = context.role_space(node, 'scalar')
    bundles_space = context.role_space(node, 'bundles')
    cross_space = (
        scalar_space is not None and bundles_space is not None and scalar_space != bundles_space
    )

    wf = pe.Workflow(name=name or node.name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['scalar', 'bundles']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']), name='outputnode')

    gunzip = pe.MapNode(Gunzip(), iterfield=['in_file'], name='gunzip')
    profile = pe.Node(
        SampleTractProfiles(n_nodes=n_nodes, out_file='tractprofile.tsv'), name='profile'
    )

    wf.connect([
        (inputnode, gunzip, [('bundles', 'in_file')]),
        (gunzip, profile, [('out_file', 'in_files')]),
        (profile, outputnode, [('out_file', 'out')]),
    ])  # fmt:skip

    if not cross_space:
        wf.connect([(inputnode, profile, [('scalar', 'scalar')])])
        return wf

    if context.provider is None:
        raise ValueError(
            f'Tract-profile node {node.name!r} has its scalar in {scalar_space!r} but its '
            f'bundles in {bundles_space!r}, so the scalar must be warped -- which needs a '
            'FactoryContext provider to resolve the reference anatomical.'
        )

    session = context.role_session(node, 'scalar')
    warp = pe.Node(
        ResolveApplyTransforms(
            source=scalar_space,
            target=bundles_space,
            interpolation='linear',  # a continuous scalar (CBF, FA), never a label map
            local_transforms=context.discover_transforms(session=session),
            reference=_find_anatomical(context, bundles_space, session),
            out_file='scalar_in_bundle_space.nii.gz',
        ),
        name='warp_scalar',
    )
    wf.connect([
        (inputnode, warp, [('scalar', 'moving')]),
        (warp, profile, [('out_file', 'scalar')]),
    ])  # fmt:skip

    if 'ACPC' in (scalar_space, bundles_space):
        _attach_acpc_bridge(wf, node, context, warp, session)

    return wf
