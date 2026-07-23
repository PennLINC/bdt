.. include:: links.rst

.. _spec:

##########################
Configuration & spec files
##########################

Both *BDT* and *BAT* are driven by a YAML **spec file**, passed with ``--spec``.
The spec file names *what* to do: which atlases to apply to which data, and which
operations to run. Run-level settings (paths, participants, resources) come from
the command line and are stored in a configuration singleton (see
:ref:`config-singleton` below).

.. warning::
   The spec grammar described here reflects the intended design of *BDT*/*BAT*
   and may not yet be fully wired up in the installed version.


.. _bdt-spec:

*******************
``bdt_spec.yaml``
*******************

The *BDT* spec is a list of ``sources``. Each source binds a derivative type
(identified by BIDS entities) to a set of atlases and operations.
Atlases are specified as lists of entity-filter dicts, which are passed directly
to PyBIDS ``.get()``. The file format (NIfTI / GIFTI / CIFTI) is auto-detected
from the matched file's extension at workflow-build time, so a single spec entry
works regardless of the format of the underlying derivative.

.. code-block:: yaml

    sources:
      - suffix: bold
        datasets: [fmriprep]
        operations: [parcellate_timeseries, functional_connectivity]
        atlases:
          - atlas: HCPMMP1
          - atlas: Schaefer2018
            seg: 17networks
            scale: 400          # scale- = region count

      - suffix: cbf
        datasets: [aslprep]
        operations: [parcellate_scalar]
        atlases:
          - atlas: HCPMMP1
            res: 2

      - suffix: asl
        datasets: [aslprep]
        operations: [parcellate_timeseries]
        atlases:
          - atlas: HCPMMP1

      - suffix: tractogram         # BEP 046
        datasets: [qsirecon]
        operations: [streamline_connectivity, bundle_stats]
        atlases:
          - atlas: Schaefer2018
            seg: 17networks
            scale: 400

      - suffix: dwimap             # BEP 016: model-/param- diffusion model
        datasets: [qsirecon]
        model: csd
        param: wm
        operations: [parcellate_odf]
        atlases:
          - atlas: HCPMMP1

Each source entry accepts:

- **BIDS entity filters** (``suffix``, ``datasets``, ``model``, ``param``,
  ``task``, ``desc``, …) used to collect the derivative files to process.
  ``datasets`` names the derivative dataset(s) registered on the command line
  with ``--datasets``.
- **operations** — a list of operations to run on each matched file (see below).
- **atlases** — a list of entity-filter dicts selecting atlases from the BIDS
  Atlas dataset. Each dict may include ``atlas``, ``seg``, ``scale``, ``res``,
  ``desc``, and so on.

Supported operations
=====================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Description
   * - ``parcellate_timeseries``
     - Extract per-parcel time series from a 4D image (BOLD, ASL), reducing each
       parcel's voxels with the ``statistics`` parameter's summaries.
   * - ``parcellate_scalar``
     - Summarize a scalar map (CBF, FA/MD, thickness, …) over parcels, likewise
       via ``statistics``.
   * - ``functional_connectivity``
     - Build a region-by-region connectivity matrix from parcellated time series.
   * - ``streamline_connectivity``
     - Warp a tractogram into the atlas's space and assign each streamline's two
       endpoints to parcels, producing an N×N connectivity matrix.
   * - ``bundle_stats``
     - Per-tractogram summary statistics (count, mean length, density). This is a
       whole-tractogram summary, **not** a connectivity matrix.
   * - ``parcellate_fixel``
     - Per-parcel fixel summary (respects fixel cardinality/direction).
   * - ``parcellate_odf``
     - Per-parcel ODF/SH-coefficient summary.

.. note::
   **Per-parcel statistics.** Both parcellation actions take a ``statistics``
   parameter naming how each parcel's voxels/vertices are reduced. The vocabulary
   is nilearn's ``NiftiLabelsMasker`` strategy set — ``mean`` (the default),
   ``median``, ``sum``, ``minimum``, ``maximum``, ``standard_deviation``,
   ``variance`` — and every one of them has a Workbench ``-method`` equivalent, so
   volumetric and grayordinate inputs accept the same list.

   .. code-block:: yaml

      parameters:
        statistics: [mean, standard_deviation]

   ``standard_deviation`` and ``variance`` are *population* moments (ddof=0).
   Workbench's sample equivalent (``SAMPSTDEV``) is deliberately not offered,
   since nilearn has no counterpart and the two backends would disagree.

   Two constraints follow from the shape of the output:

   * A **probabilistic (4D) atlas** weights every voxel, and only ``mean`` and
     ``standard_deviation`` have an agreed weighted definition, so anything else
     is an error rather than a silent substitution. ``parcellate_timeseries`` with
     such an atlas supports ``mean`` alone.
   * ``parcellate_scalar`` merges its statistics into **one** tidy table (a row
     per parcel, a column per statistic), while ``parcellate_timeseries`` is wide
     (timepoints × parcels) with nowhere to put a second statistic, so it emits a
     separate ``stat-``-labelled file per statistic instead.

   ``functional_connectivity`` correlates only the **first** statistic in the
   list; it logs which one it picked.

.. note::
   **Tractography I/O.** The only BIDS-compliant tractography format is TRX
   (``.trx``). ``.tck``/``.trk`` are accepted only as ingest conveniences and are
   converted to ``.trx`` on the way in. The tractogram suffix is ``tractogram``,
   optionally carrying ``tract-<name>`` (anatomical structure) and
   ``track-<method>``.

.. note::
   **Entities from BIDS Extension Proposals.** Because sources are selected with
   BIDS entities, the spec understands the entities defined by the relevant BEPs:
   ``model-``/``param-`` for diffusion models (`BEP 016`_), ``tract-``/``track-``
   for tractograms (`BEP 046`_), and ``seg-``/``scale-`` for atlas realizations
   (`BEP 017`_). ``scale-`` is the BIDS-blessed encoding of a parcellation's
   region count (e.g., Schaefer's 400), replacing the older
   ``desc-400Parcels17Networks`` idiom.


.. _bat-spec:

*******************
``bat_spec.yaml``
*******************

The *BAT* spec is a list of ``operations``. Each operation names an atlas-algebra
step, its inputs (entity-filter dicts), and the BIDS entities for the output
atlas. Entity values must be alphanumeric (a BIDS requirement — no hyphens or
underscores).

.. code-block:: yaml

    operations:
      - name: corticalSubcortical
        operation: union
        inputs:
          - atlas: HCPMMP1
          - atlas: Tian
            seg: S2
        output_entities:
          atlas: HCPMMPTian

      - name: networkParcels
        operation: intersect
        inputs:
          - atlas: Schaefer2018
            seg: 17networks
            scale: 400
          - atlas: RSN
            desc: networks
        output_entities:
          atlas: Schaefer2018RSN
          desc: 400Parcels17NetworksRSN

      - name: parcelBundles
        operation: outer_product
        inputs:
          - atlas: Schaefer2018
            seg: 17networks
            scale: 400
          - atlas: AFQ
            seg: bundles
        output_entities:
          atlas: Schaefer2018AFQ

Supported operations
=====================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``intersect``
     - Voxel-wise label **restriction**: keep voxels labeled in both atlases and
       relabel to the first atlas's parcel.
   * - ``union``
     - Combine two atlas label images; overlaps are resolved by a precedence rule
       rather than an error.
   * - ``outer_product``
     - Cartesian sub-parcels: combine the two atlases' labels into one label per
       overlapping region (e.g., Schaefer parcels × AFQ bundles).

.. note::
   **Scope.** *BAT* operates on discrete label images (``dseg``) only and emits
   ``dseg``. All inputs to a given operation must share the same template
   (``tpl-``) space; *BAT* raises an error if they do not. Probabilistic
   (``probseg``) atlases and streamline *data* are out of scope, but a bundle
   atlas already rasterized to a ``dseg`` label image is a valid input.


.. _config-singleton:

*******************
Run configuration
*******************

Beyond the spec file, run-level settings are held in a configuration singleton
(``config.py``), serialized to TOML so it can be shared across processes.
A copy of the resolved settings is written under the log directory for each run.
The configuration is organized into sections:

.. list-table::
   :header-rows: 1
   :widths: 22 13 65

   * - Section
     - Scope
     - Key fields
   * - ``environment``
     - shared
     - version, nipype_version, exec_env (read-only; recorded for reporting)
   * - ``execution``
     - shared
     - bids_dir, output_dir, datasets, spec, participant_label
   * - ``workflow``
     - shared
     - spaces, file_format (``auto`` / ``nifti`` / ``gifti`` / ``cifti``)
   * - ``bdt_workflow``
     - BDT only
     - min_coverage, dummy_scans, correlation_lengths, output_correlations
   * - ``bat_workflow``
     - BAT only
     - output_space, interpolation
   * - ``nipype``
     - shared
     - nprocs, omp_nthreads, plugin

With ``file_format = auto`` (the default), the format of each collected file is
detected from its extension at workflow-build time and used to route the file to
the appropriate format-specific parcellation subworkflow.
