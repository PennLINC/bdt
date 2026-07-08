.. include:: links.rst

.. _outputs:

##########################
Outputs of *BDT* and *BAT*
##########################

All *BDT* and *BAT* outputs are `BIDS Derivatives`_.
:py:class:`~bdt.interfaces.bids.DerivativesDataSink` copies every BIDS entity
from the source file and injects the atlas entities in canonical BIDS order:

    ``space- → atlas- → seg- → scale- → res- → den- → desc-``

Two rules are enforced throughout:

1. **``space-`` is REQUIRED** on every atlas-derived output — it disambiguates
   the coordinate system the parcellation ran in.
2. Measure and model semantics live in dedicated entities (``meas-``,
   ``model-``, ``param-``, ``stat-``) rather than being overloaded onto
   ``desc-``.

Suffixes follow the relevant BIDS Extension Proposals rather than the raw source
suffix (e.g., a parcellated BOLD time series uses ``timeseries``, not ``bold``).

.. note::
   Several conventions below reflect BEPs that are not yet merged into BIDS
   (`BEP 011`_, `BEP 012`_, `BEP 016`_, `BEP 017`_, `BEP 046`_). They are
   adopted here but subject to upstream change.


******
Layout
******

Assuming *BDT* is invoked with::

    bdt <bids_dir> <output_dir> participant [OPTIONS]

the output is a `BIDS Derivatives`_ dataset of the form::

    <output_dir>/
      logs/
      sub-<label>/ses-<label>/
      sub-<label>.html
      dataset_description.json

For each participant, a directory of derivatives (``sub-<label>/ses-<label>/``) and a visual
QA report (``sub-<label>.html``) are generated. The ``logs/`` directory contains
`citation boilerplate`_ text, and ``dataset_description.json`` records the
metadata recommended by BIDS.


***************
BDT derivatives
***************

Time series & functional connectivity
=====================================

``parcellate_timeseries`` produces per-parcel time series with the
``timeseries`` suffix (`BEP 012`_). ``functional_connectivity`` produces a
relationship matrix (`BEP 017`_): ``meas-<label>_relmat.dense.tsv`` with a
REQUIRED ``_relmat.json`` sidecar and a node-index sidecar. ::

    sub-<label>/ses-<label>/func/
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][desc-<label>_]timeseries.tsv
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][desc-<label>_]timeseries.json
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-<label>_relmat.dense.tsv
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-<label>_relmat.json          ← REQUIRED sidecar
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-<label>_nodeindices.tsv       ← REQUIRED node map
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-<label>_nodeindices.json

For CIFTI inputs, Connectome-Workbench-native artifacts
(``…_timeseries.ptseries.nii``, ``…_relmat.pconn.nii``) may also be emitted;
these are optional and not BEP-canonical.

- The ``meas-`` entity identifies the measure (e.g., ``pearsoncorrelation``,
  ``partialcorrelation``).
- The ``_relmat.json`` sidecar carries the REQUIRED fields ``NodeFiles``,
  ``RelationshipMeasure``, ``Weighted``, ``Directed``, ``ValidDiagonal``,
  ``StorageFormat``, and ``Software``.
- ``_nodeindices.tsv`` maps each matrix index to a source-atlas ROI. For a
  single-atlas square matrix it is an adapted copy of the atlas ``dseg.tsv``.
- The ``.dense.`` qualifier is mandatory between ``_relmat`` and the extension
  (``.sparse.`` for sparse matrices).

Scalar maps
===========

``parcellate_scalar`` mirrors the source-derivative's BEP, keeping
measure/model semantics in ``stat-`` / ``model-`` / ``param-`` rather than in the
suffix. ::

    sub-<label>/ses-<label>/func/
      sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]stat-<label>_[desc-<label>_]timeseries.tsv   ← ALFF/ReHo etc. (BEP 012)
    sub-<label>/ses-<label>/perf/
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][desc-<label>_]cbf.tsv                            ← ASL CBF (perf)
    sub-<label>/ses-<label>/dwi/
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]model-<label>_param-<label>_dwimap.tsv     ← FA/MD etc. (BEP 016)
    sub-<label>/ses-<label>/anat/
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][desc-<label>_]morph.tsv                          ← morphometrics (BEP 011)

ALFF/falff/ReHo are ``stat-`` values on a derivative map (`BEP 012`_), not a
dedicated suffix; anatomical morphometrics use the ``morph`` suffix with
`BEP 011`_ columns; diffusion scalars carry ``model-``/``param-`` per `BEP 016`_.

Streamlines
===========

``streamline_connectivity`` produces an endpoint-connectivity relationship
matrix (`BEP 017`_): the measure is a ``meas-<label>`` entity
(``count`` / ``length`` / ``density`` / ``denlen``), the suffix is ``relmat``
with the mandatory ``.dense.`` qualifier, and the ``_relmat.json`` +
``_nodeindices.*`` sidecars are REQUIRED. ``bundle_stats`` is a per-tractogram
summary, so it is a plain descriptive TSV, **not** a matrix. ::

    sub-<label>/ses-<label>/dwi/
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-count_relmat.dense.tsv   ← NxN endpoint connectivity
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-count_relmat.json         ← REQUIRED sidecar
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-count_nodeindices.tsv      ← REQUIRED node map
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]meas-count_nodeindices.json
      sub-<label>_[ses-<label>_]space-<label>_[tract-<label>_][track-<label>_]desc-bundlestats_tractogram.tsv             ← bundle_stats summary

Diffusion model
===============

``parcellate_fixel`` and ``parcellate_odf`` summarize per-parcel diffusion-model
coefficients (`BEP 016`_). ::

    sub-<label>/ses-<label>/dwi/
      sub-<label>_[ses-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_]model-<label>_param-<label>_dwimap.tsv

The model (``model-csd``, ``model-noddi``, …) and the parameter (``param-wm`` /
``param-gm`` / ``param-csf`` for CSD SH; ``param-fa``; …) are REQUIRED entities.
The BIDS-native diffusion-model image is a plain NIfTI (e.g., SH as
``IxJxKx45``); tool-internal ``.odx``/``.mif`` containers never appear in a BIDS
tree.


**************
Visual reports
**************

*BDT* writes one summary report per subject to
``<output_dir>/sub-<label>.html``. These reports provide a quick way to visually
inspect the results of parcellation and connectivity.


***********
BAT outputs
***********

*BAT* writes a new BIDS Atlas dataset to ``output_dir/``, conforming to the BIDS
*Templates and atlases* derivatives specification. Atlas files live under a
``tpl-<label>/`` directory named for the spatial reference of the input atlases. ::

    output_dir/
      dataset_description.json                       ← DatasetType: derivative
      atlas-<label>_description.json                 ← REQUIRED (validator error if absent)
      tpl-<label>/
        [<datatype>/]
          tpl-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][res-<label>_][desc-<label>_]dseg.nii.gz
          tpl-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][res-<label>_][desc-<label>_]dseg.tsv
          tpl-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][res-<label>_][desc-<label>_]dseg.json
      atlas-<label>_space-<label>_nodelabels.tsv       ← provenance for union / outer_product (BEP 017 §4.3)
      atlas-<label>_space-<label>_nodelabels.json


*BAT* should also be able to produce subject-space atlases, like so::

    output_dir/
      dataset_description.json                       ← DatasetType: derivative
      atlas-<label>_description.json                 ← REQUIRED (validator error if absent)
      sub-<label>/ses-<label>/func/
        sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][res-<label>_][desc-<label>_]dseg.nii.gz
        sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][res-<label>_][desc-<label>_]dseg.tsv
        sub-<label>_[ses-<label>_][task-<label>_]space-<label>_atlas-<label>_[seg-<label>_][scale-<label>_][res-<label>_][desc-<label>_]dseg.json
      atlas-<label>_space-<label>_nodelabels.tsv       ← provenance for union / outer_product (BEP 017 §4.3)
      atlas-<label>_space-<label>_nodelabels.json

Key BIDS-conformance requirements:

- **Canonical entity order is** ``atlas → seg → scale → res → desc``.
  ``scale-`` is the spec-blessed encoding of region count (e.g., Schaefer's 400),
  preferred over folding it into ``desc-``.
- **``atlas-<label>_description.json`` is REQUIRED** — the validator raises
  ``ATLAS_DESCRIPTION_REQUIRED`` (error) for any file carrying both ``tpl-`` and
  ``atlas-`` without a sibling description file. It must include at least
  ``AtlasName`` and ``License``; ``Description``/``Authors`` are recommended, and
  provenance (source atlases and the operation) should go in
  ``Description``/``DerivedFrom``.
- The ``dseg.tsv`` MUST have ``index`` and ``name`` columns.
- For **union** and **outer_product**, *BAT* also emits ``nodelabels.tsv``/
  ``.json`` (`BEP 017`_ §4.3) mapping each output label back to its
  ``SourceAtlasName`` + ``SourceAtlasIndex`` + ``SourceAtlasLabel``, since the
  offset-relabeling otherwise loses that provenance.
- The ``atlas-`` entity and any additional entities come from ``output_entities``
  in the *BAT* spec. The ``tpl-`` entity is inherited from the input atlases'
  shared spatial reference. If that ``tpl-`` is not a standard TemplateFlow
  identifier, ``SpatialReference`` metadata is REQUIRED on the output files.
