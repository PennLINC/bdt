.. include:: links.rst

.. _spaces:

######################################
Spaces, templates, and atlas alignment
######################################

Unlike a preprocessing pipeline, *BDT* and *BAT* do not resample data into a
list of user-requested output spaces. Instead, they *apply atlases* to data that
is already in some space, and the central spatial concern is bringing an atlas
and the data it is applied to into a **shared coordinate space**.

- *BDT* parcellates a derivative (BOLD, CBF, a scalar map, a tractogram, a
  diffusion model) using an atlas. The atlas and the data must be aligned first;
  see :ref:`atlas-alignment` below and :doc:`workflows` for the transform system.
- *BAT* combines atlases with atlas algebra. All inputs to a given *BAT*
  operation must share the same template (``tpl-``) space, and the output
  inherits that space (see :doc:`outputs`).

Every atlas-derived output carries a ``space-`` entity recording the coordinate
system the parcellation ran in; it is REQUIRED (see :doc:`outputs`).


.. _TemplateFlow:

**************
*TemplateFlow*
**************

*TemplateFlow* is a software library and a repository of neuroimaging templates
and atlases that lets applications such as *BDT* flexibly query and pull
template and atlas information.
It is how *BDT* accesses standard template spaces and the
standard-to-standard transforms used to move atlases between them, and it is
where custom templates can be registered (see below).
For more general information about *TemplateFlow*, visit
`TemplateFlow.org <https://www.templateflow.org>`__.


***************
Standard spaces
***************

Standard, stereotactic template spaces are identified by the same
``space-``/``tpl-`` identifiers used throughout BIDS and TemplateFlow — for
example ``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, or a surface space such as
``fsLR`` or ``fsaverage``.
Valid identifiers come from the
`TemplateFlow repository <https://github.com/templateflow/templateflow>`__.

Atlases in a BIDS Atlas dataset live in a template space (``tpl-``), and
derivatives carry a ``space-`` entity for the space they were resampled to
upstream. *BDT* uses the transform graph (below) to reconcile the two.


******************
Nonstandard spaces
******************

Derivatives are not always in a standard template space. *BDT* treats
subject-native and intermediate spaces as ordinary nodes in its transform graph,
including, for example:

* ``T1w`` / ``anat`` — the individual's anatomical reference.
* ``fsnative`` — the individual's FreeSurfer surface reconstruction.
* ``boldref`` / ``sbref`` — a BOLD run's own reference grid.
* ``ACPC`` / native diffusion space — common for QSIRecon tractography and
  diffusion-model outputs.

*BDT* does not require these to be pre-aligned to the atlas; it only requires
that a transform path exists between the data's space and the atlas's space (see
below).


.. _atlas-alignment:

*******************************
Aligning atlases and data
*******************************

*BDT* discovers transforms automatically from the ``*_xfm.*`` files in the
provided derivative datasets and supplements them with TemplateFlow
standard-to-standard transforms, assembling a graph whose nodes are spaces and
whose edges are transforms.
The direction in which data and atlases are moved depends on the **geometry** of
the data:

* **Grid data** (volumetric NIfTI, CIFTI dense) — the atlas is warped *into the
  data's grid* once, so signal is never re-interpolated.
* **Point data** (GIFTI vertices, TRX/TCK streamlines) — the points (or atlas
  vertices) are warped, which is lossless.
* **Diffusion-model grid** (SH/ODF/fixels) — resampled like grid data, plus SH
  reorientation.

See :doc:`workflows` for the full description of the geometry-aware transform
system and per-format dispatch.

.. note::
   *BDT* is strict rather than heuristic about alignment: if no transform path
   exists between an atlas's space and the data's space, it raises an explicit
   error naming both spaces and listing the spaces available in the graph, rather
   than guessing.


**********************
Custom standard spaces
**********************

To make a custom template or atlas space visible to *BDT*, store it under
*TemplateFlow*'s home directory.
The default is ``$HOME/.cache/templateflow``, which can be changed by setting the
``$TEMPLATEFLOW_HOME`` environment variable.
A minimal example of the files needed for a template called ``MyCustom`` follows::

  $TEMPLATEFLOW_HOME/
      tpl-MyCustom/
          template_description.json
          tpl-MyCustom_res-1_T1w.nii.gz
          tpl-MyCustom_res-1_desc-brain_mask.nii.gz
          tpl-MyCustom_res-2_T1w.nii.gz
          tpl-MyCustom_res-2_desc-brain_mask.nii.gz

For further information about how custom templates must be organized and named,
please check `the TemplateFlow tutorials
<https://www.templateflow.org/python-client/tutorials.html>`__.

.. note::
   The ``res-`` (resolution) entity of *TemplateFlow* is an **index**, not a
   voxel size: ``res-2`` does not necessarily mean 2 mm\ :sup:`3`. The mapping
   from index to physical resolution is defined per template in its
   ``template_description.json``.
