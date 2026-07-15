.. include:: links.rst

.. _Usage :

###########
Usage Notes
###########

The ``bdt`` package ships **two** BIDS Apps that share the same code base,
infrastructure, and installation:

- **BDT** (``bdt`` CLI) — the *BIDS Derivatives Transformer*.
  It applies atlases to BIDS derivative datasets and writes out parcellated
  data and connectivity matrices.
- **BAT** (``bat`` CLI) — the *BIDS Atlas Transformer*.
  It manipulates BIDS Atlas datasets via atlas algebra (intersection, union,
  and outer product).

Both tools follow the `BIDS-Apps <https://github.com/BIDS-Apps>`__ convention and
are configured primarily through a YAML *spec file* (see :doc:`spec`).

.. warning::
   *BDT*, *BAT*, and the atlas/spec machinery described here are under active
   development. Some options documented below describe the intended design and
   may not yet be wired up in the installed version. The auto-generated
   command-line table further down always reflects the options actually
   available in your installation.

.. warning::
   *BDT* includes a tracking system to report usage statistics and errors
   for debugging and grant reporting purposes.
   Users can opt out using the ``--notrack`` command-line argument.


*******************************
Execution and the BIDS format
*******************************

Both tools take as principal inputs a BIDS-valid dataset (``bids_dir``) and an
output directory (``output_dir``), followed by an analysis level.
The input dataset and any additional datasets are required to be in valid
:abbr:`BIDS (Brain Imaging Data Structure)` format.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <https://bids-standard.github.io/bids-validator/>`_.

Further information about BIDS and BIDS-Apps can be found at the
`NiPreps portal <https://www.nipreps.org/apps/framework/>`__.


BDT (BIDS Derivatives Transformer)
==================================

*BDT* runs at the **participant** analysis level, following the standard
BIDS-App pattern.
Derivative datasets (e.g., fMRIPrep, ASLPrep, QSIRecon) and the BIDS Atlas
dataset are passed together via ``--datasets``, and the atlases to apply plus
the operations to run are configured in the spec file (``--spec``).
Example: ::

    bdt <bids_dir> <output_dir> participant \
      --datasets fmriprep=/path/to/fmriprep \
                 aslprep=/path/to/aslprep \
                 qsirecon=/path/to/qsirecon \
                 atlases=/path/to/bids-atlas-dataset \
      --spec /path/to/bdt_spec.yaml \
      --participant-label sub-01

Named datasets are provided as ``name=/path`` pairs. *BDT* indexes them all,
resolves the transforms needed to bring atlases and data into a shared space,
and dispatches each matched file to the appropriate parcellation/connectivity
workflow based on its data type and the requested operations.


BAT (BIDS Atlas Transformer)
============================

*BAT* runs at the **dataset** analysis level, because atlases are not
per-subject.
For the single-dataset case, ``bids_dir`` is the input BIDS Atlas dataset and
no ``--datasets`` is needed.
Example: ::

    bat <bids_dir> <output_dir> dataset \
      --spec /path/to/bat_spec.yaml

If atlases from multiple input datasets need to be combined, ``--datasets`` can
be used to name additional atlas datasets.


****************
Input datasets
****************

*BDT* is a post-processing tool: rather than a single raw dataset, it consumes
one or more **derivative** datasets plus a **BIDS Atlas** dataset.

- **Derivative datasets** are the outputs of upstream preprocessing pipelines.
  Each is registered with a name via ``--datasets name=/path`` so the spec file
  can refer to it (e.g., ``datasets: [fmriprep]``).
  Supported inputs include volumetric NIfTI, GIFTI surface, and CIFTI dense data
  (BOLD/ASL time series, CBF, FA/MD and other scalar maps, cortical thickness),
  tractograms, and diffusion-model images (SH/ODF/fixels).
- **Transforms** are discovered automatically from the ``*_xfm.*`` files present
  in the provided derivative datasets and supplemented with TemplateFlow
  standard-to-standard transforms. This is how *BDT* knows how to move an atlas
  into the data's space (or the data's coordinates into the atlas space); see
  :doc:`workflows`.
- **BIDS Atlas dataset** provides the atlases to apply. Atlases are selected in
  the spec file using BIDS entities (``atlas``, ``seg``, ``scale``, ``res``,
  ``desc``, …). The file format (NIfTI / GIFTI / CIFTI) of each matched file is
  auto-detected from its extension at workflow-build time.

.. note::
   *BDT* is strict rather than heuristic about input selection. If a spec entry
   matches zero files, or matches more than one file for a single run, or if no
   transform path exists between an atlas's space and the data's space, *BDT*
   raises an explicit error naming the conflict rather than guessing. Add more
   BIDS entities to the offending spec entry to disambiguate.


**********************
Command-Line Arguments
**********************

.. argparse::
   :ref: bdt.cli.parser._build_parser
   :prog: bdt
   :nodefault:
   :nodefaultconst:


.. _prev_derivs:

*******************************
Reusing precomputed derivatives
*******************************

Reusing a previous, partial execution of *BDT*
==============================================

*BDT* will pick up where it left off a previous execution,
so long as the work directory
points to the same location, and this directory has not been changed/manipulated.
Some workflow nodes will rerun unconditionally, so there will always be some amount of
reprocessing.


***************
Troubleshooting
***************

Logs and crashfiles are output into the
``<output dir>/logs`` directory.
Information on how to customize and understand these files can be found on the
`Debugging Nipype Workflows <https://miykael.github.io/nipype_tutorial/notebooks/basic_debug.html>`_
page.


Support and communication
=========================

The documentation of this project is found here: https://bdt.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/nipreps/bdt/issues.

If you have a problem or would like to ask a question about how to use *BDT*,
please submit a question to
`NeuroStars.org <https://neurostars.org/tag/bdt>`_
with a ``bdt`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

Previous questions about *BDT* are available here:
https://neurostars.org/tag/bdt/


.. include:: license.rst
