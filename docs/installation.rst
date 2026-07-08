.. include:: links.rst

############
Installation
############

*BDT* should be installed using container technologies.

.. code-block:: bash

  docker pull nipreps/bdt:main


************************************************
Containerized execution (Docker and Singularity)
************************************************

*BDT* is a *NiPreps* application, and therefore follows some overarching principles
of containerized execution drawn from the BIDS-Apps protocols.
For detailed information of containerized execution of *NiPreps*, please visit the corresponding
`Docker <https://www.nipreps.org/apps/docker/>`__
or `Singularity <https://www.nipreps.org/apps/singularity/>`__ subsections.


External Dependencies
=====================

*BDT* is written using Python 3.12, and is based on nipype_.
The Python environment (nipype, `niworkflows`, `pybids`, and related NiPreps
libraries) is resolved with `pixi <https://pixi.sh>`__ from ``pixi.lock``.

The container image additionally bundles a small set of neuroimaging tools that
are not handled by Python's packaging system:

- `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_
  (version 1.5.0) — surface/CIFTI parcellation and label resampling.
- AFNI_ (a minimal subset of programs, e.g. ``3dresample``, ``3dTshift``).
- `bids-validator <https://github.com/bids-standard/bids-validator>`_
  (version 1.14.10, installed via ``npm``).

.. note::
   The full atlas-transform toolchain described in the design (ANTs
   ``ApplyTransforms`` for volumetric atlas resampling, and the Rust binaries
   ``trxrs`` / ``giftirs`` / ``odx`` for streamline, surface, and diffusion-model
   transforms) is **not yet part of the container image**. These will be added as
   the corresponding workflows are implemented. If you run *BDT* outside the
   container, install the tools required by the operations you use.


***********************************************
Not running on a local machine? - Data transfer
***********************************************

If you intend to run *BDT* on a remote system, you will need to
make your data available within that system first.

For instance, here at the Poldrack Lab we use Stanford's
:abbr:`HPC (high-performance computing)` system, called Sherlock.
Sherlock enables `the following data transfer options
<https://www.sherlock.stanford.edu/docs/user-guide/storage/data-transfer/>`_.

Alternatively, more comprehensive solutions such as `Datalad
<https://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands.
Datalad also performs version control over your data.
