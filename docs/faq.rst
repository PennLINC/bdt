.. include:: links.rst

################################
FAQ - Frequently Asked Questions
################################

.. contents::
    :local:
    :depth: 1


*************************
What are *BDT* and *BAT*?
*************************

*BDT* (BIDS Derivatives Transformer) and *BAT* (BIDS Atlas Transformer) are two
BIDS Apps that ship in the same package.
*BDT* applies atlases to BIDS derivative datasets, writing out parcellated data
and connectivity matrices; *BAT* manipulates BIDS Atlas datasets via atlas
algebra (intersection, union, outer product).
They fall under the broader umbrella of *NiPost* workflows, which perform
post-processing on BIDS-Derivative datasets.

These workflows are designed to work with derivatives from any pipeline that
produces BIDS-Derivative-compliant datasets, and are primarily tested against
the outputs of pipelines such as `fMRIPrep`_, `ASLPrep`_, and `QSIRecon`_.

.. _fMRIPrep: https://fmriprep.org
.. _ASLPrep: https://aslprep.readthedocs.io
.. _QSIRecon: https://qsirecon.readthedocs.io
