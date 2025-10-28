############################
BIDS Derivatives Transformer
############################

A post-processing BIDS App that ingresses BIDS-Atlas and BIDS Derivative datasets
and parcellated selected derivatives with the atlases.


********
Overview
********

BDT is a second-order (i.e., post-processing) BIDS App meant to parcellate derivatives.
Our motivation for this workflow is that we want to combine derivatives from different pipelines,
such as parcellating CBF maps from ASLPrep with tract masks from QSIRecon.


*****
Usage
*****

BDT's command-line interface is designed to allow users to
provide (1) BIDS datasets containing maps to be parcellated and
(2) BIDS datasets containing atlases to use for parcellation.

For example::

    bdt dset output_dir participant \
        --datasets \
            smriprep=/path/to/smriprep \
            qsirecon=/path/to/qsirecon \
            xcpd=/path/to/xcp_d \
            aslprep=/path/to/aslprep \
            atlaspack=/path/to/atlaspack \
        --atlases \
            model-msmt:bundle-AssociationArcuateFasciculusL \  # tracts
            4S156Parcels \  # NIfTI atlas
            MyersLabonte \  # CIFTI atlas
        --scalars \
            src-xcpd:stat-alff \
            src-xcpd:stat-reho \
            src-aslprep:stat-cbf:desc-basil \
        --output-formats cifti tsv
