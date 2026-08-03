bdt \
    /mnt/c/Users/tsalo/Documents/datasets/ds008325 \
    /mnt/c/Users/tsalo/Documents/datasets/ds008325-derivatives/bdt-nifti-parcellate \
    participant \
    --spec nifti_parcellate.yml \
    --participant-label 125511 \
    --datasets \
    fmriprep=/mnt/c/Users/tsalo/Documents/datasets/ds008325/derivatives/fmriprep_func \
    atlases=/mnt/c/Users/tsalo/Documents/datasets/AtlasPack \
    aslprep=/mnt/c/Users/tsalo/Documents/datasets/ds008325/derivatives/aslprep \
    difumo=/mnt/c/Users/tsalo/Documents/datasets/ds008325-derivatives/cifti-probseg \
    --work-dir /mnt/c/Users/tsalo/Documents/datasets/ds008325-derivatives/bdt-nifti-parcellate-work \
    --clean-workdir \
    --stop-on-first-crash
