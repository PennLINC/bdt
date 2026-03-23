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

**********
Packaging
**********

- Docker builds follow ASLPrep’s pixi-based multi-stage pattern:

  - `Dockerfile.base` builds the runtime base (AFNI, Workbench, system libs) and is tagged as ``pennlinc/bdt-base:<YYYYMMDD>``.
  - `Dockerfile` creates `test` and `bdt` targets from a pixi-managed environment resolved via `pixi.lock`.
  - Bump the base date tag in `Dockerfile` to trigger a base rebuild in CI when needed.

- Pixi configuration lives in `pyproject.toml` under `[tool.pixi.*]`:

  - Environments: `default` (editable), `test` (editable + tests), and `bdt` (production + container).
  - Platforms: `linux-64` only; lockfile updates must run on Linux.

- Lockfile management:

  - A GitHub Action `.github/workflows/pixi-lock.yml` updates `pixi.lock` automatically on pull requests that modify `pyproject.toml` or `pixi.lock`.
  - For fork PRs, the workflow does not push changes; update the lockfile on Linux and push from the contributor branch.

- CircleCI (`.circleci/config.yml`):

  - `image_prep` builds or reuses the `pennlinc/bdt:test` image, keyed by checksums of `Dockerfile` and `pixi.lock`.
  - `unit_tests` run inside the test image; `build_and_deploy_docker` builds and optionally pushes `pennlinc/bdt:latest` (and tag) on `main`/tags.

Local Docker testing (Linux only)::

    docker build -f Dockerfile.base -t pennlinc/bdt-base:$(date +%Y%m%d) .
    docker build --target bdt -t pennlinc/bdt:dev .
