#
# BDT Docker Container Image (pixi-based, aligned with ASLPrep)
#
# MIT License
#

ARG BASE_IMAGE=pennlinc/bdt-base:20260323

#
# Build pixi environment
#
FROM ghcr.io/prefix-dev/pixi:0.53.0 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    build-essential \
                    curl \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pixi config set --global run-post-link-scripts insecure

# Install dependencies before the package itself to leverage caching
RUN mkdir /app
COPY pixi.lock pyproject.toml /app
WORKDIR /app
# First install runs before COPY . so .git is missing.
# Use --skip bdt so pixi skips building the local package.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e bdt -e test --frozen --skip bdt
RUN --mount=type=cache,target=/root/.npm pixi run --as-is -e bdt npm install -g svgo@^3.2.0 bids-validator@1.14.10
RUN pixi shell-hook -e bdt --as-is | grep -v PATH > /shell-hook.sh
RUN pixi shell-hook -e test --as-is | grep -v PATH > /test-shell-hook.sh

# Finally, install the package
COPY . /app
# Install test and production environments separately so production does not
# inherit editable-install behavior needed for test workflows.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e test --frozen
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e bdt --frozen
# Ensure bdt is installed non-editably in the bdt env so the copied env is
# self-contained in the runtime image (lockfile may resolve to editable variant).
# Pixi envs do not include pip; use uv to install into the env's Python.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --python /app/.pixi/envs/bdt/bin/python --no-deps --force-reinstall .

#
# Main stage
#
FROM ${BASE_IMAGE} AS base

RUN useradd -m -s /bin/bash -G users bdt
WORKDIR /home/bdt
ENV HOME="/home/bdt"

RUN chmod -R go=u $HOME

ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /tmp

FROM base AS test

COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"

FROM base AS bdt

COPY --link --from=build /app/.pixi/envs/bdt /app/.pixi/envs/bdt
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/bdt/bin:$PATH"

ENV IS_DOCKER_8395080871=1

ENTRYPOINT ["/app/.pixi/envs/bdt/bin/bdt"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="bdt" \
      org.label-schema.description="BDT - A template fMRI post-processing workflow" \
      org.label-schema.url="https://github.com/nipreps/bdt" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/bdt" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
