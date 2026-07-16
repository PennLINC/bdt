# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Configure the NiWrap / Styx execution backend.

BDT runs inside a container that already has ``wb_command``, ANTs, DSI Studio,
and the Rust binaries on ``PATH`` (see the container plan), so the default
backend is the *local* runner: NiWrap shells out to those on-``PATH`` tools and
writes intermediates under the work directory.  A ``docker`` backend is exposed
for running the tools from images when they are not installed locally.
"""

from __future__ import annotations

from pathlib import Path


def configure_runner(
    work_dir: str | Path | None = None,
    *,
    engine: str = 'local',
    environ: dict[str, str] | None = None,
    docker_image: str | None = None,
):
    """Set NiWrap's global execution runner and return it.

    Parameters
    ----------
    work_dir
        Directory NiWrap uses for tool intermediates (the local runner's
        ``data_dir``).
    engine
        ``'local'`` (default; tools resolved from ``PATH``) or ``'docker'``.
    environ
        Extra environment variables to expose to invoked tools.
    docker_image
        Image to use when ``engine='docker'``.
    """
    import niwrap

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    if engine == 'local':
        return niwrap.use_local(data_dir=str(work_dir) if work_dir else None, environ=environ)
    if engine == 'docker':
        kwargs: dict = {}
        if docker_image:
            kwargs['image_overrides'] = {'*': docker_image}
        return niwrap.use_docker(data_dir=str(work_dir) if work_dir else None, **kwargs)
    raise ValueError(f'Unknown NiWrap engine {engine!r} (expected "local" or "docker").')
