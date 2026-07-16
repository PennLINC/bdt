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
"""Resolve and parse a BDT spec from a path or a pre-packaged name.

Mirrors QSIRecon's ``_load_recon_spec`` behaviour: an argument that is an
existing file is read directly; otherwise it is looked up among the pre-packaged
specs shipped under ``bdt/data/specs/``.  YAML and JSON are both accepted
(``yaml.safe_load`` parses JSON too).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bdt.spec.model import Spec, SpecError, parse_spec

# Pre-packaged specs live alongside the package data (may be empty until specs
# are shipped).  Resolved from ``__file__`` so this module needs no ``acres`` /
# ``bdt.data`` import and stays dependency-light.
_SPECS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'specs'
_SUFFIXES = ('.yaml', '.yml', '.json')


def prepackaged_specs() -> list[str]:
    """Names of the specs shipped under ``bdt/data/specs/`` (without suffix)."""
    if not _SPECS_DIR.is_dir():
        return []
    return sorted({p.stem for p in _SPECS_DIR.iterdir() if p.suffix in _SUFFIXES})


def _resolve(path_or_name: str | Path) -> Path:
    p = Path(path_or_name)
    if p.is_file():
        return p
    for suffix in _SUFFIXES:
        candidate = _SPECS_DIR / f'{path_or_name}{suffix}'
        if candidate.is_file():
            return candidate
    known = prepackaged_specs()
    raise SpecError(
        f'Could not resolve spec {str(path_or_name)!r}: not an existing file and not a '
        f'pre-packaged spec. Known pre-packaged specs: {known or "(none)"}.'
    )


def load_spec(path_or_name: str | Path) -> Spec:
    """Load, parse, and return a :class:`~bdt.spec.model.Spec`.

    Does *not* validate — call :func:`bdt.spec.validate.validate_spec` next.
    """
    path = _resolve(path_or_name)
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f'Failed to parse spec {path}: {exc}') from exc
    if doc is None:
        raise SpecError(f'Spec {path} is empty')
    return parse_spec(doc)
