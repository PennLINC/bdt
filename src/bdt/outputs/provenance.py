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
"""Provenance sidecar construction (BIDS ``GeneratedBy`` + ``Sources``).

The filename ``desc`` only has to be unique and recognizable; the authoritative
record of *what produced a file from what* lives in the JSON sidecar, per the
user-stories spec, section 1.7.
"""

from __future__ import annotations


def bids_uri(dataset: str, relpath: str) -> str:
    """A BIDS URI ``bids:<dataset>:<relative/path>`` for a ``Sources`` entry."""
    return f'bids:{dataset}:{relpath.lstrip("/")}'


def generated_by(node_name: str, action: str, parameters: dict | None = None) -> dict:
    """One ``GeneratedBy`` record describing the node that produced an output."""
    return {
        'Name': 'bdt',
        'Node': node_name,
        'Action': action,
        'Parameters': dict(parameters or {}),
    }


def build_sidecar(
    sources: list[str],
    generated_by_records: list[dict],
    extra: dict | None = None,
) -> dict:
    """Assemble a BIDS derivative sidecar dict.

    ``sources`` are ``bids:``-style URIs; ``generated_by_records`` come from
    :func:`generated_by`.  ``extra`` merges action-specific metadata on top.
    """
    sidecar: dict = {}
    if extra:
        sidecar.update(extra)
    sidecar['GeneratedBy'] = list(generated_by_records)
    sidecar['Sources'] = list(dict.fromkeys(sources))  # dedup, order-preserving
    return sidecar
