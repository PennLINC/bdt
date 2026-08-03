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
"""Dataset-level BIDS-derivative description (``dataset_description.json``).

Writes the derivatives-root ``dataset_description.json`` with a ``DatasetLinks``
dict so the ``bids:<dataset_key>:...`` ``Sources`` URIs emitted in per-file
sidecars resolve, and a ``GeneratedBy`` chain aggregated from the input
derivative datasets with BDT's own record prepended.  Kept free of a pybids
dependency, like the rest of :mod:`bdt.outputs`.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

BIDS_VERSION = '1.10.0'
_TEMPLATEFLOW_URL = 'https://github.com/templateflow/templateflow'


def dataset_generated_by() -> dict:
    """The dataset-level BDT provenance record for ``GeneratedBy[0]``.

    Distinct in shape from :func:`bdt.outputs.provenance.generated_by`, which is
    the per-*node* sidecar record.  Adds a ``Container`` sub-dict from the
    ``BDT_DOCKER_TAG`` / ``BDT_SINGULARITY_URL`` environment variables when set.
    """
    from bdt import __version__

    record: dict = {
        'Name': 'BDT',
        'Version': __version__,
        'CodeURL': f'https://github.com/nipreps/bdt/archive/{__version__}.tar.gz',
    }
    docker_tag = os.environ.get('BDT_DOCKER_TAG')
    singularity_url = os.environ.get('BDT_SINGULARITY_URL')
    if docker_tag:
        record['Container'] = {'Type': 'docker', 'Tag': f'nipreps/bdt:{docker_tag}'}
    elif singularity_url:
        record['Container'] = {'Type': 'singularity', 'URI': singularity_url}
    return record


def _inherited_generated_by(datasets: dict[str, str | Path]) -> list[dict]:
    """``GeneratedBy`` entries aggregated from input derivatives (order-preserving dedup)."""
    collected: list[dict] = []
    for key, root in datasets.items():
        if key == 'templateflow':
            continue
        desc_path = Path(root) / 'dataset_description.json'
        try:
            desc = json.loads(desc_path.read_text())
        except (OSError, ValueError):
            continue  # missing or malformed input description -> skip (best effort)
        if not isinstance(desc, dict):
            continue
        entries = desc.get('GeneratedBy')
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if entry not in collected:
                collected.append(entry)
    return collected


def write_dataset_description(
    output_dir: str | Path,
    datasets: dict[str, str | Path],
    bids_dir: str | Path | None = None,
    name: str = 'BDT derivatives',
) -> Path:
    """Write a BIDS-derivative ``dataset_description.json`` at ``output_dir``.

    ``datasets`` is the ``{key: root}`` mapping from ``--datasets``; each key
    becomes a ``DatasetLinks`` entry (absolute path, or the canonical URL for
    ``templateflow``).  When ``bids_dir`` is given it is linked as ``raw`` unless
    a ``--datasets`` key already claims that name.  Overwrites any existing file.
    """
    dataset_links: dict[str, str] = {}
    for key, root in datasets.items():
        if key == 'templateflow':
            dataset_links[key] = _TEMPLATEFLOW_URL
        else:
            dataset_links[key] = os.path.abspath(str(root))
    if bids_dir is not None:
        if 'raw' in dataset_links:
            warnings.warn(
                "A --datasets key named 'raw' shadows the raw bids_dir link; "
                'skipping the bids_dir link.',
                stacklevel=2,
            )
        else:
            dataset_links['raw'] = os.path.abspath(str(bids_dir))

    generated_by = [dataset_generated_by(), *_inherited_generated_by(datasets)]

    description = {
        'Name': name,
        'BIDSVersion': BIDS_VERSION,
        'DatasetType': 'derivative',
        'HowToAcknowledge': 'Include the generated boilerplate in the methods section.',
        'GeneratedBy': generated_by,
        'DatasetLinks': dataset_links,
    }

    dest = Path(output_dir) / 'dataset_description.json'
    dest.write_text(json.dumps(description, indent=2))
    return dest
