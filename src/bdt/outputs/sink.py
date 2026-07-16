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
"""BIDS-derivative naming and the write sink.

Composes BIDS filenames from short-name entity dicts in canonical order, applies
the ``desc`` prepend-compose rule, materializes a node's output plus its JSON
sidecar, and raises on any two outputs that resolve to the same path.  Kept
free of a pybids dependency so the naming logic is unit-testable on its own; the
canonical entity order matches BIDS + the BDT/BEP atlas-derivative ordering
(``... hemi space atlas seg stat scale res den label desc``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Canonical short-name entity order for BDT outputs.  ``sub``/``ses`` (participant)
# and ``space`` (folded into ``tpl-`` for dataset scope) form the leading token and
# are handled separately in :meth:`DerivativeSink.relpath`.
ENTITY_ORDER = (
    'sub',
    'ses',
    'task',
    'acq',
    'ce',
    'rec',
    'dir',
    'run',
    'model',
    'param',
    'hemi',
    'space',
    'atlas',
    'seg',
    'stat',
    'scale',
    'res',
    'den',
    'label',
    'desc',
)


class OutputCollisionError(ValueError):
    """Two materialized outputs resolve to the same path."""


def compose_desc(existing: str | None, node_desc: str | None) -> str | None:
    """Prepend-compose a ``desc`` token (spec 1.7).

    ``output_desc = existing + Capitalize(node_desc)`` as one BIDS-legal camelCase
    alphanumeric token, e.g. ``'geneexpression'`` + ``'strict'`` ->
    ``'geneexpressionStrict'``.  Either side may be ``None``.
    """
    if node_desc is None:
        return existing
    node_desc = str(node_desc)
    if not existing:
        return node_desc
    return f'{existing}{node_desc[:1].upper()}{node_desc[1:]}'


def bids_name(prefix: str, entities: dict, suffix: str, extension: str) -> str:
    """Build ``<prefix>[_<k>-<v> ...]_<suffix><extension>`` in canonical order.

    ``prefix`` is the leading token (``sub-01`` / ``sub-01_ses-1`` / ``tpl-fsLR``);
    ``entities`` is a short-name dict (leading-token keys already removed).
    ``extension`` includes its leading dot (e.g. ``.tsv``, ``.dscalar.nii``).
    """
    parts = [prefix]
    for key in ENTITY_ORDER:
        if key in entities and entities[key] is not None:
            parts.append(f'{key}-{entities[key]}')
    stem = '_'.join(parts)
    return f'{stem}_{suffix}{extension}'


class DerivativeSink:
    """Materialize node outputs into a BIDS-derivative tree with provenance."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._written: dict[str, str] = {}  # relpath -> node name that wrote it

    def relpath(
        self, entities: dict, suffix: str, extension: str, datatype: str, scope: str
    ) -> str:
        """Resolve the output path relative to the derivatives root.

        Participant scope -> ``sub-<sub>[/ses-<ses>]/<datatype>/<name>``; dataset
        scope -> ``tpl-<space>/<datatype>/<name>`` (``space`` becomes the ``tpl-``
        template label and is dropped from the mid-filename entities).
        """
        entities = dict(entities)
        if scope == 'dataset':
            space = entities.pop('space', None)
            if not space:
                raise ValueError(
                    "Dataset-scope output requires a 'space' entity to form the tpl- folder."
                )
            for k in ('sub', 'ses'):
                entities.pop(k, None)
            prefix = f'tpl-{space}'
            folder = prefix
        else:
            sub = entities.get('sub')
            if not sub:
                raise ValueError("Participant-scope output requires a 'sub' entity.")
            ses = entities.get('ses')
            prefix_parts = [f'sub-{sub}'] + ([f'ses-{ses}'] if ses else [])
            prefix = '_'.join(prefix_parts)
            folder = f'sub-{sub}' + (f'/ses-{ses}' if ses else '')
            # sub/ses are in the leading token; drop from mid-entities
            entities = {k: v for k, v in entities.items() if k not in ('sub', 'ses')}
        name = bids_name(prefix, entities, suffix, extension)
        return f'{folder}/{datatype}/{name}'

    def write(
        self,
        *,
        node_name: str,
        in_file: str | Path,
        entities: dict,
        suffix: str,
        extension: str,
        datatype: str,
        scope: str = 'participant',
        sidecar: dict | None = None,
    ) -> Path:
        """Copy ``in_file`` to its resolved BIDS path and write its JSON sidecar."""
        rel = self.relpath(entities, suffix, extension, datatype, scope)
        if rel in self._written:
            raise OutputCollisionError(
                f'Output path {rel!r} is produced by both node {self._written[rel]!r} and '
                f'node {node_name!r}. Add a disambiguating desc: to one of them.'
            )
        self._written[rel] = node_name

        dest = self.output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(in_file, dest)

        if sidecar is not None:
            sidecar_path = dest.parent / f'{_stem(dest.name, extension)}.json'
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=False))
        return dest


def _stem(name: str, extension: str) -> str:
    """Strip a (possibly multi-part) extension from a filename."""
    if extension and name.endswith(extension):
        return name[: -len(extension)]
    return name.rsplit('.', 1)[0]
