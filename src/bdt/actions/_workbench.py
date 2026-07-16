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
"""Thin wrappers over the Connectome-Workbench calls used by the builders.

Each function shells out via NiWrap (``niwrap.workbench.*``), which resolves
``wb_command`` from ``PATH`` under the configured runner and returns the output
path.  ``niwrap`` is imported lazily so the rest of BDT loads without it, and so
these are exercisable under NiWrap's dry runner for command-construction tests.
"""

from __future__ import annotations

from pathlib import Path


def run_cifti_parcellate(
    cifti_in: str | Path,
    atlas: str | Path,
    out: str | Path,
    *,
    direction: str = 'COLUMN',
    only_numeric: bool = True,
    method: str | None = None,
    runner=None,
) -> str:
    """``wb_command -cifti-parcellate`` — parcellate a dense CIFTI with a dlabel."""
    from niwrap import workbench

    result = workbench.cifti_parcellate(
        cifti_in=str(cifti_in),
        cifti_label=str(atlas),
        direction=direction,
        cifti_out=str(out),
        only_numeric=only_numeric,
        method=method,
        runner=runner,
    )
    return result.cifti_out


def run_cifti_correlation(
    cifti: str | Path,
    out: str | Path,
    *,
    covariance: bool = False,
    fisher_z: bool = False,
    runner=None,
) -> str:
    """``wb_command -cifti-correlation`` — correlate a parcellated series (pconn)."""
    from niwrap import workbench

    result = workbench.cifti_correlation(
        cifti=str(cifti),
        cifti_out=str(out),
        covariance=covariance,
        fisher_z=fisher_z,
        runner=runner,
    )
    return result.cifti_out


def run_volume_to_surface_mapping(
    volume: str | Path,
    midthickness: str | Path,
    white: str | Path,
    pial: str | Path,
    out: str | Path,
    runner=None,
) -> str:
    """``wb_command -volume-to-surface-mapping -ribbon-constrained`` (white..pial)."""
    from niwrap import workbench

    ribbon = workbench.volume_to_surface_mapping_ribbon_constrained_params(
        inner_surf=str(white), outer_surf=str(pial)
    )
    result = workbench.volume_to_surface_mapping(
        volume=str(volume),
        surface=str(midthickness),
        metric_out=str(out),
        ribbon_constrained=ribbon,
        runner=runner,
    )
    return result.metric_out


def run_metric_dilate(
    metric: str | Path,
    surface: str | Path,
    out: str | Path,
    *,
    distance: float = 10.0,
    nearest: bool = True,
    runner=None,
) -> str:
    """``wb_command -metric-dilate`` — fill unmapped vertices (nearest)."""
    from niwrap import workbench

    result = workbench.metric_dilate(
        metric=str(metric),
        surface=str(surface),
        distance=distance,
        metric_out=str(out),
        nearest=nearest,
        runner=runner,
    )
    return result.metric_out


def run_metric_resample(
    metric_in: str | Path,
    current_sphere: str | Path,
    new_sphere: str | Path,
    current_area: str | Path,
    new_area: str | Path,
    out: str | Path,
    *,
    method: str = 'ADAP_BARY_AREA',
    runner=None,
) -> str:
    """``wb_command -metric-resample`` (area-corrected) — move a metric to a new mesh."""
    from niwrap import workbench

    areas = workbench.metric_resample_area_surfs_params(
        current_area=str(current_area), new_area=str(new_area)
    )
    result = workbench.metric_resample(
        metric_in=str(metric_in),
        current_sphere=str(current_sphere),
        new_sphere=str(new_sphere),
        method=method,
        metric_out=str(out),
        area_surfs=areas,
        runner=runner,
    )
    return result.metric_out


def run_cifti_create_dense_scalar(
    out: str | Path,
    *,
    left_metric: str | Path | None = None,
    right_metric: str | Path | None = None,
    volume_data: str | Path | None = None,
    structure_label_volume: str | Path | None = None,
    runner=None,
) -> str:
    """``wb_command -cifti-create-dense-scalar`` — staple L/R metrics (+ subcortex)."""
    from niwrap import workbench

    kwargs: dict = {}
    if left_metric is not None:
        kwargs['left_metric'] = workbench.cifti_create_dense_scalar_left_metric_params(
            metric=str(left_metric)
        )
    if right_metric is not None:
        kwargs['right_metric'] = workbench.cifti_create_dense_scalar_right_metric_params(
            metric=str(right_metric)
        )
    if volume_data is not None and structure_label_volume is not None:
        kwargs['volume'] = workbench.cifti_create_dense_scalar_volume_params(
            volume_data=str(volume_data), structure_label_volume=str(structure_label_volume)
        )
    result = workbench.cifti_create_dense_scalar(
        cifti_out=str(out), metric=[], runner=runner, **kwargs
    )
    return result.cifti_out
