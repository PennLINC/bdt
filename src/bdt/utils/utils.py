# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""Utility functions for BDT."""

import logging

LGR = logging.getLogger(__name__)


def _get_wf_name(bold_fname, prefix):
    """Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz", "template")
    'template_task_nback_wf'
    >>> _get_wf_name(
    ...     "/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz",
    ...     "preproc",
    ... )
    'preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = '_'.join(fname.split('_')[1:-1])
    return f'{prefix}_{fname_nosub.replace("-", "_")}_wf'


def update_dict(orig_dict, new_dict):
    """Update dictionary with values from another dictionary.

    Parameters
    ----------
    orig_dict : dict
        Original dictionary.
    new_dict : dict
        Dictionary with new values.

    Returns
    -------
    updated_dict : dict
        Updated dictionary.
    """
    updated_dict = orig_dict.copy()
    for key, value in new_dict.items():
        if (orig_dict.get(key) is not None) and (value is not None):
            print(f'Updating {key} from {orig_dict[key]} to {value}')
            updated_dict[key].update(value)
        elif value is not None:
            updated_dict[key] = value

    return updated_dict


def get_col(df, col):
    """Get a column from a dataframe, with support for `_hash-<hash>` suffixes.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        The dataframe to get the column from.
    col : :obj:`str`
        The column name to get. The actual column name in the DataFrame may have a `_hash-<hash>`
        suffix.

    Returns
    -------
    :obj:`pandas.Series`
        The column from the dataframe.
    """
    import re

    # Pattern to match the base name with optional hash suffix
    # The hash can contain alphanumeric characters and plus signs
    pattern = f'^{re.escape(col)}(_hash-[0-9a-zA-Z+]+)?$'

    # Find columns that match the pattern
    matching_cols = [c for c in df.columns if re.match(pattern, c)]

    if not matching_cols:
        raise ValueError(
            f'No column found matching pattern "{pattern}" in DataFrame. '
            f'Available columns: {list(df.columns)}'
        )

    # If multiple columns match, prefer the one without hash suffix
    # (exact match first, then hash variants)
    exact_match = [c for c in matching_cols if c == col]
    if exact_match:
        return df[exact_match[0]]
    else:
        # If no exact match, use the first hash variant
        return df[matching_cols[0]]
