#!/usr/bin/env python
"""Convert the 64-map DiFuMo MNI2009c atlas to an fsLR-91k CIFTI.

The conversion has three stages:

1. Apply TemplateFlow's MNI152NLin2009cAsym -> MNI152NLin6Asym transform on
   the exact volume grid encoded by the fsLR-91k CIFTI template.
2. Sample the transformed volume onto the fsLR-32k group midthickness surfaces.
3. Combine both cortical metrics and the MNI152NLin6Asym subcortical voxels using
   the brain-model axis from a standard fsLR-91k CIFTI.

The input is probabilistic, so both volumetric resampling and surface sampling use
linear interpolation. TemplateFlow may download the transform and fsLR surfaces the
first time this script runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_INPUT = Path(
    '/mnt/c/Users/tsalo/Documents/datasets/ds008325-derivatives/'
    'cifti-probseg/tpl-MNI152NLin2009cAsym/'
    'tpl-MNI152NLin2009cAsym_res-02_atlas-DiFuMo_'
    'scale-64dimensions_probseg.nii.gz'
)
DEFAULT_OUTPUT = Path(
    '/mnt/c/Users/tsalo/Documents/datasets/ds008325-derivatives/'
    'cifti-probseg/tpl-fsLR/'
    'tpl-fsLR_atlas-DiFuMo_scale-64dimensions_den-91k_probseg.dscalar.nii'
)
DEFAULT_CIFTI_TEMPLATE = Path(
    '/mnt/c/Users/tsalo/Documents/datasets/AtlasPack/tpl-fsLR/'
    'tpl-fsLR_atlas-4S456Parcels_den-91k_dseg.dlabel.nii'
)


def _one_templateflow_file(result: object, description: str) -> Path:
    """Return one materialized TemplateFlow result or raise a useful error."""
    if result is None:
        raise RuntimeError(f'TemplateFlow did not find {description}.')
    if isinstance(result, (str, os.PathLike)):
        paths = [Path(result)]
    else:
        paths = [Path(path) for path in result]  # type: ignore[union-attr]
    if len(paths) != 1:
        raise RuntimeError(f'Expected one {description}; TemplateFlow returned {paths}.')
    if not paths[0].is_file():
        raise RuntimeError(f'TemplateFlow did not materialize {description}: {paths[0]}')
    return paths[0]


def _run(command: list[str]) -> None:
    """Print and run one external command."""
    print('+', subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def _get_templateflow_inputs() -> tuple[Path, dict[str, Path]]:
    """Fetch the MNI2009c-to-MNI6 transform and fsLR-32k surfaces."""
    from templateflow import api as tf

    transform = _one_templateflow_file(
        tf.get(
            'MNI152NLin6Asym',
            mode='image',
            suffix='xfm',
            extension='.h5',
            **{'from': 'MNI152NLin2009cAsym'},
        ),
        'MNI152NLin2009cAsym-to-MNI152NLin6Asym image transform',
    )
    surfaces = {
        hemi: _one_templateflow_file(
            tf.get(
                'fsLR',
                density='32k',
                hemi=hemi,
                suffix='midthickness',
                extension='.surf.gii',
            ),
            f'fsLR-32k hemisphere-{hemi} midthickness surface',
        )
        for hemi in ('L', 'R')
    }
    return transform, surfaces


def _warp_to_mni6(input_file: Path, reference: Path, transform: Path, output_file: Path) -> None:
    """Warp a 4D probability atlas into the standard 2-mm MNI6 grid."""
    import ants

    fixed = ants.image_read(str(reference))
    moving = ants.image_read(str(input_file))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(transform)],
        interpolator='linear',
        imagetype=3,
    )
    ants.image_write(warped, str(output_file))


def _validate_inputs(input_file: Path, cifti_template: Path, wb_command: str) -> int:
    """Validate paths and return the number of atlas maps."""
    import nibabel as nb

    if not input_file.is_file():
        raise FileNotFoundError(f'Input atlas does not exist: {input_file}')
    if not cifti_template.is_file():
        raise FileNotFoundError(f'CIFTI template does not exist: {cifti_template}')
    if shutil.which(wb_command) is None:
        raise FileNotFoundError(f'Connectome Workbench executable not found on PATH: {wb_command}')

    atlas = nb.load(input_file)
    if len(atlas.shape) != 4:
        raise ValueError(f'Expected a 4D probabilistic atlas, got shape {atlas.shape}.')
    n_maps = int(atlas.shape[3])
    if n_maps != 64:
        raise ValueError(f'Expected the 64-map DiFuMo atlas, got {n_maps} maps.')

    template = nb.load(cifti_template)
    if not isinstance(template, nb.Cifti2Image):
        raise ValueError(f'Expected a CIFTI template, got {type(template).__name__}.')
    brain_models = template.header.get_axis(1)
    if len(brain_models) != 91282:
        raise ValueError(
            f'Expected an fsLR-91k template with 91,282 brainordinates, got {len(brain_models):,}.'
        )
    return n_maps


def _set_map_names(cifti_file: Path, n_maps: int, wb_command: str) -> None:
    """Assign stable, one-based names to the CIFTI scalar maps."""
    for index in range(1, n_maps + 1):
        _run(
            [
                wb_command,
                '-set-map-name',
                str(cifti_file),
                str(index),
                f'DiFuMo-{index:03d}',
            ]
        )


def convert(
    input_file: Path,
    output_file: Path,
    cifti_template: Path,
    wb_command: str,
    work_dir: Path | None,
    keep_work_dir: bool,
) -> None:
    """Run the complete MNI2009c NIfTI -> fsLR-91k CIFTI conversion."""
    input_file = input_file.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    cifti_template = cifti_template.expanduser().resolve()
    n_maps = _validate_inputs(input_file, cifti_template, wb_command)
    transform, surfaces = _get_templateflow_inputs()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix='difumo-to-fslr-')
        working = Path(temporary.name)
    else:
        temporary = None
        working = work_dir.expanduser().resolve()
        working.mkdir(parents=True, exist_ok=True)

    try:
        mni6_reference = working / 'fslr91k_volume_space.nii.gz'
        mni6_atlas = working / 'difumo_mni6_res-2_probseg.nii.gz'
        metrics = {
            hemi: working / f'difumo_hemi-{hemi}_den-32k.func.gii' for hemi in ('L', 'R')
        }
        staged_output = working / output_file.name

        # Extract the exact volume grid from the CIFTI and use it as the ANTs
        # fixed image so Workbench accepts the warped volume for -volume-all.
        _run(
            [
                wb_command,
                '-cifti-separate',
                str(cifti_template),
                'COLUMN',
                '-volume-all',
                str(mni6_reference),
            ]
        )
        print(f'Warping {input_file} to the CIFTI volume grid', flush=True)
        _warp_to_mni6(input_file, mni6_reference, transform, mni6_atlas)

        for hemi in ('L', 'R'):
            _run(
                [
                    wb_command,
                    '-volume-to-surface-mapping',
                    str(mni6_atlas),
                    str(surfaces[hemi]),
                    str(metrics[hemi]),
                    '-trilinear',
                ]
            )

        _run(
            [
                wb_command,
                '-cifti-create-dense-from-template',
                str(cifti_template),
                str(staged_output),
                '-metric',
                'CORTEX_LEFT',
                str(metrics['L']),
                '-metric',
                'CORTEX_RIGHT',
                str(metrics['R']),
                '-volume-all',
                str(mni6_atlas),
            ]
        )
        _set_map_names(staged_output, n_maps, wb_command)

        created = __import__('nibabel').load(staged_output)
        if created.shape != (n_maps, 91282):
            raise RuntimeError(
                f'Unexpected output shape {created.shape}; expected ({n_maps}, 91282).'
            )
        # /tmp and /mnt/c are different filesystems under WSL, so an os.replace
        # directly between them fails with EXDEV. Copy to a sibling on the target
        # filesystem first, then atomically rename there.
        destination_staging = output_file.with_name(f'.{output_file.name}.tmp-{os.getpid()}')
        try:
            shutil.copyfile(staged_output, destination_staging)
            os.replace(destination_staging, output_file)
        finally:
            destination_staging.unlink(missing_ok=True)
        print(f'Wrote {output_file}', flush=True)
    finally:
        if temporary is not None:
            if keep_work_dir:
                kept = output_file.parent / f'.{output_file.name}.work'
                if kept.exists():
                    raise FileExistsError(f'Cannot preserve work directory; path exists: {kept}')
                shutil.move(working, kept)
                print(f'Kept intermediate files in {kept}', flush=True)
                temporary.cleanup = lambda: None  # type: ignore[method-assign]
            temporary.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='4D MNI2009c atlas.')
    parser.add_argument(
        '--output', type=Path, default=DEFAULT_OUTPUT, help='Output fsLR-91k dscalar.'
    )
    parser.add_argument(
        '--cifti-template',
        type=Path,
        default=DEFAULT_CIFTI_TEMPLATE,
        help='Any standard fsLR-91k CIFTI whose brain-model axis should be copied.',
    )
    parser.add_argument(
        '--wb-command', default='wb_command', help='Connectome Workbench executable.'
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        help='Directory for intermediate files (a temporary directory by default).',
    )
    parser.add_argument(
        '--keep-work-dir',
        action='store_true',
        help='Preserve an automatically created work directory next to the output.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    convert(
        input_file=args.input,
        output_file=args.output,
        cifti_template=args.cifti_template,
        wb_command=args.wb_command,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
    )
