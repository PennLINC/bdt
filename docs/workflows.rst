.. include:: links.rst

###########################
Processing pipeline details
###########################

*BDT* adapts its pipeline to the **geometry** of the data being parcellated and
to the file format of each derivative. It reads the spec file, collects the
matching files for each source entry, and dispatches each one to a data-type
workflow based on the requested operations.


*******************************
Geometry-aware transforms
*******************************

There is no single "right direction" to move data and atlases into a shared
space — it depends on the geometry of the data:

- **Grid data** (volumetric NIfTI, CIFTI dense): warping the data would
  re-interpolate signal on every step, so *BDT* warps the static label atlas
  **into the data's grid** once, using ``GenericLabel`` interpolation.
- **Point data** (GIFTI vertices, TRX/TCK streamlines): a surface or tractogram
  is a set of coordinates, and warping coordinates is lossless. So *BDT* warps
  the **points** (or atlas vertices) rather than a coarse label volume, which
  would lose small parcels under nearest-neighbor resampling.
- **Diffusion-model grid** (SH/ODF/fixels): resampled like grid data, but
  additionally requires SH reorientation (and, for fixels, cardinality handling).

This yields three transform modalities:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Data geometry
     - Operation
     - Interpolation concern
   * - Grid (NIfTI vol, CIFTI dense)
     - resample atlas → data grid
     - yes — ``GenericLabel``
   * - Points (GIFTI verts, TCK/TRX)
     - warp data coords (or atlas verts)
     - none — lossless
   * - Diffusion-model grid (SH/ODF/fixels)
     - resample + SH reorient
     - yes, + fixel cardinality

Transforms are discovered automatically from the ``*_xfm.*`` files in the
provided derivative datasets and supplemented with TemplateFlow
standard-to-standard transforms. The engine distinguishes an *image resample*
(pull semantics, for grid data) from a *point warp* (opposite direction, for
point data) so a chain is never applied backwards. If no transform path exists
between an atlas's space and the data's space, *BDT* raises an explicit error
listing the available spaces.


*******************************
File-format dispatch
*******************************

The format of each source file is detected from its extension at
workflow-build time and used to pick the parcellation tool:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Extension
     - Format
     - Parcellation tool
   * - ``.nii.gz``
     - NIfTI volumetric
     - ANTs ``ApplyTransforms`` (atlas → grid) + nilearn ``NiftiLabelsMasker``
   * - ``.func.gii`` / ``.shape.gii``
     - GIFTI surface
     - Connectome Workbench (surface atlas; ``-label-resample`` if cross-mesh)
       **or** warp surface vertices + sample a volume atlas
   * - ``.dtseries.nii`` / ``.dscalar.nii``
     - CIFTI dense
     - Connectome Workbench
   * - ``.trx`` / ``.tck`` (``.trk`` → convert)
     - Streamlines
     - warp tracts → atlas + endpoint connectivity
   * - SH/ODF/fixel
     - Diffusion model
     - resample + SH reorient, then per-parcel ODF/fixel summary


*******************************
Example workflow graph
*******************************

.. note::
   A rendered graph of the single-subject workflow will be added here once the
   workflow-build test harness (``mock_config``) and example fixtures are in
   place. *BDT* is currently pre-alpha and the workflows are still being
   implemented.
