# Image

## Image parameters

The image parameters file is defined in JSON format and looks like the
following:

```json
{
  "VERSION": 1.0,
  "nx": 192,
  "ny": 192,
  "nz": 89,
  "nt" : 1,
  "vx": 2.0,
  "vy": 2.0,
  "vz": 2.8,
  "off_x": 0.0,
  "off_y": 0.0,
  "off_z": 0.0
}
```

The properties are:

- `nx`, `ny`, `nz` is the image size in X, Y, and Z in number of voxels
- `nt` is the time dimension, which is used for dynamic reconstructions
- `vx`, `vy`, `vz` is the voxel size in X, Y, and Z in millimeters
- `off_x`, `off_y`, `off_z` is the X, Y and Z position of the *center* of the
  image.
- Optionally, one can also specify:
    - `length_x`, `length_y`, `length_z` to define the physical size of the
      image in
      millimeters.
    - If this is not defined, the lengths are computed
      as `nx*vx`, `ny*vy`, `nz*vz` respectively.

The physical position of any voxel in the X dimension is:

$$
x_p = \left(x_i - \frac{n_x - 1}{2}\right) v_x + o_x
$$

Where $o_x$ is `off_x` and $n_x$ is `nx`. $x_i$ is the *logical*
index of the voxel while $x_p$ is the *physical* position (in millimeters)
of the voxel in dimension X. The above equation also applies in dimensions
Y and Z.

## Image file format

Images are stored and read in NIfTI format.
Since NifTI files include image orientation, origin and pixel
spacing, The following requirements are imposed:
- Input images (e.g. attenuation map, sensitivity image) must have an identity
  orientation matrix (no rotation).
- In the reconstruction script, the origin/pixel size information is read from
  an image parameter file (described above), unless a sensitivity image is
  provided (with `--sens`), in which case the orientation/origin/pixel size from
  the NifTI sensitivity image is used.

## Fourth Dimension (Time)

The image can have a fourth dimension for time, enabling dynamic PET
reconstructions.
The `nt` field in `ImageParams` sets the number of temporal frames.
This is used in conjunction with {doc}`dynamic-framing` to reconstruct an image
for each time range.

The image dimensions are, from least contiguous to most contiguous,
`(nt, nz, ny, nx)`. Note that time is the first dimension (`nt`).

### Notes
When using dynamic framing:
- Each frame is reconstructed separately (i.e., independently of other frames) but
  the iterative updates are calculated for all frames simultaneously.
- Sensitivity images can also be 4D when performing a dynamic reconstruction.
- YRT-PET fully supports images with up to $2^{63}$ voxels in total
  - However, YRT-PET does not support images that have more than $2^{31}$ voxels
  in one dimension.

## For Python users and plugin developers

Note that the `ImageOwned` class has three constructors. One with a single
`ImageParams` object, one with a single `string` for the filename, and
one with both. The one that only takes a filename deduces the image parameters
from the NIfTI header while the one that takes both the NIfTI file and
the `ImageParams` object uses the latter to perform a consistency check in order
to avoid mismatches.
