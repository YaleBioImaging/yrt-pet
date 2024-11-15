# Image parameters

The image parameters file is defined in JSON format and looks like the following:
```json
{
    "VERSION": 1.0,
    "nx": 192,
    "ny": 192,
    "nz": 89,
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
- `vx`, `vy`, `vz` is the voxel size in X, Y, and Z in millimeters
- `off_x`, `off_y`, `off_z` is the X, Y and Z position of the *center* of the image
Optionally, one can also specify:
- `length_x`, `length_y`, `length_z` to define the physical size of the image in millimeters.
  - If this is not defined, the length is computed as `nx*vx`, `ny*vy`, `nz*vz`.

The physical position of any voxel in the X dimension is:
```math
x_p = x_iv_x-\frac{l_x}{2}+\frac{v_x}{2}+o_x
```
Where $`o_x`$ is `off_x` and $`l_x`$ is `length_x`. $`x_i`$ is the *logical* index of the voxel
while $`x_p`$ is the *physical* position (in millimeters) of the voxel in dimension X.
The above equation also applies in dimensions Y and Z.

# Image format
Images are stored and read in NIfTI format. This is the main reason SimpleITK is a dependency.
This means that image positioning information is potentially repeated (and therefore potentially contradicted).
This is why the reconstruction executable does not require an image parameters file if the sensitivity image is
already provided.

## For Python users and plugin developers
Note that the `ImageOwned` class has three constructors. One only with an `ImageParams` object, one only with
an `std::string` for the filename, and one with both. The one that only takes a filename deduces the image parameters
from the NIfTI header while the one that takes both the NIfTI file and the `ImageParams` object uses the latter to
perform a consistency check in order to avoid mismatches.
