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

The physical position of any voxel is:
```math
x = x_iv_x-\frac{l_x}{2}+\frac{v_x}{2}+o_x
```
```math
y = y_iv_y-\frac{l_y}{2}+\frac{v_y}{2}+o_y
```
```math
z = z_iv_z-\frac{l_z}{2}+\frac{v_z}{2}+o_z
```

Where $`o_d`$ is `off_d` and $`l_d`$ is `length_d` for $`d\in\{x,y,z\}`$.

