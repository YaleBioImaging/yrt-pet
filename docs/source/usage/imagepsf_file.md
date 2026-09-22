# Image-Based PSF File Format

YRT-PET supports image-based point spread function (PSF) input files in two
forms:

- **Uniform PSF kernel**: three separable 1D kernels, used at every voxel.
- **Spatially variant PSF kernel (PSF Look-Up Table or LUT)**: a regular grid
  of single Gaussian or dual Gaussian 3D kernels.

Both formats use standard **CSV (comma-separated values)** files.

## Selecting the PSF Format

| Image-space model | Command line option | Python OSEM mode | File contents |
|-------------------|---------------------|------------------|---------------|
| Uniform | `--psf uniform_psf.csv` | `yrt.ImagePSFMode.UNIFORM` (default) | Four rows containing separable kernels and their sizes |
| Spatially variant, single Gaussian | `--varpsf single_gaussian.csv` | `yrt.ImagePSFMode.VARIANT` | Three columns |
| Spatially variant, dual Gaussian | `--varpsf dual_gaussian.csv` | `yrt.ImagePSFMode.VARIANT` | Seven columns |

For a variant LUT, the reader **automatically selects single or dual Gaussian
from the number of CSV columns**.

---

## Uniform PSF Kernel

### Generation

A utility script is provided to generate uniform PSF kernel files:

```
yrt-pet/scripts/utils/generate_psf_kernel.py
```

```bash
generate_psf_kernel.py [-h] --fx FX --fy FY --fz FZ --vx VX --vy VY --vz VZ
                       [--size_x SIZE_X] [--size_y SIZE_Y] [--size_z SIZE_Z]
                       -o OUTPUT
```

Arguments:

- `--fx FX`: FWHM (full width at half maximum) in X direction (in mm)
- `--fy FY`: FWHM in Y direction
- `--fz FZ`: FWHM in Z direction
- `--vx VX`, `--vy VY`, `--vz VZ`: Voxel sizes in X, Y, and Z (in mm)
- `--size_x SIZE_X`, `--size_y SIZE_Y`, `--size_z SIZE_Z`: (Optional) Kernel
  sizes in X, Y, and Z (must be odd)
- `-o OUTPUT`, `--output OUTPUT`: Output CSV file path

Output Format:

The resulting CSV file contains:

```
1D convolution kernel in X direction
1D convolution kernel in Y direction
1D convolution kernel in Z direction
Kernel sizes in X, Y, and Z directions
```

This file can be used in YRT-PET reconstruction workflows that support uniform
image-based PSF kernels.

---

## Spatially Variant PSF Kernel (PSF LUT)

Each LUT entry defines an axis-aligned, centered 3D Gaussian or a mixture of
two such Gaussians. The widths may differ in X, Y, and Z and between grid
locations. Dual Gaussian refers to **two 3D components**, not a 2D kernel.

### Assumptions and Behavior

1. **Symmetry**: PSF kernels are symmetric in X, Y, and Z. Distance to the center is treated as absolute.
2. **Regular Grid**: Ranges are nonnegative and gaps must be positive, in mm.
   Each “gap” defines the spacing between kernel locations, and the specified “range” must be divisible by the gap. The
   number of samples along each axis is `floor(range / gap) + 1`.
3. **Interpolation**: Nearest-neighbor interpolation is used to determine which kernel to apply. Out-of-range queries fall back to edge values. There is no interpolation of widths or mixture weights.
4. **Order**: PSF kernels are stored in the order: X → Y → Z.

### PSF LUT CSV Format

The first three rows contain numeric metadata:

| Row | First three values | Units |
|-----|--------------------|-------|
| 1 | `range_x,range_y,range_z`: maximum offset from the center at which PSF kernels are sampled along each axis | mm |
| 2 | `gap_x,gap_y,gap_z`: spacing between adjacent PSF kernel locations along each axis | mm |
| 3 | `nStd_x,nStd_y,nStd_z`: number of Gaussian standard deviations included in the kernel support along each axis | Dimensionless |

All remaining rows contain kernel parameters. Use plain numeric values,
with no column-name header, blank lines, comments, missing fields, or extra
columns.

#### Single Gaussian (Three Columns)

Each kernel row contains `sigma_x,sigma_y,sigma_z`, in mm. These are standard
deviations, not FWHM values or voxel counts. For an individual Gaussian,
`sigma = FWHM / 2.3548`.

For ranges of 50 mm and gaps of 50 mm in all directions, the eight entries
are ordered as `(0,0,0)`, `(50,0,0)`, `(0,50,0)`, `(50,50,0)`, `(0,0,50)`,
`(50,0,50)`, `(0,50,50)`, `(50,50,50)`. A complete example is:

```text
50,50,50
50,50,50
4,4,4
1.0,1.0,1.0
1.2,1.0,1.0
1.0,1.2,1.0
1.2,1.2,1.0
1.0,1.0,1.4
1.2,1.0,1.4
1.0,1.2,1.4
1.2,1.2,1.4
```

#### Dual Gaussian (Seven Columns)

Each kernel row contains, in order:

```text
sigma_x1,sigma_y1,sigma_z1,sigma_x2,sigma_y2,sigma_z2,weight2
```

- The first three values are the standard deviations of component 1 in mm.
- The next three values are the standard deviations of component 2 in mm.
- `weight2` is the fraction assigned to component 2, between 0 and 1 inclusive.
  Component 1 has weight `1 - weight2`; no separate first weight is supplied.
- All six standard deviations must be positive, including when `weight2` is
  0 or 1. The code does not require component 2 to be broader than component 1.

**Pad the first three metadata rows with four zeros** to make seven columns.
Only their first three values are used. With the same grid and entry order
as the single Gaussian example, a complete dual Gaussian LUT is:

```text
50,50,50,0,0,0,0
50,50,50,0,0,0,0
4,4,4,0,0,0,0
1.0,1.0,1.0,2.0,2.0,2.0,0.20
1.2,1.0,1.0,2.4,2.0,2.0,0.25
1.0,1.2,1.0,2.0,2.4,2.0,0.25
1.2,1.2,1.0,2.4,2.4,2.0,0.30
1.0,1.0,1.4,2.0,2.0,2.8,0.20
1.2,1.0,1.4,2.4,2.0,2.8,0.25
1.0,1.2,1.4,2.0,2.4,2.8,0.25
1.2,1.2,1.4,2.4,2.4,2.8,0.30
```

The example parameters illustrate the format; replace them with parameters
appropriate to your scanner and acquisition.

Each component is sampled and normalized separately on the same finite 3D
support. The combined kernel is

$$
K = (1-w_2)\frac{G_1}{\sum G_1} + w_2\frac{G_2}{\sum G_2}.
$$

Thus `weight2` is a mixture fraction after discrete normalization, not a
ratio of peak amplitudes. The resulting kernel is nonnegative and sums to one.
