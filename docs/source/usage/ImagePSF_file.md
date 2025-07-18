# Image-Based PSF File Format

YRT-PET supports image-based point spread function (PSF) input files in two forms:

- **Uniform PSF kernel**
- **Spatially variant PSF kernel (PSF Look-Up Table or LUT)**

Both formats use standard **CSV (comma-separated values)** files.

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
- `--size_x SIZE_X`, `--size_y SIZE_Y`, `--size_z SIZE_Z`: (Optional) Kernel sizes in X, Y, and Z (must be odd)  
- `-o OUTPUT`, `--output OUTPUT`: Output CSV file path  

Output Format:

The resulting CSV file contains:

```
1D convolution kernel in X direction  
1D convolution kernel in Y direction  
1D convolution kernel in Z direction  
Kernel sizes in X, Y, and Z directions
```

This file can be used in YRT-PET reconstruction workflows that support uniform image-based PSF kernels.

---

## Spatially Variant PSF Kernel (PSF LUT)

YRT-PET also supports **spatially varying 3D symmetric Gaussian kernels**, organized in a structured CSV file (PSF LUT).

### Assumptions and Behavior

1. **Symmetry**: PSF kernels are symmetric in X, Y, and Z. Distance to the center is treated as absolute.
2. **Regular Grid**: PSF kernels are placed on a uniform grid. Each “gap” defines the spacing between kernel locations, and the specified "range" must be divisible by the gap.
3. **Interpolation**: Nearest-neighbor interpolation is used to determine which kernel to apply. Out-of-range queries fall back to edge values.
4. **Order**: PSF kernels are stored in the order: X → Y → Z.

### PSF LUT CSV Format

```
X,Y,Z range of PSF kernel grid in mm (max offset from center), float  
X,Y,Z gap of PSF kernel grid in mm (spacing between kernels), float  
X,Y,Z kernel size control (determines how many sigmas are included in kernel), float  
SigmaX1,SigmaY1,SigmaZ1  # (kernel at 0,0,0)
SigmaX2,SigmaY2,SigmaZ2  # (kernel at xgap,0,0)
SigmaX3,SigmaY3,SigmaZ3  # (kernel at xgap*2,0,0)
...
```

### Example

For:  
- XYZ range = 50 mm  
- XYZ gap = 50 mm  
- XYZ kernel size control = 4  
Kernel location coding:
(0,0,0)
(50,0,0)
(0,50,0)
(50,50,0)
(0,0,50)
(50,0,50)
(0,50,50)
(50,50,50)
```
50,50,50
50,50,50
4,4,4
sigmaX1,sigmaY1,sigmaZ1  
sigmaX2,sigmaY2,sigmaZ2   
sigmaX3,sigmaY3,sigmaZ3   
sigmaX4,sigmaY4,sigmaZ4   
sigmaX5,sigmaY5,sigmaZ5   
sigmaX6,sigmaY6,sigmaZ6   
sigmaX7,sigmaY7,sigmaZ7   
sigmaX8,sigmaY8,sigmaZ8   
```
