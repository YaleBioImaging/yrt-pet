# Image-space PSF Operators

Image-space PSF operators filter an image without running a reconstruction.
They can be used for image processing or combined with an
[OperatorProjector](operator-projector.md). For OSEM reconstruction, use
[`addImagePSF`](osem.md) so that OSEM applies the PSF and its adjoint (transpose) internally.

## Python Types

| Model | CPU | GPU |
|-------|-----|------------------|
| Uniform separable kernel | `yrt.OperatorPsf` | `yrt.OperatorPsfDevice` |
| Spatially variant single or dual Gaussian | `yrt.OperatorVarPsf` | `yrt.OperatorVarPsfDevice` |

Both models provide `applyA(img_in, img_out)` for the forward operation and
`applyAH(img_in, img_out)` for adjoint operation. A spatially varying PSF generally has different forward and
adjoint operations, even though each local Gaussian kernel is symmetric.

## Imports

Run the following sections in order in the same Python session. Will
generate the input image and PSF CSV files, so no external data is
needed.

```python
import itertools

import numpy as np
import pyyrtpet as yrt
```

## Preparing an Image

Below we created a small random image with 33 x 31 x 32 voxels of 1 mm.
NumPy stores the image in `(nz, ny, nx)` order, while `ImageParams` takes
`(nx, ny, nz)` followed by the physical lengths in mm.

```python
rng = np.random.default_rng(0)
nx, ny, nz = 33, 31, 32
img_params = yrt.ImageParams(nx, ny, nz, nx * 1.0, ny * 1.0, nz * 1.0)
x = rng.random((nz, ny, nx)).astype(np.float32)
image = yrt.ImageAlias(img_params)
image.bind(x)
```

## Uniform PSF

### Generating a Kernel File

A uniform PSF applies the same three separable 1D kernels at every voxel.
`sigma` and `voxel_size` are in mm; `size` is an odd number of samples.

```python
def gaussian_1d(size, sigma, voxel_size):
    u = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (u * voxel_size / sigma) ** 2)
    return kernel / kernel.sum()
```

The [uniform PSF CSV](../usage/imagepsf_file.md) has four rows: the X, Y,
and Z kernels, followed by their sizes. Pad the rows with zeros to the
width of the longest kernel.

```python
sizes = (9, 7, 5)
sigmas = (2.0, 1.5, 3.0)
voxel_sizes = (img_params.vx, img_params.vy, img_params.vz)
uniform_lut = np.zeros((4, max(sizes)))
for axis, (size, sigma, voxel_size) in enumerate(zip(sizes, sigmas, voxel_sizes)):
    uniform_lut[axis, :size] = gaussian_1d(size, sigma, voxel_size)
uniform_lut[3, :3] = sizes
np.savetxt("uniform_psf.csv", uniform_lut, delimiter=",")
```

### Applying the Forward and Adjoint Operators

Load the file and allocate output images with the same image parameters:

```python
psf_uniform = yrt.OperatorPsf("uniform_psf.csv")

filtered = yrt.ImageOwned(img_params)
filtered.allocate()
adjoint_image = yrt.ImageOwned(img_params)
adjoint_image.allocate()

psf_uniform.applyA(image, filtered)
psf_uniform.applyAH(image, adjoint_image)

np.testing.assert_allclose(
    np.asarray(filtered), np.asarray(adjoint_image), rtol=1e-5, atol=1e-6
)
print("Uniform Gaussian forward and adjoint results match")
```

`OperatorPsf` overwrites its output, so no `fill(0.0)` is needed. The adjoint uses the reversed 1D kernels.

### Constructing a Custom Kernel Directly

Alternatively, pass three nonempty odd-length coefficient arrays directly.
The operator uses them as supplied; normalize each axis to unit sum for a
normalized PSF. Here the X kernel is asymmetric, so the forward and adjoint
operations generally give different results on the same input:

```python
kernel_x = [0.1, 0.2, 0.7]
kernel_y = [0.25, 0.5, 0.25]
kernel_z = [0.3, 0.4, 0.3]
psf_custom = yrt.OperatorPsf(kernel_x, kernel_y, kernel_z)
psf_custom.applyA(image, filtered)
psf_custom.applyAH(image, adjoint_image)

difference = np.abs(np.asarray(filtered) - np.asarray(adjoint_image)).max()
print(f"Custom kernel: max |A x - A^H x| = {difference:.6f}")
```

## Spatially Variant PSF

### Generating a Dual Gaussian LUT

A spatially variant PSF selects a kernel from a regular grid using the
nearest absolute distance from the image center along each axis. Locations
outside the grid use its edge values. Each entry can contain one Gaussian
or a mixture of **two Gaussians sharing the same center**.

Below is an example of generating a dual Gaussian LUT covering the voxel centers of our image.
We sample every 8 mm from 0 to 16 mm along each axis, giving
`(16 / 8 + 1)^3 = 27` kernels. Each kernel has a narrow core and a broad
tail whose widths increase with distance from the image center.
These parameters are illustrative, not scanner calibration values.

```python
xyz_range = (16.0, 16.0, 16.0)  # mm
xyz_gap = (8.0, 8.0, 8.0)  # mm
xyz_nstd = (2.5, 2.5, 2.5)  # Kernel support controls, in sigmas
nx_lut, ny_lut, nz_lut = (
    int(np.floor(extent / gap)) + 1
    for extent, gap in zip(xyz_range, xyz_gap)
)

# Seven columns in every row, including the three metadata rows.
variant_lut = np.zeros((3 + nx_lut * ny_lut * nz_lut, 7))
variant_lut[0, :3] = xyz_range
variant_lut[1, :3] = xyz_gap
variant_lut[2, :3] = xyz_nstd

# X varies fastest, then Y, then Z.
for row, (k, j, i) in enumerate(
    itertools.product(range(nz_lut), range(ny_lut), range(nx_lut)), start=3
):
    px, py, pz = i * xyz_gap[0], j * xyz_gap[1], k * xyz_gap[2]
    radius = np.sqrt(px * px + py * py + pz * pz)
    sigma1 = 1.0 + 0.010 * radius
    sigma2 = 2.0 + 0.020 * radius
    weight2 = 0.25 + 0.002 * radius  # Stays in [0, 1] on this grid
    variant_lut[row] = [sigma1, sigma1, sigma1, sigma2, sigma2, sigma2, weight2]

np.savetxt("dual_gaussian.csv", variant_lut, delimiter=",")
print(f"Wrote {len(variant_lut) - 3} dual Gaussian kernels")
```

The first three rows contain ranges, gaps, and support controls, padded
with four zeros. Each remaining row contains
`sigma_x1, sigma_y1, sigma_z1, sigma_x2, sigma_y2, sigma_z2, weight2`.
Component 1 has weight `1 - weight2`. See the
[PSF LUT format](../usage/imagepsf_file.md) for details.

### Applying the Forward and Adjoint Operators

The constructor automatically selects the single Gaussian model for a
three-column CSV or the dual Gaussian model for a seven-column CSV.
The variant operator works on a single 3D frame, as in our input image.
We will reuse the output images allocated in the Uniform PSF section.

Use distinct input and output images with matching dimensions and voxel
sizes. Both `applyA` and `applyAH` overwrite the output, just like
`OperatorPsf`; no `fill(0.0)` is needed before a PSF call.

```python
psf = yrt.OperatorVarPsf("dual_gaussian.csv", img_params)

psf.applyA(image, filtered)

psf.applyAH(image, adjoint_image)

print(f"Input sum: {x.sum():.6f}")
print(f"Forward sum: {np.asarray(filtered).sum():.6f}")
print(f"Adjoint sum: {np.asarray(adjoint_image).sum():.6f}")
```

### Checking the Adjoint

For an independent image `y`, the inner products should satisfy
`<A x, y> = <x, A^H y>` within floating-point tolerance. `filtered` already
contains `A x`, so we only need to compute `A^H y`:

```python
y = rng.random(x.shape).astype(np.float32)
image_y = yrt.ImageAlias(img_params)
image_y.bind(y)

psf.applyAH(image_y, adjoint_image)

lhs = filtered.dotProduct(image_y)
rhs = image.dotProduct(adjoint_image)
print(f"<Ax, y> = {lhs:.6f}; <x, A^H y> = {rhs:.6f}")
np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-6)
print("Adjoint test passed")
```

## Combining with a Projector

### Preparing a Scanner and Projector

Below we combine the variant `psf` with a small cylindrical scanner and a
Siddon projector. This creates the scanner and projection data directly,
using the same image as above:

```python
scanner = yrt.Scanner(
    "PSFDemoScanner",
    axial_fov=32.0,
    crystal_size_z=8.0,
    crystal_size_trans=4.0,
    crystal_depth=10.0,
    scanner_radius=50.0,
    dets_per_ring=64,
    num_rings=4,
    num_doi=1,
    max_ring_diff=3,
    min_ang_diff=1,
    dets_per_block=8,
)
projection_data = yrt.Histogram3DOwned(scanner)
projection_data.allocate()
bin_iter = projection_data.getBinIter(num_subsets=1, idx_subset=0)
proj_params = yrt.ProjectorParams(scanner)
proj_params.setProjector("S")
oper = yrt.createOperatorProjector(proj_params, bin_iter)
```

### Forward Projection and Backprojection

The forward operation applies the PSF before projecting. Its adjoint
reverses that order: backproject first, then apply the adjoint PSF.

```python
# Forward: image -> PSF -> projection
psf.applyA(image, filtered)
projection_data.clearProjections(0.0)
oper.applyA(filtered, projection_data)

# Adjoint: projection -> backprojection -> adjoint PSF
backprojected = yrt.ImageOwned(img_params)
backprojected.allocate()
backprojected.fill(0.0)
oper.applyAH(projection_data, backprojected)
psf.applyAH(backprojected, adjoint_image)

projection_np = np.asarray(projection_data)
backprojection_np = np.asarray(adjoint_image)
assert np.isfinite(projection_np).all() and projection_np.max() > 0
assert np.isfinite(backprojection_np).all() and backprojection_np.max() > 0
print("PSF forward projection and backprojection completed")
```

For OSEM, use [`addImagePSF`](osem.md) to configure these operations
internally rather than applying them manually.

## GPU Usage (Optional)

The following examples require a CUDA-enabled YRT-PET build and a usable
GPU. Run them after the CPU sections above; they reuse the generated CSV
files, input image, and allocated outputs.

### Uniform PSF

`OperatorPsfDevice` accepts host or device images and overwrites the output.
Each kernel length must be no larger than the corresponding image
dimension; use a one-tap kernel for an axis containing only one voxel.

```python3
psf_uniform_gpu = yrt.OperatorPsfDevice("uniform_psf.csv")
psf_uniform_gpu.applyA(image, filtered)
psf_uniform_gpu.applyAH(image, adjoint_image)
```

### Spatially Variant PSF

The file-loading constructor transfers the LUT to GPU memory automatically:

```python3
psf_gpu = yrt.OperatorVarPsfDevice("dual_gaussian.csv", img_params)
psf_gpu.applyA(image, filtered)
psf_gpu.applyAH(image, adjoint_image)
```

The GPU operator accepts host `Image` objects as above (copying data to and
from the GPU) or device `ImageDevice` objects. Both `applyA` and `applyAH`
overwrite the output for either image type, matching the CPU operator.
No `fill(0.0)` is needed before either call, including when reusing a device
output.
