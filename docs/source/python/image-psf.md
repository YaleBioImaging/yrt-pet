# Image-space PSF Operators

Image-space PSF operators filter an image without running a reconstruction.
They can be used for image processing or combined with an
[OperatorProjector](operator-projector.md). For OSEM reconstruction, use
[`addImagePSF`](osem.md) so that OSEM applies the PSF and its adjoint (transpose) internally.

## Python Types

| Model | CPU | GPU (CUDA build) |
|-------|-----|------------------|
| Uniform separable kernel | `yrt.OperatorPsf` | `yrt.OperatorPsfDevice` |
| Spatially variant single or dual Gaussian | `yrt.OperatorVarPsf` | `yrt.OperatorVarPsfDevice` |

Both models provide `applyA(img_in, img_out)` for the forward operation and
`applyAH(img_in, img_out)` for adjoint operation. A spatially varying PSF generally has different forward and
adjoint operations, even though each local Gaussian kernel is symmetric.

## Spatially Variant PSF

For an allocated input image `image` with a single 3D frame:

```python
import pyyrtpet as yrt

img_params = image.getParams()
psf = yrt.OperatorVarPsf("dual_gaussian.csv", img_params)

filtered = yrt.ImageOwned(img_params)
filtered.allocate()
filtered.fill(0.0)
psf.applyA(image, filtered)

adjoint_image = yrt.ImageOwned(img_params)
adjoint_image.allocate()
adjoint_image.fill(0.0)
psf.applyAH(image, adjoint_image)
```

The constructor automatically reads either a three-column single Gaussian
or a seven-column dual Gaussian LUT.

Use distinct input and output images with matching dimensions and voxel sizes.
**Clear the output before each CPU call**: both `applyA` and `applyAH` add
their result to the existing output.

### GPU Usage

The file-loading constructor uploads the LUT to GPU memory automatically:

```python
psf_gpu = yrt.OperatorVarPsfDevice("dual_gaussian.csv", img_params)
filtered.fill(0.0)
psf_gpu.applyA(image, filtered)
adjoint_image.fill(0.0)
psf_gpu.applyAH(image, adjoint_image)
```

The GPU operator accepts host `Image` objects as above (copying data to and
from the GPU) or device `ImageDevice` objects. The GPU forward operation
overwrites the output. The GPU adjoint accumulates into a device output;
clear that output with `fill(0.0)` before each call. When a host output is
passed, the wrapper allocates a zero-initialized device buffer and copies the
result back, replacing the host output.

## Uniform PSF

Load a uniform four-row CSV, or construct a CPU operator from three normalized
1D coefficient arrays:

```python
psf_uniform = yrt.OperatorPsf("uniform_psf.csv")

# Alternatively, construct a custom separable kernel
kernel_x = [0.25, 0.5, 0.25]
kernel_y = [0.25, 0.5, 0.25]
kernel_z = [0.25, 0.5, 0.25]
psf_uniform = yrt.OperatorPsf(kernel_x, kernel_y, kernel_z)
psf_uniform.applyA(image, filtered)
psf_uniform.applyAH(image, adjoint_image)
```

Use nonempty odd-length arrays. The operator uses custom coefficients as
supplied; normalize each axis to unit sum for a normalized PSF. Uniform
operations overwrite the output and support in-place application. They
process each frame of a 4D image independently, with circular spatial
boundaries. The adjoint uses the reversed 1D kernels.

For GPU accleration, use `yrt.OperatorPsfDevice("uniform_psf.csv")`. Its
`applyA` and `applyAH` accept host or device images. Each kernel length must
be no larger than the corresponding image dimension; use a one-tap kernel
for an axis containing only one voxel.

## Combining with a Projector

Given a configured `OperatorProjector` named `oper`, an allocated writable
histogram `projection_data` (for example, `Histogram3DOwned`), and the variant
`psf` defined above:

```python
# Forward: image -> PSF -> projection
filtered.fill(0.0)
psf.applyA(image, filtered)
projection_data.clearProjections(0.0)
oper.applyA(filtered, projection_data)

# Adjoint: projection -> backprojection -> adjoint PSF
backprojected = yrt.ImageOwned(img_params)
backprojected.allocate()
backprojected.fill(0.0)
oper.applyAH(projection_data, backprojected)
adjoint_image.fill(0.0)
psf.applyAH(backprojected, adjoint_image)
```
