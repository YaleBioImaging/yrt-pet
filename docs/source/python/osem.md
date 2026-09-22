# OSEM

OSEM (Ordered Subsets Expectation Maximization) is the main reconstruction algorithm.

## Basic Usage

This is an example usage performing the OSEM reconstruction of a list-mode dataset.

```python3
import pyyrtpet as yrt

scanner = yrt.Scanner("<MyScannerFile.json>")

lm = yrt.ListModeLUTOwned(scanner, "<MyListMode.lmDat>")

# Create OSEM with scanner
osem = yrt.createOSEM(scanner)

# Set reconstruction parameters
osem.num_MLEM_iterations = 10
osem.num_OSEM_subsets = 8

# Set image parameters
img_params = yrt.ImageParams(nx=128, ny=128, nz=64,
    length_x=128.0, length_y=128.0, length_z=64.0)
osem.setImageParams(img_params)

# Set data input
osem.setDataInput(lm)

# Set projector
osem.setProjector("Siddon")

# Generate sensitivity images (Will also save the sensitivity image into a NIfTI file)
[sens_image] = osem.generateSensitivityImages("sens_image.nii.gz")

# Set sensitivity image(s)
osem.setSensitivityImage(sens_image)

# Run reconstruction (Will also save the reconstructed image into a NIfTI file)
result = osem.reconstruct("recon_image.nii.gz")

```

## Configuration

### Reconstruction Parameters

```python3
osem.num_MLEM_iterations = 20  # Number of iterations
osem.num_OSEM_subsets = 8      # Number of subsets
```

### Data Input

```python3
osem.setDataInput(projection_data)
```

### Projector

```python3
osem.setProjector("DD")      # For the distance-driven projector
osem.setProjector("SIDDON")  # For the Siddon projector

osem.setNumRays(3)                 # Multi-ray (Siddon only)
osem.addProjPSF("<proj_psf.csv>")  # Adding projection-space PSF (DD only)

# Time-of-flight (TOF width in picoseconds, number of standard deviations to fill
#    from the TOF kernel)
osem.addTOF(500.0, 3)
```

### Corrections

Sensitivity correction (sometimes also called normalization correction) can be done by
providing a histogram (which can be of any format that inherits from `Histogram`,
not necessarily `Histogram3D`) of the sensitivity coefficient for every detector pair.

Attenuation correction can be done by providing an attenuation image or by providing
a histogram of Attenuation Correction Factors (ACFs).

Randoms correction can be performed by either providing a histogram of the randoms
estimate for every detector pair, or by encoding the randoms estimate for every
event (overloaded in the `getRandomsEstimate` member function)

Scatter correction (Not to be confused with scatter *estimation*) is perfomed by providing
a histogram of the scatter estimate for every detector pair.

```python3
osem.setSensitivityHistogram(sensitivity_his)
osem.setAttenuationImage(att_image)   # Attenuation correction (From mu-map)
osem.setACFHistogram(acf_his)         # Attenuation correction (From histogram)
osem.setRandomsHistogram(randoms_his) # Randoms correction (From histogram)
osem.setScatterHistogram(scatter_his) # Scatter correction
```

### Point Spread Function (Image-space)

Set the image parameters before adding a PSF. Choose one of the following
file-based models before generating sensitivity images or reconstructing:

```python3
osem.setImageParams(img_params)

# Uniform: four-row CSV containing separable 1D kernels
osem.addImagePSF("uniform_psf.csv")

# Alternatively, spatially variant: three-column or seven-column LUT
osem.addImagePSF("dual_gaussian.csv", yrt.ImagePSFMode.VARIANT)
```

#### PSF Modes

- `yrt.ImagePSFMode.UNIFORM` - Same separable PSF for all voxels. This is the
  default when the second argument is omitted.
- `yrt.ImagePSFMode.VARIANT` - Spatially variant single or dual Gaussian PSF.
  The CSV reader automatically selects one Gaussian for three columns or two
  Gaussians for seven columns.

In particular, a variant LUT must be loaded with `yrt.ImagePSFMode.VARIANT`, even though both file types use `.csv`.
Calling `addImagePSF` again replaces the previous image-space PSF.

See [Image-Based PSF File Format](../usage/imagepsf_file.md) for complete CSV
examples, units, LUT ordering, and mixture weights.

### Output Options

To save intermediate iteration, you must provide a `RangeList` object
```python3
# Save intermediate iterations
range_list = yrt.RangeList()
range_list.insertSorted(0,1) # Save iterations 0 and 1
range_list.insertSorted(5,8) # Save iterations 5,6,7,8
range_list.insertSorted(10,10) # Save iteration 10
```

to the `OSEM` object

```python3
# The filename given will be added a prefix to specify which iteration is was saved
osem.setSaveIterRanges(range_list, "./iterations/recon_image_intermediate.nii.gz")

# Custom initial estimate (instead of a uniform image containing the value 0.1)
osem.setInitialEstimate(initial_image)

# Reconstruction mask (To disable some voxels from the reconstruction)
osem.setMaskImage(mask_image)
```

## Output

The `reconstruct()` method returns an `ImageOwned` object, but can also save the reconstructed image to disk if provided a filename.

```python3
result = osem.reconstruct() # Will not save to disk
result = osem.reconstruct("recon_image.nii.gz") # Will save to disk

# Access as numpy array
result_np = np.array(result, copy=False)
```
