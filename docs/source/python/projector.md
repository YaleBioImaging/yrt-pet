# Projector

The Projector classes compute a forward or a backward projection for individual
LORs (Line of Response).
A forward projection projects a line on an image grid and sums the intersecting voxels.
A backprojection distributes a given value into the image grid.

## Available Projector Types

YRT-PET provides two projector implementations:

- **ProjectorSiddon** - Siddon projector, faster but less accurate since it does not
    model the crystal thickness (except with the multi-ray Siddon).
- **ProjectorDD** - Distance-Driven projector, more accurate but slower.

## Python Usage

```python
import pyyrtpet as yrt
import numpy as np

# Define a scanner using geometric parameters
scanner = yrt.Scanner(
    scanner_name='MYSCANNER',
    axial_fov=25.0,         # Axial field of view in mm
    crystal_size_z=2.0,     # Crystal size in axial direction (mm)
    crystal_size_trans=2.0, # Crystal size in transaxial direction (mm)
    crystal_depth=10.0,     # Crystal depth (mm)
    scanner_radius=161.0,   # Scanner radius (mm)
    dets_per_ring=256,      # Number of detectors per ring
    num_rings=8,            # Number of detector rings
    num_doi=1,              # Number of DOI layers
    max_ring_diff=7,        # Maximum ring difference
    min_ang_diff=1,         # Minimum angular difference (in number of crystals)
    dets_per_block=32       # Number of crystals per block (transaxial)
)

# Define image grid parameters
img_params = yrt.ImageParams(
    nx=64,    # Number of voxels in x
    ny=64,    # Number of voxels in y
    nz=32,    # Number of voxels in z
    length_x=scanner.scannerRadius*2,
    length_y=scanner.scannerRadius*2,
    length_z=scanner.axialFOV
)

# Create an empty image
image = yrt.ImageOwned(img_params)
image.allocate()
image.fill(0.0)

# Create Projector Parameters
proj_params = yrt.ProjectorParams(scanner)
proj_params.addTOF(300.0, 3)
# Set the projector parameters here

# Create a Distance-Driven projector
projector = yrt.ProjectorDD(proj_params)

# Define a Line of Response (LOR) - a line connecting two detectors
# Define two points representing a line through the FOV
p1 = yrt.Vector3D(0.0, -scanner.scannerRadius, 0.0)   # Point at the top of the ring
p2 = yrt.Vector3D(0.0, scanner.scannerRadius, 0.0)    # Opposite side
lor = yrt.Line3D(p1, p2)

# For DD projector, we need detector orientation (as unit vectors).
# These are computed from the scanner geometry.
# Here we use simplified unit vectors for demonstration.
n1 = yrt.Vector3D(0.0, -1.0, 0.0)
n2 = yrt.Vector3D(0.0, 1.0, 0.0)

# Perform single backprojection
# Parameters: image, LOR, detector_orient_1, detector_orient_2, projection_value,
#   dynamic_frame, tof_helper, tof_value
# `projection_value` is the value that will be backprojected
# `tof_helper` is an object used to compute the weight of each pixel w.r.t. the TOF kernel
# `tof_value` is the TOF measure in picoseconds to use for the backprojection.
#   We use -50ps here for example
tof_helper = projector.getTOFHelper()
projector.backProjection(
    image, lor, n1, n2, 6, 0, tof_helper, -50
)

# Ensure that we populated at least some pixels
image_np = np.array(image, copy=False)
assert image_np.max() > 0

# Forward projection

#  This time, we will use the Siddon projector
projector = yrt.ProjectorSiddon(proj_params)

# Define a LOR
p1 = yrt.Vector3D(-scanner.scannerRadius, 0.0, 0.0)   # Point at the left of the ring
p2 = yrt.Vector3D(scanner.scannerRadius, 0.0, 0.0)    # Opposite side
lor = yrt.Line3D(p1, p2)

# For the single-ray Siddon projector, we do not need to specify the
#  detector orientation

# Perform single forward projection
tof_helper = projector.getTOFHelper()
forward_proj = projector.forwardProjection(
    image, lor, n1, n2, 0, 0, tof_helper, 0
)

# Ensure we intersected at least some voxels
assert forward_proj > 0
```

## Siddon vs Distance-Driven

The two projector implementations have different characteristics:

| Feature | Siddon | Distance-Driven |
|---------|--------|-----------------|
| Speed | Faster | Slower |
| Accuracy | Lower | Higher |
| Time-of-Flight support | Yes | Yes |
| Projection-space PSF | No | Yes |
| Modeling crystal thickness | Only with multi-ray Siddon | Yes |

## Siddon-specific Methods

```python
# Siddon projector supports multi-ray configuration
projector_siddon = yrt.ProjectorSiddon(proj_params)

# Set number of rays for multi-ray sampling
projector_siddon.setNumRays(5)  # Use 5 rays per LOR
assert projector_siddon.getNumRays() == 5
```

## Single LOR Projection Methods (Siddon)

The Siddon projector provides static methods for single LOR projections:

```python
# Forward projection - returns contribution of image along the LOR
value = yrt.ProjectorSiddon.singleForwardProjection(
    image,      # Input image
    lor,        # Line of Response
    0,          # Dynamic frame index
    None,       # TOF helper (or None)
    0.0         # TOF value
)

# Back projection - adds contribution to image along the LOR
yrt.ProjectorSiddon.singleBackProjection(
    image,      # Output image (modified in-place)
    lor,        # Line of Response
    1.0 ,       # Value to backproject
    0,          # Dynamic frame index
    None,       # TOF helper (or None)
    0.0         # TOF value
)
```

## Notes

- The Siddon projector does NOT support projection-space PSF. If you try to
    add a PSF to a Siddon projector, it will be ignored with a warning. Use
    the Distance-Driven projector for PSF-aware projections.
- For a full forward/backward projection on an entire dataset, use `OperatorProjector`
    See [operator-projector.md](operator-projector) for details
