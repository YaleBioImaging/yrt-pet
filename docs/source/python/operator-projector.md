# OperatorProjector

OperatorProjector wraps a Projector and performs either forward projections
or backprojections on a set of LORs provided by a given projection-space
dataset (either list-mode or histogram).

## Python Usage

```python
import pyyrtpet as yrt
import numpy as np

# ============================================================================
# Step 1: Create a Scanner
# ============================================================================

# Define a scanner using geometric parameters
scanner = yrt.Scanner(
    scanner_name='SOME_SCANNER',
    axial_fov=20.0,             # Axial field of view (mm)
    crystal_size_z=2.0,         # Crystal size in z (mm)
    crystal_size_trans=2.0,     # Crystal size in transaxial (mm)
    crystal_depth=10.0,         # Crystal depth (mm)
    scanner_radius=200.0,       # Scanner radius (mm)
    dets_per_ring=256,          # Detectors per ring
    num_rings=10,               # Number of rings
    num_doi=1,                  # DOI layers
    max_ring_diff=9,            # Max ring difference
    min_ang_diff=1,             # Min angular difference
    dets_per_block=32           # Detectors per block
)

# ============================================================================
# Step 2: Create Image Parameters
# ============================================================================

img_params = yrt.ImageParams.fromParams(
    nx=64,   # Number of voxels in x
    ny=64,   # Number of voxels in y
    nz=11,   # Number of voxels in z
    vx=2.0,  # Voxel size x (mm)
    vy=2.0,  # Voxel size y (mm)
    vz=2.0   # Voxel size z (mm)
)

# ============================================================================
# Step 3: Create Image with random values
# ============================================================================

image = yrt.ImageOwned(img_params)
image.allocate()

# Fill with random values
image_np = np.array(image, copy=False)
image_np_init_id = id(image_np)
image_np[:] = np.random.rand(*image_np.shape).astype(np.float32)

assert id(image_np) == image_np_init_id, "Image should not have been moved"
assert image_np.max() > 0, "Image should have data"

# ============================================================================
# Step 4: Create Projection Data (Histogram3D)
# ============================================================================

# Forward projection: image -> projection data
# The histogram will be modified with the forward projected values
his_fwd = yrt.Histogram3DOwned(scanner)
his_fwd.allocate()
his_fwd.clearProjections(0.0)

# Verify histogram is properly allocated
num_bins = his_fwd.count()
assert num_bins > 0

# ============================================================================
# Step 5: Create Projector Parameters
# ============================================================================

proj_params = yrt.ProjectorParams(scanner)
proj_params.setProjector("DD")  # Use Distance-Driven projector
# Other projector options would go here

# ============================================================================
# Step 6: Create Bin Iterator
# ============================================================================

# Get a bin iterator for the histogram
# Parameters: number of subsets, subset index
bin_iter = his_fwd.getBinIter(num_subsets=1, idx_subset=0)

# ============================================================================
# Step 7: Create OperatorProjector
# ============================================================================

# Create the operator projector with projector params and bin iterator
oper = yrt.OperatorProjector(proj_params, bin_iter)

# ============================================================================
# Step 9: Forward Projection (Image -> Histogram3D)
# ============================================================================

# Apply forward projection
oper.applyA(image, his_fwd)

# Check the forward projection results
fwd_np = np.array(his_fwd, copy=False)
fwd_sum = fwd_np.sum()
fwd_max = fwd_np.max()

print(f"Forward projection sum: {fwd_sum}")
print(f"Forward projection max: {fwd_max}")

assert fwd_sum > 0, "Forward projection should produce non-zero values"
assert fwd_max > 0, "Forward projection should have positive values"

# ============================================================================
# Step 10: Back Projection (Histogram3D -> Image)
# ============================================================================

# Create empty image for back projection
bp_image = yrt.ImageOwned(img_params)
bp_image.allocate()
bp_image.fill(0.0)

# Create a histogram and populate it with random values
his_for_bp = yrt.Histogram3DOwned(scanner)
his_for_bp.allocate()
his_for_bp_np = np.array(his_for_bp, copy=False)
his_for_bp_np[:] = np.random.rand(*his_for_bp_np.shape).astype(np.float32)

# Apply back projection
oper.applyAH(his_for_bp, bp_image)

# Check the back projection results
bp_image_np = np.array(bp_image, copy=False)
bp_sum = bp_image_np.sum()
bp_max = bp_image_np.max()

print(f"Back projection sum: {bp_sum}")
print(f"Back projection max: {bp_max}")

assert bp_sum > 0, "Back projection should produce non-zero values"
assert bp_max > 0, "Back projection should have positive values"

# Advanced: This is the list of properties gathered by the projection operator for
#  every event or bin
prop_types = oper.getProjectionPropertyTypes()
print(f"Required properties: {prop_types}")

```

## Important methods

- `applyA(image, projection_data)` - Forward projection: image -> projection data
- `applyAH(projection_data, image)` - Back projection: projection data -> image
- `addTOF(tof_width_ps, tof_num_std)` - Add time-of-flight configuration
- `addProjPSF(fname)` - Add projection-space PSF

**Note**: Configuration methods must be called AFTER creating the
OperatorProjector but BEFORE calling any projection operations.

