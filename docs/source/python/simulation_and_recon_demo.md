# Simulation and reconstruction demonstration

This demo will show how to:
- Create a scanner with a regular geometry
- Define a dynamic phantom (ground truth)
- Perform a forward projection and add Poisson noise ("simulation")
- Transform the histogram into a list-mode
- Reconstruct the list-mode

This demonstration is entirely done in 2D to keep it simple.

## Imports

We will need to import YRT-PET's Python interface and NumPy.

```python
import pyyrtpet as yrt
import numpy as np
```

## Scanner definition

Let's start by defining a scanner. We will define a small cylindrical 2D scanner.

```python
scanner = yrt.Scanner("My2DScanner",
                      axial_fov=5,
                      crystal_size_z=5,
                      crystal_size_trans=5,
                      crystal_depth=10,
                      scanner_radius=250,
                      dets_per_ring=300,
                      num_rings=1, # This is a 2D scanner
                      num_doi=2,
                      max_ring_diff=0,
                      min_ang_diff=11,
                      dets_per_block=10)
```

## Defining image space

Let's define an image grid with 2 mm voxels

```python
nx = 150
ny = 150
nz = 1 # This is a 2D image
vx = 2 # mm
vy = 2 # mm
vz = 2 # mm
img_params = yrt.ImageParams(nx, ny, nz, vx*nx, vy*ny, vz*nz)
```

## Generating the simulated phantom

For our demonstration, we will synthetically generate a phantom with circles defining each region.

For this we will need this helper function:
```python
def get_circle_image(img_params : yrt.ImageParams, center, radius: float):
    """
    Draws a filled circle in a 2D grid and return the grid.

    Arguments:
        img_params: Image parameters object (yrt.ImageParams object)
        center: Physical center in mm (Tuple (x_p, y_p))
        radius: Radius of the circle in mm (float)

    Returns:
        2D numpy array (ny, nx) with 1 inside the circle, 0 outside
    """
    nx, ny = img_params.nx, img_params.ny
    vx, vy = img_params.vx, img_params.vy # mm
    ox, oy = img_params.off_x, img_params.off_y # mm
    cx_p, cy_p = center # In physical coordinates (mm)

    cx = (cx_p - ox) / vx + (nx - 1) / 2
    cy = (cy_p - oy) / vy + (ny - 1) / 2

    y_idx, x_idx = np.ogrid[:ny, :nx]
    dist_squared = ((y_idx - cy) * vy) ** 2 + ((x_idx - cx) * vx) ** 2

    grid = (dist_squared <= radius ** 2).astype(np.float32)
    return grid
```

### Static phantom

Let us define a phantom that consits of a large circle (representing soft tissue) and
four circles each representing a small region.

```python
soft_tissue = get_circle_image(img_params, (0, 0), 100)
top_left_region = get_circle_image(img_params, (-50, 50), 20)
top_right_region = get_circle_image(img_params, (50, 50), 20)
bottom_left_region = get_circle_image(img_params, (-50, -50), 20)
bottom_right_region = get_circle_image(img_params, (50, -50), 20)
```

### Defining kinetics (Dynamic frames)

Now let us define a dynamic framing for a hypothetical 5 minute-scan.
The intensity of each region will vary throughout the dynamic frames.

#### Dynamic framing

We will define a set of frames of varying length.
```python
scan_duration_ms = 5 * 60 * 1000

# Seven frames. The framing starts at 0 ms
# Here we define how the dynamic framing looks w.r.t. the duration fo each frame
num_dynamic_frames = 7
dynamic_framing_lengths = [15  * 1000, # 15 s
                           15  * 1000, # 15 s
                           30  * 1000, # 30 s
                           30  * 1000, # 30 s
                           30  * 1000, # 30 s
                           120 * 1000, # 2 min
                           60  * 1000] # 1 min (Totalling 5 min)
assert len(dynamic_framing_lengths) == num_dynamic_frames

# Then we write the dynamic framing as a list of timestamps defining the start of each frame
#  (plus one other timestamp defining the end of the last frame)
dynamic_framing_np = np.zeros(shape=(num_dynamic_frames + 1), dtype=np.uint32)
curr_timestamp = 0 # Start at 0 ms
dynamic_framing_np[0] = curr_timestamp
for frame_i, l in enumerate(dynamic_framing_lengths):
    curr_timestamp += l
    dynamic_framing_np[frame_i + 1] = curr_timestamp
assert dynamic_framing_np[num_dynamic_frames] == scan_duration_ms

# Create the dynamic framing as a YRT-PET object
dynamic_framing = yrt.DynamicFraming(dynamic_framing_np)
assert dynamic_framing.isValid()
```

#### Dynamic phantom

Let us write the phantom with varying kinetics over the dynamic framing set earlier. We will not attempt to mimic biologically-realistic kinetics here as this is not within the scope of this Python interface demo.

```python
dynamic_phantom = np.zeros(shape=(num_dynamic_frames, nz, ny, nx), dtype=np.float32)

# Values at each reagion over time (Without correcting for the frame duration)
scale_soft_tissue = np.array([0.05, 0.1, 0.2, 0.3, 0.3, 0.1, 0.05])
scale_top_left_region = scale_soft_tissue * (-0.5)
scale_top_right_region = np.array([0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0])
scale_bottom_left_region = np.array([0.05, 0.2, 0.4, 0.3, 0.2, 0, -0.05])
scale_bottom_right_region = scale_soft_tissue * -1 # Zeros (Cancels soft tissue activity)

for frame_i in range(num_dynamic_frames):
    frame_phantom = soft_tissue * scale_soft_tissue[frame_i] + \
                    top_left_region * scale_top_left_region[frame_i] + \
                    top_right_region * scale_top_right_region[frame_i] + \
                    bottom_left_region * scale_bottom_left_region[frame_i] + \
                    bottom_right_region * scale_bottom_right_region[frame_i]

    # Populate the 4D ground truth
    dynamic_phantom[frame_i, 0] = frame_phantom
```

### Defining movement (Motion frames)

Now let us define the movement of the phantom.
The motion frames will be 1 second long and will each encode a random translation and a rotation.

```python
frame_duration_ms = 1000
num_motion_frames = scan_duration_ms // frame_duration_ms
lor_motion = yrt.LORMotion(num_motion_frames)

# Randomly generate the rotations and translations
rng = np.random.default_rng()

# Only define rotation around the Z axis. Varying from 0 to pi/8 degrees
rotations = rng.random(size=(num_motion_frames), dtype=np.float32) * np.pi / 8
# Translation in X and Y. Varying it from -10 mm to 10mm
translations = rng.random(size=(num_motion_frames, 2)) * 20 - 10

# Populate the LORMotion object
curr_timestamp = 0 # Start at 0 ms
for frame_i in range(num_motion_frames):
    # Set timestamp for this frame
    lor_motion.setStartingTimestamp(frame_i, curr_timestamp)

    # Compute transformation for this frame
    rotation = yrt.Vector3D(0, 0, rotations[frame_i])
    translation = yrt.Vector3D(translations[frame_i, 0],
                               translations[frame_i, 1],
                               0)
    transform = yrt.fromRotationAndTranslationVectors(rotation, translation)

    # Set transformation for this frame
    lor_motion.setTransform(frame_i, transform)
    curr_timestamp += frame_duration_ms
```

### Merging motion and kinetics to generate final phantom

We will then generate a 4-dimensional image that will track this phantom over time.
This essentially means to generate an image for each one of the *smallest* unit of
time at which the phantom varies.
Since the motion frames are 1 s each, we will create an image each frame (4th dimension)
being associated to a motion frame.
All of this while keeping track of the dynamic framing coming from the kinetics computed above.

This will give us the final image (with both motion and kinetics) that will be used
for the simulation.

```python
full_phantom = np.zeros(shape=(num_motion_frames, nz, ny, nx), dtype=np.float32)

curr_dynamic_frame = 0
transformed_phantom = yrt.ImageOwned(img_params)
transformed_phantom.allocate() # Necessary
transformed_phantom_np = np.array(transformed_phantom, copy=False)

for motion_frame_i in range(num_motion_frames):
    timestamp = lor_motion.getStartingTimestamp(motion_frame_i)
    transform = lor_motion.getTransform(motion_frame_i)

    # Find the dynamic frame associated to this motion frame
    while curr_dynamic_frame < num_dynamic_frames:
        if dynamic_framing.getStartingTimestamp(curr_dynamic_frame) <= timestamp and \
           dynamic_framing.getStoppingTimestamp(curr_dynamic_frame) > timestamp:
           break
        curr_dynamic_frame += 1 

    transformed_phantom_np[:] = dynamic_phantom[curr_dynamic_frame]
    transformed_phantom = transformed_phantom.transformImage(transform)
    # Re-init the NumPy array since the image was re-allocated
    transformed_phantom_np = np.array(transformed_phantom, copy=False)

    full_phantom[motion_frame_i] = transformed_phantom_np
```

## Simulating the phantom

Let us create a `Histogram3D` object for every motion frame and forward project the associated image from the dynamic `full_phantom` computed above.

