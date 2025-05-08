# Owned vs Alias (For Python users and Plugin developers)

Almost every data structure class in YRT-PET comes in two forms, one has the
`Owned` suffix and the other has the `Alias` suffix.

An `Owned` class is when the inner memory is managed by the object itself.
This means that the lifetime of the memory is the lifetime of the object itself.
After the creation of an `Owned` object, one has to call the `allocate()`
function before reading/writing in its memory. If this is not done, the
execution will fail into a segmentation fault.

An `Alias` class is when the memory is not managed by the object, but by an
external instance. The object only manages the internal metadata and will not
deallocate the memory on destruction. This can be seen as using a reference to
another object. The main use of `Alias` objects is when working with numpy
arrays in Python, as the example below will demonstrate. Similarly, after the
creation of an `Alias` object, one has to call the `bind()` function before
reading/writing in its memory. If this is not done, the execution will fail
into a segmentation fault.

## Examples

### Example usage of `Owned` object
```python
import pyyrtpet as yrt
import numpy as np

img_shape = (100, 100, 50)  # in x, y, z dimensions
img_size_mm = (30, 30, 25)  # in x, y, z dimensions
img_params = yrt.ImageParams(*img_shape, *img_size_mm)

# Allocate memory in YRT-PET
img_yrt = yrt.ImageOwned(img_params)
img_yrt.allocate()

# Bind to numpy
img_np = np.array(img_yrt, copy=False) # The "copy=False" is important

# Whatever is done in the np_img array,
# it will internally be done in yrt_img's memory
# and vice-versa

# Example: This will write in img_yrt's memory
img_np[:] = 1 # The "[:]" is important to avoid reassignment

# This will write the image into a file
img_yrt.writeToFile("my_image.nii")
```

### Example usage of `Alias` object
Since most data structure objects respect the Python buffer protocol, one can
bind a numpy array into an `Alias` object.
```python
import pyyrtpet as yrt
import numpy as np

img_shape = (100, 100, 50)  # in x, y, z dimensions
img_size_mm = (30, 30, 25)  # in x, y, z dimensions
img_params = yrt.ImageParams(*img_shape, *img_size_mm)

# Allocate memory in numpy
img_np = np.ones([img_params.nz, img_params.ny, img_params.nx], dtype=np.float32)

# Bind to YRT-PET
img_yrt = yrt.ImageAlias(img_params)
img_yrt.bind(img_np)

# Whatever is done in the np_img array,
# it will internally be done in yrt_img's memory,
# and vice-versa

# Example: This will write in img_yrt's memory
img_np[:, 0, :] = 2

# This will write the image into a file
img_yrt.writeToFile("my_image.nii")
```

### Example usage of `Alias` object *in GPU*
Just as numpy allows the use of `Alias` objects by managing the CPU memory of a
YRT-PET object, PyTorch can do the same for GPU memory, as the example below
demonstrates.
```python
import torch
import pyyrtpet as yrt

# %% Use CUDA device 0
cuda0 = torch.device('cuda:0')

# %% Define image parameters
img_shape = (100, 100, 50)  # in x, y, z dimensions
img_size_mm = (100.0, 100.0, 50.0)  # in x, y, z dimensions
params = yrt.ImageParams(*img_shape, *img_size_mm)

# %% Create Torch array and bind it to an ImageDeviceAlias
ones_img = torch.zeros([params.nz,params.ny,params.nx], device=cuda0,
                       dtype=torch.float32, layout=torch.strided)
img_dev = yrt.ImageDeviceAlias(params)
# Bind Torch array to YRT-PET Image
img_dev.setDevicePointer(ones_img.data_ptr())

# Now, the "ones_img" Torch array points to the same memory location
# as the "img_dev" YRT-PET ImageDevice object

# %% Initialize the scanner
scanner = yrt.Scanner("./MYSCANNER.json")

# %% Initialize an empty histogram
his = yrt.Histogram3DOwned(scanner)
his.allocate()
his.clearProjections(1.0)

# %% Initialize the projector

# Create a bin iterator that uses one subset and iterates on subset 0
bin_iter = his.getBinIter(1, 0)

# Define the projector parameters
proj_params = yrt.OperatorProjectorParams(bin_iter, scanner)

# Create the projector
oper = yrt.OperatorProjectorDD_GPU(proj_params)

# %% Create a projection-space device buffer
# Use 'his' as a reference to comute LORs and use 1 OSEM subset
his_dev = yrt.ProjectionDataDeviceAlias(scanner, his, 1)

# Important: This is needed to precompute all LORs and load them into the device
# Arguments: Load events from the batch 0 of the subset 0
his_dev.prepareBatchLORs(0, 0)

# Create a Torch array with the appropriate size
ones_proj = torch.ones([his_dev.getLoadedBatchSize()], device=cuda0,
                       dtype=torch.float32, layout=torch.strided)

# Bind Torch array to YRT-PET ProjectionData
his_dev.setProjValuesDevicePointer(ones_proj.data_ptr())

# Do the backprojection on the image
oper.applyAH(his_dev, img_dev)

# Save image
img_dev.writeToFile("./my_image.nii") # save img
```
