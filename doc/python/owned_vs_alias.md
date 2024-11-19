# Owned vs Alias (For Python users and Plugin developers)

Almost every data structure class in YRT-PET comes in two forms, one has the `Owned` suffix and the other has the
`Alias` suffix.

An `Owned` class is when the inner memory is managed by the object itself. This means that the lifetime of the memory
is the lifetime of the object itself.
After the creation of an `Owned` object, one has to call the `allocate()` function before reading/writing in its memory.
If this is not done, the execution will fail into a segmentation fault.

An `Alias` class is when the memory is not managed by the object, but by an external instance.
The object only manages the internal metadata and will not deallocate the memory on destruction.
Similarly, after the creation of an `Alias` object, one has to call the `bind()` function before
reading/writing in its memory. If this is not done, the execution will fail into a segmentation fault.

## Example usage of `Owned` object
```python
import pyyrtpet as yrt
import numpy as np

params = yrt.ImageParams(50,50,50, 50,50,50) # 50x50x50 image with voxel size 1.0
yrt_img = yrt.ImageOwned(params)
yrt_img.allocate()

np_img = np.array(yrt_img, copy=False) # The "copy=False" is important

# Whatever is done in the np_img array, it will internally be done in yrt_img's memory

# This will write in yrt_img's memory
np_img[:] = np.ones((50,50,50)) # The "[:]" is important


```

## Example usage of `Alias` object
Since most data structure objects respect the Python buffer protocol, one can bind a numpy array into an `Alias` object.
```python
import pyyrtpet as yrt
import numpy as np

params = yrt.ImageParams(50,50,50, 50,50,50) # 50x50x50 image with voxel size 1.0
np_img = np.ones([params.nz, params.ny, params.nx])

yrt_img = yrt.ImageAlias(params)
yrt_img.bind(np_img)

# Whatever is done in the np_img array, it will internally be done in yrt_img's memory

# One can then also save the image into a file
yrt_img.writeToFile("my_image.nii")
```

## Example usage of `Alias` in GPU

```python
import torch
import pyyrtpet as yrt

# %% Initialize the scanner
scanner = yrt.Scanner("./SCANNER.json")

# %% Initialize an empty histogram
his = yrt.Histogram3DOwned(scanner)
his.allocate()
his.clearProjections(1.0)

# %% Initialize the projector

# Create a bin iterator that uses one subset and iterates on subset 0
binIter = his.getBinIter(1, 0)

# Define the projector parameters
projParams = yrt.OperatorProjectorParams(binIter, scanner)

# Create the projector
oper = yrt.OperatorProjectorDD_GPU(projParams)

# %% Use CUDA device 0
cuda0 = torch.device('cuda:0')

# %% Create Torch array and bind it to an ImageDeviceAlias
params = yrt.ImageParams(100,100,100, 100,100,100, 0,0,0)
onesImg = torch.zeros([params.nz*params.ny*params.nx],device=cuda0)
imgDev = yrt.ImageDeviceAlias(params)
imgDev.setDevicePointer(onesImg.data_ptr()) # !!!

# %% Create a projection-space device buffer
# Use 'his' as a reference to comute LORs and use 1 OSEM subset
hisDev = yrt.ProjectionDataDeviceAlias(scanner, his, 1)

# Important: This is needed to precompute all LORs
# Load events from the first batch of the first subset
hisDev.loadEventLORs(0, 0, params)

# Create a torch array with the appropriate size
onesProj = torch.ones([hisDev.getCurrentBatchSize()],device=cuda0)

# Bind torch array to ProjectionDataAlias
hisDev.setProjValuesDevicePointer(onesProj.data_ptr()) # !!!

# Do the backprojection
oper.applyAH(hisDev, imgDev)

# Save image
imgDev.writeToFile("./tmp.nii") # save img

```
