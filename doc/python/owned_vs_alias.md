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

