# Array

The Array classes provide multidimensional arrays for storing data with support for
numpy memory aliasing.

## Overview

Arrays are available from 1D to 5D with two memory management variants:

- **Owned**: Allocates and manages its own memory
- **Alias**: References external memory (e.g., memory owned by numpy arrays)

## Supported Types

- `Array{ND}Float{Owned/Alias}` - Single precision float
- `Array{ND}Double{Owned/Alias}` - Double precision float
- `Array{ND}Int{Owned/Alias}` - Signed integer
- `Array{ND}Bool{Owned/Alias}` - Boolean

Replace `{ND}` with 1, 2, 3, 4, or 5 for the dimension.
Replace `{Owned/Alias}` by either `Owned` for memory managed by YRT-PET or
    `Alias` for memory managed by NumPy.

## Memory management

```python
import pyyrtpet as yrt
import numpy as np

# ============================================================================
# Owned array - allocates its own memory
# ============================================================================

# Create a 3D owned array (single precision float)
arr_3d = yrt.Array3DFloatOwned()
arr_3d.allocate([10, 20, 30])

# Verify memory is allocated
assert arr_3d.isMemoryValid(), "Memory should be allocated"

# Get dimensions
dims = arr_3d.getDims()
assert dims[0] == 10 and dims[1] == 20 and dims[2] == 30

# Get total size
total_size = arr_3d.getSizeTotal()
assert total_size == 10*20*30, f"Expected 6000, got {total_size}"

# ============================================================================
# Alias array - binds to numpy array
# ============================================================================

# Create a numpy array
arr_np = np.random.random((5, 10, 20, 30)).astype(np.float32)

# Bind to a NumPy array
arr_alias = yrt.Array4DFloatAlias()
arr_alias.bind(arr_np)

# Change value in numpy array
arr_np[1, 2, 3, 4] = np.float32(-1.2)

# Or Bind from another array
arr_alias2 = yrt.Array4DFloatAlias()
arr_alias2.bind(arr_alias)

# Verify memory is valid
assert arr_alias.isMemoryValid(), "Memory should be valid after binding"
assert arr_alias2.isMemoryValid()
assert arr_alias[1,2,3,4] == arr_np[1,2,3,4], "Memory aliasing failed"
assert arr_alias2[0,2,15,26] == arr_alias[0,2,15,26], "Memory aliasing failed"
```

## Data Access

```python
# Create a 3D array for demonstration
arr = yrt.Array3DFloatOwned()
arr.allocate([3, 4, 5])

# Multi-dimensional access (3D)
arr[0, 0, 0] = 1.0
arr[1, 2, 3] = 2.5
value = arr[1, 2, 3]
assert value == 2.5

# Flat access (linear index from 0 to total_size-1)
arr.setFlat(10, 10.0)
value = arr.getFlat(10)
assert value == 10.0

# Increment flat index (useful for parallel operations)
arr.incrementFlat(5, 1.0)  # Adds 1.0 to position 5

# Get dimensions
dims = arr.getDims()
assert len(dims) == 3
assert dims[0] == 3 and dims[1] == 4 and dims[2] == 5

# Get size of a specific dimension
assert arr.getSize(2) == 5 # Size of dim 2, should be 5

# Get strides (step size for each dimension)
strides = arr.getStrides()
# Strides represent how many elements to skip to move to next index in each dim

# Convert between flat and multi-dimensional indices
flat_idx = arr.getFlatIdx([1, 2, 3])
multi_idx = arr.unravelIdx(flat_idx)
assert multi_idx[0] == 1 and multi_idx[1] == 2 and multi_idx[2] == 3
```

### Operations

```python
# Create and fill array
arr = yrt.Array2DIntOwned()
arr.allocate([10, 10])
arr.fill(0)

# Set some values
arr[5, 5] = 100

# Sum all elements
total_sum = arr.sum()
assert total_sum == 100.0

# Get maximum value
max_val = arr.getMaxValue()
assert max_val == 100.0

# Fill with a new value
arr.fill(5)
assert arr.sum() == 500  # 10*10*5

# Arithmetic operations (in-place)
arr2 = yrt.Array2DIntOwned()
arr2.allocate([10, 10])
arr2.fill(2)

# arr += arr2
arr += arr2
assert arr.getMaxValue() == 7  # 5 + 2

# arr -= scalar
arr -= 1
assert arr.getMaxValue() == 6 # 5 + 2 - 1

# arr *= scalar
arr *= 2
assert arr.getMaxValue() == 12 # (5 + 2 - 1) * 2

# arr /= scalar
arr /= 3
assert arr.getMaxValue() == 4 # ((5 + 2 - 1) * 2) / 3

# Invert array (1/x for each element)
arr = yrt.Array2DDoubleOwned()
arr.allocate([10, 10])
arr.fill(2.0)
arr.invert()
assert abs(arr.getMaxValue() - 0.5) < 0.001

# Copy data from one array to another
arr_src = yrt.Array3DFloatOwned()
arr_src.allocate([5, 5, 5])
arr_src.fill(10.0)

arr_dst = yrt.Array3DFloatOwned()
arr_dst.allocate([5, 5, 5])
arr_dst.copy(arr_src)
assert arr_dst.sum() == 1250.0  # 5*5*5*10
```

### File I/O

```python
import os

# For this documentation's purposes only, we use a tempfile
import tempfile
with tempfile.NamedTemporaryFile(suffix='.rawd', delete=False) as f:
    my_file = f.name

# Write array to file
arr = yrt.Array3DFloatOwned()
arr.allocate([10, 20, 30])
arr.fill(42.0)
arr.writeToFile(my_file)

# Read array from file
arr2 = yrt.Array3DFloatOwned()
arr2.readFromFile(my_file)

# Verify data
assert arr2.getMaxValue() == 42.0
dims = arr2.getDims()
assert dims[0] == 10 and dims[1] == 20 and dims[2] == 30

# Clean up
os.remove(my_file)
```

## NumPy Interoperability

The Array classes implement the Python buffer protocol, allowing direct numpy
conversion:

```python
# Create owned array
arr_owned = yrt.Array3DFloatOwned()
arr_owned.allocate([10, 20, 30])

# Create a numpy array with zero-copy
arr_np = np.array(arr_owned, copy=False)

# Modify through numpy
arr_np[:] = 5.0

# arr_owned now contains 5.0
assert arr_owned.getMaxValue() == 5.0

# Alias array works similarly
arr_np_external = np.ones((5, 10, 20), dtype=np.float32)
arr_alias = yrt.Array3DFloatAlias()
arr_alias.bind(arr_np_external)

# You can even bind a numpy array to a YRT-PET array, which is itself bound to
#  a NumPy array.
arr_from_alias = np.array(arr_alias, copy=False)
arr_from_alias[0, 0, 0] = 99.0
assert arr_np_external[0, 0, 0] == 99.0
```

## Boolean Array example

```python

# Boolean array
arr_bool = yrt.Array3DBoolOwned()
arr_bool.allocate([5, 5, 5])
arr_bool.fill(False)
arr_bool[0, 0, 0] = True
assert arr_bool[0, 0, 0] == True
```
