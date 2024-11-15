# RAWD format

The RAWD file format is defined as follows:
```
    MAGIC NUMBER (int32): 732174000 in decimal, used to detect YRT-PET file type
    Number of dimensions D (int32)
    Dimension 0 (int64)
    ...
    Dimension D - 1 (int64)
    Data 0
    ...
```

Notes:

- The data format is arbitrary and must be known when reading a data file. For
  instance, images are stored in `float32`.
- The dimensions are ordered with the contiguous dimension last (e.g. Z, Y, X
  following usual 'C' conventions).
