# Command line interface

The compilation directory should contain a folder named `executables`.
The following executables might be of interest:

- `yrtpet_reconstruct`: Reconstruction executable for OSEM.
  Includes sensitivity image generation
- `yrtpet_forward_project`: Forward project an image into a fully 3D histogram
- `yrtpet_backproject`: Backproject a list-mode or a histogram into an image
- `yrtpet_convert_to_histogram`: Convert a list-mode (or any other datatype
  input) into a fully 3D histogram or a sparse histogram
- (Subject to change) `yrtpet_estimate_scatter`: Prepare a fully 3D
  histogram for usage in OSEM as scatter estimate. Currently experimental and
  incomplete

# Python interface

If the project is compiled with `BUILD_PYBIND11`, the compilation directory
should contain a folder named `pyyrtpet`.
To use the python library, add the compilation directory to your `PYTHONPATH`
environment variable:

```
export PYTHONPATH=${PYTHONPATH}:<compilation folder>
```

Almost all the functions defined in the header files have a Python bindings.
more thorough documentation on the python library is still to be written.
