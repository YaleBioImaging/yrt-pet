# YRT-PET
The Yale Reconstruction Toolkit for Positron Emission Tomography (YRT-PET)
is an image reconstruction software for PET imaging.

YRT-PET is currently focused on OSEM reconstructions in List-Mode and
Histogram format.

Current features include:
- GPU acceleration with NVIDIA CUDA
- Python bindings with pybind11
- Event-by-event motion correction
- Siddon, multi-ray Siddon and Distance-Driven projectors
  - Time-of-Flight Support
  - Projection-space PSF support for the Distance-Driven projector
- Image-space PSF
- Image-space post-reconstruction motion correction
- Additive corrections (Scatter & Randoms)
- Normalization correction (Detector sensitivity)
- Attenuation correction
- Scatter estimation (Limited support, without ToF)

Setup and usage instructions can be found in the
[documentation](https://yrt-pet.readthedocs.io/).
However, this project's documentation is still a work in progress.

# Usage

## Command line interface

The compilation directory should contain a folder named `executables`.
The following executables might be of interest:

- `yrtpet_reconstruct`: Reconstruction executable for OSEM.
  Includes sensitivity image generation
- `yrtpet_forward_project`: Forward project an image into a fully 3D histogram
- `yrtpet_convert_to_histogram`: Convert a list-mode (or any other datatype
  input) into a fully 3D histogram or a sparse histogram
- (Subject to change) `yrtpet_estimate_scatter`: Prepare a fully 3D
  histogram for usage in OSEM as scatter estimate. Currently experimental and
  incomplete

## Python interface

If the project is compiled with `BUILD_PYBIND11`, the compilation directory
should contain a folder named `pyyrtpet`.
To use the python library, add the compilation directory to your `PYTHONPATH`
environment variable:

```
export PYTHONPATH=${PYTHONPATH}:<compilation folder>
```

Almost all the functions defined in the header files have a Python bindings.
more thorough documentation on the python library is still to be written.

# Acknowledgements
- [pybind11](https://github.com/pybind/pybind11)
- [Catch2](https://github.com/catchorg/Catch2)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [The Parallel Hashmap](https://github.com/greg7mdp/parallel-hashmap)
- [NIFTI C Libraries](https://github.com/NIFTI-Imaging/nifti_clib)
- [zlib](https://www.zlib.net/)
- [cxxopts](https://github.com/jarro2783/cxxopts)
- [nlohmann::json](https://github.com/nlohmann/json)
