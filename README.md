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

Setup instructions and general information can be found in the `doc` folder.
However, this project's documentation is still a work in progress.

## Usage

### Command line interface

The compilation directory should contain a folder named `executables`.
The following executables might be of interest:

- `yrtpet_reconstruct`: Reconstruction executable for OSEM.
Includes sensitivity image generation
- `yrtpet_forward_project`: Forward project an image into a fully 3D histogram
- `yrtpet_convert_to_histogram`: Convert a list-mode (or any other datatype
  input) into a fully 3D histogram or a sparse histogram
- (Subject to change) `yrtpet_prepare_additive_correction`: Prepare a fully 3D
  histogram for usage in OSEM as additive correction. Bins will
  contain `(randoms+scatter)/(acf*sensitivity)`.

### Python interface

If the project is compiled with `BUILD_PYBIND11`, the compilation directory
should contain a folder named `pyyrtpet`.
To use the python library, add the compilation folder to your `PYTHONPATH`
environment variable:

```
export PYTHONPATH=${PYTHONPATH}:<compilation folder>
```

Almost all the functions defined in the header files have a Python bindings.
more thorough documentation on the python library is still to be written.

# Data formats

Note that all binary formats encode numerical values in little endian.

## Image format

Images are read and stored in NIfTI format.
YRT-PET also uses a JSON file to define the Image parameters
(size, voxel size, offset). See
[Documentation on the Image parameters format](doc/usage/image_parameters.md).

## YRT-PET raw data format

YRT-PET stores its array structures in the RAWD format.
See [Documentation on the RAWD file structure](doc/usage/rawd_file.md)

## Scanner parameter file

Scanners are decribed using a JSON file and a Look-Up-Table (LUT).
See [Documentation on Scanner definition](doc/usage/scanner.md)

## Listmode (``ListmodeLUT``)

YRT-PET defines a generic default List-Mode format.
When used as input, the format name is `LM`.
See [Documentation on the List-Mode file](doc/usage/list-mode_file.md)

## Sparse histogram (``SparseHistogram``)

YRT-PET defines a generic default sparse histogram format.
When used as input, the format name is `SH`.
See [Documentation on the sparse histogram file](doc/usage/sparse-histogram.md)

## Motion information

Motion information is encoded in a binary file describing the transformation
of each frame.
See [Documentation on the Motion information file](doc/usage/motion_file.md)

## Histogram (`Histogram3D`)

Fully 3D Histograms are stored in YRT-PET's RAWD format
[described earlier](doc/usage/rawd_file.md). Values are encoded in `float32`.
The histogram's dimensions are defined by the scanner properties, which are
defined in the `json` file [decribed earlier](doc/usage/scanner.md).
See [Documentation on the histogram format](doc/usage/histogram3d_format.md)
for more information.
When used as input, the format name is `H`.

# Setup

## Compilation

See [Documentation on compilation](doc/compilation/building.md).

## Testing

There are two types of tests available, unit tests and integration tests.

- Unit tests are short validation programs that check the operation of
  individual modules. To run the unit test suite, simply run `make test` after
  compiling (with `make`).
- Integration test data is currently not publicly available.

# Acknowledgements
- [pybind11](https://github.com/pybind/pybind11)
- [Catch2](https://github.com/catchorg/Catch2)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [The Parallel Hashmap](https://github.com/greg7mdp/parallel-hashmap)
- [NIFTI C Libraries](https://github.com/NIFTI-Imaging/nifti_clib)
- [zlib](https://www.zlib.net/)
- [cxxopts](https://github.com/jarro2783/cxxopts)
- [nlohmann::json](https://github.com/nlohmann/json)
