# Data formats

Note that all binary formats encode numerical values in little endian.

## Image format

Images are read and stored in NIfTI format.
YRT-PET also uses a JSON file to define the Image parameters
(size, voxel size, offset). See
[Documentation on the Image parameters format](image_parameters).

## YRT-PET raw data format

YRT-PET stores its array structures in the RAWD format.
See [Documentation on the RAWD file structure](rawd_file)

## Scanner parameter file

Scanners are decribed using a JSON file and a Look-Up-Table (LUT).
See [Documentation on Scanner definition](scanner)

## Listmode (``ListmodeLUT``)

YRT-PET defines a generic default List-Mode format.
When used as input, the format name is `LM`.
See [Documentation on the List-Mode file](list-mode_file)

## Sparse histogram (``SparseHistogram``)

YRT-PET defines a generic default sparse histogram format.
When used as input, the format name is `SH`.
See [Documentation on the sparse histogram file](sparse-histogram)

## Motion information (`LORMotion`)

Motion information is encoded in a binary file describing the transformation
of each frame.
See [Documentation on the Motion information file](motion_file)

## Histogram (`Histogram3D`)

Fully 3D Histograms are stored in YRT-PET's RAWD format
[described earlier](rawd_file). Values are encoded in `float32`.
The histogram's dimensions are defined by the scanner properties, which are
defined in the `json` file [decribed earlier](scanner).

See [Documentation on the histogram format](histogram3d_format)
for more information.
When used as input, the format name is `H`.
