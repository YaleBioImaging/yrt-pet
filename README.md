This repository contains reconstruction code developed for high resolution PET.

Setup instructions and general information on the tools used by the project can
be found in the Wiki.

# Use the program

## Command line interface

### MLEM

```
YRT-PET reconstruction driver
Usage:
  yrtpet_reconstruct [OPTION...]

  -s, --scanner arg          Scanner parameters file name
  -p, --params arg           Image parameters file
  -i, --input arg            Input file
  -f, --format arg           Input file format. Possible values: LM
                             (Listmode), LMP (ListMode with arbitrary
                             positions), LM-DOI (ListMode with DOI), LP2
                             (LabPET2) and H (Histogram)
      --sens arg             Sensitivity image files (separated by a comma)
      --att arg              Attenuation image filename
      --att_params arg       Attenuation image parameters filename
      --motion arg           Motion file for motion correction
      --ev_per_frame arg     Number of Events per frame file for motion
                             correction
      --psf arg              Image-space PSF kernel file
      --proj_psf arg         Projection-space PSF kernel file
      --add_his arg          Histogram with additive corrections (scatter &
                             randoms)
      --norm arg             Normalization file
      --norm_format arg      Normalization file format. Possible values: LM
                             (Listmode), LMP (ListMode with arbitrary
                             positions), LM-DOI (ListMode with DOI), LP2
                             (LabPET2) and H (Histogram)
      --sensdata arg         Sensitivity data input file
      --sensdata_format arg  Sensitivity data input file format. Possible
                             values: LM (Listmode), LMP (ListMode with
                             arbitrary positions), LM-DOI (ListMode with
                             DOI), LP2 (LabPET2) and H (Histogram)
      --acf arg              Attenuation Coefficient Factors (ACF)
                             histogram file
  -w, --warper arg           Path to the warp parameters file (Specify this
                             to use the MLEM with image warper algorithm)
      --projector arg        Projector to use, choices: Siddon (S),
                             Distance-Driven (D), or GPU Distance-Driven
                             (DD_GPU) The default projector is Siddon
      --num_rays arg         Number of rays to use in the Siddon projector
      --flag_tof             TOF flag
      --tof_width_ps arg     TOF Width in Picoseconds
      --tof_n_std arg        Number of standard deviations to consider for
                             TOF's Gaussian curve
      --doi_n_layers arg     Number of layers encoding DOI (LM-DOI)
  -o, --out arg              Output image filename
      --num_iterations arg   Number of MLEM Iterations
      --num_threads arg      Number of threads to use
      --num_subsets arg      Number of OSEM subsets (Default: 1)
      --hard_threshold arg   Hard Threshold
      --save_steps arg       Enable saving each MLEM iteration image (step)
      --preclinical          Enable Preclinical flag (for preclinical LP2
                             files only)
      --out_sens arg         Filename for the generated sensitivity image
                             (if it needed to be computed). Leave blank to
                             not save it
      --out_acf arg          Filename for the ACF histogram (if it needed
                             to be computed). Leave blank to not save it
      --out_sens_his arg     Filename for the generated histogram used to
                             generate the sensitivity image (if it needed
                             to be computed). Leave blank to not save it
  -h, --help                 Print help
```

## Python interface

Almost all the functions defined in the header files have an equivalent in a Python binding. A more thorough
documentation is to be written in the future.
You need to add the compilation folder to you `PYTHONPATH` environment variable:

```
export PYTHONPATH=${PYTHONPATH}:<compilation folder>
```

## Data formats

Note that all binary formats are in little endian.

### Utilities

Helper functions to read/write data for YRT-PET can be found under the `utils`
folder (MATLAB and Python versions are available in the `scripts` folder).
Most binary files are either raw binary files without header (legacy mode)
or follow the YRT-PET raw data format.

### YRT-PET raw data format

The YRT-PET format is defined as follows:

    MAGIC NUMBER (int32): 732174000 in decimal, used to detect YRT-PET file type
    Number of dimensions D (int32)
    Dimension 0 (int64)
    ...
    Dimension D - 1 (int64)
    Data 0
    ...

Notes:

- The data format is arbitrary and must be known when reading a data file. For
  instance, images are stored in `float64`.
- The dimensions are ordered with the contiguous dimension last (e.g. Z, Y, X
  following usual conventions).

### Scanner parameter file

The Scanner parameters file is a JSON file containing data about the scanner.
Here is, as an example, the SAVANT parameters file.

```JSON
{
  "VERSION": 3.0,
  "scannerName": "SAVANT",
  "detCoord": "SAVANT.lut",
  "axialFOV": 235,
  "crystalSize_z": 1.1,
  "crystalSize_trans": 1.1,
  "crystalDepth": 6,
  "scannerRadius": 197.4,
  "fwhm": 0.2,
  "energyLLD": 400,
  "collimatorRadius": 167.4,
  "dets_per_ring": 896,
  "num_rings": 144,
  "num_doi": 2,
  "max_ring_diff": 24,
  "min_ang_diff": 238,
  "dets_per_block": 8
}
```

The file specified at the `detCoord` field is the Scanner's Look-Up-Table specified below.

### Detector coordinates (`DetCoord`)

The `detCoord` value in the scanner definition allows to provide a Look-Up-Table decribing each detector's coordinates
and orientation, ordered by ID.
The detectors order must follow the following rules:

* If a scanner has several layers of crystals, the detectors closest to center must be listed first
* If the scanner has several rings, the detectors must be sorted by ascending order of z-coordinates
* Within a ring, the detectors must be sorted counter-clockwise (in a cartesian plane)
* The scanner definition must match the LUT (num_doi, num_rings, dets_per_ring)
  The file contains no header and, for each detector, the x, y, z position and x,
  y, z components of the normal orientation vector, all stored in `float32` format (4 bytes).
  The total file size is \(N_d \times 6 \times 4\) bytes where \(N_d\) is the total
  number of detectors. Note that all the values are in mm.

  Detector 0 position x (float32)
  Detector 0 position y (float32)
  Detector 0 position z (float32)
  Detector 0 orientation x (float32)
  Detector 0 orientation y (float32)
  Detector 0 orientation z (float32)
  Detector 1 position x (float32)
  ...
  The file extension used is `.lut`.

### Image (`Image`)

Images are stored in YRT-PET raw data format (i.e. with the header described
[earlier](#yrt-pet-raw-data-format)). Images are currently stored as
64-bin floating point numbers although this is likely to change in the future.
The file extension used is `.img`.

### Listmode (``ListmodeLUT``)

The listmode file is a record of all the events to be considered for the reconstruction. It is similar to the Histogram
as it has all the same fields, except for the Value (which is considered to be 1.0 in all listmode events)

    Timestamp for Event 0 (s) (float32)
    Detector1 of the Event 0 (int)
    Detector2 of the Event 0 (int)
    Timestamp for Event 1 (s) (float32)
    ...

The file extension used is `.lmDat`.

### Histogram (`Histogram3D`)

Histograms are in YRT-PET raw data format (described earlier). They are stored as `float32`. The histogram's dimensions
are defined by the scanner properties defined in the `json` file decribed earlier.
The file extension used is `.his`.

### Image parameters file

Images require a side configuration file which describes the physical
coordinates of the volume. The configuration is in `json` format.
An example is provided here for reference. Note that the values are all in mm.

```JSON
    {
  "VERSION": 1.0,
  "nx": 250,
  "ny": 250,
  "nz": 118,
  "length_x": 250.0,
  "length_y": 250.0,
  "length_z": 235.0,
  "off_x": 0.0,
  "off_y": 0.0,
  "off_z": 0.0
}
```

Note that the offset entries are currently ignored by the code, this might change in the future.

# Setup

To start working on the code,

- clone this repository,
- compile the reconstruction engine,
- start modifying the code,
- add tests,
- submit pull requests to merge the modified code to the `main` branch.

## Conventions

### Git

- The main branch is the stable branch of the project, it is designed to
  remain "clean" (i.e. tested and documented).
- Work on the project should be performed on *branches*. A branch is created
  for each feature, bug, etc. and progress is committed on that branch.
- Once the work is ready to *merge* (i.e. the code is complete, tested and
  documented) it can be submitted for review: the process is called *merge
  request*. Follow Github's instructions to submit a pull request.
- The request is then reviewed by one of the project's maintainers. After
  discussions, it should eventually be merged to the main branch by the
  maintainer.
- The branch on which the work was performed can then be deleted (the history is
  preserved). The branch commits will be squashed before the merge (this can
  be done in Github's web interface).
- If the main branch has changed between the beginning of the work and the
  pull request submission, the branch should be *rebased* to the main branch
  (in practice, tests should be run after a rebase to avoid potential
  regressions).
    - In some cases, unexpected rebase are reported (for instance if the history
      is A&#x2013;B&#x2013;C and B is merged to A, a later rebase of C to A may cause
      conflicts that should not exist). In such cases, two fixes are possible:
        - Launching an interactive rebase (`git rebase -i <main branch>`) and dropping the commits that would be
          duplicated.
    - After a rebase, `git push` by default will not allow an update that is not `fast-forward`
      with the corresponding remote branch, causing an error when trying to push.
      `git push --force-with-lease` can be used to force a push while checking that the remote branch has not changed.
      Note that this will lose history

### Coding conventions

This section will contain guidelines for writing code, once agreed upon by the
members of the team. As a starting point `clang-format` is used with a
configuration file stored in the repository at `src/.clang-format`

## Testing

There are two types of tests available, unit tests and integration tests.

- Unit tests are short validation programs that check the operation of
  individual modules. To run the unit test suite, simply run `make test` after
  compiling (with `make`).
- To run the integration tests,
    - The following environment variables have to be set:
        - `YRTPET_TEST_DATA` which refers to the path where the test files are located.
        - `YRTPET_TEST_OUT` which refers to where output files are to be written.
    - Run `pytest <build folder>/integration_tests/test_recon.py`
        - Optionally, Use the `-k` option to restrict the tests wanted
