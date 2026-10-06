# Sinogram3D format

This page describes the *Sinogram3D* data structure in YRT-PET.

Unlike the *Histogram3D* (see `histogram3d_format`), which uses logical
detector-pair indices as bin coordinates, a *Sinogram3D* uses physical
coordinates:

- **phi** — view angle in $[0, \pi)$
- **r** — signed radial offset in $(-R, R)$ where $R$ is the scanner radius
- **z** — axial plane index (segment-compressed)
- **tof** — time-of-flight bin (1 for non-TOF data)

This makes sinograms more portable across scanners with different ring counts
and ring geometries, and directly compatible with analytic reconstruction and
scatter estimation algorithms that expect standard PET sinogram coordinates.

## SinogramParams

A `SinogramParams` object fully describes a sinogram geometry.
User-specified fields:

| Field         | Description                                             |
|---------------|---------------------------------------------------------|
| `numPhi`      | Number of view-angle bins over $[0, \pi)$               |
| `numR`        | Number of radial-offset bins over $(-R, R)$             |
| `span`        | Axial compression factor (1 = no compression)           |
| `maxRingDiff` | Maximum ring difference included (0 = scanner maximum)  |
| `numTOF`      | Number of TOF bins (1 for non-TOF data)                 |

Derived (read-only) fields:

| Field        | Description                                                    |
|--------------|----------------------------------------------------------------|
| `numZ`       | Total number of axial planes across all segments               |
| `numSegSide` | Number of segments on each side: `maxRingDiff / span + 1`      |
| `phiStep`    | $\pi / \text{numPhi}$                                          |
| `rStep`      | $2R / \text{numR}$                                             |
| `rMin`       | $-R$ (center of the first radial bin)                          |

The most convenient way to build a `SinogramParams` is `fromScanner`:

```python3
params = yrt.SinogramParams.fromScanner(
    scanner,
    numPhi=200,
    numR=200,
    span=11,
    maxRingDiff=0,   # 0 = scanner maximum
    numTOF=1)
```

## Axial (z) organisation — segments and span

The axial dimension uses segment-based compression as in standard 3D PET
sinograms.

Segments are ordered as $0, +1, -1, +2, -2, \ldots$ The number of axial planes
in each segment decreases with $|s|$:

$$
N_\text{planes}(s) = N_\text{rings} - |s| \cdot \text{span}
$$

The total axial size is:

$$
\text{numZ} = N_\text{rings} + 2 \sum_{s=1}^{\text{numSegSide}-1}
              (N_\text{rings} - s \cdot \text{span})
$$

Segment $s > 0$ stores LORs where ring$_2$ - ring$_1$ = $s \cdot \text{span}$;
segment $s < 0$ is the mirror.

## Array format

The array is stored in C row-major order with shape:

$$
[\text{numTOF},\ \text{numZ},\ \text{numPhi},\ \text{numR}]
$$

Flat bin index:

$$
b = \text{tof} \cdot (\text{numZ} \cdot \text{numPhi} \cdot \text{numR})
  + z \cdot (\text{numPhi} \cdot \text{numR})
  + \phi \cdot \text{numR}
  + r
$$

## File format

Sinogram3D files use the extension `.sino`.
The binary layout (all little-endian on x86) is:

| Bytes             | Type       | Value / Description                   |
|-------------------|------------|---------------------------------------|
| 4                 | `int32`    | Magic number: `732174001`             |
| 4                 | `int32`    | Number of dimensions: `4`             |
| 8                 | `uint64`   | `numTOF`                              |
| 8                 | `uint64`   | `numZ`                                |
| 8                 | `uint64`   | `numPhi`                              |
| 8                 | `uint64`   | `numR`                                |
| 8                 | `uint64`   | `span`                                |
| 8                 | `uint64`   | `maxRingDiff`                         |
| 4 × `numTOF`×`numZ`×`numPhi`×`numR` | `float32` | Projection values |

The derived field `numZ` is stored explicitly so that readers can validate it
against the `span` / `maxRingDiff` recomputed from the scanner.  The scanner is
not embedded in the file; it must be supplied separately.

## C++ usage

```cpp
#include "yrt-pet/datastruct/projection/Sinogram3D.hpp"

// Build params from scanner
const SinogramParams params =
    SinogramParams::fromScanner(scanner, 200, 200, 11, 0, 1);

// Allocate and fill
Sinogram3DOwned sino(scanner, params);
sino.allocate();
sino.clearProjections(0.0f);
sino.setProjectionValue(42, 7.0f);

// Write / read back
sino.writeToFile("output.sino");
Sinogram3DOwned sinoIn(scanner, params, "output.sino");
```

`Sinogram3DAlias` wraps an externally-owned buffer with the same interface:

```cpp
Array4DOwned<float> buf(params.numTOF, params.numZ, params.numPhi, params.numR);
Sinogram3DAlias alias(scanner, params);
alias.bind(buf);
```

## Python usage

```python3
import pyyrtpet as yrt
import numpy as np

scanner = yrt.Scanner("myscanner.json")
params = yrt.SinogramParams.fromScanner(scanner, 200, 200, 11)

# Allocate
sino = yrt.Sinogram3DOwned(scanner, params)
sino.allocate()
sino.clearProjections(0.0)

# Zero-copy numpy view (shares memory with the C++ object)
arr = np.asarray(sino)          # shape (numTOF, numZ, numPhi, numR), float32
arr[0, 5, 10, 15] = 3.14

# Write / read back
sino.writeToFile("output.sino")
sino_in = yrt.Sinogram3DOwned(scanner, params)
sino_in.readFromFile("output.sino")

# Wrap an existing numpy array
alias = yrt.Sinogram3DAlias(scanner, params)
alias.bind(arr)
```

## Conversion between Sinogram3D and Histogram3D

Three converter functions are available (both C++ in `yrt::sino` and Python):

| Python function         | Description                                          |
|-------------------------|------------------------------------------------------|
| `fillSinoFromListMode`  | Bin a `ListMode` into a `Sinogram3D` in-place        |
| `sinoFromHistogram3D`   | Convert a `Histogram3D` to a new `Sinogram3DOwned`   |
| `sinoToHistogram3D`     | Convert a `Sinogram3D` to a new `Histogram3DOwned`   |

```python3
# List-mode → sinogram
sino = yrt.Sinogram3DOwned(scanner, params)
sino.allocate()
yrt.fillSinoFromListMode(sino, listmode, num_threads=-1)

# Histogram3D → sinogram
sino = yrt.sinoFromHistogram3D(h3d, params)

# Sinogram → Histogram3D
h3d = yrt.sinoToHistogram3D(sino)
```

The C++ headers are in
`include/yrt-pet/datastruct/projection/SinogramConverter.hpp`.

## Command-line: `yrtpet_make_sinogram`

The `yrtpet_make_sinogram` executable bins a list-mode file into one `.sino`
file per time frame plus a manifest:

```
yrtpet_make_sinogram \
    --scanner   myscanner.json \
    --format    LM \
    --input     scan.lm \
    --out       output/scan \
    --num_phi   200 \
    --num_r     200 \
    --span      11 \
    --dyn       framing.dyn
```

This produces `output/scan_frame000.sino`, `output/scan_frame001.sino`, …
and `output/scan_manifest.txt`.

When `--dyn` is omitted, uniform frames of `--time_step` milliseconds are used
(default 1000 ms).  `--max_ring_diff 0` (the default) includes all ring pairs
allowed by the scanner.

## Corrector and projector integration

`Sinogram3D` inherits from `Histogram` (and transitively from
`ProjectionData`), so it is a first-class input to the corrector and the
projector pipeline.  Pass it anywhere a `Histogram*` or `ProjectionData*` is
expected, for example as the scatter estimate in `Corrector::setScatter`.
