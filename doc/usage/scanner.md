# Scanner definition

In YRT-PET, a scanner is defined in two parts:
the scanner parameters file (JSON format) and the Look-Up-Table (LUT) file in binary format.

## Scanner parameters

The following parameters in the JSON file define the scanner:
- `VERSION` : Scanner file format version. The current version is `3.1`.  **\[Mandatory\]**
- `scannerName` : Scanner name. This is value is not used in the reconstruction.  **\[Mandatory\]**
- `detCoord`: Path to the LUT file. The path is relative to the JSON file's parent folder.
  - If this field is unspecified, a LUT will be generated from the scanner's properties.
  (deprecated feature)
- `axialFOV` : Axial Field-of-View (mm).  **\[Mandatory\]**
- `crystalSize_trans` : Crystal size in the transaxial dimension (mm).  **\[Mandatory\]**
- `crystalSize_z` : Crystal size in the axial dimension (mm).  **\[Mandatory\]**
- `crystalDepth` : Crystal size in the direction of its orientation (mm).  **\[Mandatory\]**
- `scannerRadius` : Scanner radius (mm).  **\[Mandatory\]**
- `detsPerRing` : Number of detectors per ring.  **\[Mandatory\]**
- `numRings` : Number of rings.  **\[Mandatory\]**
- `numDOI` : Number of DOI layers. **\[Mandatory\]**

The following properties are necessary to define sensitivity images and histogram shapes:
- `maxRingDiff` : Maximum ring difference two detectors must have to define a valid LOR.  **\[Mandatory\]**
- `minAngDiff` : Minimum difference two detectors must have to define a valid LOR.  **\[Mandatory\]**

The following properties are used only in scatter estimation:
- `collimatorRadius` : Collimator radius (mm). Only used in scatter estimation.
- `fwhm` : Energy resolution FWHM (keV). Only used in scatter estimation.
- `energyLLD` : Energy Low-Level-Discriminant (keV). Only used in Scatter estimation.

The following properties are deprecated:
- `detsPerBlock` : Number of detectors per block in the transaxial dimension.

### Example:

Here's an example of a Scanner's JSON file
```json
{
  "VERSION": 3.1,
  "scannerName": "myscanner",
  "detCoord": "myscanner.lut",
  "axialFOV": 250.0,
  "crystalSize_z": 1.0,
  "crystalSize_trans": 1.0,
  "crystalDepth": 8.0,
  "scannerRadius": 200.0,
  "detsPerRing": 800,
  "numRings": 150,
  "numDOI": 2,
  "maxRingDiff": 50,
  "minAngDiff": 230,
  "fwhm": 0.2,
  "energyLLD": 400,
  "collimatorRadius": 167.4
}
```
## Look-Up-Table
The LUT is a binary file that defines all the *detecting elements*, which can either be
individual crystals, or, in the case of scanners with Depth-Of-Interation (DOI),
detection positions.

The number of detection elements in the LUT must match the following:
```
detsPerRing * numRings * numDOI
```

For each element, the LUT defines, in 32-bit float:
- X, Y, Z center position of the element
- X, Y, Z orientation of the element, pointing towards the exterior of the scanner

```
<px[0]><py[0]><pz[0]>
<nx[0]><ny[0]><nz[0]>
<px[1]><py[1]><pz[1]>
<nx[1]><ny[1]><nz[1]>
```

The LUT's elements should be ordered in the following way:
- Position in the ring in either clockwise or anti-clockwise
- Ring position, either ascending or descending order
- DOI position, from inner to outer layers.

### For Python users

Due to the simplicity of this format, it can be read using the following lines:
```python
import numpy as np
lut = np.fromfile("myscanner.lut", dtype=np.float32).reshape((-1, 2, 3))
```
Then, one can use matplotlib to display the scanner's detector positions:
```python
import matplotlib.pyplot as plt
plt.scatter(lut[:N, 0, 0], lut[:N, 0, 1])
```
Or the scanner's detector orientations:
```python
plt.scatter(lut[:N, 1, 0], lut[:N, 1, 1])
```
Here, `N` is the number of detectors in a ring.

To display the scanner in three dimensions, the repository
[`plot_scanner`](https://github.com/yassirnajmaoui/plot_scanner)
can do that using [VTK](https://vtk.org/).

One can also generate the LUT using Python and save it using:
```python
my_lut.tofile("mynewscanner.lut")
```

### For plugin developers:
The value returned by the functions `getDetector1`, `getDetector2`, and `getDetectorPair` should correspond to
the index of the detector(s) in that LUT.
