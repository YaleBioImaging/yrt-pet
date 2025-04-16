# Histogram3D format

This page will describe how the *Histogram3D* structure is organised in YRT-PET.

Semantics: The word "Histogram" refers to an array of which the bins are in
logical indices. The word "Sinogram", which are not supported yet in YRT-PET,
refers to an array of which the bins point to physical coordinates, regardless
of how irregular the scanner is.

This means that Histograms point to real detector pairs present in the scanner.

## File format

The histogram file itself is a "RAWD" type. Meaning that it encodes a header
describing the array shape and the array itself in 'C'-contiguous ordering.

The file extension commonly used for a Histogram3D is `.his`.

### For Python users

It is also possible to read and write them using Python.
The file in `yrt-pet/python/pyyrtpet/_pyyrtpet.py` contains a class named
`DataFileRawd`, which allows one to read and write in this filetype.

Moreover, with the Python bindings, it is possible to use the fact that the
Histogram3D class respects the Python buffer protocol:

```python
import pyyrtpet as yrt
import numpy as np

scanner = yrt.Scanner("myscanner.json")
his = yrt.Histogram3DOwned(scanner, "myhistogram.his")
np_his = np.array(his, copy=False)  # np_his is now a 3D numpy array
```

### For MATLAB users

One can work with those files in MATLAB using `read_rawd.m` and `write_rawd.m`
in the `scripts/matlab` folder.

## Array format

Let's define what a fully 3D histogram is. It is a 3D array such that:

1. Every bin of the histogram, defined by 3 coordinates, stores a particular
   Line of Response, defined by a pair of detectors
2. Every bin of the histogram must represent a **different** pair of detectors
3. Every Line of Response allowed by the Scanner geometry must be represented
   by a bin in the histogram
4. Two different lines of responses cannot be represented by the same histogram
   bin

These rules must apply for a fully 3D scanner with DOI crystals.
Note that Time-of-flight bins are not encoded by Histogram3D.

## Crystals in the same ring

![image-20210421002446827](https://i.imgur.com/Z6CvlwW.png)

Using a certain value of integers `r` and `phi`, we calculate the coordinates of
two detectors in the same ring to respect rules 1 and 2:

$$
\rho = 
\begin{cases}
0, & \text{when } \phi \text{ is even} \\
1, & \text{when } \phi \text{ is odd}
\end{cases}
$$

$$
d_{r1}=r - \frac{n}{4}+\frac{M_a}{2}
$$

$$
d_{r2}=\frac{n}{2} - (r - \frac{n}{4}+\frac{M_a}{2}) + \rho
$$

$$
d_1=(d_{r1} + \frac{\phi}{2}) \mod n
$$

$$
d_2=(d_{r2} + \frac{\phi}{2}) \mod n
$$

Where:

$$
d_1 \text{ and } d_2 \text{ are detectors 1 and 2 of the bin}
$$

$$
n \text{ is the number of detectors in the ring}
$$

$$
M_a \text{is the minimum angular difference of the Scanner Field of view in terms of detector indices}
$$

In order to respect rules 3 and 4, a Look-Up-Table of all the pairs
`d1` and `d2` and their pair `r` and `phi` is defined for the ring.

This defines a single-ring histogram with no DOI crystals.
However, due to the ordering of the detectors in the Look-Up-Table,
it is possible to scale this to different axial positions and different DOI
layers.

## Binning for different rings

The fully 3D nature of today's scanners makes this task more complex as one LOR
can start from a ring and finish in another. To solve this issue, let's define
`z_bin`, which represents the position of the LOR in the Michelogram moving
diagonally and then from a `delta_z` to another:

![image-20210421004918289](https://i.imgur.com/XNMtT0H.png)

$$
\Delta z = |z_1-z_2|
$$

$$
R_b = \frac{(\Delta z - 1)\Delta z}{2}
$$

$$
z_{bin} = n_a\Delta z + \min(z_1,z_2) + R_b
$$

Where:

$$
z_1 \text{ and } z_2 \text{ are the ring index of detector 1 and 2}
$$

$$
n_a \text{ is the total number of rings in the scanner}
$$

I will spare the inverse function for this document.
Note that this equation only accounts for the top left half of the drawing, the
other half is managed separately afterward.

## Dealing with multiple DOI layers

A scanner with DOI layers allows for a Line of response to go from a layer to
another. To solve this issue, the `r` coordinate of the histogram is used to store this
information.

| r     | Layers for LOR |
|-------|----------------|
| `r+0` | {0,0}          |
| `r+1` | {0,1}          |
| `r+2` | {1,0}          |
| `r+3` | {1,1}          |

This has the only disadvantage of making slightly odd plottings when the
histogram is shown without taking this into account.

## Histogram Dimensions

The dimensions of the histogram are as such:

$$
N_r = N_D^2*(\frac{N_d}{2}+1-M_a)
$$

$$
N_{\phi} = N_d
$$

$$
N_{z_{\text{bin}}} = 2 * \left( (M_r + 1) * N_p - \frac{M_r * (M_r + 1)}{2} \right) - N_p
$$

Where

$$
N_D \text{ is the number of DOI layers}
$$

$$
N_r \text{ is the number of radial bins}
$$

$$
N_d \text{ is the number of detectors per ring in the scanner (not counting for DOI)}
$$

$$
N_p \text{ is the number of axial planes/rings in the scanner}
$$

$$
M_a \text{ is the minimum angle difference in the ring}
$$

$$
M_r \text{ is the maximum ring difference in the scanner}
$$

## Example

![image-20210421010439443](https://i.imgur.com/jCX1Gyr.png)

Left: Image
Right: Histogram of the forward projection of the image into a scanner.

## Small asymmetry in the Histogram

Due to the $\rho$ value in the calculation in the same ring, for every odd
value of phi, the bin at `r=0` will be an invalid bin for the scanner because
it will not respect the "minimum angle difference". It is the circled line in
the image below for example:

![image-20210421013435320](https://i.imgur.com/8r7Z9Tk.png)

This is better drawn in "The Theory and Practice of 3D PET"
by Bernard Bendriem and David W. Townsend
(From Chapter 2 by Michel Defrise and Paul Kinahan):
![expl](https://i.imgur.com/PA6J2Lq.png)

This does not cause any harm to the reconstruction since these bins will simply
never be used. The only harm that it can cause is for the sensitivity
image, but these lines are not common at all or are outside the Field of View.
If it causes harm to the image, it is still possible to use a Sensitivity
histogram
as input to the sensitivity image generation and put those bins to zero.
