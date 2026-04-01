# Simulation and reconstruction demonstration

This demo will show how to:
- Create a scanner with a regular geometry
- Define a dynamic phantom (ground truth)
- Perform a forward projection and add Poisson noise ("simulation")
- Transform the histogram into a list-mode
- Reconstruct the list-mode

This demonstration is entirely done in 2D to keep this fast.

## Imports

We will need to import YRT-PET's Python interface and NumPy.

```python
import pyyrtpet as yrt
import numpy as np
```

## Scanner definition

Let's start by defining a scanner. We will define a small 2D scanner.

```python

scanner = yrt.Scanner("MySmallScanner", axial_fov=5, crystal_size_z=5, crystal_size_trans=5, crystal_depth=10, scanner_radius=250, dets_per_ring=300, num_rings=1, num_doi=2, max_ring_diff=1, min_ang_diff=11, dets_per_block=10)

```
