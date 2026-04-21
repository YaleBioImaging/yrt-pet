# Constraints

Constraints filter which LORs are included in reconstruction.

## Python Types

```python
import pyyrtpet as yrt

# Minimum angle difference constraint
constraint = yrt.ConstraintAngleDiffDeg(30.0)

# Minimum block difference constraint
constraint = yrt.ConstraintBlockDiffIndex(2)

# Detector mask for a hypothetical scanner with 1000 crystals
detmask = yrt.DetectorMask(1000)
# Detector mask constraint
constraint = yrt.ConstraintDetectorMask(detmask)
```

Constraints are passed to the projector or OSEM to filter LORs during reconstruction.
