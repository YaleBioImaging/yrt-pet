# ProjectionList

ProjectionList stores list-mode projection values. It references a source (histogram or
list-mode) for LOR geometry and stores only the measurement values.

## Python Usage

```python
import pyyrtpet as yrt
import numpy as np

# Define a scanner using geometric parameters
scanner = yrt.Scanner(
    scanner_name='MYSCANNER',
    axial_fov=25.0,         # Axial field of view in mm
    crystal_size_z=2.0,     # Crystal size in axial direction (mm)
    crystal_size_trans=2.0, # Crystal size in transaxial direction (mm)
    crystal_depth=10.0,     # Crystal depth (mm)
    scanner_radius=161.0,   # Scanner radius (mm)
    dets_per_ring=256,      # Number of detectors per ring
    num_rings=8,            # Number of detector rings
    num_doi=1,              # Number of DOI layers
    max_ring_diff=7,        # Maximum ring difference
    min_ang_diff=1,         # Minimum angular difference (in number of crystals)
    dets_per_block=32       # Number of crystals per block (transaxial)
)

# Create a histogram reference
his = yrt.Histogram3DOwned(scanner)
his.allocate()

# Create the ProjectionList object
proj_list = yrt.ProjectionListOwned(his)
proj_list.allocate()

# Access as numpy array
proj_np = np.array(proj_list, copy=False)  # Shares memory

# Set values
proj_np[:] = 1.0

# Or use Alias to bind to external numpy
proj_np_ext = np.zeros(his.count(), dtype=np.float32)
proj_alias = yrt.ProjectionListAlias(his)
proj_alias.bind(proj_np_ext)

# The ProjectionList object can then be used just like any projection-space data
# object, including with projection operators (as input to applyA and applyAH)

# This is allows for custom algorithm implementations
```

## Owned vs Alias

- `ProjectionListOwned` - Allocates and manages its own memory
- `ProjectionListAlias` - References external array

All member methods (getProjectionValue, setProjectionValue, getDetector1, etc.)
delegate to the reference projection data source.
