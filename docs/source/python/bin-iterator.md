# Bin Iterator

Bin iterators define how to iterate over projection data structures.

## Python Types

```python
import pyyrtpet as yrt

scanner = yrt.Scanner("<MyScanner.json>")

# Create from histogram
his = yrt.Histogram3DOwned(scanner)
his.allocate()

# Get bin iterator for OSEM subsets (take the fourth out of eight subsets)
bin_iter = his.getBinIter(num_subsets=8, subset_idx=3)

# The bin iterator can then be used with `OperatorProjector` to project a
#  specific subset.
```

## Available Iterators

Here are some of the bin iterator types used in this project

- `BinIteratorRange` - Simple range iteration
- `BinIteratorVector` - Custom list of bins
- `BinIteratorChronological` - Time-ordered iteration
- `BinIteratorChronologicalInterleaved` - Interleaved time-ordered
    (For list-mode reconstruction)

The most common usage is through `getBinIter()` on histogram or list-mode data.
This function will create a bin iterator using the provided arguments and return
it.

## For plugin developers

When defining a new histogram data format, one must define the `getBinIter(...)`
function in order to allow the new histogram format to be used for
reconstruction or projection purposes.

