# Sparse histogram file

Sparse histograms are a list of detector pairs and the value associated with the
LOR.

They are structured as follows in the file:

```
Detector1 of the event 0 (uint32)
Detector2 of the event 0 (uint32)
Projection value of the event 0 (float32)
Detector1 of the event 1 (uint32)
Detector2 of the event 1 (uint32)
Projection value of the event 1 (float32)
...
```

The file extension used is `.shis` by convention.
The detectors specified in the Sparse histogram correspond to their indices in
the
scanner's LUT.

# Implementation

Sparse histograms are, in memory, stored as a hash map with the detector pair as
a key, ensuring O(1) when accessing projection values.

Currently, the implementation depends on `std::unordered_map`, which tends to be
quite slow and is not thread-safe. This will be addressed in the future.
