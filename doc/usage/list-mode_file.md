# List-mode file
The default YRT-PET list-mode file is a record of all the events to be
considered for the reconstruction.

```
Timestamp for event 0 (ms) (uint32)
Detector1 of the event 0 (uint32)
Detector2 of the event 0 (uint32)
TOF position of event 0 (ps) (float32) [Optional]
Timestamp for event 1 (ms) (uint32)
Detector1 of the event 1 (uint32)
Detector2 of the event 1 (uint32)
TOF position of event 1 (ps) (float32) [Optional]
...
```

The file extension used is `.lmDat`.
The detectors specified in the List-Mode correspond to the indices in the
scanner's LUT.

If the ListMode file contains TOF information, the option `--flag_tof` must be
used in the executable(s).

## For Python users

If using python bindings, here's how to read a ListModeLUT:

```python
import pyyrtpet as yrt

scanner = yrt.Scanner("myscanner.json")
lm = yrt.ListModeLUTOwned(scanner, "mylistmode.lmDat", flag_tof=[True / False])
```

The `flag_tof` option specifies if the list-mode contains TOF information for
each event.

# List-Mode-DOI file
A more advanced List-Mode format is available for scanners with a very high
amount of DOI layers. Documentation for this format is still in construction
