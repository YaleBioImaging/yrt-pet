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

The file extension used is `.lmDat` by convention.
The detectors specified in the List-Mode correspond to the indices in the
scanner's LUT.

If the ListMode file contains time-of-flight (TOF) information, the option
`--flag_tof` must be used in the executable(s). The TOF value is the difference
of arrival time between detector 2 ($t_2$) and detector 1 ($t_1$),
or $t_2 - t_1$, expressed in picoseconds.

## For Python users

If using python bindings, here's how to read a ListModeLUT:

```python
import pyyrtpet as yrt

scanner = yrt.Scanner("myscanner.json")
flag_tof = True  # Indicate whether the list-mode file contains a TOF field
lm = yrt.ListModeLUTOwned(scanner, "mylistmode.lmDat", flag_tof=flag_tof)
```

The `flag_tof` option specifies if the list-mode contains TOF information for
each event.

# List-mode DOI file

A more advanced List-Mode format is available for scanners with many DOI layers.
The format is similar to the regular List-mode format described above with DOI
layer (encoded in 256 bits) for each detector (from the inward face of the
detector). The `num_layers` option in the reconstruction executable allows
binning of the DOI layers from 256 layers to an arbitrary number of layers.

**Note that it is still possible to use the default list-mode format for
DOI-enabled scanners**. This additional format only allows to save some disk
space
in case the amount of DOI layers in the scanner is configurable.

```
Timestamp for event 0 (ms) (uint32)
Detector1 of the event 0 (uint32)
DOI1 of the event 0 (uint8)
Detector2 of the event 0 (uint32)
DOI2 of the event 0 (uint8)
TOF position of event 0 (ps) (float32) [Optional]
Timestamp for event 1 (ms) (uint32)
Detector1 of the event 1 (uint32)
DOI1 of the event 1 (uint8)
Detector2 of the event 1 (uint32)
DOI2 of the event 2 (uint8)
TOF position of event 1 (ps) (float32) [Optional]
...
```
