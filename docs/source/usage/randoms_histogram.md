# Randoms histogram format

A RandomsHistogram stores the singles count rate of every detector of the
scanner, together with the coincidence time window used for the randoms
estimation. The randoms estimate for a detector pair
`(d1, d2)` is computed on the fly as:

```
randoms(d1, d2) = 2 * time_window * singles_rate(d1) * singles_rate(d2)
```

When used as input, the format name is `RH`. This is a `Histogram` format, it
can therefore be used to correct for randoms for the reconstructions or for the
scatter estimation's tail-fitting step.

## File format

The file is a **plain text file**, with one value per line.
The first line holds the coincidence time window in seconds, followed by one
singles rate per line, in detector id order:

```
<time window (seconds)>
<singles rate of detector 0 (counts per second)>
<singles rate of detector 1 (counts per second)>
...
<singles rate of detector N - 1 (counts per second)>
```

where `N` is the number of detectors of the scanner
(`numDOI x numRings x detsPerRing`).

Example with a coincidence time window of 4.5 nanoseconds:

```
4.5e-9
16.2
22.3
15.1
9.7
18.1
...
```

Notes:

- The singles rates are in counts per second.
- The time window and singles rates are encoded as `float32` values in memory.
- The format is one value per line, so any whitespace separated file is
  accepted when reading.
