# Motion information file

Motion is recorded in a CSV file.
Each line represents a frame.
This is the structure of the CSV:

```
t, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, e
t, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, e
t, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, e
...
```

The timestamp `t` is the starting timestamp of the frame. It is an integer.
It is in the *same time reference* as the timestamps stored in the List-Mode
file.  The units are milliseconds.

The motion is defined by a rotation matrix and a translation vector.
The rotation matrix is defined as:
```
r00 r01 r02
r10 r11 r12
r20 r21 r22
```

The translation vector is defined by `tx`, `ty`, and `tz`.

The error value `e` is between 0 and 1.
