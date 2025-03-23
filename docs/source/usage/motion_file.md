# Motion information file

Motion is recorded in a binary file with `float32` values.

This is the structure of the file:

```
Frame 0: Timestamp (uint32)
Frame 0: r00 (float32)
Frame 0: r01 (float32)
Frame 0: r02 (float32)
Frame 0: r10 (float32)
Frame 0: r11 (float32)
Frame 0: r12 (float32)
Frame 0: r20 (float32)
Frame 0: r21 (float32)
Frame 0: r22 (float32)
Frame 0: tx (float32)
Frame 0: ty (float32)
Frame 0: tz (float32)
Frame 1: Timestamp (uint32)
Frame 1: r00 (float32)
Frame 1: r01 (float32)
...
```

The timestamp encoded is the starting timestamp of the frame.
It is the same timestamp stored in the List-Mode file.
The units are milliseconds.

The motion encoded is defined by a rotation matrix and a translation vector.
The rotation matrix is defined as:
```
r00 r01 r02
r10 r11 r12
r20 r21 r22
```

The translation vector is `tx`, `ty`, and `tz`.
