# Vector3D and Line3D

Vector3D and Line3D are geometric classes for representing 3D vectors and lines
in the YRT-PET framework.
These classes are used to define Lines of Response (LORs) as 3D lines, which
are used in various cases, most notably in projection operations.

## Vector3D

Vector3D represents a 3D vector with single or double precision.
By default, we use single precision (32 bits).

### Python Usage

```python
import pyyrtpet as yrt
import math

# ============================================================================
# Creating Vectors
# ============================================================================

# Create a vector with x, y, z components
v = yrt.Vector3D(1.0, 2.0, 3.0)

# Access individual components
x = v.x
y = v.y
z = v.z

assert x == 1.0
assert y == 2.0
assert z == 3.0

# Create a default vector (all zeros)
v_zero = yrt.Vector3D()

# ============================================================================
# Vector Operations
# ============================================================================

v1 = yrt.Vector3D(1.0, 2.0, 3.0)
v2 = yrt.Vector3D(4.0, 5.0, 6.0)

# Addition
v_sum = v1 + v2
assert v_sum.x == 5.0
assert v_sum.y == 7.0
assert v_sum.z == 9.0

# Subtraction
v_diff = v2 - v1
assert v_diff.x == 3.0
assert v_diff.y == 3.0
assert v_diff.z == 3.0

# Scalar multiplication
v_mult = v1 * 2.0
assert v_mult.x == 2.0
assert v_mult.y == 4.0
assert v_mult.z == 6.0

# Scalar addition
v_add = v1 + 1.0
assert v_add.x == 2.0
assert v_add.y == 3.0
assert v_add.z == 4.0

# Scalar subtraction
v_sub = v1 - 1.0
assert v_sub.x == 0.0
assert v_sub.y == 1.0
assert v_sub.z == 2.0

# Scalar division
v_div = v1 / 2.0
assert v_div.x == 0.5
assert v_div.y == 1.0
assert v_div.z == 1.5

# ============================================================================
# Vector Properties and Methods
# ============================================================================

v = yrt.Vector3D(3.0, 4.0, 0.0)

# Get the Euclidean norm (length) of the vector
norm = v.getNorm()
assert abs(norm - 5.0) < 0.001, "3-4-0 triangle has norm 5"

# Normalize the vector (make it a unit vector)
v_normalized = v.getNormalized()
assert abs(v_normalized.getNorm() - 1.0) < 0.001

# In-place normalization
v_to_normalize = yrt.Vector3D(3.0, 4.0, 0.0)
v_to_normalize.normalize()
assert abs(v_to_normalize.getNorm() - 1.0) < 0.001

# Check if vector is normalized
assert v_normalized.isNormalized() == True
assert v_to_normalize.isNormalized() == True
assert v.isNormalized() == False

# Update vector components
v = yrt.Vector3D(1.0, 2.0, 3.0)
v.update(10.0, 20.0, 30.0)
assert v.x == 10.0
assert v.y == 20.0
assert v.z == 30.0

# Update from another vector
v1 = yrt.Vector3D(1.0, 2.0, 3.0)
v2 = yrt.Vector3D(4.0, 5.0, 6.0)
v1.update(v2)
assert v1.x == 4.0
assert v1.y == 5.0
assert v1.z == 6.0

# ============================================================================
# Dot product and cross product
# ============================================================================

v1 = yrt.Vector3D(1.0, 0.0, 0.0)
v2 = yrt.Vector3D(0.0, 1.0, 0.0)

# Dot product
dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
assert dot == 0.0  # Orthogonal vectors

# Cross product (vector product)
# Note: We use the multiplication operator (*) for cross product (not dot product)
cross = v1 * v2  # Using the * operator for cross product
assert cross.x == 0.0
assert cross.y == 0.0
assert cross.z == 1.0

# ============================================================================
# Comparison and Equality
# ============================================================================

v1 = yrt.Vector3D(1.0, 2.0, 3.0)
v2 = yrt.Vector3D(1.0, 2.0, 3.0)
v3 = yrt.Vector3D(4.0, 5.0, 6.0)

assert (v1 == v2) == True
assert (v1 == v3) == False

# ============================================================================
# String Representation
# ============================================================================

v = yrt.Vector3D(1.5, 2.5, 3.5)
repr_str = str(v)
# Output: (1.5, 2.5, 3.5)
assert "1.5" in repr_str
assert "2.5" in repr_str
assert "3.5" in repr_str

# ============================================================================
# Double Precision Version
# ============================================================================

# For higher precision calculations, use Vector3DDouble
v_double = yrt.Vector3DDouble(1.0, 2.0, 3.0)
norm_double = v_double.getNorm()
```

## Line3D

Line3D represents a 3D line segment defined by two endpoints (point1 and point2).

### Python Usage

```python
import pyyrtpet as yrt
import math

# ============================================================================
# Creating Lines
# ============================================================================

# Create a line from two Vector3D points
p1 = yrt.Vector3D(0.0, -100.0, 0.0)
p2 = yrt.Vector3D(0.0, 100.0, 0.0)
line = yrt.Line3D(p1, p2)

# Access the endpoints
assert line.point1.x == 0.0
assert line.point1.y == -100.0
assert line.point2.x == 0.0
assert line.point2.y == 100.0

# Create a default line (both points at origin)
line_default = yrt.Line3D()

# ============================================================================
# Line Properties
# ============================================================================

# Get the length (distance between endpoints)
length = line.getNorm()
assert abs(length - 200.0) < 0.001

# ============================================================================
# Line Operations
# ============================================================================

# Update line endpoints
p1_new = yrt.Vector3D(-50.0, -50.0, -50.0)
p2_new = yrt.Vector3D(50.0, 50.0, 50.0)
line.update(p1_new, p2_new)

assert line.point1.x == -50.0
assert line.point2.x == 50.0

# ============================================================================
# Line Comparison
# ============================================================================

# Check if two lines are equal (same endpoints)
line1 = yrt.Line3D(yrt.Vector3D(0, 0, 0), yrt.Vector3D(1, 1, 1))
line2 = yrt.Line3D(yrt.Vector3D(0, 0, 0), yrt.Vector3D(1, 1, 1))
line3 = yrt.Line3D(yrt.Vector3D(1, 1, 1), yrt.Vector3D(2, 2, 2))

assert line1.isEqual(line2) == True
assert line1.isEqual(line3) == False

# ============================================================================
# Line Parallelism
# ============================================================================

# Check if two lines are parallel
line_a = yrt.Line3D(yrt.Vector3D(0, 0, 0), yrt.Vector3D(1, 0, 0))
line_b = yrt.Line3D(yrt.Vector3D(0, 1, 0), yrt.Vector3D(1, 1, 0))
line_c = yrt.Line3D(yrt.Vector3D(0, 0, 0), yrt.Vector3D(0, 0, 1))

assert line_a.isParallel(line_b) == True  # Both along x-axis
assert line_a.isParallel(line_c) == False  # Different directions

# ============================================================================
# Tuple Conversion
# ============================================================================

# Convert line to tuple format
p1 = yrt.Vector3D(1.0, 2.0, 3.0)
p2 = yrt.Vector3D(4.0, 5.0, 6.0)
line = yrt.Line3D(p1, p2)

# toTuple returns ((x1, y1, z1), (x2, y2, z2))
tup = line.toTuple()
assert tup == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

# ============================================================================
# String Representation
# ============================================================================

line = yrt.Line3D(yrt.Vector3D(1.0, 2.0, 3.0), yrt.Vector3D(4.0, 5.0, 6.0))
repr_str = str(line)
# Output contains both points

# ============================================================================
# Double Precision Version
# ============================================================================

# For higher precision, use Line3DDouble
p1d = yrt.Vector3DDouble(0.0, -100.0, 0.0)
p2d = yrt.Vector3DDouble(0.0, 100.0, 0.0)
line_double = yrt.Line3DDouble(p1d, p2d)
```

## Practical Example: Creating LORs for Projection

```python
import pyyrtpet as yrt

# Create a scanner with a regular geometry (without specifying a custom a LUT)
scanner = yrt.Scanner(
    scanner_name='EXAMPLE',
    axial_fov=100.0,
    crystal_size_z=2.0,
    crystal_size_trans=2.0,
    crystal_depth=10.0,
    scanner_radius=200.0,
    dets_per_ring=64,
    num_rings=10,
    num_doi=1,
    max_ring_diff=9,
    min_ang_diff=1,
    dets_per_block=16
)

# Gather the 17th detector's position
p1 = scanner.getDetectorPos(17)

# Gather another detector
p2 = scanner.getDetectorPos(142)

# Build an LOR
lor = yrt.Line3D(p1, p2)

# You can then use this LOR with the Siddon or DD projector
# for forward/backward projections
```

## Notes

- Both Vector3D and Line3D support single (float) and double precision versions
- The single precision versions are named `Vector3D` and `Line3D`
- The double precision versions are named `Vector3DDouble` and `Line3DDouble`
- Line3D stores two Vector3D endpoints, so any vector operations can be
  performed on `line.point1` and `line.point2` individually
