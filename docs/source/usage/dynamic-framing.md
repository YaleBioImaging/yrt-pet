# Dynamic Framing

Dynamic framing manages timestamp ranges for dynamic reconstructions.

The framing defines the time ranges used for assigning a dynamic frame to each
PET list-mode event. This is used for dynamic reconstruction as well as simple
forward-backward projection operations.

This is used along the fourth dimension of the Image class, which only defines
the *number* of dynamic frames, not the time range associated to each one.

The `DynamicFraming` object is used by the `OSEM` or the `OperatorProjector`
to correctly map each list-mode event to the appropriate frame in the fourth
dimension of the image space, allowing 4D reconstruction or projection.

The timestamps provided are in milliseconds and are in the same referrential as
the timestamps of the list-mode or the motion file.

## Python Usage

```python
import pyyrtpet as yrt
import numpy as np

# ============================================================================
# Method 1: Create dynamic framing with a specified number of frames
# ============================================================================

# Create a DynamicFraming with 10 frames (requires setting timestamps afterwards)
framing = yrt.DynamicFraming(num_frames=10)

# Set frame start times - timestamps must be in chronological order
# Frame 0 starts at 0 ms
framing.setStartingTimestamp(frame=0, timestamp=0)
# Frame 1 starts at 10 ms
framing.setStartingTimestamp(frame=1, timestamp=10)
# Frame 2 starts at 20 ms
framing.setStartingTimestamp(frame=2, timestamp=20)
# Continue for all frames...
framing.setStartingTimestamp(frame=3, timestamp=30)
framing.setStartingTimestamp(frame=4, timestamp=45)
framing.setStartingTimestamp(frame=5, timestamp=60)
framing.setStartingTimestamp(frame=6, timestamp=80)
framing.setStartingTimestamp(frame=7, timestamp=100)
framing.setStartingTimestamp(frame=8, timestamp=125)
framing.setStartingTimestamp(frame=9, timestamp=135)
# The last timestamp marks the end of the last frame
framing.setLastTimestamp(timestamp=150)

# Verify the framing is valid (timestamps in chronological order)
# This will also fail if some frames were left unset.
assert framing.isValid(), "Timestamps must be in chronological order"

# Get the number of frames
num_frames = framing.getNumFrames()
assert num_frames == 10, f"Expected 10 frames, got {num_frames}"

# ============================================================================
# Method 2: Create dynamic framing from a numpy array of timestamps
# ============================================================================

# Frame timestamps: start of each frame + end timestamp
# This creates 3 frames: [0-10ms], [10-30ms], [30-60ms]
# The dtype of the numpy array must be `uint32`
timestamps = np.array([0, 10, 30, 60], dtype=np.uint32)
framing_from_array = yrt.DynamicFraming(frame_timestamps=timestamps)
assert framing_from_array.getNumFrames() == 3
# Note that the array has 4 elements, but we defined 3 frames.
# This is because the last timestamp defines the end of the last frame.

# ============================================================================
# Method 3: Load dynamic framing from a file (.dyn extension)
# ============================================================================

# For this documentation's purposes only, we use a tempfile
import tempfile
with tempfile.NamedTemporaryFile(suffix='.dyn', delete=False) as f:
    my_file = f.name

# Framing data is stored as a text file containing timestamps separated by a
#  whitespace

# To save:
framing.writeToFile(my_file)
# To load:
framing_from_file = yrt.DynamicFraming(my_file)

# Check integrity
assert framing_from_file.getNumFrames() == framing.getNumFrames()
for i in range(framing.getNumFrames()):
    assert framing_from_file.getStartingTimestamp(i) == framing.getStartingTimestamp(i)
assert framing_from_file.getLastTimestamp() == framing.getLastTimestamp()

# ============================================================================
# Query methods
# ============================================================================

# Get total duration of the dynamic framing (last timestamp - first timestamp)
total_duration = framing.getTotalDuration()
assert total_duration == 150, f"Expected 150ms, got {total_duration}"

# Get duration of a specific frame
frame_duration = framing.getDuration(frame=0)  # Duration of frame 0 (10 ms)
assert frame_duration == 10, f"Expected 10ms, got {frame_duration}"

# Get the timestamp when a specific frame starts
start_ts = framing.getStartingTimestamp(frame=0)  # 0
start_ts = framing.getStartingTimestamp(frame=5)  # 60

# Get the timestamp when a specific frame ends (i.e., next frame starts)
stop_ts = framing.getStoppingTimestamp(frame=0)  # 10 (start of frame 1)
stop_ts = framing.getStoppingTimestamp(frame=4)  # 80 (start of frame 5)

# Get number of timestamps (frames + 1 for the final timestamp)
num_timestamps = framing.getNumTimestamps()  # 11 (10 frames + 1 end)

```

## File Format

The dynamic framing can be saved to a text file with the `.dyn` extension.
Each line contains a single timestamp (in ms):

```
0
10
30
60
120
180
```

The file must contain at least 2 timestamps (defining 1 frame).

## Notes

- Timestamps must be strictly increasing (each frame must start after the previous, without overlap)
- The number of timestamps equals `num_frames + 1` (frame starts + final timestamp)
