# YRT-PET Configuration

## Number of threads
Since YRT-PET currently uses the `std::thread` library to parallelize work, the
thread selection is managed by that library.  YRT-PET uses the maximum number of
available threads unless the `--num_threads` is passed to the executables.

Alternatively, one can run YRT-PET (or any process) using `taskset` to limit CPU
core selection.

### From Python
Using the Python bindings, it is possible to call `yrt.setNumThreads(...)` to
set the number of threads that will used for parallelized operations. This will
only affect the current process.

The `yrt.getNumThreads()` function also exists to gather that information.

## Disabling page-locked memory (or pinned memory)
For GPU operations, the intermediary buffers are allocated as page-locked
memory. This increases speed as it can allow for asynchronous copies between
host and device.

It is possible, however, to disable this behavior by setting the
`YRTPET_DISABLE_PINNED_MEMORY` environment variable to `yes`.

## From Python
Using the Python bindings, it is possible to call
`yrt.setPinnedMemoryEnabled(...)` to define this option.  This will not alter
any environment variable, it will only affect the current process.

The `yrt.isPinnedMemoryEnabled()` function also exists to gather that
information.
