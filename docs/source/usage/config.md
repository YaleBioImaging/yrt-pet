# YRT-PET Configuration

## Number of threads
Since YRT-PET currently use the OpenMP library to parallelize work, the thread
selection is managed by that library.
Check if the environment variable `OMP_NUM_THREADS` is set. This variable will
act as the default number of threads used when calling YRT-PET without
specifying `--num_threads`.
If `OMP_NUM_THREADS` is unset, by default, OpenMP will select all available
threads on the machine, regardless of other processes.

Alternatively, one run call YRT-PET (or any process) using `taskset` to limit
CPU core selection.

## From Python
Using the Python bindings, it is possible to call `yrt.setNumThreads(...)` to set the number of
threads OpenMP will use for parallelized operations. This will not alter any
environment variable, it will only affect the current process.

The `yrt.getNumThreads()` function also exists to gather that information.

## Disabling page-locked memory (or pinned memory)
For GPU operations, the intermediary buffers are allocated as page-locked
memory. This increases speed as it can allow for asynchronous copies between
host and device.

It is possible, however, to disable this behavior by setting the
`YRTPET_DISABLE_PINNED_MEMORY` environment variable to `yes`.

## From Python
Using the Python bindings, it is possible to call
`yrt.setPinnedMemoryEnabled(...)` to define this option.
This will not alter any environment variable, it will only affect the current
process.

The `yrt.isPinnedMemoryEnabled()` function also exists to gather that
information.
