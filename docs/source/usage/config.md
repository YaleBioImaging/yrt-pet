# YRT-PET Configuration

## Number of threads
Since YRT-PET currently use the OpenMP library to parallelize work, the thread
selection is managed by that library.
Check if the environment variable `OMP_NUM_THREADS` is set. This variable will
act as the default number of threads used when calling YRT-PET without
specifying `--num_threads`.
If `OMP_NUM_THREADS` is unset, by default, OpenMP will select all available
threads on the machine, regardless of other processes.

## Disabling page-locked memory (or pinned memory)
For GPU operations, the intermediary buffers are allocated as page-locked
memory. This increases speed as it can allow for asynchronous copies between
host and device.

It is possible, however, to disable this behavior by setting the
`YRTPET_DISABLE_PINNED_MEMORY` environment variable to `yes`.
