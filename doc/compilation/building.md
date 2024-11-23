# Building YRT-PET

## Requirements

- SimpleITK
    - Need to set `SimpleITK_DIR` environment variable
      to `<SimpleITK build directory>/SimpleITK-build/`
    - Need to set `ITK_DIR` environment variable
      to `<SimpleITK build directory>/ITK-build/`
- `pybind11` if compiling the python bindings (ON by default)
- CUDA toolkit if compiling using GPU (ON by default)
- An internet connection to download the `cxxopts`, `nlohmann/json`,
  and `catch2` libraries
- OpenMP, but this is baked into most compilers

## Configuration and Compilation

From the root of the repository:\
`mkdir build`\
`cd build`\
With `[ON/OFF]` being replaced by the desired configuration:\
`cmake ../yrt-pet/ -DUSE_CUDA=[ON/OFF] -DBUILD_PYBIND11=[ON/OFF]`

- The `-DUSE_CUDA` option enables or disables GPU accelerated code
    - This option is `ON` by default
- The `-DBUILD_PYBIND11` option enables or disables YRT-PET's python bindings
    - This option is `ON` by default

## Caveats

- If compiling with GPU acceleration enabled, note that by default, the
  architecture
  that the code will be compiled towards will be `native`. This means that
  YRT-PET will be compiled for the architecture of the host's GPU.
    - One can bypass this behavior using one of the two following ways:
        - Set the `CUDAARCHS` environment variable.
        - Add `-DCMAKE_CUDA_ARCHITECTURES=[CUDA architectures list]`.
    - Example: `-DCMAKE_CUDA_ARCHITECTURES="61;75"`
    - [More information here](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
- If compiling with GPU acceleration enabled, make sure the `CUDACXX`
  environment variable
  is set to the location of `nvcc`. Run `echo $CUDACXX` to verify this.
- In order to compile with the python bindings, one needs to have a working
  python installation or activate a virtual environment.
    - The python bindings will only work for the host's Python version and only
      for the CPython implementations.
        - Example: One cannot compile YRT-PET with python bindings using Python
          3.10 and expect them to work within Python 3.11
    - To add the YRT-PET python bindings to the python environment, add
      the `build` folder to the `PYTHONPATH` environment variable.
    - To test the python bindings,
      run: `python -c "import pyyrtpet as yrt; print(yrt.compiledWithCuda());"`
        - If the `PYTHONPATH` environment variable is not set or misplaced,
          the error will look like:
            - `ModuleNotFoundError: No module named 'pyyrtpet'`
        - If there is a mismatch between the python version used to compile
          YRT-PET and the environment's python version, the error will look
          like:
            - `ImportError: No module named pyyrtpet_wrapper._pyyrtpet`
        - If the `PYTHONPATH` environment is properly set and the python
          versions match, the command will either print `True` or `False`.
            - `True` if the project was compiled with `-DUSE_CUDA=ON`
            - `False` if the project was compiled with `-DUSE_CUDA=OFF`
- YRT-PET uses OpenMP to parallelize most of the code. By default,
  YRT-PET will use the maximum amount of threads available. If the environment
  variable `OMP_NUM_THREADS` is set, YRT-PET will use that number instead.
    - It is common within HPC cluster infrastructures to override this
      environment variable to a smaller number like 4. Beware of such situations
      if computation time is a constraint.
