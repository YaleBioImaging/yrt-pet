# CUDA-to-backend inventory

Date: 2026-05-24

Scope: checkpoint the current macOS CPU-only compatibility work, verify the CPU
build still works, and inventory the CUDA surface before any Metal port starts.

## Checkpoint

No Metal implementation was started in this checkpoint.

Build artifact hygiene:

- `install_cpu/` is excluded locally through `.git/info/exclude`.
- Existing build directories such as `build_cpu/`, `build_cpu_py/`, and
  `build_cpu_xcode_sdk/` remain ignored by the repository ignore rules.
- `git status --short` does not list `install_cpu/`.

CPU build verification:

```sh
cmake -S yrt-pet -B build_cpu_xcode_sdk -DUSE_CUDA=OFF -DBUILD_PYBIND11=OFF -DBUILD_TESTS=OFF -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk
cmake --build build_cpu_xcode_sdk -j 8
build_cpu_xcode_sdk/executables/yrtpet_reconstruct --version
build_cpu_xcode_sdk/executables/yrtpet_reconstruct --help
```

Result: build succeeded. `yrtpet_reconstruct --version` printed
`2.0.4-fa07acc-dirty`, and `--help` ran successfully.

Python CPU binding verification:

```sh
cmake -S yrt-pet -B build_cpu_py -DUSE_CUDA=OFF -DBUILD_PYBIND11=ON -DBUILD_TESTS=OFF -DPython_EXECUTABLE=/Users/yanischemli/miniconda3/bin/python3 -Dpybind11_DIR=/Users/yanischemli/miniconda3/lib/python3.13/site-packages/pybind11/share/cmake/pybind11 -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk
cmake --build build_cpu_py -j 8
/Users/yanischemli/miniconda3/bin/python3 -u -c "import sys; sys.path.insert(0, '/Users/yanischemli/Documents/Codex/2026-05-24/yalebioimaging-yrt-pet-https-github-com/src/build_cpu_py'); import pyyrtpet as yrt; print(yrt.getVersion()); print(yrt.compiledWithCuda())"
```

Result: build succeeded. Python imported `pyyrtpet`, printed
`2.0.4-fa07acc-dirty`, and `compiledWithCuda()` returned `False`.

## Changed files

| File | CPU-only compatibility purpose |
| --- | --- |
| `yrt-pet/CMakeLists.txt` | Uses CMake's `LINK_LIBRARY:WHOLE_ARCHIVE` generator expression instead of GNU-only linker flags, and avoids linking `Python_LIBRARIES` into the pybind module on Apple while still linking Python libraries into executables when needed. |
| `yrt-pet/executables/conversion/ConvertToLowCountListModeLUT.cpp` | Disambiguates project `yrt::size_t` from platform `size_t`. |
| `yrt-pet/executables/conversion/Histogram3DToListMode.cpp` | Disambiguates project `yrt::size_t`. |
| `yrt-pet/executables/recon/EstimateScatter.cpp` | Disambiguates project `yrt::size_t`. |
| `yrt-pet/executables/utils/AccumulateToDetectors.cpp` | Disambiguates project `yrt::size_t`. |
| `yrt-pet/executables/utils/PostReconMotionCorrection.cpp` | Disambiguates project `yrt::size_t`. |
| `yrt-pet/include/yrt-pet/operators/TimeOfFlight.hpp` | Adds the missing standard include needed for `std::min`/`std::max`. |
| `yrt-pet/include/yrt-pet/utils/AtomicUtils.hpp` | Adds `yrt::util::atomicAdd`, a portable host-side atomic-add wrapper that uses `std::atomic_ref` when available and a compare-exchange fallback for Apple toolchains where `atomic_ref::fetch_add` is unavailable for float/double. |
| `yrt-pet/include/yrt-pet/utils/Timer.hpp` | Aligns the stored time point type with `std::chrono::high_resolution_clock`. |
| `yrt-pet/src/datastruct/image/ImageParams.cpp` | Adds a missing `<sstream>` include. |
| `yrt-pet/src/datastruct/projection/LORMotion.cpp` | Replaces float `std::from_chars` use with a `strtof` fallback path for macOS libc++ compatibility. |
| `yrt-pet/src/operators/OperatorVarPsf.cpp` | Uses `yrt::util::atomicAdd` instead of direct `std::atomic_ref` arithmetic. |
| `yrt-pet/src/operators/ProjectorDD.cpp` | Uses `yrt::util::atomicAdd`. |
| `yrt-pet/src/operators/ProjectorSiddon.cpp` | Uses `yrt::util::atomicAdd`. |
| `yrt-pet/src/operators/ProjectorUpdater.cpp` | Uses `yrt::util::atomicAdd` for CPU projector updater accumulation. |
| `yrt-pet/src/scatter/ScatterSpace.cpp` | Uses `yrt::util::atomicAdd`. |
| `yrt-pet/src/utils/FileReader.cpp` | Fixes a strict type mismatch around `std::min` and stream sizes. |
| `yrt-pet/src/utils/ReconstructionUtils.cpp` | Uses `yrt::util::atomicAdd` and macOS-safe project-size type spelling. |

## CUDA file inventory

| CUDA file | Classes/functions supported | Kernels defined or declared | Closest CPU equivalent |
| --- | --- | --- | --- |
| `yrt-pet/include/yrt-pet/datastruct/image/ImageDevice.cuh` | Declares `ImageDevice`, `ImageDeviceOwned`, and `ImageDeviceAlias`, the device-backed `ImageBase` implementation with host/device transfers and GPU image operations. | None in this file. | `yrt-pet/include/yrt-pet/datastruct/image/Image.hpp`, `yrt-pet/src/datastruct/image/Image.cpp`, `ImageBase.cpp`. |
| `yrt-pet/include/yrt-pet/datastruct/image/ImageSpaceKernels.cuh` | Declares image-space kernels for EM updates, thresholding, fill, addition, scaling, motion averaging, and image PSF convolution. | Declares `updateEM_kernel`, `updateEMDynamic_kernel`, `applyThreshold_kernel`, `applyThresholdBroadcast_kernel`, `fill_kernel`, `addFirstImageToSecond_kernel`, `addFirstImage3DToSecond4D_kernel`, `multWithScalar_kernel`, `timeAverageMoveImage_kernel`, `convolve3DSeparable_kernel`, `convolve3D_kernel`. | `Image.cpp` voxel operations, `OperatorPsf.cpp`, `OperatorVarPsf.cpp`, `ReconstructionUtils.cpp` motion averaging. |
| `yrt-pet/include/yrt-pet/datastruct/image/ImageUtils.cuh` | Provides host/device interpolation helpers: `trilinearInterpolateCore`, `trilinearInterpolate`, and `indexToPosition`. | None. | `Image.cpp` interpolation and transform/resample helpers. |
| `yrt-pet/include/yrt-pet/datastruct/projection/LORsDevice.cuh` | Declares `LORsDevice`, which precomputes, stages, and uploads LOR/projection-property batches through `BinLoader` and `PropStructDevice`. | None. | `BinLoader`, `ProjectionData`, `ProjectionList`, and CPU projector property collection. |
| `yrt-pet/include/yrt-pet/datastruct/projection/ProjectionListDevice.cuh` | Declares `ProjectionListDevice`, `ProjectionListDeviceOwned`, and `ProjectionListDeviceAlias`, including device projection-value buffers, batching, LOR staging, and projection vector operations. | None in this file; methods launch projection-space kernels. | `ProjectionList`, `ProjectionListOwned`, `ProjectionData`, and `Histogram` classes. |
| `yrt-pet/include/yrt-pet/datastruct/scanner/ScannerDevice.cuh` | Declares `ScannerDevice`, a device upload/cache for detector positions and orientations. | None. | `Scanner` and `DetectorSetup` host accessors. |
| `yrt-pet/include/yrt-pet/operators/DDKernels.cuh` | Implements distance-driven projection device helpers: `getOverlap_safe` overloads and `projectDD`. | Defines template kernel `projectDD_kernel<IsForward, HasTOF, HasProjPSF, UseUpdater>`. | `ProjectorDD.hpp`, `ProjectorDD.cpp` (`dd_project_ref`, forward/back projection, projection PSF weighting). |
| `yrt-pet/include/yrt-pet/operators/DeviceSynchronized.cuh` | Declares `DeviceSynchronized`, `initiateDeviceParameters`, and converters to `CUScannerParams`, `CUImageParams`, and `CUImage`. | None. | No direct CPU equivalent; closest host orchestration is the CPU operator/reconstruction setup code. |
| `yrt-pet/include/yrt-pet/operators/OperatorProjectorDD_GPU.cuh` | Declares `OperatorProjectorDD_GPU`, the GPU distance-driven projector operator. | None in this file; declares templated `launchKernel`. | `OperatorProjector`, `ProjectorDD`. |
| `yrt-pet/include/yrt-pet/operators/OperatorProjectorDevice.cuh` | Declares `OperatorProjectorDevice`, the common GPU projector base for batching, TOF setup, projection PSF setup, and updater setup. | None. | `OperatorProjector`, `Projector`, `ProjectorUpdater`, `ProjectionPsfManager`. |
| `yrt-pet/include/yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh` | Declares `OperatorProjectorSiddon_GPU`, including `getNumRays`, `setNumRays`, batch apply methods, and templated `launchKernel`. | None in this file. | `OperatorProjector`, `ProjectorSiddon`. |
| `yrt-pet/include/yrt-pet/operators/OperatorPsfDevice.cuh` | Declares `OperatorPsfDevice`, the GPU uniform/separable image PSF operator with device kernel buffers and temporary device image storage. | None in this file. | `OperatorPsf`. |
| `yrt-pet/include/yrt-pet/operators/OperatorVarPsfDevice.cuh` | Declares `OperatorVarPsfDevice` and `DeviceVarPsf` for variable image PSF convolution using flattened kernel LUT storage. | None in this file. | `OperatorVarPsf`. |
| `yrt-pet/include/yrt-pet/operators/ProjectionPsfManagerDevice.cuh` | Declares `ProjectionPsfProperties`, `ProjectionPsfKernelStruct`, and `ProjectionPsfManagerDevice` for projection-space PSF LUT upload. | None. | `ProjectionPsfManager`. |
| `yrt-pet/include/yrt-pet/operators/ProjectionPsfUtils.cuh` | Provides device helpers `util::getKernel` and `util::getWeight` for projection-space PSF lookup and integration. | None. | `ProjectionPsfManager::getKernel`, `ProjectionPsfManager::getWeight`, `ProjectorDD` PSF code. |
| `yrt-pet/include/yrt-pet/operators/ProjectionSpaceKernels.cuh` | Declares projection-value array kernels used by `ProjectionListDevice`. | Declares `divideMeasurements_kernel`, `addProjValues_kernel`, `invertProjValues_kernel`, `convertToACFs_kernel`, two `multiplyProjValues_kernel` overloads, and `clearProjections_kernel`. | `ProjectionData::operationOnEachBin`, `ProjectionList`, and `ReconstructionUtils::convertProjectionValuesToACF`. |
| `yrt-pet/include/yrt-pet/operators/ProjectorUpdaterDevice.cuh` | Declares device updater hierarchy: `ProjectorUpdaterDevice`, `ProjectorUpdaterDeviceDefault4D`, `ProjectorUpdaterDeviceLRUnrolled`, `ProjectorUpdaterDeviceLR`, `ProjectorUpdaterDeviceWrapper`, `dispatchCreateUpdaterLR`. | Defines `constructUpdaterOnDevice` and `destroyUpdaterOnDevice`. | `ProjectorUpdater`, `ProjectorUpdaterDefault4D`, `ProjectorUpdaterLR`, `ProjectorUpdaterLRDualUpdate`. |
| `yrt-pet/include/yrt-pet/operators/ProjectorWrapper.cuh` | Provides `projectAny`, a device dispatcher between Siddon and distance-driven projectors for OSEM kernels. | None. | CPU dispatch through `Projector` subclasses and `OperatorProjector`. |
| `yrt-pet/include/yrt-pet/operators/SiddonKernels.cuh` | Implements Siddon device helpers: `setupMultiRays`, `moveLineToRandomOffset`, `projectSiddon`, `projectSiddonDefault`. | Defines template kernel `projectSiddon_kernel<IsForward, HasTOF, IsIncremental, IsMultiRay, UseUpdater>`. | `ProjectorSiddon.hpp`, `ProjectorSiddon.cpp` (`project_helper`, single forward/back projection). |
| `yrt-pet/include/yrt-pet/recon/Corrector_GPU.cuh` | Declares `Corrector_GPU`, the GPU-flavored attenuation/precorrection factor precompute path. | None. | `Corrector_CPU`, `Corrector`. |
| `yrt-pet/include/yrt-pet/recon/LREM_GPU.cuh` | Declares `LREM_GPU`, the low-rank EM GPU reconstruction subclass with device `cW`/`cH` update buffers and HBasis sync methods. | None. | `LREM_CPU`, `LREM`, `OSEM_CPU`. |
| `yrt-pet/include/yrt-pet/recon/OSEM_GPU.cuh` | Declares `OSEM_GPU`, its device buffers, batch property staging, GPU streams, updater, TOF, projection PSF, and sensitivity/EM-update methods. | Declares `generateSensImage_kernel` and `computeEMUpdateImage_kernel<UseUpdater>`. | `OSEM_CPU`, `OSEM`. |
| `yrt-pet/include/yrt-pet/utils/DeviceArray.cuh` | Declares `DeviceArray<T>`, a one-dimensional device allocation/copy/memset wrapper. | None. | Standard host containers/arrays; backend equivalent should be a generic `DeviceBuffer<T>`. |
| `yrt-pet/include/yrt-pet/utils/DeviceObject.cuh` | Declares `DeviceObject<T>`, `makeDeviceObject`, and `makeDeviceObjectDerived` for device-side object allocation and placement construction. | Defines `constructOnDevice`, `constructOnDeviceNoArgs`, `destroyOnDevice`. | No CPU equivalent; CPU uses normal object lifetime. |
| `yrt-pet/include/yrt-pet/utils/DeviceSynchronizedObject.cuh` | Declares `DeviceSynchronizedObject<T>`, which mirrors a small host object to device memory. | None. | Plain host objects in CPU operators. |
| `yrt-pet/include/yrt-pet/utils/GPUKernelUtils.cuh` | Provides device `float3` arithmetic, `normalize`, `cross`, variadic `min`/`max`, and attenuation kernel declaration. | Declares `applyAttenuationFactors_kernel`. | `Vector3D`/geometry helpers and attenuation conversion in `ReconstructionUtils.cpp`; the kernel appears currently unused by call sites. |
| `yrt-pet/include/yrt-pet/utils/GPUMemory.cuh` | Provides template memory helpers: `allocateDevice`, `deallocateDevice`, `copyHostToDevice`, `copyDeviceToHost`, `copyDeviceToDevice`, `memsetDevice`. | None. | Host allocation/copy; backend equivalent should abstract allocation/copy for CUDA, Metal, and CPU fallback. |
| `yrt-pet/include/yrt-pet/utils/GPUStream.cuh` | Declares `GPUStream`, an RAII wrapper over `cudaStream_t`. | None. | No CPU equivalent; backend equivalent should be a stream/command-queue wrapper. |
| `yrt-pet/include/yrt-pet/utils/GPUTypes.cuh` | Declares `GPUBatchSetup`, `GPULaunchConfig`, `GPULaunchParams`, `GPULaunchParams3D`, and `synchronizeIfNeeded`. | None. | CPU batching loops and backend launch configuration. |
| `yrt-pet/include/yrt-pet/utils/GPUUtils.cuh` | Declares CUDA error helpers, available-VRAM query, and `HOST_DEVICE_CALLABLE`. | None. | CPU build has no direct equivalent; backend equivalent should centralize backend availability and error reporting. |
| `yrt-pet/include/yrt-pet/utils/PageLockedBuffer.cuh` | Declares `PageLockedBuffer<T>`, a CUDA pinned-host-memory wrapper with fallback to normal heap allocation. | None. | Normal host buffers such as `std::unique_ptr<T[]>`/vectors. |
| `yrt-pet/include/yrt-pet/utils/ReconstructionUtilsDevice.cuh` | Declares GPU reconstruction utilities: motion-averaged image creation, GPU OSEM factory, GPU projector factory. | None. | `ReconstructionUtils.hpp`, `ReconstructionUtils.cpp`, `createOSEM_CPU`, CPU `timeAverageMoveImage`. |
| `yrt-pet/src/datastruct/image/ImageDevice.cu` | Implements `ImageDevice`, `ImageDeviceOwned`, and `ImageDeviceAlias`; handles transfers, thresholding, EM updates, fill, scaling, additions, file IO via host images. | Defines no kernels. Launches image-space kernels from `ImageSpaceKernels.cu`. | `Image.cpp`, `ImageOwned`, `ImageAlias`, `ImageBase`. |
| `yrt-pet/src/datastruct/image/ImageSpaceKernels.cu` | Implements image-space GPU kernels and helper functions. | Defines `updateEM_kernel`, `updateEMDynamic_kernel`, `applyThreshold_kernel`, `applyThresholdBroadcast_kernel`, `fill_kernel`, `addFirstImage3DToSecond4D_kernel`, `multWithScalar_kernel`, `addFirstImageToSecond_kernel`, `timeAverageMoveImage_kernel`, device helpers `circular` and `idx3`, `convolve3DSeparable_kernel`, and `convolve3D_kernel`. | `Image.cpp`, `OperatorPsf.cpp`, `OperatorVarPsf.cpp`, `ReconstructionUtils.cpp`. |
| `yrt-pet/src/datastruct/projection/LORsDevice.cu` | Implements LOR/property precompute and upload for batches through `BinLoader` and `PropStructDevice`. | None. | `BinLoader`, `ProjectionData::collectProjectionProperties`, CPU operator setup. |
| `yrt-pet/src/datastruct/projection/ProjectionListDevice.cu` | Implements device projection-list construction, batching, host/device value transfers, clear/divide/invert/add/ACF/multiply operations, owned/alias device pointer management. | Defines no kernels. Launches projection-space kernels from `ProjectionSpaceKernels.cu`. | `ProjectionList.cpp`, `ProjectionData.cpp`, `Histogram`, `SparseHistogram`, `UniformHistogram`. |
| `yrt-pet/src/datastruct/scanner/ScannerDevice.cu` | Implements detector-position/orientation upload to device arrays. | None. | `Scanner`, `DetectorSetup`. |
| `yrt-pet/src/operators/DeviceSynchronized.cu` | Implements launch-parameter selection and conversion to compact CUDA-side scanner/image structs. | None. | CPU setup code inside operators/reconstruction; backend equivalent should be shared launch/context configuration. |
| `yrt-pet/src/operators/OperatorProjectorDD_GPU.cu` | Implements GPU distance-driven operator construction, needed property reporting, forward/back batch apply, and templated kernel launch selection for TOF/projection PSF/updater variants. | Defines no kernels. Launches `projectDD_kernel`. | `OperatorProjector`, `ProjectorDD`. |
| `yrt-pet/src/operators/OperatorProjectorDevice.cu` | Implements common GPU projector batching, factory, TOF/projection PSF setup, updater setup, and `applyA`/`applyAH` orchestration. | None. | `OperatorProjector`, `Projector`, `ProjectorUpdater`. |
| `yrt-pet/src/operators/OperatorProjectorSiddon_GPU.cu` | Implements GPU Siddon operator construction, multi-ray configuration, needed property reporting, forward/back batch apply, and templated kernel launch selection. | Defines no kernels. Launches `projectSiddon_kernel`. | `OperatorProjector`, `ProjectorSiddon`. |
| `yrt-pet/src/operators/OperatorPsfDevice.cu` | Implements GPU uniform image PSF setup, kernel upload, `applyA`/`applyAH`, host/device overloads, and separable convolution staging. | Defines no kernels. Launches `convolve3DSeparable_kernel<0/1/2>`. | `OperatorPsf`. |
| `yrt-pet/src/operators/OperatorVarPsfDevice.cu` | Implements GPU variable image PSF LUT upload, `applyA`/`applyAH`, and variable convolution staging. | Defines no kernels. Launches `convolve3D_kernel<false>` and `convolve3D_kernel<true>`. | `OperatorVarPsf`. |
| `yrt-pet/src/operators/ProjectionPsfManagerDevice.cu` | Implements projection-space PSF LUT upload and flipped-kernel upload. | None. | `ProjectionPsfManager`. |
| `yrt-pet/src/operators/ProjectionSpaceKernels.cu` | Implements projection-value array kernels. | Defines `divideMeasurements_kernel`, `addProjValues_kernel`, `invertProjValues_kernel`, `convertToACFs_kernel`, two `multiplyProjValues_kernel` overloads, and `clearProjections_kernel`. | `ProjectionData::operationOnEachBin`, `ProjectionList`, `ReconstructionUtils::convertProjectionValuesToACF`. |
| `yrt-pet/src/recon/Corrector_GPU.cu` | Implements GPU-flavored attenuation factor precomputation by forward-projecting attenuation images with GPU projectors, then using CPU projection-value conversion. | None. | `Corrector_CPU`, `Corrector`. |
| `yrt-pet/src/recon/LREM_GPU.cu` | Implements low-rank GPU reconstruction buffer setup, reset/update steps, HBasis/cW sync, and updater wiring. | None. | `LREM_CPU`, `LREM`, `OSEM_CPU`. |
| `yrt-pet/src/recon/OSEM_GPU.cu` | Implements GPU OSEM orchestration: sensitivity-image generation, reconstruction buffers, batch property loading, EM update generation, correction-factor handling, GPU streams, and projector/updater setup. | Defines and launches `generateSensImage_kernel` and `computeEMUpdateImage_kernel<UseUpdater>`. | `OSEM_CPU`, `OSEM`, CPU sensitivity and EM update loops. |
| `yrt-pet/src/utils/GPUKernelUtils.cu` | Implements attenuation-factor projection-value kernel. | Defines `applyAttenuationFactors_kernel`; no current launch site found. | `ReconstructionUtils::convertProjectionValuesToACF` and attenuation correction code in `Corrector`. |
| `yrt-pet/src/utils/GPUStream.cu` | Implements `GPUStream` creation, access, and destruction. | None. | No CPU equivalent; backend command queue/stream wrapper. |
| `yrt-pet/src/utils/GPUTypes.cu` | Implements `synchronizeIfNeeded`. | None. | Backend synchronization helper. |
| `yrt-pet/src/utils/GPUUtils.cu` | Implements CUDA error checking, `gpuErrchk`, and `globals::getAvailableVRAM`; also exposes `getAvailableVRAM` to Python when pybind is enabled. | None. | CPU has no equivalent; backend device selection/error reporting. |
| `yrt-pet/src/utils/PageLockedBuffer.cu` | Implements `PageLockedBuffer<T>` pinned host allocation with heap fallback. | None. | Normal host buffers. |
| `yrt-pet/src/utils/ReconstructionUtilsDevice.cu` | Implements GPU motion-average image utilities, pybind bindings for GPU utilities, GPU OSEM factory, and GPU projector factory. | Defines no kernels. Launches `timeAverageMoveImage_kernel<true>` and calls GPU projector/OSEM constructors. | `ReconstructionUtils.cpp`, `createOSEM_CPU`, CPU `timeAverageMoveImage`. |
| `yrt-pet/unit_tests/operators/test_Siddon_GPU.cu` | Tests GPU-vs-CPU Siddon forward/back projection and multi-ray random offsets. | Defines test kernel `getMultiRayPos`. | `unit_tests/operators/test_Siddon.cpp` and CPU `ProjectorSiddon`. |
| `yrt-pet/unit_tests/utils/test_CUDA.cu` | Simple CUDA availability/vector-add smoke test. | Defines test kernel `cudaSum`. | CPU/unit-test smoke equivalent would be a backend availability and buffer-copy test. |

## Recommended first Metal milestone

Start with a backend foundation and the simple vector/image kernels, not the
Siddon/DD projectors.

Milestone 1 acceptance target:

1. Add a backend-neutral layer for device buffers, streams/command queues,
   copy/memset, launch sizing, synchronization, and error reporting. Keep CUDA
   behavior intact and make Metal optional at configure time.
2. Port only the small, regular kernels first:
   `ProjectionSpaceKernels` plus scalar `ImageSpaceKernels` operations
   (`fill`, thresholding, add, multiply, and static/dynamic EM update).
3. Add focused CPU-vs-backend golden tests for those kernels. The tests should
   compare against existing `Image`/`ProjectionData` CPU behavior.
4. Keep macOS CPU-only and Python CPU builds passing with `USE_CUDA=OFF` and
   no Metal dependency.

Why this first: these kernels exercise memory ownership, async copies,
synchronization, launch dispatch, Python-safe build configuration, and numerical
comparison without immediately taking on the hardest CUDA-specific parts:
`curand`, virtual device updater objects, many template-specialized projector
kernels, atomics-heavy backprojection, and TOF/projection-PSF branching.

Next likely milestone after that: `OperatorPsfDevice`/separable image PSF,
because it is self-contained and depends mostly on the backend foundation plus
3D image kernels. The Siddon/DD projectors and OSEM kernels should come later,
after backend buffers, simple kernels, and PSF behavior are proven on macOS.
