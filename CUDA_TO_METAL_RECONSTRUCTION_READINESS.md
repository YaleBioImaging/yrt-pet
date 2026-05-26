# CUDA-to-Metal reconstruction readiness plan

This plan describes the next safe Metal milestones after the current
projection-vector, image-scalar, image-PSF, dispatcher/facade, and file-backed
PSF work. It intentionally keeps Metal outside the production reconstruction
dispatch path until the prerequisite kernel and ownership pieces are validated
with CPU-vs-Metal golden tests.

## Current checkpoint

Metal is still opt-in through `USE_METAL=ON`. CPU and CUDA behavior remain
unchanged by default. The current Metal code lives under
`yrt-pet/include/yrt-pet/backends/metal/`,
`yrt-pet/src/backends/metal/`, `yrt-pet/executables/utils/`, and
`yrt-pet/unit_tests/metal/`.

The first tiny projector-adjacent slice added here is
`ProjectionGeometryOps`/`ProjectionGeometryKernels`, with the
`projection_siddon_entry_range` Metal kernel. It computes the same
FOV/volume entry alpha range used at the start of the CPU/CUDA Siddon path for
image-centered LOR endpoints. This is isolated in the experimental backend and
is not wired into `ProjectionListDevice`, projectors, OSEM, LREM, or Python.

The follow-on projection ownership slice is `ProjectionBatchMetal`, which owns
a host-provided LOR batch in a Metal buffer plus a projection-values buffer. It
validates host/Metal roundtrips and can run the entry-range primitive over the
owned LOR buffer. It is still isolated under `yrt::backend::metal`.

The first projector kernel slice is now implemented as
`SiddonProjectorOps`/`SiddonProjectorKernels`. It runs single-ray Siddon forward
and adjoint projection with no TOF and no updater, writes through explicit
`ProjectionBatchMetal`/`Image` arguments, and stays out of all production
projector and reconstruction dispatch.

## CUDA and CPU inventory

| Area | CUDA files | Classes/functions/kernels | Closest CPU equivalent |
| --- | --- | --- | --- |
| Siddon projector | `include/yrt-pet/operators/SiddonKernels.cuh`, `src/operators/OperatorProjectorSiddon_GPU.cu`, `include/yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh` | `projectSiddon_kernel`, `projectSiddon`, multi-ray setup, TOF range clipping, updater-aware forward/back updates | `src/operators/ProjectorSiddon.cpp`, `include/yrt-pet/operators/ProjectorSiddon.hpp`, especially `ProjectorSiddon::project_helper` |
| DD projector | `include/yrt-pet/operators/DDKernels.cuh`, `src/operators/OperatorProjectorDD_GPU.cu`, `include/yrt-pet/operators/OperatorProjectorDD_GPU.cuh` | `projectDD_kernel`, `projectDD`, detector overlap helpers, optional projection PSF and TOF, updater-aware forward/back updates | `src/operators/ProjectorDD.cpp`, `include/yrt-pet/operators/ProjectorDD.hpp`, especially `ProjectorDD::dd_project_ref` |
| Shared projector dispatch | `include/yrt-pet/operators/ProjectorWrapper.cuh`, `src/operators/OperatorProjectorDevice.cu`, `include/yrt-pet/operators/OperatorProjectorDevice.cuh` | `projectAny`, `OperatorProjectorDevice::applyA/applyAH`, batch loading, image/projection host-device staging, projector factory | `src/operators/OperatorProjector.cpp`, `src/operators/OperatorProjectorBase.cpp`, CPU `Projector` classes |
| Projection list/device data | `src/datastruct/projection/ProjectionListDevice.cu`, `include/yrt-pet/datastruct/projection/ProjectionListDevice.cuh`, `src/datastruct/projection/LORsDevice.cu`, `include/yrt-pet/datastruct/projection/LORsDevice.cuh` | `ProjectionListDeviceOwned/Alias`, `prepareBatchLORs`, `loadProjValuesFromHost`, `transferProjValuesToHost`, LOR/property gathering | `src/datastruct/projection/ProjectionList.cpp`, `ProjectionData.cpp`, `ProjectionProperties.cpp`, `BinLoader.cpp`, `BinIterator.cpp` |
| Scanner/device structs | `src/datastruct/scanner/ScannerDevice.cu`, `include/yrt-pet/datastruct/scanner/ScannerDevice.cuh`, `include/yrt-pet/recon/CUParameters.hpp` | `CUScannerParams`, `CUImageParams`, scanner constants copied to device | `src/datastruct/scanner/Scanner.cpp`, `src/datastruct/image/ImageParams.cpp` |
| Image/device support | `src/datastruct/image/ImageDevice.cu`, `include/yrt-pet/datastruct/image/ImageDevice.cuh`, `src/datastruct/image/ImageSpaceKernels.cu`, `include/yrt-pet/datastruct/image/ImageSpaceKernels.cuh` | device allocation/copy plus image scalar/update kernels | `src/datastruct/image/Image.cpp`; most scalar/update kernels already have isolated Metal equivalents |
| Projection-space scalar support | `src/operators/ProjectionSpaceKernels.cu`, `include/yrt-pet/operators/ProjectionSpaceKernels.cuh` | projection clear/add/multiply/divide/invert/ACF kernels | `ProjectionList`/`ProjectionData` host operations; isolated Metal equivalents already exist |
| Projector updaters | `include/yrt-pet/operators/ProjectorUpdaterDevice.cuh` | virtual device updater, `DEFAULT4D`, LR updater, HBasis/H write accumulation | `src/operators/ProjectorUpdater.cpp`, `include/yrt-pet/operators/ProjectorUpdater.hpp` |
| Projection PSF for DD | `src/operators/ProjectionPsfManagerDevice.cu`, `include/yrt-pet/operators/ProjectionPsfManagerDevice.cuh` | projection PSF kernel storage, flipped kernels, DD kernel lookup | `src/operators/ProjectionPsfManager.cpp`, `include/yrt-pet/operators/ProjectionPsfManager.hpp` |
| OSEM GPU | `src/recon/OSEM_GPU.cu`, `include/yrt-pet/recon/OSEM_GPU.cuh` | sensitivity image batches, projector calls, correction, PSF, EM image update | `src/recon/OSEM_CPU.cpp`, `include/yrt-pet/recon/OSEM_CPU.hpp`, `src/recon/OSEM.cpp` |
| LREM GPU | `src/recon/LREM_GPU.cu`, `include/yrt-pet/recon/LREM_GPU.cuh` | LR updater setup, W/H update scaling, device HBasis sync | `src/recon/LREM_CPU.cpp`, `include/yrt-pet/recon/LREM_CPU.hpp`, `src/recon/LREM.cpp` |
| Corrector GPU | `src/recon/Corrector_GPU.cu`, `include/yrt-pet/recon/Corrector_GPU.cuh` | multiplicative/additive correction buffers and device access | `src/recon/Corrector_CPU.cpp`, `include/yrt-pet/recon/Corrector_CPU.hpp` |

## Prerequisites for a minimal Metal projector path

Must port or design before any projector/OSEM production wiring:

- Metal projection batch ownership: host-to-Metal buffers for LOR endpoints,
  detector orientations, TOF values, dynamic frame ids, and projection values.
- Metal scanner/image parameter structs that match the fields consumed by
  Siddon/DD kernels, without depending on CUDA-only `CU*` types.
- A single-ray, no-TOF, no-updater Siddon forward kernel validated against CPU
  on tiny images and a tiny projection batch.
- A matching single-ray Siddon adjoint kernel with deterministic accumulation
  expectations for small tests.
- Explicit context/lifetime ownership for projection buffers and images so the
  first integration point does not hide CPU/Metal copies.
- Golden tests that compare CPU and Metal for geometry, forward projection,
  adjoint projection, and simple A/AH consistency.

Can remain CPU for the first hybrid milestone:

- OSEM/LREM outer loops, subset iteration, saving, and reconstruction control.
- Corrector objects and correction-factor generation.
- Sensitivity image generation orchestration.
- Image-space PSF dispatch beyond the already explicit Metal file helper.
- Projection PSF for DD.
- TOF.
- Multi-ray Siddon.
- DD projector.
- LR updater, HBasis/H write updates, and dual-update LREM.
- Python data/plugin exposure beyond the explicit OSEM CPU opt-in flag.

## Recommended smallest next Metal kernel milestone

The safest first post-PSF projector-adjacent slice is now implemented:

- `ProjectionGeometryOps.hpp/.cpp`
- `ProjectionGeometryKernels.hpp/.cpp`
- `projection_siddon_entry_range` in `SmokeKernels.metal`
- `projection_geometry_siddon_entry_range_golden` in `yrtpet_metal_tests`

This slice validates the FOV/volume entry alpha calculation that Siddon and DD
both need before walking voxels. It does not read image voxels, does not write
projection values, and avoids atomics.

The next slice, a Metal projection batch container, is now implemented and is
still experimental and still not wired into production:

```cpp
namespace yrt::backend::metal {

struct ProjectionBatchShape {
    std::size_t eventCount;
    bool hasDetectorOrientations;
    bool hasTof;
    bool hasDynamicFrames;
};

class ProjectionBatchMetal {
public:
    ProjectionBatchMetal(const Context& context,
                         std::vector<ProjectionLineEndpoints> lors);
    ProjectionBatchMetal(const Context& context,
                         std::vector<ProjectionLineEndpoints> lors,
                         std::vector<float> projectionValues);

    bool isValid() const;
    std::size_t size() const;
    const Buffer& lorBuffer() const;
    Buffer& projectionValuesBuffer();
    bool setProjectionValues(const std::vector<float>& values);
    bool copyProjectionValuesToHost(std::vector<float>& values) const;
    bool computeSiddonEntryRanges(const ProjectionImageBounds& bounds,
                                  std::vector<ProjectionAlphaRange>& ranges)
        const;
};

}
```

It includes a CPU-vs-Metal golden test for buffer ownership, projection-value
roundtrip, and Siddon entry range over the owned LOR buffer.

The single-ray Siddon forward and adjoint kernels are now implemented through:

```cpp
yrt::backend::metal::forwardProjectSiddonSingleRay(context, image, batch,
                                                   frame);
yrt::backend::metal::backProjectSiddonSingleRay(context, batch, image,
                                                frame);
```

It is covered by `siddon_single_ray_forward_golden`, including explicit frame
selection on a tiny 4D image. The adjoint is covered by
`siddon_single_ray_adjoint_golden`, including duplicate-LOR accumulation, a
missed LOR, a zero projection value, and explicit frame selection. The first
adjoint kernel uses deterministic per-voxel accumulation instead of atomics.

The small `SiddonProjectorMetal` adapter is now implemented on top of those
same low-level helpers. It owns a `Context`, creates `ProjectionBatchMetal`
inputs, and exposes only the isolated single-ray forward and adjoint calls. It
is covered by `siddon_projector_metal_adjointness_golden`, which compares the
adapter to CPU `ProjectorSiddon` and checks `<Ax, y> == <x, A^H y>` on a tiny
4D image. The adapter hardening tests now also cover miss/empty behavior,
invalid batches and invalid frames, and explicit frame isolation.

This is enough coverage to define the first production-facing projector
integration. The `OperatorProjectorMetalBridge` described below is implemented
with CPU-vs-Metal tests, and `OperatorProjector` now has an explicit opt-in
dispatch flag that can call the bridge for supported configurations. The flag
is disabled by default and falls back to CPU for unsupported configurations.

## First production-facing projector bridge

Status: bridge implemented in this checkpoint; production dispatch is touched
only through an explicit opt-in `OperatorProjector` flag.

The bridge keeps `SiddonProjectorMetal` as the tested low-level adapter and adds
a narrow `OperatorProjectorMetalBridge` that mirrors the data flow used by
`OperatorProjector::applyA` and `OperatorProjector::applyAH`. The bridge is
compiled only when `USE_METAL=ON`, lives under `yrt::backend::metal`, and
explicitly reports whether a given `OperatorProjector` configuration is
supported.

The bridge supports exactly this first production-facing subset:

- `ProjectorType::SIDDON`
- `UpdaterType::DEFAULT4D`
- one-ray Siddon, inferred by requiring no `DET_ORIENT` projection property
- no TOF
- no projection-space PSF
- host `Image` and host `ProjectionData`
- LOR properties gathered from the existing `BinLoader`/`BinIterator`
- optional dynamic frames by grouping events per frame; negative dynamic frames
  are skipped like CPU Siddon

Everything else returns "unsupported". The future dispatch hook must use that
result to fall back to the existing CPU or CUDA path.

Proposed C++ API:

```cpp
namespace yrt::backend::metal {

struct OperatorProjectorMetalSupport {
    bool supported = false;
    std::string reason;
};

class OperatorProjectorMetalBridge {
public:
    explicit OperatorProjectorMetalBridge(const Context& context);

    OperatorProjectorMetalSupport canRunSiddon(
        const OperatorProjector& projector) const;

    bool applyA(const OperatorProjector& projector,
                const Image& image,
                ProjectionData& projectionData,
                const BinIterator& binIterator,
                const BinLoader& binLoader) const;

    bool applyAH(const OperatorProjector& projector,
                 const ProjectionData& projectionData,
                 Image& image,
                 const BinIterator& binIterator,
                 const BinLoader& binLoader) const;
};

}
```

The first real-dispatch hook is implemented in `OperatorProjector`:

```cpp
void setExperimentalMetalProjectorEnabled(bool enabled);
bool isExperimentalMetalProjectorEnabled() const;
```

When disabled, `OperatorProjector::applyA` and `OperatorProjector::applyAH`
execute the same CPU code they did before. When enabled in a `USE_METAL=ON`
build, they call the Metal bridge only if `canRunSiddon()` succeeds; otherwise
they fall back to the existing CPU implementation without changing results. In
non-Metal builds, the flag can be set but no Metal bridge code is compiled, so
execution falls back to CPU.

The first reconstruction-facing hook is also implemented as an explicit
C++/Python opt-in on `OSEM_CPU`:

```cpp
void setExperimentalMetalProjectorEnabled(bool enabled);
bool isExperimentalMetalProjectorEnabled() const;
bool didLastExperimentalMetalProjectorRun() const;
```

It uses the same `OperatorProjectorMetalBridge` for only the projector portion
of `OSEM_CPU::computeEMUpdateImage()`: Metal Siddon forward projection, CPU
correction/ratio calculation with `Corrector_CPU`, and Metal Siddon adjoint
backprojection into the EM update image. The flag is disabled by default,
exposed to Python only as this explicit experimental method set, not wired into
command-line reconstruction, and falls back to the existing fused CPU loop when
Metal is unavailable, image PSF is enabled, or the projector configuration is
unsupported.

The first file-backed OSEM validation utility is implemented as
`yrtpet_metal_osem_compare`. It accepts a sensitivity image and arbitrary-LOR
measurement CSV, runs CPU `OSEM_CPU` and opt-in Metal `OSEM_CPU`, confirms the
Metal projector hook actually ran, and compares output images.

Bridge-only files touched:

- `include/yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp`
- `src/backends/metal/OperatorProjectorMetalBridge.cpp`
- `src/CMakeLists.txt` inside the existing `USE_METAL` source list
- `unit_tests/metal/MetalTests.cpp`
- `docs/source/compilation/metal_backend.md`
- `CUDA_TO_METAL_RECONSTRUCTION_READINESS.md`

Files touched for the first real-dispatch milestone:

- `include/yrt-pet/operators/OperatorProjector.hpp`
- `src/operators/OperatorProjector.cpp`
- `unit_tests/metal/MetalTests.cpp`

Bridge tests added before touching dispatch:

- gather a tiny list of LOR bins through `BinLoader`/`BinIterator`, run bridge
  forward, and compare the written `ProjectionData` to CPU
  `OperatorProjector::applyA`
- run bridge adjoint into a tiny 4D image and compare to CPU
  `OperatorProjector::applyAH`
- verify dynamic-frame grouping for at least two frames and one negative frame
- verify unsupported cases return unsupported: DD, Siddon multi-ray, TOF, LR,
  and LRDUALUPDATE

First dispatch tests added:

- construct a normal CPU `OperatorProjector` with the Metal opt-in disabled and
  compare output to the current CPU baseline
- enable the opt-in for the supported Siddon subset and compare CPU vs Metal
  for `applyA`
- enable the opt-in for the supported Siddon subset and compare CPU vs Metal
  for `applyAH`
- enable the opt-in for an unsupported TOF configuration and prove it falls
  back to CPU with identical output
- enable the opt-in for unsupported DD, DD plus projection PSF, Siddon
  multi-ray, LR, and LRDUALUPDATE configurations and prove they fall back to
  CPU with identical output
- write a small NIfTI image and arbitrary-LOR CSV fixture, reload both from
  disk, and compare CPU, direct Metal bridge, and opt-in `OperatorProjector`
  dispatch for supported Siddon forward and adjoint projection

Rollback plan:

- remove `OperatorProjectorMetalBridge` from the `USE_METAL` source list
- remove the opt-in members and early bridge calls from `OperatorProjector`
- remove the optional `yrtpet_metal_projector_compare` executable from the
  `USE_METAL` executable list
- leave `SiddonProjectorMetal`, low-level kernels, CPU, and CUDA untouched
- configure with `-DUSE_METAL=OFF` to remove the entire bridge from the build

Still out of scope for this dispatch touch:

- changing default `OperatorProjector` behavior
- Python exposure
- `ProjectionListDevice`
- `OperatorProjectorDevice` or any CUDA classes
- DD, TOF, projection PSF, Siddon multi-ray, LR/LRDUALUPDATE
- OSEM, LREM, sensitivity generation, or reconstruction loop dispatch

## First golden tests before reconstruction dispatch

Before touching `OperatorProjector`, `ProjectorSiddon`, OSEM, or LREM dispatch,
add these tests under `yrt-pet/unit_tests/metal/`:

- `projection_geometry_siddon_entry_range_golden`: already added; validates
  FOV/volume alpha ranges and invalid LORs.
- `projection_batch_metal_buffer_golden`: already added; creates a small host
  LOR batch, copies it to Metal, verifies projection-value roundtrip, runs the
  geometry primitive from the owned LOR buffer, and verifies unchanged host
  metadata.
- `siddon_single_ray_forward_golden`: already added; uses a tiny 4D image and
  several centered LORs, no TOF, no updater, and compares explicit frame 0/1
  projection values against CPU `ProjectorSiddon`.
- `siddon_single_ray_adjoint_golden`: already added; compares CPU adjoint image
  output, including duplicate-LOR accumulation and missed/zero contributions.
- `siddon_projector_metal_adjointness_golden`: already added; validates the
  `SiddonProjectorMetal` adapter against CPU forward/adjoint and checks the
  forward/adjoint dot-product identity.
- `siddon_empty_or_miss_golden`: already added; LORs outside image/FOV produce
  zero projection and no image updates.
- `siddon_projector_metal_failure_modes_golden`: already added; invalid frames
  and invalid batches return failure without mutating projection values or
  image data.
- `siddon_projector_metal_frame_isolation_golden`: already added; explicit
  frame selection is validated for forward and adjoint adapter calls.
- `projection_values_roundtrip_golden`: Metal projection values return to host
  without changing `ProjectionData` or `ProjectionListDevice`.

## Target file layout

Keep all new Metal projector work under `yrt::backend::metal`:

- `include/yrt-pet/backends/metal/ProjectionGeometryOps.hpp`
- `include/yrt-pet/backends/metal/ProjectionGeometryKernels.hpp`
- `include/yrt-pet/backends/metal/ProjectionBatchMetal.hpp`
- `include/yrt-pet/backends/metal/SiddonProjectorKernels.hpp`
- `include/yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp`
- `include/yrt-pet/backends/metal/SiddonProjectorOps.hpp`
- `src/backends/metal/ProjectionGeometryOps.cpp`
- `src/backends/metal/ProjectionGeometryKernels.cpp`
- `src/backends/metal/ProjectionBatchMetal.cpp`
- `src/backends/metal/SiddonProjectorKernels.cpp`
- `src/backends/metal/OperatorProjectorMetalBridge.cpp`
- `src/backends/metal/SiddonProjectorOps.cpp`
- Metal kernels in the current generated `.metallib` source until the kernel
  set becomes large enough to split into multiple `.metal` files.

Do not put these classes under `datastruct/projection`, `operators`, or
`recon` until CPU-vs-Metal parity is proven.

## Risks

- The CUDA projector kernels depend on CUDA-only object and virtual-dispatch
  patterns, especially `ProjectorUpdaterDevice`; Metal should avoid copying
  that design too early.
- Backprojection atomics can be nondeterministic; tests need tolerances and
  fixtures small enough to diagnose accumulation differences.
- `ProjectionPropertyManager` and packed `PropertyUnit` layout are flexible but
  not yet a stable Metal ABI.
- CPU Siddon/DD behavior contains edge-case details such as FOV clipping,
  volume clipping, TOF alpha clipping, dynamic-frame skips, and updater paths.
- Host-copy helpers are correctness tools, not performance architecture.
- Metal and CUDA builds must remain independently configurable.

## Rollback plan

- Remove the new Metal source files from the `USE_METAL` block in
  `yrt-pet/src/CMakeLists.txt`.
- Remove the isolated `.metal` kernel and its test case.
- Leave CPU/CUDA files untouched; no production dispatch points depend on the
  new Metal geometry helper.
- Disable `USE_METAL` at configure time to return to the prior CPU/CUDA-only
  behavior.

## Explicitly out of scope

- Wiring Metal into `ImageDevice`, `ProjectionListDevice`,
  `OperatorProjector`, `ProjectorSiddon`, `ProjectorDD`,
  `ProjectorUpdater`, OSEM, LREM, or Python bindings.
- Porting full Siddon, DD, OSEM, LREM, projection PSF, TOF, multi-ray, or
  updater kernels in this checkpoint.
- Replacing CUDA behavior or making Metal a default backend.
- Performance optimization, persistent reconstruction-resident Metal buffers,
  or asynchronous multi-command scheduling.
