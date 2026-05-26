# Metal reconstruction opt-in wiring plan

This is the next proposed production-facing step after the experimental Metal
adapters, the opt-in `OperatorProjector` bridge, and the real-input projector
compare executable. It intentionally avoids full reconstruction dispatch and
keeps CPU/CUDA behavior unchanged by default.

## Current checkpoint

The current Metal projector path is opt-in and limited to:

- `ProjectorType::SIDDON`
- `UpdaterType::DEFAULT4D`
- one-ray Siddon
- no TOF
- no projection-space PSF
- host `Image` and host `ProjectionData`
- LOR and dynamic-frame gathering through the existing
  `BinLoader`/`BinIterator`

The direct bridge and opt-in `OperatorProjector` dispatch are covered by:

- CPU-vs-Metal bridge forward and adjoint golden tests
- default CPU dispatch golden test
- opt-in Siddon forward and adjoint dispatch golden tests
- unsupported fallback equality tests for TOF, DD, DD plus projection PSF,
  multi-ray Siddon, LR, and LRDUALUPDATE
- `yrtpet_metal_projector_compare`, a real-input utility that compares CPU
  `OperatorProjector`, the direct Metal bridge, and the opt-in
  `OperatorProjector` flag for forward and adjoint projection

Build and install outputs remain ignored artifacts. The source checkpoint is
commit-sized as:

- macOS CPU compatibility fixes
- experimental Metal backend foundation and kernels already ported
- opt-in Metal adapters and compare utilities
- opt-in `OperatorProjector` bridge and tests
- Metal backend developer documentation and readiness notes

## Recommended first reconstruction-facing hook

Add an explicitly opt-in OSEM CPU projector-step path that uses the existing
Metal Siddon forward and adjoint kernels for only the projector portion of
`OSEM_CPU::computeEMUpdateImage()`.

The outer OSEM reconstruction flow should remain CPU:

- subset iteration
- sensitivity image selection
- correction factor lookup
- measurement ratio calculation
- EM image update
- image PSF handling
- saving and reconstruction orchestration

The first hook should be a hybrid projector step:

1. Apply image PSF on CPU if already enabled, exactly as today. For the first
   implementation, the experimental Metal projector path should be disabled
   when image PSF is enabled unless we deliberately test that combination.
2. Use the Metal bridge to forward-project the current image into a temporary
   projection-value overlay.
3. Compute denominator corrections and measurement ratios on CPU using the
   existing `Corrector_CPU` caches.
4. Use the Metal bridge to backproject the corrected ratio values into the EM
   update image.
5. Fall back to the existing fused CPU loop if the Metal bridge is unavailable,
   unsupported, or fails.

This keeps the first reconstruction-facing implementation close to the tested
`OperatorProjector` behavior without introducing Metal correction kernels,
projection-list device ownership, CUDA-style batching, OSEM GPU dispatch, or
Python defaults.

## Implementation status

Implemented as an explicit C++/Python opt-in on `OSEM_CPU`. The flag remains
disabled by default:

```cpp
osemCpu.setExperimentalMetalProjectorEnabled(true);
osemCpu.isExperimentalMetalProjectorEnabled();
osemCpu.didLastExperimentalMetalProjectorRun();
```

The implementation uses the existing `OperatorProjectorMetalBridge` directly
inside `OSEM_CPU::computeEMUpdateImage()` only when image PSF is disabled and
the bridge reports the projector configuration as supported. The EM correction
ratio math stays on CPU with the existing `Corrector_CPU` caches. Unsupported
or unavailable Metal configurations return to the existing fused CPU loop.

The opt-in/status methods are now exposed to Python so a `USE_METAL=ON` Python
build can run the same guarded hook once a real `ProjectionData` source is
available:

```python
osem.setExperimentalMetalProjectorEnabled(True)
out_img = osem.reconstruct()
assert osem.didLastExperimentalMetalProjectorRun()
```

This is deliberately limited to the explicit experimental methods; command-line
reconstruction and Python defaults remain unchanged.

Covered by Metal golden tests:

- `osem_cpu_experimental_metal_projector_golden`
- `osem_cpu_experimental_metal_projector_dd_fallback_golden`

Covered by a file-backed compare executable:

- `yrtpet_metal_osem_compare`

## Proposed API

Add a narrow C++ opt-in API on `OSEM_CPU` only:

```cpp
void setExperimentalMetalProjectorEnabled(bool enabled);
bool isExperimentalMetalProjectorEnabled() const;
bool didLastExperimentalMetalProjectorRun() const;
```

Default value: `false`.

Do not expose this in Python or command-line reconstruction yet. The first
consumer should be a C++/Metal golden test. Command-line exposure can follow
only after the one-iteration CPU-vs-Metal reconstruction test is stable.

## Proposed implementation shape

Add a private helper to `OSEM_CPU`:

```cpp
bool computeEMUpdateImageWithExperimentalMetalProjector(
    const Image& inputImageForForwardProj,
    Image& destImageForBackproj,
    const ProjectionData& measurements,
    const BinIterator& binIter);
```

The helper should:

- be compiled only when `BUILD_METAL` is enabled
- construct or reuse an `OperatorProjector` with the current `projectorParams`
  and `binIter`
- call `OperatorProjectorMetalBridge::canRunSiddon()` before doing work
- allocate temporary projection-value overlays backed by `std::vector<float>`
  and delegating all geometry/property access to `measurements`
- run direct Metal bridge `applyA`
- compute CPU correction/ratio values into the overlay
- run direct Metal bridge `applyAH`
- return `true` only when the Metal projector path actually completed
- return `false` for unsupported configs so the existing CPU loop runs

The temporary overlay should be reusable outside OSEM tests if useful:

```cpp
class ProjectionDataValuesOverlay final : public ProjectionData {
public:
    ProjectionDataValuesOverlay(const ProjectionData& source,
                                std::vector<float> values);

    float getProjectionValue(bin_t id) const override;
    void setProjectionValue(bin_t id, float value) override;

    // Delegate geometry/properties to source:
    det_id_t getDetector1(bin_t id) const override;
    det_id_t getDetector2(bin_t id) const override;
    std::unique_ptr<BinIterator> getBinIter(int numSubsets,
                                            int idxSubset) const override;
    timestamp_t getTimestamp(bin_t id) const override;
    frame_t getDynamicFrame(bin_t id) const override;
    bool hasDynamicFraming() const override;
    size_t getNumDynamicFrames() const override;
    bool hasTOF() const override;
    float getTOFValue(bin_t id) const override;
    bool hasArbitraryLORs() const override;
    Line3D getArbitraryLOR(bin_t id) const override;
    std::set<ProjectionPropertyType> getProjectionPropertyTypes()
        const override;
};
```

For the first implementation, place the overlay under
`yrt::backend::metal` only if it is Metal-specific. If it is generally useful
for CPU tests, place it under a neutral test/helper path instead of production
data structures.

## Files likely touched

Minimum production files:

- `yrt-pet/include/yrt-pet/recon/OSEM_CPU.hpp`
- `yrt-pet/src/recon/OSEM_CPU.cpp`

Metal/helper files, only if the overlay is shared:

- `yrt-pet/include/yrt-pet/backends/metal/ProjectionDataValuesOverlay.hpp`
- `yrt-pet/src/backends/metal/ProjectionDataValuesOverlay.cpp`
- `yrt-pet/src/CMakeLists.txt`

Tests/docs:

- `yrt-pet/unit_tests/metal/MetalTests.cpp`
- `docs/source/compilation/metal_backend.md`
- `CUDA_TO_METAL_RECONSTRUCTION_READINESS.md`
- this plan file

Do not touch:

- `OSEM_GPU`
- `LREM_CPU` or `LREM_GPU`
- `OperatorProjectorDevice`
- `ProjectionListDevice`
- `ProjectorSiddon` or `ProjectorDD` production dispatch
- Python bindings
- reconstruction command-line options

## First tests before implementation is considered production-adjacent

Add one focused Metal golden test before exposing the flag beyond C++:

- construct a tiny 3D or 4D image
- construct a tiny arbitrary-LOR `ProjectionData` fixture with dynamic frames
  and one negative frame
- construct a small sensitivity image
- run one CPU `OSEM_CPU::computeEMUpdateImage()` equivalent through the normal
  reconstruction flow with the flag disabled
- run the same setup with the experimental Metal projector flag enabled
- verify the EM update image matches CPU within tolerance
- verify unsupported configurations with the flag enabled fall back to CPU:
  TOF, DD, multi-ray Siddon, image PSF enabled, LR/LRDUALUPDATE

Then add a tiny one-iteration reconstruction smoke test:

- one subset
- one iteration
- no TOF
- no projection PSF
- no image PSF for the first pass
- no scatter/randoms/attenuation for the first pass
- pre-provided sensitivity image
- CPU result equals opt-in Metal result within tolerance

The existing `yrtpet_metal_projector_compare` should remain the real-input
debug tool for failures.

## Risks

- The existing CPU OSEM implementation fuses forward projection, correction,
  ratio calculation, and backprojection in one loop. The Metal path would split
  this into forward, CPU correction, and adjoint passes, so floating-point order
  will differ slightly.
- The split path needs temporary projection buffers with size proportional to
  `measurements.count()`.
- Fallback must be explicit and tested so unsupported configs do not silently
  change behavior.
- Image PSF buffer swapping in `OSEM_CPU::computeEMUpdateImage()` makes PSF
  combinations easy to get wrong. Keep image PSF out of the first hook.
- List-mode sensitivity scaling and multiple subsets are sensitive control
  paths. Start with one subset and expand only after the one-subset case is
  golden.

## Rollback plan

- Remove the OSEM CPU opt-in flag and helper branch.
- Remove any temporary overlay helper if it is only used by the hook.
- Keep `OperatorProjectorMetalBridge`, `SiddonProjectorMetal`,
  `yrtpet_metal_projector_compare`, and low-level Metal tests intact.
- Configure with `-DUSE_METAL=OFF` to remove all Metal-only code from the
  build.

## Explicitly out of scope

- Making Metal the default anywhere
- Python defaults or GE plugin wiring
- command-line reconstruction exposure through `yrtpet_reconstruct`
- full OSEM dispatch replacement
- LREM
- CUDA changes
- `ProjectionListDevice`
- `OperatorProjectorDevice`
- DD projector
- TOF
- projection-space PSF
- multi-ray Siddon
- LR/LRDUALUPDATE
- Metal correction kernels
- Metal EM update kernels beyond the image scalar kernels already isolated in
  the backend
