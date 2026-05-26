# Experimental Metal backend

YRT-PET has an experimental Metal backend for macOS. It is disabled by default
and must be enabled explicitly with CMake. The current backend is a small
developer-facing path for validating Metal kernels against CPU results; it is
not a replacement for the CUDA device path.

## CMake options

- `-DUSE_METAL=ON` enables the experimental Metal backend. The default is
  `OFF`.
- `-DUSE_CUDA=ON/OFF` remains independent. Enabling Metal does not make Metal
  the default GPU backend and does not change CUDA behavior.
- `-DBUILD_TESTS=ON` builds the Metal test executable when `USE_METAL=ON`.
- `-DBUILD_PYBIND11=ON/OFF` controls Python bindings as usual. Only the
  explicit experimental `OSEM_CPU` Metal projector opt-in/status methods are
  exposed to Python; the lower-level Metal adapters are not Python APIs.

## Supported platforms

The Metal backend requires macOS on an Apple platform with a Metal-capable GPU,
Apple's Metal framework, Foundation, QuartzCore, and Xcode command line tools.
CMake uses `xcrun`, `metal`, and `metallib` to build the `.metal` source into a
`.metallib`.

Configuring with `-DUSE_METAL=ON` on non-Apple platforms is expected to fail.

The `metal-cpp` headers are vendored under
`yrt-pet/third_party/metal-cpp/SingleHeader` and are only added to include paths
when `USE_METAL=ON`.

## Build and run

From the repository root:

```sh
cmake -S yrt-pet -B build_metal \
  -DUSE_CUDA=OFF \
  -DUSE_METAL=ON \
  -DBUILD_PYBIND11=OFF \
  -DBUILD_TESTS=ON
cmake --build build_metal --target yrtpet_metal_smoke -j 8
cmake --build build_metal --target yrtpet_metal_backend_sample -j 8
cmake --build build_metal --target yrtpet_metal_psf_compare -j 8
cmake --build build_metal --target yrtpet_metal_projector_compare -j 8
cmake --build build_metal --target yrtpet_metal_osem_compare -j 8
cmake --build build_metal --target yrtpet_metal_tests -j 8
```

Run the standalone smoke executable:

```sh
./build_metal/executables/yrtpet_metal_smoke
```

Run the experimental facade sample executable:

```sh
./build_metal/executables/yrtpet_metal_backend_sample
```

Run the real-input PSF comparison executable:

```sh
./build_metal/executables/yrtpet_metal_psf_compare \
  --input input_image.nii \
  --psf image_psf.csv \
  --cpu-out cpu_psf.nii \
  --metal-out metal_psf.nii \
  --diff-out metal_minus_cpu.nii
```

Run the real-input projector comparison executable:

```sh
./build_metal/executables/yrtpet_metal_projector_compare \
  --input input_image.nii \
  --lors siddon_lors.csv
```

The LOR CSV format is
`x1,y1,z1,x2,y2,z2[,projection_value][,dynamic_frame]`. Forward mode compares
CPU `OperatorProjector::applyA`, the direct Metal bridge, and the opt-in
`OperatorProjector` dispatch flag. `--adjoint` compares `applyAH`; in adjoint
mode the input image provides the output image shape and initial values. This
utility supports only the same experimental Siddon subset as the bridge:
one-ray Siddon, no TOF, no projection PSF, and `DEFAULT4D`.

Run the real-input OSEM comparison executable:

```sh
./build_metal/executables/yrtpet_metal_osem_compare \
  --sensitivity sensitivity.nii \
  --lors siddon_measurements.csv \
  --initial initial_estimate.nii \
  --cpu-out cpu_osem.nii \
  --metal-out metal_osem.nii \
  --diff-out metal_minus_cpu.nii
```

The OSEM LOR CSV format is
`x1,y1,z1,x2,y2,z2[,measurement][,dynamic_frame]`. If `--initial` is omitted,
the executable fills an initial estimate from the sensitivity image geometry
and `--initial-value`; when dynamic frames are present, the initial image `nt`
is inferred from the frame column. The executable runs CPU `OSEM_CPU` and then
`OSEM_CPU` with `setExperimentalMetalProjectorEnabled(true)`, confirms that
the Metal projector step actually ran, and reports max absolute/relative image
differences. It currently supports the same guarded subset as the OSEM hook:
one-ray Siddon, no TOF, no projection-space PSF, no image PSF, and host
projection data.

Run the Metal golden tests:

```sh
ctest --test-dir build_metal -R yrtpet_metal_tests --output-on-failure
```

The test executable returns CTest skip code `77` when no Metal device is
available. A host process with access to the Mac GPU is required for a real
Metal pass/fail result.

## Opt-in integration test

The experimental backend owns Metal runtime state through:

```cpp
yrt::backend::metal::Context context;
```

`Context` creates the Metal device, loads the generated `.metallib`, creates a
command queue, and reports availability through `isValid()` and
`errorMessage()`.

The current opt-in adapter surface is intentionally small and host-owned:

| Adapter | Owns | Delegates to | Not wired into |
| --- | --- | --- | --- |
| `ProjectionVectorMetal` | `std::vector<float>` and `Context` | `ProjectionVectorOps` | `ProjectionListDevice` or projection production paths |
| `ProjectionBatchMetal` | LOR and projection-value Metal buffers plus host LOR metadata | `ProjectionGeometryKernels` | `ProjectionListDevice`, projectors, or reconstruction |
| `ImageMetal` | allocated `ImageOwned` and `Context` | `ImageOps` | `ImageDevice`, reconstruction, or projectors |
| `OperatorPsfMetal` | PSF kernels, flipped AH kernels, and `Context` | `PsfOps` | `OperatorPsf`, `OperatorPsfDevice`, or reconstruction |
| `OperatorProjectorMetalBridge` | `Context` and temporary Metal projection batches | `SiddonProjectorMetal` | `OperatorProjector`, `OperatorProjectorDevice`, or reconstruction dispatch |

Each adapter is a developer-facing convenience wrapper around the corresponding
host API. They are only compiled when `USE_METAL=ON`, require explicit
construction by the caller, and do not replace CPU or CUDA dispatch.

The first production-facing experimental entry point is
`yrt-pet/backends/metal/ExperimentalBackend.hpp`:

```cpp
yrt::backend::metal::ExperimentalBackend backend;
auto values = backend.makeProjectionVector(hostValues);
auto image = backend.makeImage(hostImage);
auto psf = backend.makeOperatorPsf(kernelX, kernelY, kernelZ);
backend.applyOperatorPsfForward(inputImage, outputImage, kernelX, kernelY,
                                kernelZ);
backend.applyOperatorPsfForward(inputImage, outputImage, "image_psf.csv");
```

`ExperimentalBackend` owns one `Context` and constructs the existing opt-in
adapters with that context. It centralizes device/metallib validation through
`isAvailable()`, `isValid()`, and `errorMessage()`, but it still does not
participate in runtime dispatch or production reconstruction paths. The
`applyOperatorPsfForward` and `applyOperatorPsfAdjoint` helpers are explicit
Metal PSF calls for production-adjacent experiments; they accept either
explicit uniform kernels or the same uniform image-space PSF CSV file format
used by CPU `OperatorPsf`. They do not alter `OperatorPsf` or
`OperatorPsfDevice` dispatch.

The first production-facing opt-in helper is
`yrt-pet/backends/metal/PsfFileOps.hpp`:

```cpp
yrt::backend::metal::applyPsfForward(inputImage, outputImage,
                                     "image_psf.csv");
yrt::backend::metal::applyPsfAdjoint(inputImage, outputImage,
                                     "image_psf.csv");
```

These helpers are thin, explicit Metal-only wrappers around the same
file-backed PSF path. They are only available in `USE_METAL=ON` builds, create
their own Metal context, return `false` when Metal is unavailable or input and
output images are incompatible, and propagate PSF CSV parsing errors. They do
not participate in default `OperatorPsf`, `OperatorPsfDevice`, CPU, CUDA, or
reconstruction dispatch.

The `yrtpet_metal_backend_sample` executable demonstrates this facade by
running one projection-vector operation, one image operation, and one PSF
operation against CPU references. It is a developer sample/smoke path, not a
production command-line workflow.

The `yrtpet_metal_psf_compare` executable reads a real input image and a
uniform image-space PSF CSV file, applies CPU `OperatorPsf` and the explicit
Metal PSF helper, compares the results with configurable tolerances, and can
write CPU, Metal, and signed difference images. It is the first real-input
production-adjacent check, but it still does not change reconstruction,
projector, or operator dispatch behavior.

The `yrtpet_metal_projector_compare` executable reads a real input image and a
small arbitrary-LOR CSV file, validates that the explicit Metal Siddon bridge
supports the configuration, then compares CPU `OperatorProjector` with both
the direct Metal bridge and the explicit opt-in `OperatorProjector` dispatch
flag. It supports forward and adjoint modes, but only for the current
experimental subset: one-ray Siddon, no TOF, no projection-space PSF, and
`DEFAULT4D`. It does not change default projector, reconstruction, CUDA, or
Python behavior.

The Metal test executable also includes a file-backed PSF golden test that
writes a small NIfTI image and uniform PSF CSV fixture, reloads the image from
disk, and compares the public opt-in Metal PSF file helpers against CPU
`OperatorPsf` from the same files. A companion error-path test covers missing
PSF files, malformed CSV files, even kernel sizes, and mismatched input/output
image dimensions.

The Metal tests also include `operator_projector_metal_file_input_golden`,
which writes a small NIfTI image and arbitrary-LOR CSV fixture, reloads both
from disk, and compares CPU, direct Metal bridge, and opt-in
`OperatorProjector` dispatch for forward and adjoint projection.

The first reconstruction-facing experiment is an explicit C++/Python flag on
`OSEM_CPU`:

```cpp
osemCpu.setExperimentalMetalProjectorEnabled(true);
osemCpu.isExperimentalMetalProjectorEnabled();
osemCpu.didLastExperimentalMetalProjectorRun();
```

The flag is disabled by default and is exposed to Python only as this explicit
experimental API; it is not wired into command-line reconstruction tools. When
enabled in a `USE_METAL=ON` build, the `OSEM_CPU::computeEMUpdateImage()`
projector step can split the existing fused CPU loop into: Metal Siddon forward
projection, CPU correction/ratio
calculation using the existing `Corrector_CPU` caches, and Metal Siddon adjoint
backprojection into the EM update image. It falls back to the existing CPU loop
when Metal is unavailable, image PSF is enabled, or the projector configuration
is outside the current bridge subset. The supported Metal projector subset is
the same as `OperatorProjectorMetalBridge`: one-ray Siddon, no TOF, no
projection-space PSF, and `DEFAULT4D`.

This hook still does not make Metal a default reconstruction backend and does
not wire Metal into `OSEM_GPU`, `LREM`, `ImageDevice`, `ProjectionListDevice`,
`OperatorProjectorDevice`, Siddon/DD production dispatch, or Python bindings.
The Metal test suite covers `osem_cpu_experimental_metal_projector_golden` for
a one-iteration CPU-vs-Metal OSEM CPU reconstruction and
`osem_cpu_experimental_metal_projector_dd_fallback_golden` for DD fallback
equality with the opt-in flag enabled.

For Python experiments, build with both `USE_METAL=ON` and
`BUILD_PYBIND11=ON`. External projection-data plugins can be enabled using the
standard plugin symlink mechanism. For the GE plugin:

```sh
ln -s /Users/yanischemli/Documents/Codex/2026-05-24/yrt-pet-ge \
  yrt-pet/plugins/yrt-pet-ge
cmake -S yrt-pet -B build_metal \
  -DUSE_METAL=ON \
  -DBUILD_PYBIND11=ON \
  -DBUILD_TESTS=ON
cmake --build build_metal --target pyyrtpet -j 8
```

Then construct the data path normally, and enable the flag on the returned CPU
OSEM object:

```python
osem = yrt.createOSEM(scanner, False)
osem.num_MLEM_iterations = 1
osem.num_OSEM_subsets = 1
osem.setProjector(yrt.ProjectorType.SIDDON)
osem.setSensitivityImage(sens_img)
osem.setDataInput(proj_data)
osem.setExperimentalMetalProjectorEnabled(True)
out_img = osem.reconstruct()
assert osem.didLastExperimentalMetalProjectorRun()
```

A repeatable real-data GE script is available at
`yrt-pet/python/examples/metal_ge_osem_smoke.py`. It follows the mini-hot-spot
CPU reconstruction setup: it reads `PseudoListMode.yrt` as
`ListModeLUTAlias` triples `(timestamp, detector1, detector2)`, loads the real
normalization/ACF/randoms/scatter histogram files, generates the sensitivity
image from the real correction histogram, optionally applies motion and image
PSF, and writes a CPU OSEM reconstruction. By default it preserves the CPU
workflow and does not enable Metal:

```sh
PYTHONPATH=build_metal \
  python yrt-pet/python/examples/metal_ge_osem_smoke.py \
  --base /Users/yanischemli/Desktop/mini_hot_spot
```

For the current experimental Metal projector subset, run a CPU-vs-Metal
comparison with image PSF disabled and a small validation-sized event count:

```sh
PYTHONPATH=build_metal \
  python yrt-pet/python/examples/metal_ge_osem_smoke.py \
  --base /Users/yanischemli/Desktop/mini_hot_spot \
  --compare-metal \
  --no-psf \
  --max-events 4096 \
  --iterations 1 \
  --subsets 1
```

The script reports setup time, CPU reconstruction time, Metal reconstruction
time, whether the Metal projector path actually ran, image sum/max/nonzero
counts, CPU-vs-Metal max absolute/relative differences, mismatch count, and the
Metal/CPU time ratio. `--pct` selects a percentage of the list-mode file and
`--max-events` caps it for development runs. The current Metal OSEM projector
hook is explicitly incompatible with image PSF, so the script exits early if
`--compare-metal` is combined with PSF. The Metal adjoint/backprojection path
now uses a per-LOR atomic accumulation kernel instead of the original
per-voxel validation kernel, but the script still refuses `--compare-metal`
runs above the default `--metal-event-limit` or with more than one
iteration/subset unless `--allow-unsafe-metal` is passed. Larger runs should be
unlocked gradually from C++/repo-side tests first. Use `--fail-on-mismatch` to
make tolerance mismatches fatal; otherwise mismatches are reported as
diagnostics.

The Metal OSEM hook itself does not depend on the GE plugin, but the plugin is
needed for the GE `ProjectionData` classes used by this real-data smoke.

The first production-facing projector bridge is implemented and can be reached
through an explicit `OperatorProjector` opt-in flag. The flag is disabled by
default:

```cpp
operatorProjector.setExperimentalMetalProjectorEnabled(true);
operatorProjector.isExperimentalMetalProjectorEnabled();
```

When enabled in a `USE_METAL=ON` build, `OperatorProjector::applyA` and
`OperatorProjector::applyAH` first ask `OperatorProjectorMetalBridge` whether
the configuration is supported. The current supported subset is Siddon, one
ray, no TOF, no projection PSF, and `DEFAULT4D` host
`Image`/`ProjectionData` inputs. The bridge gathers LOR and dynamic-frame
properties through the same `BinLoader`/`BinIterator` surface used by CPU
`OperatorProjector`, then delegates to `SiddonProjectorMetal`. If the flag is
disabled, Metal is unavailable, or the projector configuration is unsupported,
the existing CPU path is used. The opt-in flag is not exposed through Python.
The Metal test suite covers CPU fallback equality when the opt-in flag is
enabled for unsupported TOF, DD, DD with projection-space PSF, multi-ray
Siddon, LR, and LR dual-update configurations.

The experimental host-facing projection vector API is available through
`yrt-pet/backends/metal/ProjectionVectorOps.hpp`:

```cpp
yrt::backend::metal::clear(context, values, value);
yrt::backend::metal::add(context, input, output);
yrt::backend::metal::multiplyByScalar(context, values, scalar);
yrt::backend::metal::multiplyElementwise(context, input, output);
yrt::backend::metal::divideMeasurements(context, measurements, output);
yrt::backend::metal::invert(context, input, output);
yrt::backend::metal::convertToACF(context, input, output, unitFactor);
```

These functions accept host `std::vector<float>` values, require non-empty
matching vector sizes for input/output operations, copy data to Metal buffers,
run existing Metal kernels, and copy mutated outputs back to host vectors.

An experimental projection vector adapter is available through
`yrt-pet/backends/metal/ProjectionVectorMetal.hpp`:

```cpp
yrt::backend::metal::ProjectionVectorMetal values(context, hostValues);
values.add(inputValues);
values.multiplyByScalar(scale);
values.invert();
```

This adapter owns a host `std::vector<float>` and a Metal `Context`. Each
operation delegates to the projection vector host API above and copies results
back into the owned host vector. It is an opt-in convenience wrapper only; it
does not change `ProjectionListDevice` or any production projection data path.

The experimental projector-adjacent geometry API is available through
`yrt-pet/backends/metal/ProjectionGeometryOps.hpp`:

```cpp
yrt::backend::metal::ProjectionImageBounds bounds{lengthX, lengthY, lengthZ,
                                                  fovRadius};
yrt::backend::metal::computeSiddonEntryRanges(context, centeredLorEndpoints,
                                              bounds, alphaRanges);
```

This helper computes the Siddon-style FOV/volume entry alpha range for
image-centered LOR endpoints. It is a small readiness primitive for future
projector work only; it does not own projection data, does not touch
`ProjectionListDevice`, and does not launch Siddon, DD, or reconstruction
kernels.

An experimental projection batch adapter is available through
`yrt-pet/backends/metal/ProjectionBatchMetal.hpp`:

```cpp
yrt::backend::metal::ProjectionBatchMetal batch(context,
                                                centeredLorEndpoints,
                                                projectionValues);
batch.setProjectionValues(updatedProjectionValues);
batch.copyProjectionValuesToHost(hostProjectionValues);
batch.computeSiddonEntryRanges(bounds, alphaRanges);
```

This adapter owns Metal buffers for a small host-provided LOR batch and its
projection values. It is a buffer ownership and roundtrip validation surface
for future projector work; it does not read or write `ProjectionData`, does not
use `ProjectionListDevice`, and does not participate in production projection
or reconstruction dispatch.

The experimental Siddon projector API is available through
`yrt-pet/backends/metal/SiddonProjectorOps.hpp` and the small adapter in
`yrt-pet/backends/metal/SiddonProjectorMetal.hpp`:

```cpp
yrt::backend::metal::forwardProjectSiddonSingleRay(context, image, batch,
                                                   frame);
yrt::backend::metal::backProjectSiddonSingleRay(context, batch, image,
                                                frame);
batch.copyProjectionValuesToHost(hostProjectionValues);

yrt::backend::metal::SiddonProjectorMetal projector(context);
auto adapterBatch = projector.makeBatch(centeredLorEndpoints,
                                        projectionValues);
projector.forwardProjectSingleRay(image, adapterBatch, frame);
projector.backProjectSingleRay(adapterBatch, image, frame);
```

These helpers run only the simplest Siddon paths: single ray, no TOF, no
updater, explicit frame selection, and explicit `ProjectionBatchMetal` data.
The adjoint helper uses per-LOR Metal threads with atomic float accumulation
into the image buffer, which removes the original O(voxels x LORs) validation
kernel. This is still a CPU-vs-Metal validation surface only; it does not alter
`ProjectorSiddon`, `OperatorProjector`, `ProjectionListDevice`, or
reconstruction dispatch. The adapter owns only a `Context` and creates
`ProjectionBatchMetal` inputs for the existing isolated single-ray kernels.

The experimental host-facing image API is available through
`yrt-pet/backends/metal/ImageOps.hpp`:

```cpp
yrt::backend::metal::fill(context, image, value);
yrt::backend::metal::multiplyByScalar(context, image, scalar);
yrt::backend::metal::add3DTo3D(context, input3D, output3D);
yrt::backend::metal::add3DTo4D(context, input3D, output4D);
yrt::backend::metal::applyThreshold(context, image3D, mask3D, ...);
yrt::backend::metal::applyThresholdBroadcast(context, image4D, mask3D, ...);
yrt::backend::metal::updateEMStatic(context, image3D, update3D, sensitivity3D,
                                    threshold);
yrt::backend::metal::updateEMDynamic(context, image4D, update4D, sensitivity3D,
                                     threshold);
```

These functions accept host `Image` objects, copy image data to Metal buffers,
run existing Metal kernels, and copy mutated outputs back to the host images.
They do not change `ImageDevice` or the normal CPU image path.

An experimental image adapter is available through
`yrt-pet/backends/metal/ImageMetal.hpp`:

```cpp
yrt::backend::metal::ImageMetal image(context, hostImage);
image.fill(value);
image.multiplyByScalar(scale);
image.updateEMStatic(updateImage, sensitivityImage, threshold);
```

This adapter owns an allocated host `ImageOwned` and a Metal `Context`. Each
operation delegates to the image host API above and copies results back into the
owned host image. It is an opt-in convenience wrapper only; it does not change
`ImageDevice`, reconstruction, projectors, or the normal CPU image path.

The opt-in host PSF API is available through
`yrt-pet/backends/metal/PsfOps.hpp`:

```cpp
yrt::backend::metal::convolve3DSeparableHost(context, ...)
```

It follows the same host-copy pattern and runs the existing separable PSF Metal
kernels.

An experimental PSF adapter is available through
`yrt-pet/backends/metal/OperatorPsfMetal.hpp`:

```cpp
yrt::backend::metal::OperatorPsfMetal psf(context, kernelX, kernelY, kernelZ);
yrt::backend::metal::OperatorPsfMetal psfFromFile(context, "image_psf.csv");
psf.applyA(inputImage, outputImage);
psf.applyAH(inputImage, outputImage);
```

This adapter does not inherit from `OperatorPsf` and is not used by the default
`OperatorPsf` dispatch path. It is an explicit Metal-only wrapper around the
host PSF API. The file constructor is a convenience for opt-in experiments that
already have a uniform image-space PSF CSV on disk; it still stays entirely
inside `yrt::backend::metal`.

A convenience `convolve3DSeparableHost` overload without an explicit `Context`
also exists for single-call smoke usage. These APIs are only available in
builds configured with `USE_METAL=ON`.

The `yrtpet_metal_tests` executable includes `backend_context_valid` and
CPU-vs-Metal host API tests named `projection_vector_ops_host_api_golden`,
`projection_vector_metal_golden`,
`projection_geometry_siddon_entry_range_golden`,
`projection_batch_metal_buffer_golden`,
`siddon_single_ray_forward_golden`,
`siddon_single_ray_adjoint_golden`,
`siddon_projector_metal_adjointness_golden`,
`siddon_empty_or_miss_golden`,
`siddon_projector_metal_failure_modes_golden`,
`siddon_projector_metal_frame_isolation_golden`,
`operator_projector_metal_bridge_forward_golden`,
`operator_projector_metal_bridge_adjoint_golden`,
`operator_projector_metal_bridge_unsupported_golden`,
`operator_projector_metal_dispatch_default_golden`,
`operator_projector_metal_dispatch_forward_golden`,
`operator_projector_metal_dispatch_adjoint_golden`,
`operator_projector_metal_dispatch_fallback_golden`,
`operator_projector_metal_dispatch_dd_fallback_golden`,
`operator_projector_metal_dispatch_multi_ray_fallback_golden`,
`operator_projector_metal_dispatch_projection_psf_fallback_golden`,
`operator_projector_metal_dispatch_lr_fallback_golden`,
`operator_projector_metal_file_input_golden`,
`image_ops_host_api_golden`, `image_metal_golden`,
`psf_ops_host_api_golden`, and `operator_psf_metal_golden`. The facade
composition test is named `experimental_backend_golden` and covers both
explicit PSF kernels and the uniform PSF CSV file overload. The public
file-backed PSF helper tests are named `psf_file_ops_real_input_golden` and
`psf_file_ops_error_paths`.

## Implemented Metal kernels

The current `.metal` file contains:

- backend smoke: `smoke_add_one`
- projection-space vector ops: clear, add, multiply by scalar, multiply
  elementwise, divide measurements, invert, and convert projection values to
  ACFs
- projector-adjacent geometry: Siddon-style FOV/volume entry alpha range for
  image-centered LOR endpoints
- Siddon projector: single-ray forward projection and per-LOR atomic adjoint
  projection with no TOF and no updater
- image scalar/update ops: fill, multiply by scalar, add 3D image to 3D image,
  add 3D image to 4D image, `applyThreshold`, `applyThresholdBroadcast`,
  static EM update, dynamic EM update, and dynamic EM update with sensitivity
  scaling
- image PSF ops: separable 3D convolution over X, Y, and Z with circular
  boundary wrapping

## Known limitations before reconstruction wiring

The current Metal backend is intentionally host-copy based. Each host API or
adapter call copies data from CPU memory into Metal buffers, launches one or
more kernels, and copies results back to CPU memory. This is useful for
CPU-vs-Metal validation, but it is not yet a performance-oriented
reconstruction data path.

Before any reconstruction wiring, the project still needs:

- a clear ownership model for where image and projection data live during an
  iteration
- a synchronization model that avoids hidden CPU/Metal transfer costs
- broader PSF validation on representative reconstruction inputs and tolerances
- explicit configuration semantics for selecting Metal without changing CPU or
  CUDA defaults
- tolerance and reproducibility expectations for iterative reconstruction
  outputs
- a rollback path that can disable Metal without changing reconstruction
  results

Until those are settled, Metal should remain limited to explicit
`yrt::backend::metal` calls, smoke/sample executables, and golden tests.

## Not wired into production

Metal is not currently wired into:

- `ImageDevice`
- `ProjectionListDevice` or other projection device data paths
- `OperatorPsfDevice` or the default `OperatorPsf` dispatch path
- reconstruction code
- projectors, Siddon, DD, OSEM, or LREM
- Python bindings

The existing CPU and CUDA behavior remains the default behavior unless a caller
explicitly builds with `USE_METAL=ON` and directly calls the experimental
`yrt::backend::metal` API or Metal smoke/test executables.
