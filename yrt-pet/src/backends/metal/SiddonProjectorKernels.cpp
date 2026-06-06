/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"

#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

#include <cstdlib>

namespace yrt::backend::metal
{
namespace
{

bool coversFloatCount(const Buffer& buffer, std::size_t count)
{
	return buffer.isValid() && buffer.byteCount() >= sizeof(float) * count;
}

bool coversUint32Count(const Buffer& buffer, std::size_t count)
{
	return buffer.isValid() && buffer.byteCount() >= sizeof(std::uint32_t) * count;
}

bool coversLineCount(const Buffer& buffer, std::size_t count)
{
	return buffer.isValid() &&
	       buffer.byteCount() >= sizeof(ProjectionLineEndpoints) * count;
}

bool areParamsValid(const SiddonForwardImageParams& params)
{
	return params.nx > 0 && params.ny > 0 && params.nz > 0 && params.nt > 0 &&
	       params.frame < params.nt && params.lengthX > 0.0f &&
	       params.lengthY > 0.0f && params.lengthZ > 0.0f &&
	       params.voxelX > 0.0f && params.voxelY > 0.0f &&
	       params.voxelZ > 0.0f && params.invVoxelX > 0.0f &&
	       params.invVoxelY > 0.0f && params.invVoxelZ > 0.0f &&
	       params.halfLengthX > 0.0f && params.halfLengthY > 0.0f &&
	       params.halfLengthZ > 0.0f && params.fovRadius > 0.0f;
}

std::size_t voxelCount(const SiddonForwardImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

KernelLaunchOptions launchOptions(const ProjectorKernelOptions* options)
{
	return options == nullptr ? KernelLaunchOptions{} : options->launchOptions;
}

bool useNativeFloatAtomicsForAdjoint(const ProjectorKernelOptions* options)
{
	if (options != nullptr && options->nativeFloatAtomicsExplicit)
	{
		return options->nativeFloatAtomics;
	}
	const char* value = std::getenv("YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}  // namespace

bool launchSiddonForwardSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& image, const Buffer& lines, Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    const ProjectorKernelOptions* options)
{
	return areParamsValid(params) && coversFloatCount(image, voxelCount(params)) &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       launchKernel1D(device, library, commandQueue,
	           "siddon_forward_single_ray",
	           {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	           {{&params, sizeof(params), 3}}, lineCount,
	           launchOptions(options));
}

bool launchSiddonBackProjectSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    Buffer& image, const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    const ProjectorKernelOptions* options)
{
	return areParamsValid(params) && lineCount > 0 &&
	       coversFloatCount(image, voxelCount(params)) &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       launchKernel1D(device, library, commandQueue,
	           useNativeFloatAtomicsForAdjoint(options)
	               ? "siddon_backproject_single_ray_native_atomic_float"
	               : "siddon_backproject_single_ray",
	           {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	           {{&params, sizeof(params), 3}}, lineCount,
	           launchOptions(options));
}

bool launchSiddonBackProjectSingleRayUpdateCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues, Buffer& updateCounts,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    const ProjectorKernelOptions* options)
{
	return areParamsValid(params) && lineCount > 0 &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       coversUint32Count(updateCounts, lineCount) &&
	       launchKernel1D(device, library, commandQueue,
	           "siddon_backproject_single_ray_update_count",
	           {{&lines, 0}, {&projectionValues, 1}, {&updateCounts, 2}},
	           {{&params, sizeof(params), 3}}, lineCount,
	           launchOptions(options));
}

bool launchSiddonBackProjectSingleRayVoxelHitCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues,
    Buffer& voxelHitCounts, const SiddonForwardImageParams& params,
    std::size_t lineCount, const ProjectorKernelOptions* options)
{
	return areParamsValid(params) && lineCount > 0 &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       coversUint32Count(voxelHitCounts, voxelCount(params)) &&
	       launchKernel1D(device, library, commandQueue,
	           "siddon_backproject_single_ray_voxel_hit_count",
	           {{&lines, 0}, {&projectionValues, 1}, {&voxelHitCounts, 2}},
	           {{&params, sizeof(params), 3}}, lineCount,
	           launchOptions(options));
}

}  // namespace yrt::backend::metal
