/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/JosephProjectorKernels.hpp"

#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

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
	return buffer.isValid() &&
	       buffer.byteCount() >= sizeof(std::uint32_t) * count;
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

bool useNativeFloatAtomicsForAdjoint()
{
	const char* value = std::getenv("YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::uint32_t josephSampleStride()
{
	const char* value = std::getenv("YRTPET_METAL_JOSEPH_SAMPLE_STRIDE");
	if (value == nullptr || value[0] == '\0' || value[0] == '-')
	{
		return 1;
	}

	char* end = nullptr;
	const unsigned long parsed = std::strtoul(value, &end, 10);
	if (end == value || parsed <= 1)
	{
		return 1;
	}
	return static_cast<std::uint32_t>(
	    std::min<unsigned long>(
	        parsed, static_cast<unsigned long>(
	                    std::numeric_limits<int>::max())));
}

bool useThreadgroupSampleAccumulationForAdjoint()
{
	const char* value =
	    std::getenv("YRTPET_METAL_JOSEPH_ADJOINT_ACCUMULATION");
	return value != nullptr &&
	       (std::strcmp(value, "threadgroup-sample") == 0 ||
	           std::strcmp(value, "tg-sample") == 0 ||
	           std::strcmp(value, "threadgroup") == 0);
}

const char* axisSuffix(std::uint32_t axis)
{
	if (axis == 0)
	{
		return "x";
	}
	if (axis == 1)
	{
		return "y";
	}
	return axis == 2 ? "z" : nullptr;
}

}  // namespace

bool launchJosephForwardSingleRay(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& image, const Buffer& lines,
    Buffer& projectionValues, const SiddonForwardImageParams& params,
    std::size_t lineCount)
{
	const std::uint32_t sampleStride = josephSampleStride();
	return areParamsValid(params) && coversFloatCount(image, voxelCount(params)) &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       (sampleStride <= 1 ?
	            launchKernel1D(device, library, commandQueue,
	                "joseph_forward_single_ray",
	                {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	                {{&params, sizeof(params), 3}}, lineCount) :
	            launchKernel1D(device, library, commandQueue,
	                "joseph_forward_single_ray_sample_stride",
	                {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	                {{&params, sizeof(params), 3},
	                    {&sampleStride, sizeof(sampleStride), 4}},
	                lineCount));
}

bool launchJosephForwardSingleRayAxis(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& image, const Buffer& lines, Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    std::uint32_t axis)
{
	const char* suffix = axisSuffix(axis);
	const std::uint32_t sampleStride = josephSampleStride();
	if (suffix == nullptr || sampleStride > 1)
	{
		return launchJosephForwardSingleRay(device, library, commandQueue,
		    image, lines, projectionValues, params, lineCount);
	}
	if (!areParamsValid(params) || !coversFloatCount(image, voxelCount(params)) ||
	    !coversLineCount(lines, lineCount) ||
	    !coversFloatCount(projectionValues, lineCount))
	{
		return false;
	}

	const std::string kernelName =
	    std::string("joseph_forward_single_ray_axis_") + suffix;
	return launchKernel1D(device, library, commandQueue, kernelName.c_str(),
	    {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	    {{&params, sizeof(params), 3}}, lineCount);
}

bool launchJosephForwardSingleRayTexture(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Texture3D& image, const Sampler& sampler, const Buffer& lines,
    Buffer& projectionValues, const SiddonForwardImageParams& params,
    std::size_t lineCount)
{
	const std::uint32_t sampleStride = josephSampleStride();
	return areParamsValid(params) && image.isValid() &&
	       image.width() == params.nx && image.height() == params.ny &&
	       image.depth() == params.nz && sampler.isValid() &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       (sampleStride <= 1 ?
	            launchKernel1D(device, library, commandQueue,
	                "joseph_forward_single_ray_texture",
	                {{&lines, 0}, {&projectionValues, 1}},
	                {{&params, sizeof(params), 2}}, {{&image, 0}},
	                {{&sampler, 0}}, lineCount) :
	            launchKernel1D(device, library, commandQueue,
	                "joseph_forward_single_ray_texture_sample_stride",
	                {{&lines, 0}, {&projectionValues, 1}},
	                {{&params, sizeof(params), 2},
	                    {&sampleStride, sizeof(sampleStride), 3}},
	                {{&image, 0}}, {{&sampler, 0}}, lineCount));
}

bool launchJosephBackProjectSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image,
    const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount)
{
	const std::uint32_t sampleStride = josephSampleStride();
	if (!areParamsValid(params) || lineCount == 0 ||
	    !coversFloatCount(image, voxelCount(params)) ||
	    !coversLineCount(lines, lineCount) ||
	    !coversFloatCount(projectionValues, lineCount))
	{
		return false;
	}

	const bool useNativeFloatAtomics = useNativeFloatAtomicsForAdjoint();
	if (sampleStride <= 1)
	{
		const char* kernelName =
		    useNativeFloatAtomics &&
		            useThreadgroupSampleAccumulationForAdjoint() ?
		        "joseph_backproject_single_ray_threadgroup_sample_"
		        "native_atomic_float" :
		    useNativeFloatAtomics ?
		        "joseph_backproject_single_ray_native_atomic_float" :
		        "joseph_backproject_single_ray";
		return launchKernel1D(device, library, commandQueue, kernelName,
		    {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
		    {{&params, sizeof(params), 3}}, lineCount);
	}

	return launchKernel1D(device, library, commandQueue,
	    useNativeFloatAtomics ?
	        "joseph_backproject_single_ray_sample_stride_"
	        "native_atomic_float" :
	        "joseph_backproject_single_ray_sample_stride",
	    {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	    {{&params, sizeof(params), 3},
	        {&sampleStride, sizeof(sampleStride), 4}},
	    lineCount);
}

bool launchJosephBackProjectSingleRayAxis(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image,
    const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    std::uint32_t axis)
{
	const char* suffix = axisSuffix(axis);
	const std::uint32_t sampleStride = josephSampleStride();
	if (suffix == nullptr || sampleStride > 1 ||
	    useThreadgroupSampleAccumulationForAdjoint())
	{
		return launchJosephBackProjectSingleRay(device, library, commandQueue,
		    image, lines, projectionValues, params, lineCount);
	}
	if (!areParamsValid(params) || lineCount == 0 ||
	    !coversFloatCount(image, voxelCount(params)) ||
	    !coversLineCount(lines, lineCount) ||
	    !coversFloatCount(projectionValues, lineCount))
	{
		return false;
	}

	const bool useNativeFloatAtomics = useNativeFloatAtomicsForAdjoint();
	const std::string kernelName =
	    std::string("joseph_backproject_single_ray_axis_") + suffix +
	    (useNativeFloatAtomics ? "_native_atomic_float" : "");
	return launchKernel1D(device, library, commandQueue, kernelName.c_str(),
	    {{&image, 0}, {&lines, 1}, {&projectionValues, 2}},
	    {{&params, sizeof(params), 3}}, lineCount);
}

bool launchJosephBackProjectSingleRayUpdateCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues, Buffer& updateCounts,
    const SiddonForwardImageParams& params, std::size_t lineCount)
{
	const std::uint32_t sampleStride = josephSampleStride();
	return areParamsValid(params) && lineCount > 0 &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       coversUint32Count(updateCounts, lineCount) &&
	       (sampleStride <= 1 ?
	            launchKernel1D(device, library, commandQueue,
	                "joseph_backproject_single_ray_update_count",
	                {{&lines, 0}, {&projectionValues, 1}, {&updateCounts, 2}},
	                {{&params, sizeof(params), 3}}, lineCount) :
	            launchKernel1D(device, library, commandQueue,
	                "joseph_backproject_single_ray_update_count_sample_stride",
	                {{&lines, 0}, {&projectionValues, 1}, {&updateCounts, 2}},
	                {{&params, sizeof(params), 3},
	                    {&sampleStride, sizeof(sampleStride), 4}},
	                lineCount));
}

bool launchJosephBackProjectSingleRayVoxelHitCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues,
    Buffer& voxelHitCounts, const SiddonForwardImageParams& params,
    std::size_t lineCount)
{
	const std::uint32_t sampleStride = josephSampleStride();
	return areParamsValid(params) && lineCount > 0 &&
	       coversLineCount(lines, lineCount) &&
	       coversFloatCount(projectionValues, lineCount) &&
	       coversUint32Count(voxelHitCounts, voxelCount(params)) &&
	       (sampleStride <= 1 ?
	            launchKernel1D(device, library, commandQueue,
	                "joseph_backproject_single_ray_voxel_hit_count",
	                {{&lines, 0}, {&projectionValues, 1}, {&voxelHitCounts, 2}},
	                {{&params, sizeof(params), 3}}, lineCount) :
	            launchKernel1D(device, library, commandQueue,
	                "joseph_backproject_single_ray_voxel_hit_count_"
	                "sample_stride",
	                {{&lines, 0}, {&projectionValues, 1}, {&voxelHitCounts, 2}},
	                {{&params, sizeof(params), 3},
	                    {&sampleStride, sizeof(sampleStride), 4}},
	                lineCount));
}

}  // namespace yrt::backend::metal
