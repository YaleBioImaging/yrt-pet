/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ProjectionVectorKernels.hpp"

namespace yrt::backend::metal
{

bool launchProjectionClear(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& values, float value,
    std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue, "projection_clear",
	    {{&values, 0}}, {{&value, sizeof(value), 1}}, valueCount);
}

bool launchProjectionAdd(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue, "projection_add",
	    {{&input, 0}, {&output, 1}}, {}, valueCount);
}

bool launchProjectionMultiplyScalar(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& output, float scalar,
    std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue,
	    "projection_multiply_scalar", {{&output, 0}},
	    {{&scalar, sizeof(scalar), 1}}, valueCount);
}

bool launchProjectionMultiplyElementwise(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue,
	    "projection_multiply_elementwise", {{&input, 0}, {&output, 1}}, {},
	    valueCount);
}

bool launchProjectionDivideMeasurements(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& measurements, Buffer& output, std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue,
	    "projection_divide_measurements", {{&measurements, 0}, {&output, 1}},
	    {}, valueCount);
}

bool launchProjectionInvert(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue, "projection_invert",
	    {{&input, 0}, {&output, 1}}, {}, valueCount);
}

bool launchProjectionConvertToACF(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    float unitFactor, std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue, "projection_to_acf",
	    {{&input, 0}, {&output, 1}}, {{&unitFactor, sizeof(unitFactor), 2}},
	    valueCount);
}

bool launchProjectionOsemRatio(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& estimatesAndOutput,
    const Buffer& measurements, const Buffer& sensitivity,
    const Buffer& attenuation, const Buffer& randoms, const Buffer& scatter,
    const Buffer& inVivoAttenuation,
    const ProjectionOsemRatioParams& params, std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue,
	    "projection_osem_ratio",
	    {{&estimatesAndOutput, 0},
	        {&measurements, 1},
	        {&sensitivity, 2},
	        {&attenuation, 3},
	        {&randoms, 4},
	        {&scatter, 5},
	        {&inVivoAttenuation, 6}},
	    {{&params, sizeof(params), 7}}, valueCount);
}

bool launchProjectionCompactOsemRatio(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    Buffer& estimatesAndOutput, const Buffer& measurements,
    const Buffer& multiplicative, const Buffer& additive,
    const Buffer& inVivoAttenuation,
    const ProjectionCompactOsemRatioParams& params, std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue,
	    "projection_compact_osem_ratio",
	    {{&estimatesAndOutput, 0},
	        {&measurements, 1},
	        {&multiplicative, 2},
	        {&additive, 3},
	        {&inVivoAttenuation, 4}},
	    {{&params, sizeof(params), 5}}, valueCount);
}

}  // namespace yrt::backend::metal
