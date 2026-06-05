/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>

namespace yrt::backend::metal
{

struct ProjectionOsemRatioParams
{
	float globalScaleFactor;
	float denomThreshold;
	std::uint32_t hasSensitivity;
	std::uint32_t hasAttenuation;
	std::uint32_t hasRandoms;
	std::uint32_t hasScatter;
	std::uint32_t hasInVivoAttenuation;
};

struct ProjectionCompactOsemRatioParams
{
	float denomThreshold;
	std::uint32_t hasInVivoAttenuation;
};

bool launchProjectionClear(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& values, float value,
    std::size_t valueCount);
bool launchProjectionAdd(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    std::size_t valueCount);
bool launchProjectionMultiplyScalar(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& output, float scalar,
    std::size_t valueCount);
bool launchProjectionMultiplyElementwise(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, std::size_t valueCount);
bool launchProjectionDivideMeasurements(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& measurements, Buffer& output, std::size_t valueCount);
bool launchProjectionInvert(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    std::size_t valueCount);
bool launchProjectionConvertToACF(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    float unitFactor, std::size_t valueCount);
bool launchProjectionOsemRatio(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& estimatesAndOutput,
    const Buffer& measurements, const Buffer& sensitivity,
    const Buffer& attenuation, const Buffer& randoms, const Buffer& scatter,
    const Buffer& inVivoAttenuation,
    const ProjectionOsemRatioParams& params, std::size_t valueCount);
bool launchProjectionCompactOsemRatio(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    Buffer& estimatesAndOutput, const Buffer& measurements,
    const Buffer& multiplicative, const Buffer& additive,
    const Buffer& inVivoAttenuation,
    const ProjectionCompactOsemRatioParams& params, std::size_t valueCount);

}  // namespace yrt::backend::metal
