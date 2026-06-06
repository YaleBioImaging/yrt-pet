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

struct SiddonForwardImageParams
{
	std::uint32_t nx;
	std::uint32_t ny;
	std::uint32_t nz;
	std::uint32_t nt;
	std::uint32_t frame;
	float lengthX;
	float lengthY;
	float lengthZ;
	float voxelX;
	float voxelY;
	float voxelZ;
	float invVoxelX;
	float invVoxelY;
	float invVoxelZ;
	float halfLengthX;
	float halfLengthY;
	float halfLengthZ;
	float fovRadius;
};

struct ProjectorKernelOptions
{
	bool nativeFloatAtomicsExplicit = false;
	bool nativeFloatAtomics = false;
	bool josephAdjointAxisSwitchOnceExplicit = false;
	bool josephAdjointAxisSwitchOnce = false;
	KernelLaunchOptions launchOptions;
};

bool launchSiddonForwardSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& image, const Buffer& lines, Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    const ProjectorKernelOptions* options = nullptr);
bool launchSiddonBackProjectSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    Buffer& image, const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    const ProjectorKernelOptions* options = nullptr);
bool launchSiddonBackProjectSingleRayUpdateCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues, Buffer& updateCounts,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    const ProjectorKernelOptions* options = nullptr);
bool launchSiddonBackProjectSingleRayVoxelHitCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues,
    Buffer& voxelHitCounts, const SiddonForwardImageParams& params,
    std::size_t lineCount, const ProjectorKernelOptions* options = nullptr);

}  // namespace yrt::backend::metal
