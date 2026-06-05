/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"

#include <cstddef>
#include <cstdint>

namespace yrt::backend::metal
{

bool launchJosephForwardSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& image, const Buffer& lines, Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount);
bool launchJosephForwardSingleRayAxis(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& image, const Buffer& lines, Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    std::uint32_t axis);
bool launchJosephForwardSingleRayTexture(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Texture3D& image, const Sampler& sampler, const Buffer& lines,
    Buffer& projectionValues, const SiddonForwardImageParams& params,
    std::size_t lineCount);
bool launchJosephBackProjectSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image,
    const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount);
bool launchJosephBackProjectSingleRayAxis(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image,
    const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount,
    std::uint32_t axis);
bool launchJosephBackProjectSingleRayUpdateCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues, Buffer& updateCounts,
    const SiddonForwardImageParams& params, std::size_t lineCount);
bool launchJosephBackProjectSingleRayVoxelHitCount(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, const Buffer& projectionValues,
    Buffer& voxelHitCounts, const SiddonForwardImageParams& params,
    std::size_t lineCount);

}  // namespace yrt::backend::metal
