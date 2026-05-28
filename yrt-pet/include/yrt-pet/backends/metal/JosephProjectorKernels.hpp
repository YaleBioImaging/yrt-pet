/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"

#include <cstddef>

namespace yrt::backend::metal
{

bool launchJosephForwardSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& image, const Buffer& lines, Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount);
bool launchJosephBackProjectSingleRay(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image,
    const Buffer& lines, const Buffer& projectionValues,
    const SiddonForwardImageParams& params, std::size_t lineCount);

}  // namespace yrt::backend::metal
