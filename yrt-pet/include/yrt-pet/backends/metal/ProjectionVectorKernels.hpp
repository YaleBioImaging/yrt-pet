/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <cstddef>

namespace yrt::backend::metal
{

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

}  // namespace yrt::backend::metal
