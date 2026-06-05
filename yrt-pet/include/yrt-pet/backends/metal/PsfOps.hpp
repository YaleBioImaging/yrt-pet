/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"

#include <cstdint>
#include <vector>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class Context;
class Buffer;

// Explicit opt-in helper: copies host Images to Metal buffers, runs the
// existing separable PSF kernels, and copies the result back to output.
bool convolve3DSeparableHost(const Context& context, const Image& input,
                             Image& output,
                             const std::vector<float>& kernelX,
                             const std::vector<float>& kernelY,
                             const std::vector<float>& kernelZ);
bool convolve3DSeparableHost(const Image& input, Image& output,
                             const std::vector<float>& kernelX,
                             const std::vector<float>& kernelY,
                             const std::vector<float>& kernelZ);

// Resident-buffer helper: caller owns input/output/temp/kernel buffers. No host
// image upload or download is performed by this path.
bool convolve3DSeparableBuffer(const Context& context, const Buffer& input,
                               Buffer& output, Buffer& temp,
                               const Buffer& kernelX,
                               std::uint32_t kernelXSize,
                               const Buffer& kernelY,
                               std::uint32_t kernelYSize,
                               const Buffer& kernelZ,
                               std::uint32_t kernelZSize,
                               const ImageShape& shape);

}  // namespace yrt::backend::metal
