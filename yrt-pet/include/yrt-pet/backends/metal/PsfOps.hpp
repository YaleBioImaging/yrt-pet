/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class Context;

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

}  // namespace yrt::backend::metal
