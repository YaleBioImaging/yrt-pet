/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <string>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

// Explicit opt-in helper for uniform image-space PSF CSV files. This is only
// compiled when USE_METAL=ON and does not alter CPU or CUDA dispatch.
bool applyPsfForward(const Image& input, Image& output,
                     const std::string& imagePsfFilename);
bool applyPsfAdjoint(const Image& input, Image& output,
                     const std::string& imagePsfFilename);

}  // namespace yrt::backend::metal
