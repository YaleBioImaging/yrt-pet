/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Types.hpp"

#include <memory>

namespace yrt
{
class Image;
class ImageOwned;
class LORMotion;
}

namespace yrt::backend::metal
{

class Context;

std::unique_ptr<ImageOwned> timeAverageMoveImage(
    const Context& context, const LORMotion& lorMotion,
    const Image& unmovedImage, timestamp_t timeStart, timestamp_t timeStop);

bool timeAverageMoveImage(const Context& context, const LORMotion& lorMotion,
    const Image& unmovedImage, Image& outImage, timestamp_t timeStart,
    timestamp_t timeStop, frame_t outDynamicFrame = 0);

}  // namespace yrt::backend::metal
