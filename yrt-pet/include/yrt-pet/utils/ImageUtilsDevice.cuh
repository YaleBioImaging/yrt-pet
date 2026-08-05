/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"

#include <memory>

#if BUILD_CUDA

namespace yrt
{

class DynamicFraming;
class LORMotion;

namespace util
{
std::unique_ptr<ImageDevice>
    timeAverageMoveImageDevice(const LORMotion& lorMotion,
                               const ImageBase* unmovedImage,
                               GPULaunchConfig launchConfig);
void timeAverageMoveImageDevice(const LORMotion& lorMotion,
                                const ImageBase* unmovedImage,
                                ImageDevice* outImage, frame_t outDynamicFrame,
                                GPULaunchConfig launchConfig);

std::unique_ptr<ImageDevice> timeAverageMoveImageDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    timestamp_t timeStart, timestamp_t timeStop, GPULaunchConfig launchConfig);
void timeAverageMoveImageDevice(const LORMotion& lorMotion,
                                const ImageBase* unmovedImage,
                                ImageDevice* outImage, timestamp_t timeStart,
                                timestamp_t timeStop, frame_t outDynamicFrame,
                                GPULaunchConfig launchConfig);

std::unique_ptr<ImageDevice> timeAverageMoveImageDynamicDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    const DynamicFraming& dynamicFraming, GPULaunchConfig launchConfig);
void timeAverageMoveImageDynamicDevice(const LORMotion& lorMotion,
                                       const ImageBase* unmovedImage,
                                       ImageDevice* outImage,
                                       const DynamicFraming& dynamicFraming,
                                       GPULaunchConfig launchConfig);
}  // namespace util
}  // namespace yrt

#endif
