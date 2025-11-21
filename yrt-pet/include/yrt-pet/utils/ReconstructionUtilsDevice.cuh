/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/utils/GPUTypes.cuh"

#include <memory>

namespace yrt::util
{

std::unique_ptr<ImageDevice>
    timeAverageMoveImageDevice(const LORMotion& lorMotion,
                               const ImageBase* unmovedImage,
                               GPULaunchConfig launchConfig);
std::unique_ptr<ImageDevice> timeAverageMoveImageDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    timestamp_t timeStart, timestamp_t timeStop, GPULaunchConfig launchConfig);
std::unique_ptr<OperatorProjectorBase> createOperatorProjectorDevice(
    OperatorProjectorBase::ProjectorType projType,
    const OperatorProjectorParams& projParams,
    const std::vector<Constraint*>& constraintsPtr,
    const cudaStream_t* mainStream, const cudaStream_t* auxStream);

}  // namespace yrt::util
