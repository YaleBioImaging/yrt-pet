/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/recon/OSEM.hpp"
#include "yrt-pet/utils/GPUTypes.cuh"

#include <memory>

#if BUILD_CUDA

namespace yrt::util
{

// TODO NOW: Add a function that would do this but with an ImageDevice given as
//  input
// TODO NOW: Make the kernel receive a dynamic frame as input
std::unique_ptr<ImageDevice>
    timeAverageMoveImageDevice(const LORMotion& lorMotion,
                               const ImageBase* unmovedImage,
                               GPULaunchConfig launchConfig);
std::unique_ptr<ImageDevice> timeAverageMoveImageDevice(
    const LORMotion& lorMotion, const ImageBase* unmovedImage,
    timestamp_t timeStart, timestamp_t timeStop, GPULaunchConfig launchConfig);

// This function throws an error if the project was not compiled with CUDA
//  enabled
std::unique_ptr<OSEM> createOSEM_GPU(const Scanner& scanner,
                                     bool isLowRank = false);
//  This function create either a OperatorProjectorSiddon_GPU or a
//  OperatorProjectorDD_GPU. It will also throw an error if the project was not
//  compiled with CUDA enabled. The bin iterator is mandatory, which is why it
//  is being passed by reference here instead of by pointer. This is to ensure
//  the given projector operator is usable after this
std::unique_ptr<OperatorProjectorBase> createOperatorProjectorDevice(
    const ProjectorParams& projParams, const BinIterator& binIter,
    const std::vector<Constraint*>& constraintsPtr,
    const cudaStream_t* mainStream, const cudaStream_t* auxStream);

}  // namespace yrt::util

#endif
