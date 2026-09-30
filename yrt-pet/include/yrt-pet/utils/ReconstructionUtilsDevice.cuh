/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/recon/OSEM.hpp"

#include <memory>

#if BUILD_CUDA

namespace yrt
{

class DynamicFraming;
class LORMotion;

namespace util
{

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

}  // namespace util
}  // namespace yrt

#endif
