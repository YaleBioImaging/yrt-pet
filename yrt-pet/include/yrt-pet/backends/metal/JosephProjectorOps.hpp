/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"

#include <cstdint>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class Context;
class Buffer;
class ProjectionBatchMetal;
struct SiddonProjectorKernelProfile;

bool forwardProjectJosephSingleRay(const Context& context, const Image& image,
    ProjectionBatchMetal& batch, std::uint32_t frame = 0,
    SiddonProjectorKernelProfile* profile = nullptr);
bool forwardProjectJosephSingleRay(const Context& context,
    const Buffer& imageBuffer, ProjectionBatchMetal& batch,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile = nullptr);
bool backProjectJosephSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame = 0,
    SiddonProjectorKernelProfile* profile = nullptr);
bool backProjectJosephSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Buffer& imageBuffer,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile = nullptr);

}  // namespace yrt::backend::metal
