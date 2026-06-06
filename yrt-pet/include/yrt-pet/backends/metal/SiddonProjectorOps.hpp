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

bool makeSiddonForwardImageParams(const Image& image, std::uint32_t frame,
                                  SiddonForwardImageParams& params);
bool uploadSiddonImageBuffer(const Context& context, const Image& image,
                             Buffer& imageBuffer,
                             SiddonProjectorKernelProfile* profile = nullptr);
bool downloadSiddonImageBuffer(const Buffer& imageBuffer, Image& image,
                               SiddonProjectorKernelProfile* profile = nullptr);
bool downloadSiddonImageBuffer(const Context& context,
                               const Buffer& imageBuffer, Image& image,
                               SiddonProjectorKernelProfile* profile = nullptr);

bool forwardProjectSiddonSingleRay(const Context& context, const Image& image,
    ProjectionBatchMetal& batch, std::uint32_t frame = 0,
    SiddonProjectorKernelProfile* profile = nullptr,
    const ProjectorKernelOptions* options = nullptr);
bool forwardProjectSiddonSingleRay(const Context& context,
    const Buffer& imageBuffer, ProjectionBatchMetal& batch,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile = nullptr,
    const ProjectorKernelOptions* options = nullptr);
bool backProjectSiddonSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame = 0,
    SiddonProjectorKernelProfile* profile = nullptr,
    const ProjectorKernelOptions* options = nullptr);
bool backProjectSiddonSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Buffer& imageBuffer,
    const SiddonForwardImageParams& params,
    SiddonProjectorKernelProfile* profile = nullptr,
    const ProjectorKernelOptions* options = nullptr);

}  // namespace yrt::backend::metal
