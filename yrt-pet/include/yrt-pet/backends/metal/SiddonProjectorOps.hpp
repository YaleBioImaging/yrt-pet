/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cstdint>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class Context;
class ProjectionBatchMetal;

bool forwardProjectSiddonSingleRay(const Context& context, const Image& image,
    ProjectionBatchMetal& batch, std::uint32_t frame = 0);
bool backProjectSiddonSingleRay(const Context& context,
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame = 0);

}  // namespace yrt::backend::metal
