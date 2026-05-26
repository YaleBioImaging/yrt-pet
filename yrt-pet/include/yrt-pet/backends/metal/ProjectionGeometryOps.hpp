/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

#include <vector>

namespace yrt
{
class ImageParams;
}

namespace yrt::backend::metal
{

class Context;

ProjectionImageBounds makeProjectionImageBounds(const yrt::ImageParams& params);

bool computeSiddonEntryRanges(const Context& context,
    const std::vector<ProjectionLineEndpoints>& lines,
    const ProjectionImageBounds& bounds,
    std::vector<ProjectionAlphaRange>& ranges);

}  // namespace yrt::backend::metal
