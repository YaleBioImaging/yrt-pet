/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ProjectionGeometryOps.hpp"

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/datastruct/image/ImageParams.hpp"

#include <cstddef>
#include <utility>

namespace yrt::backend::metal
{
namespace
{

std::size_t byteCount(const std::vector<ProjectionLineEndpoints>& values)
{
	return sizeof(ProjectionLineEndpoints) * values.size();
}

std::size_t byteCount(const std::vector<ProjectionAlphaRange>& values)
{
	return sizeof(ProjectionAlphaRange) * values.size();
}

bool areBoundsValid(const ProjectionImageBounds& bounds)
{
	return bounds.lengthX > 0.0f && bounds.lengthY > 0.0f &&
	       bounds.lengthZ > 0.0f && bounds.fovRadius > 0.0f;
}

}  // namespace

ProjectionImageBounds makeProjectionImageBounds(const yrt::ImageParams& params)
{
	return {params.length_x, params.length_y, params.length_z,
	    params.fovRadius};
}

bool computeSiddonEntryRanges(const Context& context,
    const std::vector<ProjectionLineEndpoints>& lines,
    const ProjectionImageBounds& bounds,
    std::vector<ProjectionAlphaRange>& ranges)
{
	if (!context.isValid() || lines.empty() || !areBoundsValid(bounds))
	{
		return false;
	}

	std::vector<ProjectionAlphaRange> output(lines.size());
	Buffer linesBuffer =
	    Buffer::copyFromHost(context.device(), lines.data(), byteCount(lines));
	Buffer rangesBuffer =
	    Buffer::allocate(context.device(), byteCount(output));
	if (!linesBuffer.isValid() || !rangesBuffer.isValid() ||
	    !launchProjectionSiddonEntryRange(context.device(), context.library(),
	        context.commandQueue(), linesBuffer, rangesBuffer, bounds,
	        lines.size()) ||
	    !rangesBuffer.copyToHost(output.data(), byteCount(output)))
	{
		return false;
	}

	ranges = std::move(output);
	return true;
}

}  // namespace yrt::backend::metal
