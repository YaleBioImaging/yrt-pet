/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

namespace yrt::backend::metal
{
namespace
{

bool coversLineCount(const Buffer& buffer, std::size_t count)
{
	return buffer.isValid() &&
	       buffer.byteCount() >= sizeof(ProjectionLineEndpoints) * count;
}

bool coversRangeCount(const Buffer& buffer, std::size_t count)
{
	return buffer.isValid() &&
	       buffer.byteCount() >= sizeof(ProjectionAlphaRange) * count;
}

bool areBoundsValid(const ProjectionImageBounds& bounds)
{
	return bounds.lengthX > 0.0f && bounds.lengthY > 0.0f &&
	       bounds.lengthZ > 0.0f && bounds.fovRadius > 0.0f;
}

}  // namespace

bool launchProjectionSiddonEntryRange(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, Buffer& ranges, const ProjectionImageBounds& bounds,
    std::size_t lineCount)
{
	return areBoundsValid(bounds) && coversLineCount(lines, lineCount) &&
	       coversRangeCount(ranges, lineCount) &&
	       launchKernel1D(device, library, commandQueue,
	           "projection_siddon_entry_range", {{&lines, 0}, {&ranges, 1}},
	           {{&bounds, sizeof(bounds), 2}}, lineCount);
}

}  // namespace yrt::backend::metal
