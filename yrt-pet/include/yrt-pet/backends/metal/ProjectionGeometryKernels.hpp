/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>

namespace yrt::backend::metal
{

struct ProjectionLineEndpoints
{
	float p1x;
	float p1y;
	float p1z;
	float p2x;
	float p2y;
	float p2z;
};

struct ProjectionImageBounds
{
	float lengthX;
	float lengthY;
	float lengthZ;
	float fovRadius;
};

struct ProjectionAlphaRange
{
	float alphaMin;
	float alphaMax;
	std::uint32_t valid;
};

bool launchProjectionSiddonEntryRange(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& lines, Buffer& ranges, const ProjectionImageBounds& bounds,
    std::size_t lineCount);

}  // namespace yrt::backend::metal
