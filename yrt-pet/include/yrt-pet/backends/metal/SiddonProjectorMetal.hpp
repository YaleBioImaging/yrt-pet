/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace yrt
{
class Image;
}

namespace yrt::backend::metal
{

class ProjectionBatchMetal;

class SiddonProjectorMetal
{
public:
	SiddonProjectorMetal();
	explicit SiddonProjectorMetal(const Context& context);

	bool isValid() const;
	const std::string& errorMessage() const;
	const Context& context() const;

	ProjectionBatchMetal makeBatch(
	    std::vector<ProjectionLineEndpoints> lines) const;
	ProjectionBatchMetal makeBatch(
	    std::vector<ProjectionLineEndpoints> lines,
	    std::vector<float> projectionValues) const;

	bool forwardProjectSingleRay(const Image& image, ProjectionBatchMetal& batch,
	                             std::uint32_t frame = 0) const;
	bool backProjectSingleRay(const ProjectionBatchMetal& batch, Image& image,
	                          std::uint32_t frame = 0) const;

private:
	Context m_context;
};

}  // namespace yrt::backend::metal
