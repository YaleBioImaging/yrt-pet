/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/SiddonProjectorMetal.hpp"

#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"

#include <utility>

namespace yrt::backend::metal
{

SiddonProjectorMetal::SiddonProjectorMetal()
    : SiddonProjectorMetal{Context{}}
{
}

SiddonProjectorMetal::SiddonProjectorMetal(const Context& context)
    : m_context{context}
{
}

bool SiddonProjectorMetal::isValid() const
{
	return m_context.isValid();
}

const std::string& SiddonProjectorMetal::errorMessage() const
{
	return m_context.errorMessage();
}

const Context& SiddonProjectorMetal::context() const
{
	return m_context;
}

ProjectionBatchMetal SiddonProjectorMetal::makeBatch(
    std::vector<ProjectionLineEndpoints> lines) const
{
	return ProjectionBatchMetal{m_context, std::move(lines)};
}

ProjectionBatchMetal SiddonProjectorMetal::makeBatch(
    std::vector<ProjectionLineEndpoints> lines,
    std::vector<float> projectionValues) const
{
	return ProjectionBatchMetal{m_context, std::move(lines),
	                            std::move(projectionValues)};
}

bool SiddonProjectorMetal::forwardProjectSingleRay(
    const Image& image, ProjectionBatchMetal& batch, std::uint32_t frame) const
{
	return forwardProjectSiddonSingleRay(m_context, image, batch, frame);
}

bool SiddonProjectorMetal::backProjectSingleRay(
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame) const
{
	return backProjectSiddonSingleRay(m_context, batch, image, frame);
}

}  // namespace yrt::backend::metal
