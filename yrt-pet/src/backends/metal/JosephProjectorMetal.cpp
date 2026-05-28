/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/JosephProjectorMetal.hpp"

#include "yrt-pet/backends/metal/JosephProjectorOps.hpp"
#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"

#include <utility>

namespace yrt::backend::metal
{

JosephProjectorMetal::JosephProjectorMetal()
    : JosephProjectorMetal{Context{}}
{
}

JosephProjectorMetal::JosephProjectorMetal(const Context& context)
    : m_context{context}
{
}

bool JosephProjectorMetal::isValid() const
{
	return m_context.isValid();
}

const std::string& JosephProjectorMetal::errorMessage() const
{
	return m_context.errorMessage();
}

const Context& JosephProjectorMetal::context() const
{
	return m_context;
}

ProjectionBatchMetal JosephProjectorMetal::makeBatch(
    std::vector<ProjectionLineEndpoints> lines) const
{
	return ProjectionBatchMetal{m_context, std::move(lines)};
}

ProjectionBatchMetal JosephProjectorMetal::makeBatch(
    std::vector<ProjectionLineEndpoints> lines,
    std::vector<float> projectionValues) const
{
	return ProjectionBatchMetal{m_context, std::move(lines),
	                            std::move(projectionValues)};
}

bool JosephProjectorMetal::forwardProjectSingleRay(
    const Image& image, ProjectionBatchMetal& batch, std::uint32_t frame,
    SiddonProjectorKernelProfile* profile) const
{
	return forwardProjectJosephSingleRay(m_context, image, batch, frame,
	                                     profile);
}

bool JosephProjectorMetal::backProjectSingleRay(
    const ProjectionBatchMetal& batch, Image& image, std::uint32_t frame,
    SiddonProjectorKernelProfile* profile) const
{
	return backProjectJosephSingleRay(m_context, batch, image, frame, profile);
}

}  // namespace yrt::backend::metal
