/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"

#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

#include <utility>

namespace yrt::backend::metal
{
namespace
{

std::size_t lineByteCount(std::size_t count)
{
	return sizeof(ProjectionLineEndpoints) * count;
}

std::size_t projectionValueByteCount(std::size_t count)
{
	return sizeof(float) * count;
}

std::size_t rangeByteCount(std::size_t count)
{
	return sizeof(ProjectionAlphaRange) * count;
}

std::vector<float> zeroProjectionValues(std::size_t count)
{
	return std::vector<float>(count, 0.0f);
}

}  // namespace

ProjectionBatchMetal::ProjectionBatchMetal()
    : ProjectionBatchMetal{std::vector<ProjectionLineEndpoints>{}}
{
}

ProjectionBatchMetal::ProjectionBatchMetal(
    std::vector<ProjectionLineEndpoints> lines)
    : ProjectionBatchMetal{Context{}, std::move(lines)}
{
}

ProjectionBatchMetal::ProjectionBatchMetal(
    std::vector<ProjectionLineEndpoints> lines,
    std::vector<float> projectionValues)
    : ProjectionBatchMetal{Context{}, std::move(lines),
          std::move(projectionValues)}
{
}

ProjectionBatchMetal::ProjectionBatchMetal(
    const Context& context, std::vector<ProjectionLineEndpoints> lines)
    : m_context{context}, m_lines{std::move(lines)}
{
	m_shape.eventCount = m_lines.size();
	if (!initializeBuffers(zeroProjectionValues(m_lines.size())))
	{
		return;
	}
}

ProjectionBatchMetal::ProjectionBatchMetal(
    const Context& context, std::vector<ProjectionLineEndpoints> lines,
    std::vector<float> projectionValues)
    : m_context{context}, m_lines{std::move(lines)}
{
	m_shape.eventCount = m_lines.size();
	if (!initializeBuffers(projectionValues))
	{
		return;
	}
}

bool ProjectionBatchMetal::isValid() const
{
	return m_context.isValid() && !m_lines.empty() && m_lorBuffer.isValid() &&
	       m_projectionValuesBuffer.isValid() && m_errorMessage.empty();
}

const std::string& ProjectionBatchMetal::errorMessage() const
{
	return m_errorMessage.empty() ? m_context.errorMessage() : m_errorMessage;
}

std::size_t ProjectionBatchMetal::size() const
{
	return m_shape.eventCount;
}

bool ProjectionBatchMetal::empty() const
{
	return size() == 0;
}

const ProjectionBatchShape& ProjectionBatchMetal::shape() const
{
	return m_shape;
}

const std::vector<ProjectionLineEndpoints>& ProjectionBatchMetal::lines() const
{
	return m_lines;
}

const Buffer& ProjectionBatchMetal::lorBuffer() const
{
	return m_lorBuffer;
}

Buffer& ProjectionBatchMetal::projectionValuesBuffer()
{
	return m_projectionValuesBuffer;
}

const Buffer& ProjectionBatchMetal::projectionValuesBuffer() const
{
	return m_projectionValuesBuffer;
}

bool ProjectionBatchMetal::setProjectionValues(
    const std::vector<float>& projectionValues)
{
	if (!isValid())
	{
		return false;
	}
	if (projectionValues.size() != size())
	{
		return false;
	}
	return m_projectionValuesBuffer.copyFromHost(projectionValues.data(),
	    projectionValueByteCount(projectionValues.size()));
}

bool ProjectionBatchMetal::copyProjectionValuesToHost(
    std::vector<float>& projectionValues) const
{
	if (!isValid())
	{
		return false;
	}

	std::vector<float> output(size(), 0.0f);
	if (!m_projectionValuesBuffer.copyToHost(output.data(),
	        projectionValueByteCount(output.size())))
	{
		return false;
	}
	projectionValues = std::move(output);
	return true;
}

bool ProjectionBatchMetal::computeSiddonEntryRanges(
    const ProjectionImageBounds& bounds,
    std::vector<ProjectionAlphaRange>& ranges) const
{
	if (!isValid())
	{
		return false;
	}

	std::vector<ProjectionAlphaRange> output(size());
	Buffer rangesBuffer = Buffer::allocate(m_context.device(),
	    rangeByteCount(output.size()));
	if (!rangesBuffer.isValid() ||
	    !launchProjectionSiddonEntryRange(m_context.device(),
	        m_context.library(), m_context.commandQueue(), m_lorBuffer,
	        rangesBuffer, bounds, size()) ||
	    !rangesBuffer.copyToHost(output.data(), rangeByteCount(output.size())))
	{
		return false;
	}
	ranges = std::move(output);
	return true;
}

bool ProjectionBatchMetal::initializeBuffers(
    const std::vector<float>& projectionValues)
{
	if (!m_context.isValid())
	{
		return setError("Metal context is not valid");
	}
	if (m_lines.empty())
	{
		return setError("Projection batch has no events");
	}
	if (projectionValues.size() != m_lines.size())
	{
		return setError("Projection value count does not match LOR count");
	}

	m_lorBuffer = Buffer::copyFromHost(m_context.device(), m_lines.data(),
	    lineByteCount(m_lines.size()));
	m_projectionValuesBuffer =
	    Buffer::copyFromHost(m_context.device(), projectionValues.data(),
	        projectionValueByteCount(projectionValues.size()));
	if (!m_lorBuffer.isValid() || !m_projectionValuesBuffer.isValid())
	{
		return setError("Failed to allocate Metal projection batch buffers");
	}
	return true;
}

bool ProjectionBatchMetal::setError(std::string errorMessage)
{
	m_errorMessage = std::move(errorMessage);
	return false;
}

}  // namespace yrt::backend::metal
