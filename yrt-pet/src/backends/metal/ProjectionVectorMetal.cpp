/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ProjectionVectorMetal.hpp"

#include "yrt-pet/backends/metal/ProjectionVectorOps.hpp"

#include <utility>

namespace yrt::backend::metal
{

ProjectionVectorMetal::ProjectionVectorMetal()
    : ProjectionVectorMetal{std::vector<float>{}}
{
}

ProjectionVectorMetal::ProjectionVectorMetal(std::vector<float> values)
    : ProjectionVectorMetal{Context{}, std::move(values)}
{
}

ProjectionVectorMetal::ProjectionVectorMetal(std::size_t size,
                                             float initialValue)
    : ProjectionVectorMetal{Context{}, size, initialValue}
{
}

ProjectionVectorMetal::ProjectionVectorMetal(const Context& context,
                                             std::vector<float> values)
    : m_context{context}, m_values{std::move(values)}
{
}

ProjectionVectorMetal::ProjectionVectorMetal(const Context& context,
                                             std::size_t size,
                                             float initialValue)
    : ProjectionVectorMetal{context, std::vector<float>(size, initialValue)}
{
}

bool ProjectionVectorMetal::isValid() const
{
	return m_context.isValid();
}

const std::string& ProjectionVectorMetal::errorMessage() const
{
	return m_context.errorMessage();
}

std::size_t ProjectionVectorMetal::size() const
{
	return m_values.size();
}

bool ProjectionVectorMetal::empty() const
{
	return m_values.empty();
}

const std::vector<float>& ProjectionVectorMetal::values() const
{
	return m_values;
}

std::vector<float>& ProjectionVectorMetal::values()
{
	return m_values;
}

bool ProjectionVectorMetal::clear(float value)
{
	return yrt::backend::metal::clear(m_context, m_values, value);
}

bool ProjectionVectorMetal::add(const std::vector<float>& input)
{
	return yrt::backend::metal::add(m_context, input, m_values);
}

bool ProjectionVectorMetal::add(const ProjectionVectorMetal& input)
{
	return add(input.values());
}

bool ProjectionVectorMetal::multiplyByScalar(float scalar)
{
	return yrt::backend::metal::multiplyByScalar(m_context, m_values, scalar);
}

bool ProjectionVectorMetal::multiplyElementwise(const std::vector<float>& input)
{
	return yrt::backend::metal::multiplyElementwise(m_context, input,
	                                               m_values);
}

bool ProjectionVectorMetal::multiplyElementwise(
    const ProjectionVectorMetal& input)
{
	return multiplyElementwise(input.values());
}

bool ProjectionVectorMetal::divideMeasurements(
    const std::vector<float>& measurements)
{
	return yrt::backend::metal::divideMeasurements(m_context, measurements,
	                                              m_values);
}

bool ProjectionVectorMetal::divideMeasurements(
    const ProjectionVectorMetal& measurements)
{
	return divideMeasurements(measurements.values());
}

bool ProjectionVectorMetal::invert()
{
	std::vector<float> output(m_values.size());
	if (!yrt::backend::metal::invert(m_context, m_values, output))
	{
		return false;
	}
	m_values = std::move(output);
	return true;
}

bool ProjectionVectorMetal::convertToACF(float unitFactor)
{
	std::vector<float> output(m_values.size());
	if (!yrt::backend::metal::convertToACF(m_context, m_values, output,
	        unitFactor))
	{
		return false;
	}
	m_values = std::move(output);
	return true;
}

}  // namespace yrt::backend::metal
