/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalContext.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace yrt::backend::metal
{

class ProjectionVectorMetal
{
public:
	ProjectionVectorMetal();
	explicit ProjectionVectorMetal(std::vector<float> values);
	ProjectionVectorMetal(std::size_t size, float initialValue);
	ProjectionVectorMetal(const Context& context, std::vector<float> values);
	ProjectionVectorMetal(const Context& context, std::size_t size,
	                      float initialValue);

	bool isValid() const;
	const std::string& errorMessage() const;

	std::size_t size() const;
	bool empty() const;
	const std::vector<float>& values() const;
	std::vector<float>& values();

	bool clear(float value);
	bool add(const std::vector<float>& input);
	bool add(const ProjectionVectorMetal& input);
	bool multiplyByScalar(float scalar);
	bool multiplyElementwise(const std::vector<float>& input);
	bool multiplyElementwise(const ProjectionVectorMetal& input);
	bool divideMeasurements(const std::vector<float>& measurements);
	bool divideMeasurements(const ProjectionVectorMetal& measurements);
	bool invert();
	bool convertToACF(float unitFactor);

private:
	Context m_context;
	std::vector<float> m_values;
};

}  // namespace yrt::backend::metal
