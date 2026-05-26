/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace yrt::backend::metal
{

struct ProjectionBatchShape
{
	std::size_t eventCount = 0;
	bool hasDetectorOrientations = false;
	bool hasTof = false;
	bool hasDynamicFrames = false;
};

class ProjectionBatchMetal
{
public:
	ProjectionBatchMetal();
	explicit ProjectionBatchMetal(std::vector<ProjectionLineEndpoints> lines);
	ProjectionBatchMetal(std::vector<ProjectionLineEndpoints> lines,
	                     std::vector<float> projectionValues);
	ProjectionBatchMetal(const Context& context,
	                     std::vector<ProjectionLineEndpoints> lines);
	ProjectionBatchMetal(const Context& context,
	                     std::vector<ProjectionLineEndpoints> lines,
	                     std::vector<float> projectionValues);

	bool isValid() const;
	const std::string& errorMessage() const;

	std::size_t size() const;
	bool empty() const;
	const ProjectionBatchShape& shape() const;
	const std::vector<ProjectionLineEndpoints>& lines() const;

	const Buffer& lorBuffer() const;
	Buffer& projectionValuesBuffer();
	const Buffer& projectionValuesBuffer() const;

	bool setProjectionValues(const std::vector<float>& projectionValues);
	bool copyProjectionValuesToHost(std::vector<float>& projectionValues) const;
	bool computeSiddonEntryRanges(const ProjectionImageBounds& bounds,
	    std::vector<ProjectionAlphaRange>& ranges) const;

private:
	bool initializeBuffers(const std::vector<float>& projectionValues);
	bool setError(std::string errorMessage);

	Context m_context;
	ProjectionBatchShape m_shape;
	std::vector<ProjectionLineEndpoints> m_lines;
	Buffer m_lorBuffer;
	Buffer m_projectionValuesBuffer;
	std::string m_errorMessage;
};

}  // namespace yrt::backend::metal
