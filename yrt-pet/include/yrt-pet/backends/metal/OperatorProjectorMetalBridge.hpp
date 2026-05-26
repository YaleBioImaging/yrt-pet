/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalContext.hpp"

#include <string>

namespace yrt
{
class BinIterator;
class BinLoader;
class Image;
class OperatorProjector;
class ProjectionData;
}

namespace yrt::backend::metal
{

struct OperatorProjectorMetalSupport
{
	bool supported = false;
	std::string reason;
};

class OperatorProjectorMetalBridge
{
public:
	explicit OperatorProjectorMetalBridge(const Context& context);

	OperatorProjectorMetalSupport
	    canRunSiddon(const OperatorProjector& projector) const;

	bool applyA(const OperatorProjector& projector, const Image& image,
	            ProjectionData& projectionData, const BinIterator& binIterator,
	            const BinLoader& binLoader) const;
	bool applyAH(const OperatorProjector& projector,
	             const ProjectionData& projectionData, Image& image,
	             const BinIterator& binIterator,
	             const BinLoader& binLoader) const;

private:
	Context m_context;
};

}  // namespace yrt::backend::metal
