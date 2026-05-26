/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"

#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorMetal.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"

#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace yrt::backend::metal
{
namespace
{

struct BridgeEvent
{
	bin_t bin = 0;
	ProjectionLineEndpoints line;
	frame_t frame = 0;
	float projectionValue = 0.0f;
};

OperatorProjectorMetalSupport supported()
{
	return {true, {}};
}

OperatorProjectorMetalSupport unsupported(std::string reason)
{
	return {false, std::move(reason)};
}

bool hasOnlySupportedProjectionProperties(
    const std::set<ProjectionPropertyType>& projectionProperties)
{
	for (const ProjectionPropertyType property : projectionProperties)
	{
		if (property != ProjectionPropertyType::LOR &&
		    property != ProjectionPropertyType::DYNAMIC_FRAME)
		{
			return false;
		}
	}
	return true;
}

ProjectionLineEndpoints makeCenteredLine(const Line3D& line,
                                         const ImageParams& params)
{
	return {line.point1.x - params.off_x, line.point1.y - params.off_y,
	        line.point1.z - params.off_z, line.point2.x - params.off_x,
	        line.point2.y - params.off_y, line.point2.z - params.off_z};
}

bool gatherBridgeEvents(const ProjectionData& projectionData,
                        const Image& image, const BinIterator& binIterator,
                        const BinLoader& binLoader,
                        bool includeProjectionValues,
                        std::vector<BridgeEvent>& events)
{
	if (!image.isMemoryValid() ||
	    !binLoader.getPropertyManager().has(ProjectionPropertyType::LOR) ||
	    binLoader.getProjectionPropertiesRawPointer() == nullptr)
	{
		return false;
	}

	BinFilter::CollectInfoFlags collectInfoFlags(false);
	binLoader.collectFlags(collectInfoFlags);
	const ProjectionPropertyManager& propertyManager =
	    binLoader.getPropertyManager();
	PropertyUnit* properties = binLoader.getProjectionPropertiesRawPointer();
	const ImageParams& imageParams = image.getParams();

	std::vector<BridgeEvent> gatheredEvents;
	gatheredEvents.reserve(binIterator.size());
	for (std::size_t i = 0; i < binIterator.size(); ++i)
	{
		const bin_t bin = binIterator.get(static_cast<bin_t>(i));
		binLoader.collectInfo(bin, 0, 0, projectionData, collectInfoFlags);
		if (!binLoader.verifyConstraints(0))
		{
			continue;
		}

		projectionData.collectProjectionProperties(propertyManager, properties,
		                                           0, bin);
		const Line3D line = propertyManager.getDataValue<Line3D>(
		    properties, 0, ProjectionPropertyType::LOR);
		frame_t frame = 0;
		if (propertyManager.has(ProjectionPropertyType::DYNAMIC_FRAME))
		{
			frame = propertyManager.getDataValue<frame_t>(
			    properties, 0, ProjectionPropertyType::DYNAMIC_FRAME);
		}
		if (frame >= imageParams.nt)
		{
			return false;
		}

		BridgeEvent event;
		event.bin = bin;
		event.line = makeCenteredLine(line, imageParams);
		event.frame = frame;
		if (includeProjectionValues)
		{
			event.projectionValue = projectionData.getProjectionValue(bin);
		}
		gatheredEvents.push_back(event);
	}

	events = std::move(gatheredEvents);
	return true;
}

std::map<frame_t, std::vector<std::size_t>>
    groupNonNegativeFrames(const std::vector<BridgeEvent>& events)
{
	std::map<frame_t, std::vector<std::size_t>> frameGroups;
	for (std::size_t i = 0; i < events.size(); ++i)
	{
		if (events[i].frame >= 0)
		{
			frameGroups[events[i].frame].push_back(i);
		}
	}
	return frameGroups;
}

}  // namespace

OperatorProjectorMetalBridge::OperatorProjectorMetalBridge(
    const Context& context)
    : m_context{context}
{
}

OperatorProjectorMetalSupport OperatorProjectorMetalBridge::canRunSiddon(
    const OperatorProjector& projector) const
{
	if (!m_context.isValid())
	{
		return unsupported(m_context.errorMessage());
	}
	if (projector.getProjectorType() != ProjectorType::SIDDON)
	{
		return unsupported("Only Siddon projection is supported");
	}
	if (projector.getUpdaterType() != UpdaterType::DEFAULT4D)
	{
		return unsupported("Only DEFAULT4D updater mode is supported");
	}
	if (projector.getTOFHelper() != nullptr)
	{
		return unsupported("TOF projection is not supported");
	}
	if (projector.getProjectionPsfManager() != nullptr)
	{
		return unsupported("Projection-space PSF is not supported");
	}

	const std::set<ProjectionPropertyType> projectionProperties =
	    projector.getProjectionPropertyTypes();
	if (projectionProperties.count(ProjectionPropertyType::DET_ORIENT) != 0)
	{
		return unsupported("Siddon multi-ray projection is not supported");
	}
	if (!hasOnlySupportedProjectionProperties(projectionProperties))
	{
		return unsupported("Projection property set is not supported");
	}
	return supported();
}

bool OperatorProjectorMetalBridge::applyA(
    const OperatorProjector& projector, const Image& image,
    ProjectionData& projectionData, const BinIterator& binIterator,
    const BinLoader& binLoader) const
{
	if (!canRunSiddon(projector).supported)
	{
		return false;
	}

	std::vector<BridgeEvent> events;
	if (!gatherBridgeEvents(projectionData, image, binIterator, binLoader,
	        false, events))
	{
		return false;
	}
	if (events.empty())
	{
		return true;
	}

	std::vector<float> results(events.size(), 0.0f);
	const SiddonProjectorMetal metalProjector(m_context);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		std::vector<ProjectionLineEndpoints> lines;
		lines.reserve(indices.size());
		for (const std::size_t index : indices)
		{
			lines.push_back(events[index].line);
		}

		auto batch = metalProjector.makeBatch(
		    lines, std::vector<float>(lines.size(), 0.0f));
		std::vector<float> frameResults;
		if (!batch.isValid() ||
		    !metalProjector.forwardProjectSingleRay(
		        image, batch, static_cast<std::uint32_t>(frame)) ||
		    !batch.copyProjectionValuesToHost(frameResults) ||
		    frameResults.size() != indices.size())
		{
			return false;
		}

		for (std::size_t i = 0; i < indices.size(); ++i)
		{
			results[indices[i]] = frameResults[i];
		}
	}

	for (std::size_t i = 0; i < events.size(); ++i)
	{
		projectionData.setProjectionValue(events[i].bin, results[i]);
	}
	return true;
}

bool OperatorProjectorMetalBridge::applyAH(
    const OperatorProjector& projector, const ProjectionData& projectionData,
    Image& image, const BinIterator& binIterator,
    const BinLoader& binLoader) const
{
	if (!canRunSiddon(projector).supported)
	{
		return false;
	}

	std::vector<BridgeEvent> events;
	if (!gatherBridgeEvents(projectionData, image, binIterator, binLoader, true,
	        events))
	{
		return false;
	}
	if (events.empty())
	{
		return true;
	}

	ImageOwned workingImage(image.getParams());
	workingImage.allocate();
	workingImage.copyFromImage(&image);

	const SiddonProjectorMetal metalProjector(m_context);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		std::vector<ProjectionLineEndpoints> lines;
		std::vector<float> projectionValues;
		lines.reserve(indices.size());
		projectionValues.reserve(indices.size());
		for (const std::size_t index : indices)
		{
			if (std::abs(events[index].projectionValue) == 0.0f)
			{
				continue;
			}
			lines.push_back(events[index].line);
			projectionValues.push_back(events[index].projectionValue);
		}
		if (lines.empty())
		{
			continue;
		}

		auto batch = metalProjector.makeBatch(lines, projectionValues);
		if (!batch.isValid() ||
		    !metalProjector.backProjectSingleRay(
		        batch, workingImage, static_cast<std::uint32_t>(frame)))
		{
			return false;
		}
	}

	image.copyFromImage(&workingImage);
	return true;
}

}  // namespace yrt::backend::metal
