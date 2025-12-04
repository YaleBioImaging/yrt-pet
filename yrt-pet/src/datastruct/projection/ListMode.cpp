/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace py::literals;

namespace yrt
{
void py_setup_listmode(py::module& m)
{
	auto c = py::class_<ListMode, ProjectionData>(m, "ListMode");
	c.def("addLORMotion",
	      static_cast<void (ListMode::*)(const std::string& lorMotion_fname)>(
	          &ListMode::addLORMotion),
	      "lorMotion_fname"_a);
	c.def("addLORMotion",
	      static_cast<void (ListMode::*)(
	          const std::shared_ptr<LORMotion>& pp_lorMotion)>(
	          &ListMode::addLORMotion),
	      "lorMotion_fname"_a);
}
}  // namespace yrt
#endif  // if BUILD_PYBIND11

namespace yrt
{
ListMode::ListMode(const Scanner& pr_scanner) : ProjectionData{pr_scanner} {}

float ListMode::getProjectionValue(bin_t id) const
{
	(void)id;
	return 1.0f;
}

void ListMode::setProjectionValue(bin_t id, float val)
{
	(void)id;
	(void)val;
	throw std::logic_error("setProjectionValue unimplemented");
}

timestamp_t ListMode::getScanDuration() const
{
	// By default, return timestamp of the last event - timestamp of first event
	return getTimestamp(count() - 1) - getTimestamp(0);
}

std::unique_ptr<BinIterator> ListMode::getBinIter(int numSubsets,
                                                  int idxSubset) const
{
	ASSERT_MSG(idxSubset < numSubsets,
	           "The subset index has to be smaller than the number of subsets");
	ASSERT_MSG(
	    idxSubset >= 0 && numSubsets > 0,
	    "The subset index cannot be negative, the number of subsets cannot "
	    "be less than or equal to zero");

	size_t numEvents = count();
	return std::make_unique<BinIteratorChronologicalInterleaved>(
	    numSubsets, numEvents, idxSubset);
}

void ListMode::addLORMotion(const std::string& lorMotion_fname)
{
	const auto lorMotion = std::make_shared<LORMotion>(lorMotion_fname);
	addLORMotion(lorMotion);
}

void ListMode::addLORMotion(const std::shared_ptr<LORMotion>& pp_lorMotion)
{
	mp_lorMotion = pp_lorMotion;
	mp_frames = std::make_unique<Array1D<frame_t>>();
	const size_t numEvents = count();
	mp_frames->allocate(numEvents);

	// Populate the frames
	const frame_t numFrames =
	    static_cast<frame_t>(mp_lorMotion->getNumFrames());
	bin_t evId = 0;

	// Skip the events that are before the first frame
	const timestamp_t firstTimestamp = mp_lorMotion->getStartingTimestamp(0);
	while (getTimestamp(evId) < firstTimestamp)
	{
		mp_frames->setFlat(evId, -1);
		evId++;
	}

	// Fill the events in the middle
	frame_t currentFrame;
	for (currentFrame = 0; currentFrame < numFrames - 1; currentFrame++)
	{
		const timestamp_t endingTimestamp =
		    mp_lorMotion->getStartingTimestamp(currentFrame + 1);
		while (evId < numEvents && getTimestamp(evId) < endingTimestamp)
		{
			mp_frames->setFlat(evId, currentFrame);
			evId++;
		}
	}

	// Fill the events at the end
	for (; evId < numEvents; evId++)
	{
		mp_frames->setFlat(evId, currentFrame);
	}
}

bool ListMode::hasMotion() const
{
	return mp_lorMotion != nullptr;
}

frame_t ListMode::getFrame(bin_t id) const
{
	if (mp_lorMotion != nullptr)
	{
		return mp_frames->getFlat(id);
	}
	return ProjectionData::getFrame(id);
}

size_t ListMode::getNumFrames() const
{
	if (mp_lorMotion != nullptr)
	{
		return mp_lorMotion->getNumFrames();
	}
	return ProjectionData::getNumFrames();
}

transform_t ListMode::getTransformOfFrame(frame_t frame) const
{
	ASSERT(mp_lorMotion != nullptr);
	if (frame >= 0)
	{
		return mp_lorMotion->getTransform(frame);
	}
	// For the events before the beginning of the frame
	return ProjectionData::getTransformOfFrame(frame);
}

float ListMode::getDurationOfFrame(frame_t frame) const
{
	ASSERT(mp_lorMotion != nullptr);
	if (frame >= 0)
	{
		return mp_lorMotion->getDuration(frame);
	}
	// For the events before the beginning of the frame
	return ProjectionData::getDurationOfFrame(frame);
}
}  // namespace yrt
