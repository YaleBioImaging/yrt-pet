/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

	// bind the shared_ptr<DynamicFraming> overload
	c.def(
	    "addDynamicFraming",
	    py::overload_cast<const std::shared_ptr<DynamicFraming>&>(
	        &ListMode::addDynamicFraming),
	    py::arg("framing"));

	// Possibility to bind a NumPy 1D array
//	c.def("addDynamicFraming",
//	       [](ListMode &self,
//	          py::array_t<timestamp_t, py::array::c_style|py::array::forcecast> np_data) {
//		       auto buffer = np_data.request();
//		       if (buffer.ndim != 1)
//			       throw std::invalid_argument("addDynamicFraming() expects a 1D array");
//
//		       // Copy into std::vector<timestamp_t>
//		       size_t n = buffer.shape[0];
//		       std::vector<timestamp_t> v(n);
//		       std::memcpy(v.data(), buffer.ptr, n * sizeof(timestamp_t));
//
//		       // Build a DynamicFraming and hand it over
//		       auto df = std::make_shared<DynamicFraming>(v);
//		       self.addDynamicFraming(df);
//	       },
//	       py::arg("timestamps"));
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
	return std::make_unique<BinIteratorChronological>(numSubsets, numEvents,
	                                                  idxSubset);
}

void ListMode::addLORMotion(const std::string& lorMotion_fname)
{
	const auto lorMotion = std::make_shared<LORMotion>(lorMotion_fname);
	addLORMotion(lorMotion);
}

void ListMode::addLORMotion(const std::shared_ptr<LORMotion>& pp_lorMotion)
{
	mp_lorMotion = pp_lorMotion;
	mp_motionFrames = std::make_unique<Array1D<frame_t>>();
	const size_t numEvents = count();
	mp_motionFrames->allocate(numEvents);

	// Populate the frames
	const frame_t numFrames =
	    static_cast<frame_t>(mp_lorMotion->getNumFrames());
	bin_t evId = 0;

	// Skip the events that are before the first frame
	const timestamp_t firstTimestamp = mp_lorMotion->getStartingTimestamp(0);
	while (getTimestamp(evId) < firstTimestamp)
	{
		mp_motionFrames->setFlat(evId, -1);
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
			mp_motionFrames->setFlat(evId, currentFrame);
			evId++;
		}
	}

	// Fill the events at the end
	for (; evId < numEvents; evId++)
	{
		mp_motionFrames->setFlat(evId, currentFrame);
	}
}

void ListMode::addDynamicFraming(const std::vector<timestamp_t>& dynamicFramingVector)
{
	const auto dynamicFraming = std::make_shared<DynamicFraming>(dynamicFramingVector);
	addDynamicFraming(dynamicFraming);
}


void ListMode::addDynamicFraming(
    const std::shared_ptr<DynamicFraming>& pp_dynamicFraming)
{
	mp_dynamicFraming = pp_dynamicFraming;
	mp_dynamicFrames = std::make_unique<Array1D<frame_t>>();
	const size_t numEvents = count();
	mp_dynamicFrames->allocate(numEvents);

	const auto numFrames =
	    static_cast<frame_t>(mp_dynamicFraming->getNumFrames());
	if (numFrames == 0) {
		// no frames defined — mark all events out of range
		throw std::invalid_argument("Number of frames cannot be zero.");
	}

	// get the frame bounds from mp_dynamicFraming
	const timestamp_t firstTimestamp  = mp_dynamicFraming->getStartingTimestamp(0);
	bin_t evId = 0;

	// Skip the events that are before the first frame
	while (evId < numEvents && getTimestamp(evId) < firstTimestamp) {
		mp_dynamicFrames->setFlat(evId, -1);
		++evId;
	}

	// Fill the events in the middle
	for (frame_t currentFrame = 0; currentFrame < numFrames; currentFrame++)
	{
		const timestamp_t endingTimestamp =
		    mp_dynamicFraming->getStoppingTimestamp(currentFrame);
		while (evId < numEvents && getTimestamp(evId) < endingTimestamp)
		{
			mp_dynamicFrames->setFlat(evId, currentFrame);
			evId++;
		}
	}

	// Disable the events after the last frame
	for (; evId < numEvents; evId++)
	{
		mp_dynamicFrames->setFlat(evId, frame_t(-1));
	}
}



bool ListMode::hasMotion() const
{
	return mp_lorMotion != nullptr;
}

frame_t ListMode::getDynamicFrame(bin_t id) const
{
	if (mp_dynamicFraming != nullptr)
	{
		return mp_dynamicFrames->getFlat(id);
	}
	return ProjectionData::getDynamicFrame(id);
}

size_t ListMode::getNumDynamicFrames() const
{
	if (mp_dynamicFraming != nullptr)
	{
		return mp_dynamicFraming->getNumFrames();
	}
	return ProjectionData::getNumDynamicFrames();
}

frame_t ListMode::getMotionFrame(bin_t id) const
{
	if (mp_lorMotion != nullptr)
	{
		return mp_motionFrames->getFlat(id);
	}
	return ProjectionData::getMotionFrame(id);
}

size_t ListMode::getNumMotionFrames() const
{
	if (mp_lorMotion != nullptr)
	{
		return mp_lorMotion->getNumFrames();
	}
	return ProjectionData::getNumMotionFrames();
}

transform_t ListMode::getTransformOfMotionFrame(frame_t frame) const
{
	ASSERT(mp_lorMotion != nullptr);
	if (frame >= 0)
	{
		return mp_lorMotion->getTransform(frame);
	}
	// For the events before the beginning of the frame
	return ProjectionData::getTransformOfMotionFrame(frame);
}

float ListMode::getDurationOfMotionFrame(frame_t frame) const
{
	ASSERT(mp_lorMotion != nullptr);
	if (frame >= 0)
	{
		return mp_lorMotion->getDuration(frame);
	}
	// For the events before the beginning of the frame
	return ProjectionData::getDurationOfMotionFrame(frame);
}

}  // namespace yrt
