/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"
#include <iostream>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_dynaming_framing(py::module& m)
{
	auto c = py::class_<DynamicFraming, std::shared_ptr<DynamicFraming>>(m, "DynamicFraming");

	c.def(py::init([](pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_frameTimestamps) {

		               pybind11::buffer_info buffer = p_frameTimestamps.request();
		               if (buffer.ndim != 1)
		               {
			               throw std::invalid_argument(
			                   "The frames buffer has to be 1-dimensional");
		               }

		               size_t n = buffer.shape[0];
		               std::vector<timestamp_t> v(n);
		               std::memcpy(v.data(), buffer.ptr, n* sizeof(timestamp_t));
		                return std::make_shared<DynamicFraming>(v);
	                }));
//	c.def(py::init<size_t>(), py::arg("numFrames"));

}
}  // namespace yrt

#endif


namespace yrt
{

DynamicFraming::DynamicFraming(const std::vector<timestamp_t>& dynamicFramingVector)
    : m_frameTimestamps(dynamicFramingVector)
{
	if (m_frameTimestamps.size() < 2) {
		throw std::invalid_argument(
		    "Need at least two timestamps to define one frame");
	}
}

size_t DynamicFraming::getNumTimestamps() const
{
	return m_frameTimestamps.size();
}

size_t DynamicFraming::getNumFrames() const
{
	return m_frameTimestamps.size() - 1;
}

void DynamicFraming::setStartingTimestamp(frame_t frame, timestamp_t timestamp)
{
	m_frameTimestamps[frame] = timestamp;
}


timestamp_t DynamicFraming::getStartingTimestamp(frame_t frame) const
{
	return m_frameTimestamps[frame];
}

timestamp_t DynamicFraming::getStoppingTimestamp(frame_t frame) const
{
	return m_frameTimestamps[frame + 1];
}

float DynamicFraming::getDuration(frame_t frame) const
{
	const size_t numFrames = getNumFrames();
	const frame_t lastFrame = numFrames - 1;

	if (frame < lastFrame)
	{
		return m_frameTimestamps[frame + 1] - m_frameTimestamps[frame];
	}
	if (frame == lastFrame)
	{
		// Last frame, take duration of second-to-last frame
		return m_frameTimestamps[lastFrame] - m_frameTimestamps[lastFrame - 1];
	}
	throw std::runtime_error("Frame index out of range");
}

float DynamicFraming::getTotalDuration() const
{
	const frame_t lastFrame = getNumFrames() - 1;
	return m_frameTimestamps[lastFrame + 1] - m_frameTimestamps[0];
}

}