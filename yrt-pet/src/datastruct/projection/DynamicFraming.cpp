/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"

#include "yrt-pet/utils/Assert.hpp"

#include <fstream>
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
	auto c = py::class_<DynamicFraming, std::shared_ptr<DynamicFraming>>(
	    m, "DynamicFraming");

	c.def(py::init(
	          [](pybind11::array_t<timestamp_t, pybind11::array::c_style>&
	                 p_frameTimestamps)
	          {
		          pybind11::buffer_info buffer = p_frameTimestamps.request();
		          if (buffer.ndim != 1)
		          {
			          throw std::invalid_argument(
			              "The frames buffer has to be 1-dimensional");
		          }

		          size_t n = buffer.shape[0];
		          std::vector<timestamp_t> v(n);
		          std::memcpy(v.data(), buffer.ptr, n * sizeof(timestamp_t));
		          return std::make_shared<DynamicFraming>(v);
	          }),
	      "frame_timestamps"_a);
	c.def(py::init<size_t>(), "num_frames"_a);
	c.def(py::init<const std::string&>(), "fname"_a);
	c.def("__repr__",
	      [](const DynamicFraming& self)
	      {
		      std::stringstream ss;
		      for (frame_t f = 0; f < static_cast<frame_t>(self.getNumFrames());
		           f++)
		      {
			      ss << self.getStartingTimestamp(f) << "\n";
		      }
		      ss << self.getLastTimestamp() << std::endl;
		      return ss.str();
	      });

	c.def("writeToFile", &DynamicFraming::writeToFile, "fname"_a);
	c.def("readFromFile", &DynamicFraming::readFromFile, "fname"_a);

	c.def("getNumFrames", &DynamicFraming::getNumFrames,
	      "Return the number of frames");
	c.def("getDuration", &DynamicFraming::getDuration, "frame"_a,
	      "Return the timestamp of the frame \"frame\" minus the subsequent "
	      "timestamp");
	c.def("getTotalDuration", &DynamicFraming::getTotalDuration,
	      "Return the last timestamp minus the first timestamp");
	c.def(
	    "getNumTimestamps", &DynamicFraming::getNumTimestamps,
	    "Different from \"getNumFrames\" as this includes the last timestamp");
	c.def("setStartingTimestamp", &DynamicFraming::setStartingTimestamp,
	      "frame"_a, "timestamp"_a,
	      "Set the timestamp at which the frame \"frame\" starts");
	c.def("setLastTimestamp", &DynamicFraming::setLastTimestamp, "timestamp"_a,
	      "Set the timestamp at which the entire dynamic framing ends");
	c.def("isValid", &DynamicFraming::isValid,
	      "Validate the chronological order of the timestamps");
	c.def("getStartingTimestamp", &DynamicFraming::getStartingTimestamp,
	      "frame"_a, "Get the timestamp at which a frame starts");
	c.def("getStoppingTimestamp", &DynamicFraming::getStoppingTimestamp,
	      "frame"_a, "Get the timestamp at which a frame ends");
}
}  // namespace yrt

#endif


namespace yrt
{

DynamicFraming::DynamicFraming(size_t numFrames)
{
	ASSERT_MSG(numFrames >= 1, "The number of frames must be greater than 0");
	m_frameTimestamps = std::vector<timestamp_t>(numFrames + 1, 0);
}

DynamicFraming::DynamicFraming(
    const std::vector<timestamp_t>& dynamicFramingVector)
    : m_frameTimestamps(dynamicFramingVector)
{
	ASSERT_MSG(m_frameTimestamps.size() > 1,
	           "Need at least two timestamps to define one frame");
}

DynamicFraming::DynamicFraming(const std::string& fname)
{
	readFromFile(fname);
}

void DynamicFraming::readFromFile(const std::string& fname)
{
	std::ifstream infile(fname);
	ASSERT_MSG(infile.is_open(),
	           ("Error opening file \"" + fname + "\"").c_str());
	std::vector<timestamp_t> dynamicFramingVector;
	timestamp_t timestamp;
	while (infile >> timestamp)
	{
		dynamicFramingVector.push_back(timestamp);
	}
	m_frameTimestamps = dynamicFramingVector;
}

void DynamicFraming::writeToFile(const std::string& fname) const
{
	std::ofstream outfile(fname);
	ASSERT_MSG(outfile.is_open(),
	           ("Error opening file \"" + fname + "\"").c_str());
	for (auto& timestamp : m_frameTimestamps)
	{
		outfile << timestamp << "\n";
	}
}

size_t DynamicFraming::getNumFrames() const
{
	return m_frameTimestamps.size() - 1;
}

timestamp_t DynamicFraming::getDuration(frame_t frame) const
{
	const size_t numFrames = getNumFrames();

	if (frame >= 0 && static_cast<size_t>(frame) < numFrames)
	{
		ASSERT(m_frameTimestamps[frame + 1] > m_frameTimestamps[frame]);
		return m_frameTimestamps[frame + 1] - m_frameTimestamps[frame];
	}
	throw std::runtime_error("Frame index out of range");
}

timestamp_t DynamicFraming::getTotalDuration() const
{
	ASSERT(m_frameTimestamps[getNumFrames()] > m_frameTimestamps[0]);
	return m_frameTimestamps[getNumFrames()] - m_frameTimestamps[0];
}

size_t DynamicFraming::getNumTimestamps() const
{
	return m_frameTimestamps.size();
}

void DynamicFraming::setStartingTimestamp(frame_t frame, timestamp_t timestamp)
{
	m_frameTimestamps[frame] = timestamp;
}

void DynamicFraming::setLastTimestamp(timestamp_t timestamp)
{
	m_frameTimestamps[getNumFrames()] = timestamp;
}

bool DynamicFraming::isValid() const
{
	if (m_frameTimestamps.size() <= 1)
	{
		return false;
	}

	timestamp_t t = m_frameTimestamps[0];

	for (size_t frame_i = 1; frame_i < m_frameTimestamps.size(); frame_i++)
	{
		if (m_frameTimestamps[frame_i] <= t)
		{
			return false;
		}

		t = m_frameTimestamps[frame_i];
	}

	return true;
}

timestamp_t DynamicFraming::getStartingTimestamp(frame_t frame) const
{
	return m_frameTimestamps[frame];
}

timestamp_t DynamicFraming::getStoppingTimestamp(frame_t frame) const
{
	return m_frameTimestamps[frame + 1];
}

timestamp_t DynamicFraming::getLastTimestamp() const
{
	return m_frameTimestamps[getNumFrames()];
}

}  // namespace yrt
