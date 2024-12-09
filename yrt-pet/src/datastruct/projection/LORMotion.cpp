/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/LORMotion.hpp"
#include "utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

void py_setup_lormotion(py::module& m)
{
	auto c = py::class_<LORMotion>(m, "LORMotion");

	c.def(py::init<const std::string&>(), py::arg("filename"));
	c.def(py::init<size_t>(), py::arg("numFrames"));

	c.def("writeToFile", &LORMotion::writeToFile);
	c.def("getTransform", &LORMotion::getTransform, "frame"_a);
	c.def("getDuration", &LORMotion::getDuration, "frame"_a);
	c.def("getStartingTimestamp", &LORMotion::getStartingTimestamp, "frame"_a);
	c.def("setTransform", &LORMotion::setTransform, "frame"_a, "transform"_a);
	c.def("setStartingTimestamp", &LORMotion::setStartingTimestamp, "frame"_a,
	      "timestamp"_a);
	c.def("getNumFrames", &LORMotion::getNumFrames);
}

#endif

LORMotion::LORMotion(const std::string& filename)
    : m_structure{filename, 1 << 20}
{
	setupPointers();
}

LORMotion::LORMotion(size_t numFrames) : m_structure{numFrames}
{
	setupPointers();
}

transform_t LORMotion::getTransform(frame_t frame) const
{
	return mp_transforms[frame];
}

float LORMotion::getDuration(frame_t frame) const
{
	const size_t numFrames = getNumFrames();

	if (frame < static_cast<int32_t>(numFrames - 1))
	{
		return mp_startingTimestamps[frame + 1] - mp_startingTimestamps[frame];
	}

	// Last frame, take duration of second-to-last frame
	return mp_startingTimestamps[numFrames - 1] -
	       mp_startingTimestamps[numFrames - 2];
}

timestamp_t LORMotion::getStartingTimestamp(frame_t frame) const
{
	return mp_startingTimestamps[frame];
}

void LORMotion::setTransform(frame_t frame, const transform_t& transform)
{
	mp_transforms[frame] = transform;
}

void LORMotion::setStartingTimestamp(frame_t frame, timestamp_t timestamp)
{
	mp_startingTimestamps[frame] = timestamp;
}

size_t LORMotion::getNumFrames() const
{
	return m_structure.get_num_columns();
}

void LORMotion::writeToFile(const std::string& filename) const
{
	m_structure.save_transpose(filename);
}

void LORMotion::setupPointers()
{
	ASSERT_MSG(getNumFrames() > 0, "There has to be at least one frame");
	mp_startingTimestamps = m_structure.get_pointer<0>();
	mp_transforms = m_structure.get_pointer<1>();
}
