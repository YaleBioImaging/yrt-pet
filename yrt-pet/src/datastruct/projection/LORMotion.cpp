/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/utils/Assert.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_lormotion(py::module& m)
{
	auto c = py::class_<LORMotion, std::shared_ptr<LORMotion>>(m, "LORMotion");

	c.def(py::init<const std::string&>(), py::arg("filename"));
	c.def(py::init<size_t>(), py::arg("numFrames"));

	c.def("getTransform", &LORMotion::getTransform, "frame"_a);
	c.def("getStartingTimestamp", &LORMotion::getStartingTimestampSafe,
	      "frame"_a);
	c.def("setTransform", &LORMotion::setTransformSafe, "frame"_a,
	      "transform"_a);
	c.def("setStartingTimestamp", &LORMotion::setStartingTimestampSafe,
	      "frame"_a, "timestamp"_a);
	c.def("getDuration", &LORMotion::getDuration, "frame"_a);
	c.def("getError", &LORMotion::getErrorSafe, "frame"_a);
	c.def("setError", &LORMotion::setErrorSafe, "frame"_a, "error"_a);
	c.def("getNumFrames", &LORMotion::getNumFrames);
	c.def("readFromFile", &LORMotion::readFromFile, "filename"_a);
	c.def("writeToFile", &LORMotion::writeToFile, "filename"_a);
	c.def("getTotalDuration", &LORMotion::getTotalDuration);
}
}  // namespace yrt

#endif

namespace yrt
{
LORMotion::LORMotion(const std::string& filename)
{
	readFromFile(filename);
}

LORMotion::LORMotion(size_t numFrames)
{
	resize(numFrames);
}

transform_t LORMotion::getTransform(frame_t frame) const
{
	if (frame < 0)
	{
		// If before frame start, return identity
		return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
	}
	ASSERT_MSG(frame < static_cast<frame_t>(getNumFrames()),
	           "Frame index out of range");
	return m_records[frame].transform;
}

float LORMotion::getDuration(frame_t frame) const
{
	const size_t numFrames = getNumFrames();
	const frame_t lastFrame = numFrames - 1;

	if (frame < lastFrame)
	{
		return m_records[frame + 1].timestamp - m_records[frame].timestamp;
	}
	if (frame == lastFrame)
	{
		// Last frame, take duration of second-to-last frame
		return m_records[lastFrame].timestamp -
		       m_records[lastFrame - 1].timestamp;
	}
	throw std::runtime_error("Frame index out of range");
}

timestamp_t LORMotion::getStartingTimestamp(frame_t frame) const
{
	return m_records[frame].timestamp;
}

void LORMotion::setTransform(frame_t frame, const transform_t& transform)
{
	m_records[frame].transform = transform;
}

void LORMotion::setStartingTimestamp(frame_t frame, timestamp_t timestamp)
{
	m_records[frame].timestamp = timestamp;
}

size_t LORMotion::getNumFrames() const
{
	return m_records.size();
}

void LORMotion::readFromFile(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file)
	{
		throw std::runtime_error("Failed to open file: " + filename);
	}

	std::string line;
	int lineNumber = 0;
	m_records.clear();
	m_records.reserve(1 << 16);

	while (std::getline(file, line))
	{
		++lineNumber;
		if (line.empty())
			continue;

		const char* ptr = line.c_str();
		const char* end = ptr + line.size();

		Record rec;
		auto fptr = reinterpret_cast<float*>(&rec.transform);

		// Parse timestamp
		auto [p, ec] = std::from_chars(ptr, end, rec.timestamp);
		if (ec != std::errc())
		{
			std::cerr << "Line " << lineNumber << ": invalid timestamp\n";
			continue;
		}
		ptr = p;

		// Parse 12 floats
		bool failed = false;
		for (int i = 0; i < 12; ++i)
		{
			while (ptr < end && (*ptr == ' ' || *ptr == ','))
				++ptr;
			if (ptr >= end)
			{
				std::cerr << "Line " << lineNumber << ": too few values\n";
				failed = true;
				break;
			}

			float val;
			auto [np, ecf] = std::from_chars(ptr, end, val);
			if (ecf != std::errc())
			{
				std::cerr << "Line " << lineNumber << ": invalid float\n";
				failed = true;
				break;
			}

			fptr[i] = val;
			ptr = np;
		}
		if (failed)
		{
			continue;
		}

		// Parse error value
		while (ptr < end && (*ptr == ' ' || *ptr == ','))
			++ptr;
		if (ptr >= end)
		{
			std::cerr << "Line " << lineNumber << ": missing error value\n";
			continue;
		}
		auto [p3, ec3] = std::from_chars(ptr, end, rec.error);
		if (ec3 != std::errc())
		{
			std::cerr << "Line " << lineNumber << ": invalid error value\n";
			continue;
		}
		ptr = p3;

		// Skip trailing whitespace
		while (ptr < end && std::isspace(*ptr))
			++ptr;

		// Check for unexpected trailing content
		if (ptr != end)
		{
			std::cerr << "Line " << lineNumber << ": too many values\n";
			continue;
		}

		// Line is valid, store record
		m_records.push_back(rec);
	}
}

void LORMotion::writeToFile(const std::string& filename) const
{
	std::ofstream out(filename);
	if (!out)
	{
		throw std::runtime_error("Failed to open output file: " + filename);
	}

	// Optional: set fixed float format with 9 decimal places
	out << std::fixed << std::setprecision(9);

	for (const auto& rec : m_records)
	{
		out << rec.timestamp << ',' << rec.transform.r00 << ','
		    << rec.transform.r01 << ',' << rec.transform.r02 << ','
		    << rec.transform.tx << ',' << rec.transform.r10 << ','
		    << rec.transform.r11 << ',' << rec.transform.r12 << ','
		    << rec.transform.ty << ',' << rec.transform.r20 << ','
		    << rec.transform.r21 << ',' << rec.transform.r22 << ','
		    << rec.transform.tz << ',' << rec.error << '\n';
	}
	out << std::flush;
}

float LORMotion::getTotalDuration() const
{
	const frame_t lastFrame = getNumFrames() - 1;
	return m_records[lastFrame].timestamp - m_records[0].timestamp +
	       getDuration(lastFrame);
}

timestamp_t LORMotion::getStartingTimestampSafe(frame_t frame) const
{
	ASSERT_MSG(frame < static_cast<frame_t>(getNumFrames()),
	           "Frame index out of range");
	ASSERT_MSG(frame >= 0, "Frame index must be positive");
	return getStartingTimestamp(frame);
}

void LORMotion::setTransformSafe(frame_t frame, const transform_t& transform)
{
	ASSERT_MSG(frame < static_cast<frame_t>(getNumFrames()),
	           "Frame index out of range");
	ASSERT_MSG(frame >= 0, "Frame index must be positive");
	setTransform(frame, transform);
}

void LORMotion::setStartingTimestampSafe(frame_t frame, timestamp_t timestamp)
{
	ASSERT_MSG(frame < static_cast<frame_t>(getNumFrames()),
	           "Frame index out of range");
	ASSERT_MSG(frame >= 0, "Frame index must be positive");
	setStartingTimestamp(frame, timestamp);
}

void LORMotion::setErrorSafe(frame_t frame, float error)
{
	ASSERT_MSG(frame < static_cast<frame_t>(getNumFrames()),
	           "Frame index out of range");
	ASSERT_MSG(frame >= 0, "Frame index must be positive");
	setError(frame, error);
}

float LORMotion::getErrorSafe(frame_t frame) const
{
	ASSERT_MSG(frame < static_cast<frame_t>(getNumFrames()),
	           "Frame index out of range");
	ASSERT_MSG(frame >= 0, "Frame index must be positive");
	return getError(frame);
}

void LORMotion::setError(frame_t frame, float error)
{
	m_records[frame].error = error;
}

float LORMotion::getError(frame_t frame) const
{
	return m_records[frame].error;
}

void LORMotion::resize(size_t newSize)
{
	m_records.resize(newSize);
}
}  // namespace yrt
