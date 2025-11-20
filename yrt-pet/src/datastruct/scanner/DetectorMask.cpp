/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"

#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_detectormask(pybind11::module& m)
{
	auto c = py::class_<DetectorMask>(m, "DetectorMask", py::buffer_protocol());
	c.def(py::init<size_t>(), "numDets"_a);
	c.def(py::init<const std::string&>(), "fname"_a);
	c.def(py::init<const Array1DBase<bool>&>(), "maskArray"_a);
	c.def(py::init<const Array3DBase<float>&>(), "maskArray"_a);
	c.def(py::init<const DetectorMask&>(), "other"_a);
	c.def("setNumDets", &DetectorMask::setNumDets);
	c.def("readFromFile", &DetectorMask::readFromFile, "fname"_a);
	c.def_buffer(
	    [](DetectorMask& self) -> py::buffer_info
	    {
		    Array1D<bool>& d = self.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(bool),
		                           py::format_descriptor<bool>::format(), 1,
		                           d.getDims(), d.getStrides());
	    });
	c.def("getData",
	      static_cast<const Array1D<bool>& (DetectorMask::*)() const>(
	          &DetectorMask::getData));
	c.def("enableAllDetectors", &DetectorMask::enableAllDetectors);
	c.def("disableAllDetectors", &DetectorMask::disableAllDetectors);
	c.def("enableDetector", &DetectorMask::enableDetector, "detId"_a);
	c.def("disableDetector", &DetectorMask::disableDetector, "detId"_a);
	c.def("checkAgainstScanner", &DetectorMask::checkAgainstScanner);
	c.def("getNumDets", &DetectorMask::getNumDets);
	c.def("checkDetector", &DetectorMask::checkDetector, "detId"_a);
	c.def("isDetectorEnabled", &DetectorMask::isDetectorEnabled, "detId"_a);
	c.def("areAllDetectorsEnabled", &DetectorMask::areAllDetectorsEnabled);
	c.def("areAllDetectorsDisabled", &DetectorMask::areAllDetectorsDisabled);
	c.def("writeToFile", &DetectorMask::writeToFile, "fname"_a);
	c.def("logicalAndWithOther", &DetectorMask::logicalAndWithOther, "other"_a);
	c.def("logicalOrWithOther", &DetectorMask::logicalOrWithOther, "other"_a);
	c.def("logicalXorWithOther", &DetectorMask::logicalXorWithOther, "other"_a);
	c.def("logicalNandWithOther", &DetectorMask::logicalNandWithOther,
	      "other"_a);
	c.def("countEnabledDetectors", &DetectorMask::countEnabledDetectors);
	c.def("countDisabledDetectors", &DetectorMask::countDisabledDetectors);
}

}  // namespace yrt
#endif

namespace yrt
{

DetectorMask::DetectorMask(size_t numDets)
{
	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->allocate(numDets);

	// By default, all detectors are enabled
	enableAllDetectors();
}

DetectorMask::DetectorMask(const std::string& pr_fname)
{
	readFromFile(pr_fname);
}

DetectorMask::DetectorMask(const Array1DBase<bool>& pr_data)
{
	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->allocate(pr_data.getSizeTotal());
	mp_data->copy(pr_data);
}

DetectorMask::DetectorMask(const Array3DBase<float>& pr_data)
{
	mp_data = std::make_unique<Array1D<bool>>();

	const size_t size = pr_data.getSizeTotal();
	mp_data->allocate(size);

	for (size_t i = 0; i < size; ++i)
	{
		mp_data->setFlat(i, pr_data.getFlat(i) > 0.0f);
	}
}

DetectorMask::DetectorMask(const DetectorMask& other)
{
	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->allocate(other.getNumDets());
	mp_data->copy(other.getData());
}

void DetectorMask::setNumDets(size_t numDets)
{
	const auto oldData = std::move(mp_data);

	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->allocate(numDets);

	// By default, all detectors are enabled
	enableAllDetectors();

	// Keep the previous mask if there was one
	if (oldData != nullptr)
	{
		const size_t minNumDets =
		    std::min(oldData->getSizeTotal(), mp_data->getSizeTotal());
		for (size_t detId = 0; detId < minNumDets; detId++)
		{
			mp_data->setFlat(detId, oldData->getFlat(detId));
		}
	}
}

void DetectorMask::readFromFile(const std::string& fname)
{
	mp_data = std::make_unique<Array1D<bool>>();

	// Open the file
	std::ifstream file;
	file.open(fname.c_str(), std::ios::binary | std::ios::in);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "The file given \"" + fname + "\" could not be opened",
		    std::make_error_code(std::errc::no_such_file_or_directory));
	}

	// Get the file size
	file.seekg(0, std::ios::end);
	const size_t fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	// Allocate and read the file
	mp_data->allocate(fileSize);
	file.read(reinterpret_cast<char*>(mp_data->getRawPointer()),
	          fileSize * sizeof(bool));
}

Array1D<bool>& DetectorMask::getData()
{
	ASSERT(mp_data != nullptr);
	return *mp_data;
}

const Array1D<bool>& DetectorMask::getData() const
{
	ASSERT(mp_data != nullptr);
	return *mp_data;
}

void DetectorMask::enableAllDetectors()
{
	mp_data->fill(true);
}

void DetectorMask::disableAllDetectors()
{
	mp_data->fill(false);
}

void DetectorMask::enableDetector(det_id_t detId)
{
	setDetectorEnabled(detId, true);
}

void DetectorMask::disableDetector(det_id_t detId)
{
	setDetectorEnabled(detId, false);
}

bool DetectorMask::checkAgainstScanner(const Scanner& scanner) const
{
	return scanner.getNumDets() == mp_data->getSizeTotal();
}

size_t DetectorMask::getNumDets() const
{
	return mp_data->getSizeTotal();
}

bool DetectorMask::checkDetector(det_id_t detId) const
{
	return isDetectorEnabled(detId);
}

bool DetectorMask::isDetectorEnabled(det_id_t detId) const
{
	// Assumes that mp_maskArray != nullptr
	return mp_data->getFlat(detId);
}

bool DetectorMask::areAllDetectorsEnabled() const
{
	for (det_id_t i = 0; i < getNumDets(); i++)
	{
		if (!isDetectorEnabled(i))
		{
			return false;
		}
	}
	return true;
}

bool DetectorMask::areAllDetectorsDisabled() const
{
	for (det_id_t i = 0; i < getNumDets(); i++)
	{
		if (isDetectorEnabled(i))
		{
			return false;
		}
	}
	return true;
}

void DetectorMask::writeToFile(const std::string& fname) const
{
	std::ofstream file;
	file.open(fname.c_str(), std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "The file given \"" + fname + "\" could not be opened",
		    std::make_error_code(std::errc::io_error));
	}
	file.write(reinterpret_cast<char*>(mp_data->getRawPointer()),
	           getNumDets() * sizeof(bool));
}

void DetectorMask::logicalAndWithOther(const DetectorMask& other)
{
	logicalOperWithOther<BinaryOperations::AND>(other);
}

void DetectorMask::logicalOrWithOther(const DetectorMask& other)
{
	logicalOperWithOther<BinaryOperations::OR>(other);
}

void DetectorMask::logicalXorWithOther(const DetectorMask& other)
{
	logicalOperWithOther<BinaryOperations::XOR>(other);
}

void DetectorMask::logicalNandWithOther(const DetectorMask& other)
{
	logicalOperWithOther<BinaryOperations::NAND>(other);
}

size_t DetectorMask::countEnabledDetectors() const
{
	size_t numEnabledDetectors = 0;
	const size_t numDets = getNumDets();
	for (det_id_t detId = 0; detId < numDets; ++detId)
	{
		if (isDetectorEnabled(detId))
		{
			numEnabledDetectors++;
		}
	}
	return numEnabledDetectors;
}

size_t DetectorMask::countDisabledDetectors() const
{
	return getNumDets() - countEnabledDetectors();
}

void DetectorMask::setDetectorEnabled(det_id_t detId, bool enabled)
{
	mp_data->setFlat(detId, enabled);
}

template <DetectorMask::BinaryOperations Oper>
void DetectorMask::logicalOperWithOther(const DetectorMask& other)
{
	size_t numDets = getNumDets();
	ASSERT_MSG(getNumDets() == other.getNumDets(),
	           "Size mismatch between the two detector masks");
	util::parallelForChunked(
	    numDets, globals::getNumThreads(),
	    [&](det_id_t detId, unsigned int /*threadId*/)
	    {
		    if constexpr (Oper == BinaryOperations::AND)
		    {
			    setDetectorEnabled(detId, isDetectorEnabled(detId) &&
			                                  other.isDetectorEnabled(detId));
		    }
		    else if constexpr (Oper == BinaryOperations::OR)
		    {
			    setDetectorEnabled(detId, isDetectorEnabled(detId) ||
			                                  other.isDetectorEnabled(detId));
		    }
		    else if constexpr (Oper == BinaryOperations::XOR)
		    {
			    setDetectorEnabled(detId, isDetectorEnabled(detId) !=
			                                  other.isDetectorEnabled(detId));
		    }
		    else if constexpr (Oper == BinaryOperations::NAND)
		    {
			    setDetectorEnabled(detId, !(isDetectorEnabled(detId) &&
			                                other.isDetectorEnabled(detId)));
		    }
	    });
}

}  // namespace yrt