/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"

#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_listmodelut(py::module& m)
{
	auto c = py::class_<ListModeLUT, ListMode>(m, "ListModeLUT");

	c.def("getTOFValue", &ListModeLUT::getTOFValue_safe);
	c.def("getRandomsEstimate", &ListModeLUT::getRandomsEstimate_safe);

	c.def("setDetectorId1OfEvent", &ListModeLUT::setDetectorId1OfEvent,
	      "event_id"_a, "d1"_a);
	c.def("setDetectorId2OfEvent", &ListModeLUT::setDetectorId2OfEvent,
	      "event_id"_a, "d2"_a);
	c.def("setDetectorIdsOfEvent", &ListModeLUT::setDetectorIdsOfEvent,
	      "event_id"_a, "d1"_a, "d2"_a);
	c.def("setTimestampOfEvent", &ListModeLUT::setTimestampOfEvent,
	      "event_id"_a, "timestamp"_a);
	c.def("setTOFValueOfEvent", &ListModeLUT::setTOFValueOfEvent, "event_id"_a,
	      "tof_value"_a);

	c.def("getTimestampArray",
	      [](const ListModeLUT& self) -> py::array_t<timestamp_t>
	      {
		      Array1DBase<timestamp_t>* arr = self.getTimestampArrayPtr();
		      auto buf_info = py::buffer_info(
		          arr->getRawPointer(), sizeof(timestamp_t),
		          py::format_descriptor<timestamp_t>::format(), 1,
		          {arr->getSizeTotal()}, {sizeof(timestamp_t)});
		      return py::array_t<timestamp_t>(buf_info);
	      });
	c.def("getDetector1Array",
	      [](const ListModeLUT& self) -> py::array_t<det_id_t>
	      {
		      Array1DBase<det_id_t>* arr = self.getDetector1ArrayPtr();
		      auto buf_info =
		          py::buffer_info(arr->getRawPointer(), sizeof(det_id_t),
		                          py::format_descriptor<det_id_t>::format(), 1,
		                          {arr->getSizeTotal()}, {sizeof(det_id_t)});
		      return py::array_t<det_id_t>(buf_info);
	      });
	c.def("getDetector2Array",
	      [](const ListModeLUT& self) -> py::array_t<det_id_t>
	      {
		      Array1DBase<det_id_t>* arr = self.getDetector2ArrayPtr();
		      auto buf_info =
		          py::buffer_info(arr->getRawPointer(), sizeof(det_id_t),
		                          py::format_descriptor<det_id_t>::format(), 1,
		                          {arr->getSizeTotal()}, {sizeof(det_id_t)});
		      return py::array_t<det_id_t>(buf_info);
	      });
	c.def("getTOFArray",
	      [](const ListModeLUT& self) -> py::array_t<float>
	      {
		      ASSERT_MSG(self.hasTOF(),
		                 "ListModeLUT object does not hold TOF information");
		      Array1DBase<float>* arr = self.getTOFArrayPtr();
		      auto buf_info =
		          py::buffer_info(arr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
	c.def("getRandomsEstimatesArray",
	      [](const ListModeLUT& self) -> py::array_t<float>
	      {
		      ASSERT_MSG(self.hasRandomsEstimates(),
		                 "ListModeLUT object does not hold randoms estimates");
		      Array1DBase<float>* arr = self.getRandomsEstimatesArrayPtr();
		      auto buf_info =
		          py::buffer_info(arr->getRawPointer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {arr->getSizeTotal()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });

	c.def("isMemoryValid", &ListModeLUT::isMemoryValid);

	c.def("writeToFile", &ListModeLUT::writeToFile, "filename"_a);
	c.def("getNativeLORFromId", &ListModeLUT::getNativeLORFromId, "event_id"_a);

	auto c_alias =
	    py::class_<ListModeLUTAlias, ListModeLUT>(m, "ListModeLUTAlias");
	c_alias.def(py::init<const Scanner&, bool, bool>(), "scanner"_a,
	            "flag_tof"_a = false, "flag_randoms"_a = false);

	c_alias.def("bind",
	            static_cast<void (ListModeLUTAlias::*)(
	                pybind11::array_t<timestamp_t, pybind11::array::c_style>*,
	                pybind11::array_t<det_id_t, pybind11::array::c_style>*,
	                pybind11::array_t<det_id_t, pybind11::array::c_style>*,
	                pybind11::array_t<float, pybind11::array::c_style>*,
	                pybind11::array_t<float, pybind11::array::c_style>*)>(
	                &ListModeLUTAlias::bind),
	            "timestamps"_a, "detector_ids1"_a, "detector_ids2"_a,
	            "tof_ps"_a, "randoms"_a);

	auto c_owned =
	    py::class_<ListModeLUTOwned, ListModeLUT>(m, "ListModeLUTOwned");
	c_owned.def(py::init<const Scanner&, bool, bool>(), "scanner"_a,
	            "flag_tof"_a = false, "flag_randoms"_a = false);
	c_owned.def(py::init<const Scanner&, const std::string&, bool, bool>(),
	            "scanner"_a, "listMode_fname"_a, "flag_tof"_a = false,
	            "flag_randoms"_a = false);
	c_owned.def("readFromFile", &ListModeLUTOwned::readFromFile, "filename"_a);
	c_owned.def("allocate", &ListModeLUTOwned::allocate, "num_events"_a);
	c_owned.def(
	    "createFromHistogram3D",
	    [](ListModeLUTOwned* self, const Histogram3D* histo, size_t num_events)
	    { util::histogram3DToListModeLUT(histo, self, num_events); }, "histo"_a,
	    "num_events"_a);
}
}  // namespace yrt
#endif  // if BUILD_PYBIND11

namespace yrt
{
ListModeLUT::ListModeLUT(const Scanner& pr_scanner) : ListMode(pr_scanner) {}

ListModeLUTOwned::ListModeLUTOwned(const Scanner& pr_scanner, bool p_flagTOF,
                                   bool p_flagRandoms)
    : ListModeLUT(pr_scanner)
{
	mp_timestamps = std::make_unique<Array1D<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1D<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1D<det_id_t>>();
	if (p_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1D<float>>();
	}
	if (p_flagRandoms)
	{
		mp_randoms = std::make_unique<Array1D<float>>();
	}
}

ListModeLUTOwned::ListModeLUTOwned(const Scanner& pr_scanner,
                                   const std::string& listMode_fname,
                                   bool p_flagTOF, bool p_flagRandoms)
    : ListModeLUTOwned(pr_scanner, p_flagTOF, p_flagRandoms)
{
	ListModeLUTOwned::readFromFile(listMode_fname);
}

ListModeLUTAlias::ListModeLUTAlias(const Scanner& pr_scanner, bool p_flagTOF,
                                   bool p_flagRandoms)
    : ListModeLUT(pr_scanner)
{
	mp_timestamps = std::make_unique<Array1DAlias<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1DAlias<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1DAlias<det_id_t>>();
	if (p_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1DAlias<float>>();
	}
	if (p_flagRandoms)
	{
		mp_randoms = std::make_unique<Array1DAlias<float>>();
	}
}

void ListModeLUTOwned::readFromFile(const std::string& listMode_fname)
{
	std::ifstream fin(listMode_fname, std::ios::in | std::ios::binary);

	if (!fin.good())
	{
		throw std::runtime_error("Error reading input file " + listMode_fname);
	}

	const det_id_t numDets = mr_scanner.getNumDets();
	const bool hasTOF = mp_tof_ps != nullptr;
	const bool hasRandoms = mp_randoms != nullptr;

	// first check that file has the right size:
	fin.seekg(0, std::ios::end);
	size_t end = fin.tellg();
	fin.seekg(0, std::ios::beg);
	size_t begin = fin.tellg();
	size_t fileSize = end - begin;
	int numFields = 3;
	if (hasTOF)
	{
		numFields++;
	}
	if (hasRandoms)
	{
		numFields++;
	}

	size_t sizeOfAnEvent = numFields * sizeof(float);
	if (fileSize <= 0 || (fileSize % sizeOfAnEvent) != 0)
	{
		throw std::runtime_error("Error: Input file has incorrect size in "
		                         "ListModeLUTOwned::readFromFile.");
	}

	// Allocate the memory
	size_t numEvents = fileSize / sizeOfAnEvent;
	allocate(numEvents);

	// Compute buffer size (No need to allocate more than needed)
	size_t bufferSize = 1ull << 30;
	bufferSize = std::min(fileSize, bufferSize);

	// Read content of file
	auto buff = std::make_unique<uint32_t[]>(bufferSize);
	size_t posStart = 0;
	while (posStart < numEvents)
	{
		size_t readSize =
		    std::min(bufferSize, numFields * (numEvents - posStart));
		fin.read(reinterpret_cast<char*>(buff.get()),
		         (readSize / numFields) * numFields * sizeof(float));

#pragma omp parallel for default(none),                                     \
    shared(mp_timestamps, mp_detectorId1, mp_detectorId2, buff, mp_tof_ps), \
    firstprivate(readSize, numFields, posStart, numDets, hasTOF, hasRandoms)
		for (size_t i = 0; i < readSize / numFields; i++)
		{
			const size_t eventPos = posStart + i;
			size_t bufferPos = numFields * i;

			const timestamp_t timestamp = buff[bufferPos++];
			const det_id_t d1 = buff[bufferPos++];
			const det_id_t d2 = buff[bufferPos++];

			if (CHECK_LIKELY(d1 < numDets && d2 < numDets))
			{
				(*mp_timestamps)[eventPos] = timestamp;
				(*mp_detectorId1)[eventPos] = d1;
				(*mp_detectorId2)[eventPos] = d2;
				if (hasTOF)
				{
					std::memcpy(&mp_tof_ps->getRawPointer()[eventPos],
					            &buff[bufferPos++], sizeof(float));
				}
				if (hasRandoms)
				{
					std::memcpy(&mp_randoms->getRawPointer()[eventPos],
					            &buff[bufferPos++], sizeof(float));
				}
			}
			else
			{
				throw std::invalid_argument(
				    "Detectors invalid in list-mode event " +
				    std::to_string(eventPos));
			}
		}
		posStart += readSize / numFields;
	}
}

void ListModeLUT::writeToFile(const std::string& listMode_fname) const
{
	const bool hasTOF = mp_tof_ps != nullptr;
	const bool hasRandoms = mp_randoms != nullptr;
	int numFields = 3;
	if (hasTOF)
	{
		numFields++;
	}
	if (hasRandoms)
	{
		numFields++;
	}

	const size_t numEvents = count();
	std::ofstream file;
	file.open(listMode_fname.c_str(), std::ios::binary | std::ios::out);

	size_t bufferSize = 1ull << 30;
	const size_t maxBufferSize = numFields * numEvents;
	bufferSize = std::min(bufferSize, maxBufferSize);
	auto buff = std::make_unique<uint32_t[]>(bufferSize);
	// This is done assuming that "int" and "float" are of the same size
	// (4bytes)
	size_t posStart = 0;
	while (posStart < numEvents)
	{
		const size_t writeSize =
		    std::min(bufferSize, numFields * (numEvents - posStart));
		for (size_t i = 0; i < writeSize / numFields; i++)
		{
			size_t bufferPos = numFields * i;

			buff[bufferPos++] = mp_timestamps->getFlat(posStart + i);
			buff[bufferPos++] = mp_detectorId1->getFlat(posStart + i);
			buff[bufferPos++] = mp_detectorId2->getFlat(posStart + i);
			if (hasTOF)
			{
				std::memcpy(&buff[bufferPos++],
				            &mp_tof_ps->getRawPointer()[posStart + i],
				            sizeof(float));
			}
			if (hasRandoms)
			{
				std::memcpy(&buff[bufferPos++],
				            &mp_randoms->getRawPointer()[posStart + i],
				            sizeof(float));
			}
		}
		file.write(reinterpret_cast<char*>(buff.get()),
		           (writeSize / numFields) * numFields * sizeof(uint32_t));
		posStart += writeSize / numFields;
	}
	file.close();
}

void ListModeLUT::addLORMotion(const std::shared_ptr<LORMotion>& pp_lorMotion)
{
	ASSERT_MSG(isMemoryValid(), "List-mode data not allocated yet");
	ListMode::addLORMotion(pp_lorMotion);
}

bool ListModeLUT::isMemoryValid() const
{
	return mp_timestamps->getRawPointer() != nullptr;
}

timestamp_t ListModeLUT::getTimestamp(bin_t eventId) const
{
	return (*mp_timestamps)[eventId];
}

size_t ListModeLUT::count() const
{
	return mp_timestamps->getSize(0);
}

bool ListModeLUT::isUniform() const
{
	return true;
}

void ListModeLUT::setTimestampOfEvent(bin_t eventId, timestamp_t ts)
{
	(*mp_timestamps)[eventId] = ts;
}

void ListModeLUT::setDetectorId1OfEvent(bin_t eventId, det_id_t d1)
{
	(*mp_detectorId1)[eventId] = d1;
}

void ListModeLUT::setDetectorId2OfEvent(bin_t eventId, det_id_t d2)
{
	(*mp_detectorId2)[eventId] = d2;
}

void ListModeLUT::setDetectorIdsOfEvent(bin_t eventId, det_id_t d1, det_id_t d2)
{
	(*mp_detectorId1)[eventId] = d1;
	(*mp_detectorId2)[eventId] = d2;
}

void ListModeLUT::setTOFValueOfEvent(bin_t eventId, float tofValue)
{
	ASSERT_MSG(hasTOF(), "TOF not enabled in the list-mode");
	(*mp_tof_ps)[eventId] = tofValue;
}

void ListModeLUT::setRandomsEstimateOfEvent(bin_t eventId, float randoms)
{
	ASSERT_MSG(hasRandomsEstimates(), "Randoms not enabled in the list-mode");
	(*mp_randoms)[eventId] = randoms;
}

Array1DBase<timestamp_t>* ListModeLUT::getTimestampArrayPtr() const
{
	return mp_timestamps.get();
}

Array1DBase<det_id_t>* ListModeLUT::getDetector1ArrayPtr() const
{
	return mp_detectorId1.get();
}

Array1DBase<det_id_t>* ListModeLUT::getDetector2ArrayPtr() const
{
	return mp_detectorId2.get();
}

Array1DBase<float>* ListModeLUT::getTOFArrayPtr() const
{
	return mp_tof_ps.get();
}

Array1DBase<float>* ListModeLUT::getRandomsEstimatesArrayPtr() const
{
	return mp_randoms.get();
}

Line3D ListModeLUT::getNativeLORFromId(bin_t eventId) const
{
	return util::getNativeLOR(mr_scanner, *this, eventId);
}

bool ListModeLUT::hasTOF() const
{
	return mp_tof_ps != nullptr;
}

void ListModeLUTOwned::allocate(size_t numEvents)
{
	static_cast<Array1D<timestamp_t>*>(mp_timestamps.get())
	    ->allocate(numEvents);
	static_cast<Array1D<det_id_t>*>(mp_detectorId1.get())->allocate(numEvents);
	static_cast<Array1D<det_id_t>*>(mp_detectorId2.get())->allocate(numEvents);
	if (hasTOF())
	{
		static_cast<Array1D<float>*>(mp_tof_ps.get())->allocate(numEvents);
	}
	if (hasRandomsEstimates())
	{
		static_cast<Array1D<float>*>(mp_randoms.get())->allocate(numEvents);
	}
}

det_id_t ListModeLUT::getDetector1(bin_t eventId) const
{
	return (*mp_detectorId1)[eventId];
}

det_id_t ListModeLUT::getDetector2(bin_t eventId) const
{
	return (*mp_detectorId2)[eventId];
}

float ListModeLUT::getTOFValue(bin_t eventId) const
{
	return (*mp_tof_ps)[eventId];
}

float ListModeLUT::getTOFValue_safe(bin_t eventId) const
{
	ASSERT_MSG(hasTOF(), "The given ListMode does not have any TOF values");
	return getTOFValue(eventId);
}

bool ListModeLUT::hasRandomsEstimates() const
{
	return mp_randoms != nullptr;
}

float ListModeLUT::getRandomsEstimate(bin_t eventId) const
{
	return (*mp_randoms)[eventId];
}

float ListModeLUT::getRandomsEstimate_safe(bin_t eventId) const
{
	ASSERT_MSG(mp_randoms != nullptr,
	           "The given ListMode does not have randoms estimates");
	return getRandomsEstimate(eventId);
}

void ListModeLUTAlias::bind(ListModeLUT* listMode)
{
	bind(listMode->getTimestampArrayPtr(), listMode->getDetector1ArrayPtr(),
	     listMode->getDetector2ArrayPtr());
}

void ListModeLUTAlias::bind(const Array1DBase<timestamp_t>* pp_timestamps,
                            const Array1DBase<det_id_t>* pp_detector_ids1,
                            const Array1DBase<det_id_t>* pp_detectorIds2,
                            const Array1DBase<float>* pp_tof_ps,
                            const Array1DBase<float>* pp_randoms)
{
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
	    ->bind(*pp_timestamps);
	if (mp_timestamps->getRawPointer() == nullptr)
		throw std::runtime_error("The timestamps array could not be bound");

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
	    ->bind(*pp_detector_ids1);
	if (mp_detectorId1->getRawPointer() == nullptr)
		throw std::runtime_error("The detector_ids1 array could not be bound");

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
	    ->bind(*pp_detectorIds2);
	if (mp_detectorId2->getRawPointer() == nullptr)
		throw std::runtime_error("The detector_ids2 array could not be bound");

	if (pp_tof_ps != nullptr)
	{
		ASSERT_MSG(hasTOF(),
		           "The list-mode was initialized without TOF enabled");
		static_cast<Array1DAlias<float>*>(mp_tof_ps.get())->bind(*pp_tof_ps);
		if (mp_tof_ps->getRawPointer() == nullptr)
			throw std::runtime_error("The tof_ps array could not be bound");
	}

	if (pp_randoms != nullptr)
	{
		ASSERT_MSG(hasRandomsEstimates(),
		           "The list-mode was initialized without Randoms enabled");
		static_cast<Array1DAlias<float>*>(mp_randoms.get())->bind(*pp_randoms);
		if (mp_randoms->getRawPointer() == nullptr)
			throw std::runtime_error(
			    "The randoms estimates array could not be bound");
	}
}

#if BUILD_PYBIND11
void ListModeLUTAlias::bind(
    pybind11::array_t<timestamp_t, pybind11::array::c_style>* pp_timestamps,
    pybind11::array_t<det_id_t, pybind11::array::c_style>* pp_detectorIds1,
    pybind11::array_t<det_id_t, pybind11::array::c_style>* pp_detectorIds2,
    pybind11::array_t<float, pybind11::array::c_style>* pp_tof_ps,
    pybind11::array_t<float, pybind11::array::c_style>* pp_randoms)
{
	ssize_t count = 0;
	ASSERT_MSG(pp_timestamps != nullptr, "The timestamps array is null");
	const pybind11::buffer_info buffer1 = pp_timestamps->request();
	ASSERT_MSG(buffer1.ndim == 1,
	           "The timestamps array has to be 1-dimensional");
	ASSERT_MSG(buffer1.shape[0] > 0, "Timestamps array is empty");
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
	    ->bind(reinterpret_cast<timestamp_t*>(buffer1.ptr), buffer1.shape[0]);
	count = buffer1.shape[0];

	ASSERT_MSG(pp_detectorIds1 != nullptr, "The detector_ids1 array is null");
	const pybind11::buffer_info buffer2 = pp_detectorIds1->request();
	ASSERT_MSG(buffer2.ndim == 1,
	           "The detector_ids1 array has to be 1-dimensional");
	ASSERT_MSG(
	    buffer2.shape[0] == count,
	    "detector_ids1 array is not of coherent size with the other arrays");
	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
	    ->bind(reinterpret_cast<det_id_t*>(buffer2.ptr), buffer2.shape[0]);

	ASSERT_MSG(pp_detectorIds2 != nullptr, "The detector_ids2 array is null");
	const pybind11::buffer_info buffer3 = pp_detectorIds2->request();
	ASSERT_MSG(buffer3.ndim == 1,
	           "The detector_ids2 array has to be 1-dimensional");
	ASSERT_MSG(
	    buffer3.shape[0] == count,
	    "detector_ids2 array is not of coherent size with the other arrays");
	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
	    ->bind(reinterpret_cast<det_id_t*>(buffer3.ptr), buffer3.shape[0]);

	if (pp_tof_ps != nullptr)
	{
		ASSERT_MSG(hasTOF(),
		           "The ListMode was not created with flag_tof at true");
		pybind11::buffer_info buffer = pp_tof_ps->request();
		ASSERT_MSG(buffer.ndim == 1, "The TOF array has to be 1-dimensional");
		ASSERT_MSG(buffer.shape[0] == count,
		           "TOF array is not of coherent size with the other arrays");
		static_cast<Array1DAlias<float>*>(mp_tof_ps.get())
		    ->bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0]);
	}

	if (pp_randoms != nullptr)
	{
		ASSERT_MSG(hasRandomsEstimates(),
		           "The ListMode was not created with flag_tof at true");
		pybind11::buffer_info buffer = pp_randoms->request();
		ASSERT_MSG(buffer.ndim == 1,
		           "The randoms estimates array has to be 1-dimensional");
		ASSERT_MSG(buffer.shape[0] == count,
		           "randoms estimates array is not of coherent size with the "
		           "other arrays");
		static_cast<Array1DAlias<float>*>(mp_tof_ps.get())
		    ->bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0]);
	}
}
#endif

std::unique_ptr<ProjectionData>
    ListModeLUTOwned::create(const Scanner& scanner,
                             const std::string& filename,
                             const io::OptionsResult& options)
{
	const auto flagTOFVariant = options.at("flag_tof");
	bool flagTOF = false;
	if (!std::holds_alternative<std::monostate>(flagTOFVariant))
	{
		ASSERT(std::holds_alternative<bool>(flagTOFVariant));
		flagTOF = std::get<bool>(flagTOFVariant);
	}

	const auto flagRandomsVariant = options.at("flag_randoms");
	bool flagRandoms = false;
	if (!std::holds_alternative<std::monostate>(flagRandomsVariant))
	{
		ASSERT(std::holds_alternative<bool>(flagRandomsVariant));
		flagRandoms = std::get<bool>(flagRandomsVariant);
	}

	auto lm = std::make_unique<ListModeLUTOwned>(scanner, filename, flagTOF,
	                                             flagRandoms);

	return lm;
}

plugin::OptionsListPerPlugin ListModeLUTOwned::getOptions()
{
	return {
	    {"flag_tof", {"Flag for reading TOF column", io::TypeOfArgument::BOOL}},
	    {"flag_randoms",
	     {"Flag for reading Randoms estimates column",
	      io::TypeOfArgument::BOOL}}};
}

REGISTER_PROJDATA_PLUGIN("LM", ListModeLUTOwned, ListModeLUTOwned::create,
                         ListModeLUTOwned::getOptions)

}  // namespace yrt
