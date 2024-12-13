/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ListModeLUT.hpp"

#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"

#include <cmath>
#include <fstream>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_listmodelut(py::module& m)
{
	auto c = py::class_<ListModeLUT, ListMode>(m, "ListModeLUT");
	c.def("setDetectorId1OfEvent", &ListModeLUT::setDetectorId1OfEvent);
	c.def("setDetectorId2OfEvent", &ListModeLUT::setDetectorId2OfEvent);
	c.def("setDetectorIdsOfEvent", &ListModeLUT::setDetectorIdsOfEvent);

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
	c.def("addLORMotion", &ListModeLUT::addLORMotion);

	c.def("writeToFile", &ListModeLUT::writeToFile);
	c.def("getNativeLORFromId", &ListModeLUT::getNativeLORFromId);

	auto c_alias =
		py::class_<ListModeLUTAlias, ListModeLUT>(m, "ListModeLUTAlias");
	c_alias.def(py::init<const Scanner&, bool>(), py::arg("scanner"),
	            py::arg("flag_tof") = false);

	c_alias.def("bind",
	            static_cast<void (ListModeLUTAlias::*)(
		            pybind11::array_t<timestamp_t, pybind11::array::c_style>&,
		            pybind11::array_t<det_id_t, pybind11::array::c_style>&,
		            pybind11::array_t<det_id_t, pybind11::array::c_style>&)>(
		            &ListModeLUTAlias::bind),
	            py::arg("timestamps"), py::arg("detector_ids1"),
	            py::arg("detector_ids2"));
	c_alias.def("bind",
	            static_cast<void (ListModeLUTAlias::*)(
		            pybind11::array_t<timestamp_t, pybind11::array::c_style>&,
		            pybind11::array_t<det_id_t, pybind11::array::c_style>&,
		            pybind11::array_t<det_id_t, pybind11::array::c_style>&,
		            pybind11::array_t<float, pybind11::array::c_style>&)>(
		            &ListModeLUTAlias::bind),
	            py::arg("timestamps"), py::arg("detector_ids1"),
	            py::arg("detector_ids2"), py::arg("tof_ps"));


	auto c_owned =
		py::class_<ListModeLUTOwned, ListModeLUT>(m, "ListModeLUTOwned");
	c_owned.def(py::init<const Scanner&, bool>(), py::arg("scanner"),
	            py::arg("flag_tof") = false);
	c_owned.def(py::init<const Scanner&, const std::string&, bool>(),
	            py::arg("scanner"), py::arg("listMode_fname"),
	            py::arg("flag_tof") = false);
	c_owned.def("readFromFile", &ListModeLUTOwned::readFromFile);
	c_owned.def("allocate", &ListModeLUTOwned::allocate);
	c_owned.def("createFromHistogram3D",
	            [](ListModeLUTOwned* self, const Histogram3D* histo,
	               size_t num_events)
	            {
		            Util::histogram3DToListModeLUT(histo, self, num_events);
	            });
}

#endif  // if BUILD_PYBIND11


ListModeLUT::ListModeLUT(const Scanner& pr_scanner, bool p_flagTOF)
	: ListMode(pr_scanner),
	  m_flagTOF(p_flagTOF) {}

ListModeLUTOwned::ListModeLUTOwned(const Scanner& pr_scanner, bool p_flagTOF)
	: ListModeLUT(pr_scanner, p_flagTOF)
{
	mp_timestamps = std::make_unique<Array1D<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1D<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1D<det_id_t>>();
	if (m_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1D<float>>();
	}
}

ListModeLUTOwned::ListModeLUTOwned(const Scanner& pr_scanner,
                                   const std::string& listMode_fname,
                                   bool p_flagTOF)
	: ListModeLUTOwned(pr_scanner, p_flagTOF)
{
	ListModeLUTOwned::readFromFile(listMode_fname);
}

ListModeLUTAlias::ListModeLUTAlias(const Scanner& pr_scanner, bool p_flagTOF)
	: ListModeLUT(pr_scanner, p_flagTOF)
{
	mp_timestamps = std::make_unique<Array1DAlias<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1DAlias<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1DAlias<det_id_t>>();
	if (m_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1DAlias<float>>();
	}
}

void ListModeLUTOwned::readFromFile(const std::string& listMode_fname)
{
	std::ifstream fin(listMode_fname, std::ios::in | std::ios::binary);

	if (!fin.good())
	{
		throw std::runtime_error("Error reading input file " + listMode_fname +
		                         "ListModeLUTOwned::readFromFile.");
	}

	// first check that file has the right size:
	fin.seekg(0, std::ios::end);
	size_t end = fin.tellg();
	fin.seekg(0, std::ios::beg);
	size_t begin = fin.tellg();
	size_t file_size = end - begin;
	int num_fields = m_flagTOF ? 4 : 3;
	size_t sizeOfAnEvent = num_fields * sizeof(float);
	if (file_size <= 0 || (file_size % sizeOfAnEvent) != 0)
	{
		throw std::runtime_error("Error: Input file has incorrect size in "
			"ListModeLUTOwned::readFromFile.");
	}

	// Allocate the memory
	size_t numEvents = file_size / sizeOfAnEvent;
	allocate(numEvents);

	// Read content of file
	size_t buffer_size = (size_t(1) << 30);
	auto buff = std::make_unique<float[]>(buffer_size);
	size_t pos_start = 0;
	while (pos_start < numEvents)
	{
		size_t read_size =
			std::min(buffer_size, num_fields * (numEvents - pos_start));
		fin.read((char*)buff.get(),
		         (read_size / num_fields) * num_fields * sizeof(float));

		int num_threads = Globals::get_num_threads();
#pragma omp parallel for num_threads(num_threads)
		for (size_t i = 0; i < read_size / num_fields; i++)
		{
			(*mp_timestamps)[pos_start + i] = buff[num_fields * i];
			(*mp_detectorId1)[pos_start + i] =
				*(reinterpret_cast<det_id_t*>(&(buff[num_fields * i + 1])));
			(*mp_detectorId2)[pos_start + i] =
				*(reinterpret_cast<det_id_t*>(&(buff[num_fields * i + 2])));
			if (m_flagTOF)
			{
				(*mp_tof_ps)[pos_start + i] = buff[num_fields * i + 3];
			}
		}
		pos_start += read_size / num_fields;
	}
}

void ListModeLUT::writeToFile(const std::string& listMode_fname) const
{
	int num_fields = m_flagTOF ? 4 : 3;
	size_t numEvents = count();
	std::ofstream file;
	file.open(listMode_fname.c_str(), std::ios::binary | std::ios::out);

	size_t bufferSize = (size_t(1) << 30);
	auto buff = std::make_unique<float[]>(bufferSize);
	// This is done assuming that "int" and "float" are of the same size
	// (4bytes)
	size_t posStart = 0;
	while (posStart < numEvents)
	{
		size_t writeSize =
			std::min(bufferSize, num_fields * (numEvents - posStart));
		for (size_t i = 0; i < writeSize / num_fields; i++)
		{
			buff[num_fields * i] = (*mp_timestamps)[posStart + i];
			buff[num_fields * i + 1] =
				*(reinterpret_cast<float*>(&(*mp_detectorId1)[posStart + i]));
			buff[num_fields * i + 2] =
				*(reinterpret_cast<float*>(&(*mp_detectorId2)[posStart + i]));
			if (m_flagTOF)
			{
				buff[num_fields * i + 3] = (*mp_tof_ps)[posStart + i];
			}
		}
		file.write((char*)buff.get(),
		           (writeSize / num_fields) * num_fields * sizeof(float));
		posStart += writeSize / num_fields;
	}
	file.close();
}

void ListModeLUT::addLORMotion(const std::string& lorMotion_fname)
{
	mp_lorMotion = std::make_unique<LORMotion>(lorMotion_fname);
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

bool ListModeLUT::hasMotion() const
{
	return mp_lorMotion != nullptr;
}

frame_t ListModeLUT::getFrame(bin_t id) const
{
	if (mp_lorMotion != nullptr)
	{
		return mp_frames->getFlat(id);
	}
	return ProjectionData::getFrame(id);
}

size_t ListModeLUT::getNumFrames() const
{
	if (mp_lorMotion != nullptr)
	{
		return mp_lorMotion->getNumFrames();
	}
	return ProjectionData::getNumFrames();
}

transform_t ListModeLUT::getTransformOfFrame(frame_t frame) const
{
	ASSERT(mp_lorMotion != nullptr);
	if (frame >= 0)
	{
		return mp_lorMotion->getTransform(frame);
	}
	// For the events before the beginning of the frame
	return ProjectionData::getTransformOfFrame(frame);
}

float ListModeLUT::getDurationOfFrame(frame_t frame) const
{
	ASSERT(mp_lorMotion != nullptr);
	if (frame >= 0)
	{
		return mp_lorMotion->getDuration(frame);
	}
	// For the events before the beginning of the frame
	return ProjectionData::getDurationOfFrame(frame);
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

void ListModeLUT::setDetectorIdsOfEvent(bin_t eventId, det_id_t d1,
                                        det_id_t d2)
{
	(*mp_detectorId1)[eventId] = d1;
	(*mp_detectorId2)[eventId] = d2;
}

Array1DBase<timestamp_t>* ListModeLUT::getTimestampArrayPtr() const
{
	return (mp_timestamps.get());
}

Array1DBase<det_id_t>* ListModeLUT::getDetector1ArrayPtr() const
{
	return (mp_detectorId1.get());
}

Array1DBase<det_id_t>* ListModeLUT::getDetector2ArrayPtr() const
{
	return (mp_detectorId2.get());
}

Line3D ListModeLUT::getNativeLORFromId(bin_t id) const
{
	return Util::getNativeLOR(mr_scanner, *this, id);
}

bool ListModeLUT::hasTOF() const
{
	return m_flagTOF;
}

void ListModeLUTOwned::allocate(size_t numEvents)
{
	static_cast<Array1D<timestamp_t>*>(mp_timestamps.get())
		->allocate(numEvents);
	static_cast<Array1D<det_id_t>*>(mp_detectorId1.get())->allocate(numEvents);
	static_cast<Array1D<det_id_t>*>(mp_detectorId2.get())->allocate(numEvents);
	if (m_flagTOF)
	{
		static_cast<Array1D<float>*>(mp_tof_ps.get())->allocate(numEvents);
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
	if (m_flagTOF)
		return (*mp_tof_ps)[eventId];
	else
		throw std::logic_error(
			"The given ListMode does not have any TOF values");
}

void ListModeLUTAlias::bind(ListModeLUT* listMode)
{
	bind(listMode->getTimestampArrayPtr(), listMode->getDetector1ArrayPtr(),
	     listMode->getDetector2ArrayPtr());
}

void ListModeLUTAlias::bind(Array1DBase<timestamp_t>* pp_timestamps,
                            Array1DBase<det_id_t>* pp_detectorIds1,
                            Array1DBase<det_id_t>* pp_detectorIds2,
                            Array1DBase<float>* pp_tof_ps)
{
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
		->bind(*pp_timestamps);
	if (mp_timestamps->getRawPointer() == nullptr)
		throw std::runtime_error("The timestamps array could not be bound");

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
		->bind(*pp_detectorIds1);
	if (mp_detectorId1->getRawPointer() == nullptr)
		throw std::runtime_error("The detector_ids1 array could not be bound");

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
		->bind(*pp_detectorIds2);
	if (mp_detectorId2->getRawPointer() == nullptr)
		throw std::runtime_error("The detector_ids2 array could not be bound");

	if (mp_tof_ps != nullptr && pp_tof_ps != nullptr)
	{
		static_cast<Array1DAlias<float>*>(mp_tof_ps.get())->bind(*pp_tof_ps);
		if (mp_tof_ps->getRawPointer() == nullptr)
			throw std::runtime_error("The tof_ps array could not be bound");
	}
}

#if BUILD_PYBIND11
void ListModeLUTAlias::bind(
	pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detectorIds1,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detectorIds2)
{
	pybind11::buffer_info buffer1 = p_timestamps.request();
	if (buffer1.ndim != 1)
	{
		throw std::invalid_argument(
			"The timestamps buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
		->bind(reinterpret_cast<timestamp_t*>(buffer1.ptr), buffer1.shape[0]);

	pybind11::buffer_info buffer2 = p_detectorIds1.request();
	if (buffer2.ndim != 1)
	{
		throw std::invalid_argument(
			"The detector_ids1 buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
		->bind(reinterpret_cast<det_id_t*>(buffer2.ptr), buffer2.shape[0]);

	pybind11::buffer_info buffer3 = p_detectorIds2.request();
	if (buffer3.ndim != 1)
	{
		throw std::invalid_argument(
			"The detector_ids2 buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
		->bind(reinterpret_cast<det_id_t*>(buffer3.ptr), buffer3.shape[0]);
}

void ListModeLUTAlias::bind(
	pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	pybind11::array_t<float, pybind11::array::c_style>& p_tof_ps)
{
	bind(p_timestamps, p_detector_ids1, p_detector_ids2);
	if (!m_flagTOF)
		throw std::logic_error(
			"The ListMode was not created with flag_tof at true");
	pybind11::buffer_info buffer = p_tof_ps.request();
	if (buffer.ndim != 1)
	{
		throw std::invalid_argument("The TOF buffer has to be 1-dimensional");
	}
	static_cast<Array1DAlias<float>*>(mp_tof_ps.get())
		->bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0]);
}
#endif

std::unique_ptr<ProjectionData>
	ListModeLUTOwned::create(const Scanner& scanner,
	                         const std::string& filename,
	                         const Plugin::OptionsResult& pluginOptions)
{
	const auto flagTOF_it = pluginOptions.find("flag_tof");
	bool flagTOF = flagTOF_it != pluginOptions.end();
	auto lm = std::make_unique<ListModeLUTOwned>(scanner, filename, flagTOF);

	if (pluginOptions.count("lor_motion"))
	{
		lm->addLORMotion(pluginOptions.at("lor_motion"));
	}
	return lm;
}

Plugin::OptionsListPerPlugin ListModeLUTOwned::getOptions()
{
	return {{"flag_tof", {"Flag for reading TOF column", true}},
	        {"lor_motion", {"LOR motion file for motion correction", false}}};
}

REGISTER_PROJDATA_PLUGIN("LM", ListModeLUTOwned, ListModeLUTOwned::create,
                         ListModeLUTOwned::getOptions)