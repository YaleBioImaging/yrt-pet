/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ListModeLUTDOI.hpp"

#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"

#include <cmath>
#include <cstring>
#include <memory>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_listmodelutdoi(py::module& m)
{
	auto c = py::class_<ListModeLUTDOI, ListModeLUT>(m, "ListModeLUTDOI");

	c.def("writeToFile", &ListModeLUTDOI::writeToFile);

	auto c_alias = py::class_<ListModeLUTDOIAlias, ListModeLUTDOI>(
	    m, "ListModeLUTDOIAlias");
	c_alias.def(py::init<const Scanner&, bool, bool, int>(), "scanner"_a,
	            "flag_tof"_a = false, "flag_randoms"_a = false,
	            "numLayers"_a = 256);

	c_alias.def("bind",
	            static_cast<void (ListModeLUTDOIAlias::*)(
	                pybind11::array_t<timestamp_t, pybind11::array::c_style>*,
	                pybind11::array_t<det_id_t, pybind11::array::c_style>*,
	                pybind11::array_t<det_id_t, pybind11::array::c_style>*,
	                pybind11::array_t<unsigned char, pybind11::array::c_style>*,
	                pybind11::array_t<unsigned char, pybind11::array::c_style>*,
	                pybind11::array_t<float, pybind11::array::c_style>*,
	                pybind11::array_t<float, pybind11::array::c_style>*)>(
	                &ListModeLUTDOIAlias::bind),
	            "timestamps"_a, "detector_ids1"_a, "detector_ids2"_a, "doi1"_a,
	            "doi2"_a, "tof_ps"_a, "randoms"_a);


	auto c_owned = py::class_<ListModeLUTDOIOwned, ListModeLUTDOI>(
	    m, "ListModeLUTDOIOwned");
	c_owned.def(py::init<const Scanner&, bool, bool, int>(), "scanner"_a,
	            "flag_tof"_a = false, "flag_randoms"_a = false,
	            "numLayers"_a = 256);
	c_owned.def(py::init<const Scanner&, std::string, bool, bool, int>(),
	            "scanner"_a, "listMode_fname"_a, "flag_tof"_a = false,
	            "flag_randoms"_a = false, "numLayers"_a = 256);
	c_owned.def("readFromFile", &ListModeLUTDOIOwned::readFromFile);
	c_owned.def("allocate", &ListModeLUTDOIOwned::allocate);
}
}  // namespace yrt
#endif  // if BUILD_PYBIND11


namespace yrt
{

ListModeLUTDOI::ListModeLUTDOI(const Scanner& pr_scanner, int numLayers)
    : ListModeLUT(pr_scanner), m_numLayers(numLayers)
{
}

ListModeLUTDOIOwned::ListModeLUTDOIOwned(const Scanner& pr_scanner,
                                         bool p_flagTOF, bool p_flagRandoms,
                                         int numLayers)
    : ListModeLUTDOI(pr_scanner, numLayers)
{
	mp_timestamps = std::make_unique<Array1D<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1D<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1D<det_id_t>>();
	mp_doi1 = std::make_unique<Array1D<unsigned char>>();
	mp_doi2 = std::make_unique<Array1D<unsigned char>>();
	if (p_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1D<float>>();
	}
	if (p_flagRandoms)
	{
		mp_randoms = std::make_unique<Array1D<float>>();
	}
}

ListModeLUTDOIOwned::ListModeLUTDOIOwned(const Scanner& pr_scanner,
                                         const std::string& listMode_fname,
                                         bool p_flagTOF, bool p_flagRandoms,
                                         int numLayers)
    : ListModeLUTDOIOwned(pr_scanner, p_flagTOF, p_flagRandoms, numLayers)
{
	readFromFile(listMode_fname);
}

ListModeLUTDOIAlias::ListModeLUTDOIAlias(const Scanner& pr_scanner,
                                         bool p_flagTOF, bool p_flagRandoms,
                                         int numLayers)
    : ListModeLUTDOI(pr_scanner, numLayers)
{
	mp_timestamps = std::make_unique<Array1DAlias<timestamp_t>>();
	mp_detectorId1 = std::make_unique<Array1DAlias<det_id_t>>();
	mp_detectorId2 = std::make_unique<Array1DAlias<det_id_t>>();
	mp_doi1 = std::make_unique<Array1DAlias<unsigned char>>();
	mp_doi2 = std::make_unique<Array1DAlias<unsigned char>>();
	if (p_flagTOF)
	{
		mp_tof_ps = std::make_unique<Array1DAlias<float>>();
	}
	if (p_flagRandoms)
	{
		mp_randoms = std::make_unique<Array1DAlias<float>>();
	}
}

void ListModeLUTDOIOwned::readFromFile(const std::string& listMode_fname)
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
	int numFields = 3;  // Number of 4-byte fields
	if (hasTOF)
	{
		numFields++;
	}
	if (hasRandoms)
	{
		numFields++;
	}

	size_t sizeOfAnEvent =
	    numFields * sizeof(float) + 2 * sizeof(unsigned char);
	if (fileSize <= 0 || (fileSize % sizeOfAnEvent) != 0)
	{
		throw std::runtime_error("Error: Input file has incorrect size in "
		                         "ListModeLUTDOIOwned::readFromFile.");
	}

	// Allocate the memory
	size_t numEvents = fileSize / sizeOfAnEvent;
	allocate(numEvents);

	// Read content of file
	size_t numEventsBatch = 1ull << 15;
	auto buff =
	    std::make_unique<unsigned char[]>(numEventsBatch * sizeOfAnEvent);
	auto buffPtr = buff.get();
	size_t eventStart = 0;
	while (eventStart < numEvents)
	{
		size_t numEventsBatchCurr =
		    std::min(numEventsBatch, numEvents - eventStart);
		size_t readSize = numEventsBatchCurr * sizeOfAnEvent;
		fin.read((char*)buff.get(), readSize);

		util::parallel_for_chunked(
		    numEventsBatchCurr, globals::numThreads(),
		    [eventStart, sizeOfAnEvent, buffPtr, numDets, hasTOF, hasRandoms,
		     this](size_t i, size_t /*tid*/)
		    {
			    const size_t eventPos = eventStart + i;
			    size_t bufferPos = sizeOfAnEvent * i;

			    const timestamp_t timestamp =
			        *reinterpret_cast<timestamp_t*>(&(buffPtr[bufferPos]));
			    bufferPos += sizeof(timestamp_t);

			    const det_id_t d1 =
			        *(reinterpret_cast<det_id_t*>(&(buffPtr[bufferPos])));
			    bufferPos += sizeof(det_id_t);

			    const unsigned char doi1 = buffPtr[bufferPos];
			    bufferPos += sizeof(unsigned char);

			    const det_id_t d2 =
			        *(reinterpret_cast<det_id_t*>(&(buffPtr[bufferPos])));
			    bufferPos += sizeof(det_id_t);

			    const unsigned char doi2 = buffPtr[bufferPos];
			    bufferPos += sizeof(unsigned char);

			    if (CHECK_LIKELY(d1 < numDets && d2 < numDets))
			    {
				    (*mp_timestamps)[eventPos] = timestamp;
				    (*mp_detectorId1)[eventPos] = d1;
				    (*mp_doi1)[eventPos] = doi1;
				    (*mp_detectorId2)[eventPos] = d2;
				    (*mp_doi2)[eventPos] = doi2;
				    if (hasTOF)
				    {
					    (*mp_tof_ps)[eventPos] =
					        *(reinterpret_cast<float*>(&(buffPtr[bufferPos])));
					    bufferPos += sizeof(float);
				    }
				    if (hasRandoms)
				    {
					    (*mp_randoms)[eventPos] =
					        *(reinterpret_cast<float*>(&(buffPtr[bufferPos])));
					    // Uncomment this if we add another field:
					    // bufferPos += sizeof(float);
				    }
			    }
			    else
			    {
				    throw std::invalid_argument(
				        "Detectors invalid in list-mode event " +
				        std::to_string(eventPos));
			    }
		    });
		eventStart += numEventsBatchCurr;
	}
}

bool ListModeLUTDOI::hasArbitraryLORs() const
{
	return true;
}

Line3D ListModeLUTDOI::getArbitraryLOR(bin_t id) const
{
	const det_id_t detId1 = getDetector1(id);
	const det_id_t detId2 = getDetector2(id);
	const Vector3D p1 = mr_scanner.getDetectorPos(detId1);
	const Vector3D p2 = mr_scanner.getDetectorPos(detId2);
	const Vector3D n1 = mr_scanner.getDetectorOrient(detId1);
	const Vector3D n2 = mr_scanner.getDetectorOrient(detId2);
	const float layerSize = (1 << 8) / static_cast<float>(m_numLayers);
	const float doi1_t = std::floor((*mp_doi1)[id] / layerSize) *
	                     mr_scanner.crystalDepth /
	                     static_cast<float>(m_numLayers);
	const float doi2_t = std::floor((*mp_doi2)[id] / layerSize) *
	                     mr_scanner.crystalDepth /
	                     static_cast<float>(m_numLayers);
	const Vector3D p1_doi{
	    p1.x + (doi1_t - 0.5f * mr_scanner.crystalDepth) * n1.x,
	    p1.y + (doi1_t - 0.5f * mr_scanner.crystalDepth) * n1.y,
	    p1.z + (doi1_t - 0.5f * mr_scanner.crystalDepth) * n1.z};
	const Vector3D p2_doi{
	    p2.x + (doi2_t - 0.5f * mr_scanner.crystalDepth) * n2.x,
	    p2.y + (doi2_t - 0.5f * mr_scanner.crystalDepth) * n2.y,
	    p2.z + (doi2_t - 0.5f * mr_scanner.crystalDepth) * n2.z};
	return Line3D{{p1_doi.x, p1_doi.y, p1_doi.z},
	              {p2_doi.x, p2_doi.y, p2_doi.z}};
}

void ListModeLUTDOI::writeToFile(const std::string& listMode_fname) const
{
	const bool hasTOF = mp_tof_ps != nullptr;
	const bool hasRandoms = mp_randoms != nullptr;

	int numFields = 3;  // Number of 4-byte fields
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
	const size_t sizeOfAnEvent =
	    numFields * sizeof(float) + (2 * sizeof(unsigned char));

	constexpr size_t numEventsBatch = 1ull << 15;
	auto buff =
	    std::make_unique<unsigned char[]>(numEventsBatch * sizeOfAnEvent);
	size_t eventStart = 0;
	while (eventStart < numEvents)
	{
		const size_t numEventsBatchCurr =
		    std::min(numEventsBatch, numEvents - eventStart);
		const size_t writeSize = numEventsBatchCurr * sizeOfAnEvent;
		for (size_t i = 0; i < numEventsBatchCurr; i++)
		{
			size_t bufferPos = sizeOfAnEvent * i;
			const size_t arrayPos = eventStart + i;

			memcpy(&buff[bufferPos], &(*mp_timestamps)[arrayPos],
			       sizeof(timestamp_t));
			bufferPos += sizeof(timestamp_t);

			memcpy(&buff[bufferPos], &(*mp_detectorId1)[arrayPos],
			       sizeof(det_id_t));
			bufferPos += sizeof(det_id_t);

			buff[bufferPos] = (*mp_doi1)[arrayPos];
			bufferPos += sizeof(unsigned char);

			memcpy(&buff[bufferPos], &(*mp_detectorId2)[arrayPos],
			       sizeof(det_id_t));
			bufferPos += sizeof(det_id_t);

			buff[bufferPos] = (*mp_doi2)[arrayPos];
			bufferPos += sizeof(unsigned char);

			if (hasTOF)
			{
				memcpy(&buff[bufferPos], &(*mp_tof_ps)[arrayPos],
				       sizeof(float));
				bufferPos += sizeof(float);
			}

			if (hasRandoms)
			{
				memcpy(&buff[bufferPos], &(*mp_randoms)[arrayPos],
				       sizeof(float));
				// Uncomment this if we add another field:
				// bufferPos += sizeof(float);
			}
		}
		file.write(reinterpret_cast<char*>(buff.get()), writeSize);
		eventStart += numEventsBatchCurr;
	}
	file.close();
}

void ListModeLUTDOIOwned::allocate(size_t num_events)
{
	static_cast<Array1D<timestamp_t>*>(mp_timestamps.get())
	    ->allocate(num_events);
	static_cast<Array1D<det_id_t>*>(mp_detectorId1.get())->allocate(num_events);
	static_cast<Array1D<det_id_t>*>(mp_detectorId2.get())->allocate(num_events);
	static_cast<Array1D<unsigned char>*>(mp_doi1.get())->allocate(num_events);
	static_cast<Array1D<unsigned char>*>(mp_doi2.get())->allocate(num_events);
	if (hasTOF())
	{
		static_cast<Array1D<float>*>(mp_tof_ps.get())->allocate(num_events);
	}
	if (hasRandomsEstimates())
	{
		static_cast<Array1D<float>*>(mp_randoms.get())->allocate(num_events);
	}
}

void ListModeLUTDOIAlias::bind(const Array1DBase<timestamp_t>* pp_timestamps,
                               const Array1DBase<det_id_t>* pp_detector_ids1,
                               const Array1DBase<det_id_t>* pp_detector_ids2,
                               const Array1DBase<unsigned char>* pp_doi1,
                               const Array1DBase<unsigned char>* pp_doi2,
                               const Array1DBase<float>* pp_tof_ps,
                               const Array1DBase<float>* pp_randoms)
{
	static_cast<Array1DAlias<timestamp_t>*>(mp_timestamps.get())
	    ->bind(*pp_timestamps);
	if (mp_timestamps->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The timestamps array could not be bound");
	}

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId1.get())
	    ->bind(*pp_detector_ids1);
	if (mp_detectorId1->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The detector_ids1 array could not be bound");
	}

	static_cast<Array1DAlias<det_id_t>*>(mp_detectorId2.get())
	    ->bind(*pp_detector_ids2);
	if (mp_detectorId2->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The detector_ids2 array could not be bound");
	}

	static_cast<Array1DAlias<unsigned char>*>(mp_doi1.get())->bind(*pp_doi1);
	if (mp_doi1->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The doi1 array could not be bound");
	}
	static_cast<Array1DAlias<unsigned char>*>(mp_doi2.get())->bind(*pp_doi2);
	if (mp_doi2->getRawPointer() == nullptr)
	{
		throw std::runtime_error("The doi2 array could not be bound");
	}

	if (mp_tof_ps != nullptr)
	{
		ASSERT_MSG(
		    pp_tof_ps != nullptr,
		    "TOF is enabled for the list-mode but no TOF array was provided");
		static_cast<Array1DAlias<float>*>(mp_tof_ps.get())->bind(*pp_tof_ps);
		if (mp_tof_ps->getRawPointer() == nullptr)
			throw std::runtime_error("The tof_ps array could not be bound");
	}

	if (mp_randoms != nullptr)
	{
		ASSERT_MSG(pp_randoms != nullptr,
		           "Randoms are enabled for the list-mode but no Randoms "
		           "estimates array was provided");
		static_cast<Array1DAlias<float>*>(mp_randoms.get())->bind(*pp_randoms);
		if (mp_randoms->getRawPointer() == nullptr)
			throw std::runtime_error("The randoms array could not be bound");
	}
}

#if BUILD_PYBIND11
void ListModeLUTDOIAlias::bind(
    pybind11::array_t<timestamp_t, pybind11::array::c_style>* pp_timestamps,
    pybind11::array_t<det_id_t, pybind11::array::c_style>* pp_detectorIds1,
    pybind11::array_t<det_id_t, pybind11::array::c_style>* pp_detectorIds2,
    pybind11::array_t<unsigned char, pybind11::array::c_style>* pp_doi1,
    pybind11::array_t<unsigned char, pybind11::array::c_style>* pp_doi2,
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

	// To avoid code duplication:
	using doi_pair_t =
	    std::pair<py::array_t<unsigned char, py::array::c_style>*,
	              Array1DBase<unsigned char>*>;
	std::array<doi_pair_t, 2> doiPairs;
	doiPairs[0] = {pp_doi1, mp_doi1.get()};
	doiPairs[1] = {pp_doi2, mp_doi2.get()};
	for (auto doiPair : doiPairs)
	{
		auto [pp_doi, mp_doi] = doiPair;
		ASSERT_MSG(pp_doi != nullptr, "The DOI array is null");
		const pybind11::buffer_info buffer = pp_doi->request();
		ASSERT_MSG(buffer.ndim == 1, "The DOI array has to be 1-dimensional");
		ASSERT_MSG(buffer.shape[0] == count,
		           "DOI array is not of coherent size with the other arrays");
		static_cast<Array1DAlias<unsigned char>*>(mp_doi)->bind(
		    reinterpret_cast<unsigned char*>(buffer.ptr), buffer.shape[0]);
	}

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
		static_cast<Array1DAlias<float>*>(mp_randoms.get())
		    ->bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0]);
	}
}
#endif


std::unique_ptr<ProjectionData>
    ListModeLUTDOIOwned::create(const Scanner& scanner,
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

	const auto numLayersVariant = options.at("num_layers");

	std::unique_ptr<ListModeLUTDOIOwned> lm;
	if (std::holds_alternative<std::monostate>(numLayersVariant))
	{
		lm = std::make_unique<ListModeLUTDOIOwned>(scanner, filename, flagTOF,
		                                           flagRandoms);
	}
	else
	{
		ASSERT(std::holds_alternative<int>(numLayersVariant));
		const int numLayers = std::get<int>(numLayersVariant);

		lm = std::make_unique<ListModeLUTDOIOwned>(scanner, filename, flagTOF,
		                                           flagRandoms, numLayers);
	}

	return lm;
}

plugin::OptionsListPerPlugin ListModeLUTDOIOwned::getOptions()
{
	return {
	    {"flag_tof", {"Flag for reading TOF column", io::TypeOfArgument::BOOL}},
	    {"flag_randoms",
	     {"Flag for reading randoms estimates column",
	      io::TypeOfArgument::BOOL}},
	    {"num_layers", {"Number of layers", io::TypeOfArgument::INT}}};
}


REGISTER_PROJDATA_PLUGIN("LM-DOI", ListModeLUTDOIOwned,
                         ListModeLUTDOIOwned::create,
                         ListModeLUTDOIOwned::getOptions)

}  // namespace yrt
