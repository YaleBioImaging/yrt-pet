/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/SparseHistogram.hpp"
#include "utils/Assert.hpp"
#include "utils/ProgressDisplay.hpp"

#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;
namespace py = pybind11;

void py_setup_sparsehistogram(py::module& m)
{
	auto c = py::class_<SparseHistogram, Histogram>(m, "SparseHistogram");
	c.def(py::init<const Scanner&>(), "scanner"_a);
	c.def(py::init<const Scanner&, const std::string&>(), "scanner"_a,
	      "filename"_a);
	c.def(py::init<const Scanner&, const ProjectionData&>(), "scanner"_a,
	      "projectionData"_a);
	c.def("allocate", &SparseHistogram::allocate, "numBins"_a);
	c.def(
	    "accumulate",
	    static_cast<void (SparseHistogram::*)(
	        det_pair_t detPair, float projValue)>(&SparseHistogram::accumulate),
	    "detPair"_a, "projValue"_a);
	c.def(
	    "accumulate",
	    [](SparseHistogram& self, const ProjectionData& projData,
	       bool ignoreZeros)
	    {
		    if (ignoreZeros)
		    {
			    self.accumulate<true>(projData);
		    }
		    else
		    {
			    self.accumulate<false>(projData);
		    }
	    },
	    "projData"_a, "ignoreZeros"_a = true);
	c.def("getProjectionValueFromDetPair",
	      &SparseHistogram::getProjectionValueFromDetPair, "detPair"_a);
	c.def("readFromFile", &SparseHistogram::readFromFile, "filename"_a);
	c.def("writeToFile", &SparseHistogram::writeToFile, "filename"_a);
	c.def("getProjValuesArray",
	      [](SparseHistogram& self) -> pybind11::array_t<float>
	      {
		      const auto buf_info = py::buffer_info(
		          self.getProjectionValuesBuffer(), sizeof(float),
		          py::format_descriptor<float>::format(), 1, {self.count()},
		          {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
}
#endif

SparseHistogram::SparseHistogram(const Scanner& pr_scanner)
    : Histogram(pr_scanner)
{
}

SparseHistogram::SparseHistogram(const Scanner& pr_scanner,
                                 const std::string& filename)
    : SparseHistogram(pr_scanner)
{
	readFromFile(filename);
}

SparseHistogram::SparseHistogram(const Scanner& pr_scanner,
                                 const ProjectionData& pr_projData)
    : SparseHistogram(pr_scanner)
{
	accumulate(pr_projData);
}

void SparseHistogram::allocate(size_t numBins)
{
	m_detectorMap.reserve(numBins);
	m_projValues.reserve(numBins);
	m_detPairs.reserve(numBins);
}

template <bool IgnoreZeros>
void SparseHistogram::accumulate(const ProjectionData& projData)
{
	const size_t numBins = projData.count();

	allocate(count() + numBins);

	Util::ProgressDisplay progress{static_cast<int64_t>(numBins), 5};

	for (bin_t binId = 0; binId < numBins; binId++)
	{
		progress.progress(binId);
		const float projValue = projData.getProjectionValue(binId);
		if constexpr (IgnoreZeros)
		{
			if (projValue == 0.0f)
			{
				continue;
			}
		}
		const det_pair_t detPair = projData.getDetectorPair(binId);
		// Add to the histogram
		accumulate(detPair, projValue);
	}
}
template void SparseHistogram::accumulate<true>(const ProjectionData& projData);
template void
    SparseHistogram::accumulate<false>(const ProjectionData& projData);

void SparseHistogram::accumulate(det_pair_t detPair, float projValue)
{
	const det_pair_t newPair = SwapDetectorPairIfNeeded(detPair);

	auto [detectorMapLocation, newlyInserted] =
	    m_detectorMap.try_emplace(newPair, m_detectorMap.size());
	if (newlyInserted)
	{
		// Add the pair
		m_detPairs.push_back(newPair);
		// Initialize its value
		m_projValues.push_back(projValue);
	}
	else
	{
		// Get the proper bin
		const bin_t bin = detectorMapLocation->second;
		// Accumulate the value
		m_projValues[bin] += projValue;
	}
	ASSERT(m_detPairs.size() == m_detectorMap.size());
	ASSERT(m_projValues.size() == m_detectorMap.size());
}

float SparseHistogram::getProjectionValueFromDetPair(det_pair_t detPair) const
{
	const auto detectorMapLocation =
	    m_detectorMap.find(SwapDetectorPairIfNeeded(detPair));
	if (detectorMapLocation != m_detectorMap.end())
	{
		// Get the proper bin
		const bin_t bin = detectorMapLocation->second;
		// Return the value
		return m_projValues[bin];
	}
	return 0.0f;
}

size_t SparseHistogram::count() const
{
	return m_projValues.size();
}

det_id_t SparseHistogram::getDetector1(bin_t id) const
{
	return m_detPairs[id].d1;
}

det_id_t SparseHistogram::getDetector2(bin_t id) const
{
	return m_detPairs[id].d2;
}

det_pair_t SparseHistogram::getDetectorPair(bin_t id) const
{
	return m_detPairs[id];
}

std::unique_ptr<BinIterator> SparseHistogram::getBinIter(int numSubsets,
                                                         int idxSubset) const
{
	ASSERT_MSG(idxSubset < numSubsets,
	           "The subset index has to be smaller than the number of subsets");
	ASSERT_MSG(idxSubset == 0 && numSubsets == 1,
	           "Multiple subsets are not supported in sparse histograms");

	return std::make_unique<BinIteratorChronological>(numSubsets, count(),
	                                                  idxSubset);
}

float SparseHistogram::getProjectionValue(bin_t id) const
{
	return m_projValues[id];
}

void SparseHistogram::setProjectionValue(bin_t id, float val)
{
	m_projValues[id] = val;
}

float SparseHistogram::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	ASSERT(std::holds_alternative<det_pair_t>(histoBinId));
	const auto detPair = std::get<det_pair_t>(histoBinId);
	return getProjectionValueFromDetPair(detPair);
}

void SparseHistogram::writeToFile(const std::string& filename) const
{
	std::ofstream ofs{filename.c_str(), std::ios::binary | std::ios::out};

	if (!ofs.good())
	{
		throw std::runtime_error("Error opening file " + filename);
	}

	constexpr long long sizeOfAnEvent_bytes =
	    sizeof(det_pair_t) + sizeof(float);

	constexpr long long bufferSize_fields = (1ll << 30);
	auto buff = std::make_unique<float[]>(bufferSize_fields);
	constexpr int numFieldsPerEvent = 3;
	static_assert(numFieldsPerEvent * sizeof(float) == sizeOfAnEvent_bytes);
	const long long numEvents = count();

	long long posStart_events = 0;
	while (posStart_events < numEvents)
	{
		const long long writeSize_fields =
		    std::min(bufferSize_fields,
		             numFieldsPerEvent * (numEvents - posStart_events));
		const long long writeSize_events = writeSize_fields / numFieldsPerEvent;

		for (long long i = 0; i < writeSize_events; i++)
		{
			std::memcpy(&buff[numFieldsPerEvent * i + 0], &m_detPairs[i].d1,
			            sizeof(det_id_t));
			std::memcpy(&buff[numFieldsPerEvent * i + 1], &m_detPairs[i].d2,
			            sizeof(det_id_t));
			buff[numFieldsPerEvent * i + 2] = m_projValues[i];
		}

		ofs.write(reinterpret_cast<char*>(buff.get()),
		          sizeOfAnEvent_bytes * writeSize_events);

		posStart_events += writeSize_events;
	}

	ofs.close();
}

void SparseHistogram::readFromFile(const std::string& filename)
{
	std::ifstream ifs{filename, std::ios::in | std::ios::binary};

	if (!ifs.good())
	{
		throw std::runtime_error("Error reading input file " + filename);
	}

	constexpr long long sizeOfAnEvent_bytes =
	    sizeof(det_pair_t) + sizeof(float);

	// Check that file has a proper size:
	const long long fileSize_bytes =
	    static_cast<long long>(fs::file_size(filename));

	if (fileSize_bytes <= 0 || (fileSize_bytes % sizeOfAnEvent_bytes) != 0)
	{
		throw std::runtime_error("Error: Input file has incorrect size in "
		                         "SparseHistogram::readFromFile.");
	}
	// Compute the number of events using its size
	const long long numEvents = fileSize_bytes / sizeOfAnEvent_bytes;

	// Allocate the memory
	allocate(numEvents);

	// Prepare buffer of 3 4-byte fields
	constexpr long long bufferSize_fields = (1ll << 30);
	auto buff = std::make_unique<float[]>(bufferSize_fields);
	constexpr long long numFieldsPerEvent = 3ll;
	static_assert(numFieldsPerEvent * sizeof(float) == sizeOfAnEvent_bytes);

	long long posStart_events = 0;
	while (posStart_events < numEvents)
	{
		const long long readSize_fields =
		    std::min(bufferSize_fields,
		             numFieldsPerEvent * (numEvents - posStart_events));
		const long long readSize_events = readSize_fields / numFieldsPerEvent;

		ifs.read(reinterpret_cast<char*>(buff.get()),
		         sizeOfAnEvent_bytes * readSize_events);

		for (long long i = 0; i < readSize_events; i++)
		{
			const det_id_t d1 = *reinterpret_cast<det_id_t*>(
			    &buff[numFieldsPerEvent * i + 0ll]);
			const det_id_t d2 = *reinterpret_cast<det_id_t*>(
			    &buff[numFieldsPerEvent * i + 1ll]);
			const float projValue = buff[numFieldsPerEvent * i + 2ll];
			accumulate({d1, d2}, projValue);
		}

		posStart_events += readSize_events;
	}
	ifs.close();
}

float* SparseHistogram::getProjectionValuesBuffer()
{
	return m_projValues.data();
}

det_pair_t* SparseHistogram::getDetectorPairBuffer()
{
	return m_detPairs.data();
}

const float* SparseHistogram::getProjectionValuesBuffer() const
{
	return m_projValues.data();
}

const det_pair_t* SparseHistogram::getDetectorPairBuffer() const
{
	return m_detPairs.data();
}

det_pair_t SparseHistogram::SwapDetectorPairIfNeeded(det_pair_t detPair)
{
	auto [d1, d2] = std::minmax({detPair.d1, detPair.d2});
	return det_pair_t{d1, d2};
}

std::unique_ptr<ProjectionData>
    SparseHistogram::create(const Scanner& scanner, const std::string& filename,
                            const Plugin::OptionsResult& pluginOptions)
{
	(void)pluginOptions;
	return std::make_unique<SparseHistogram>(scanner, filename);
}

Plugin::OptionsListPerPlugin SparseHistogram::getOptions()
{
	return {};
}

REGISTER_PROJDATA_PLUGIN("SH", SparseHistogram, SparseHistogram::create,
                         SparseHistogram::getOptions)
