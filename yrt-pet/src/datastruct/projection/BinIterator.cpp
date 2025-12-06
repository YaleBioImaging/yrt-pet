/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/BinIterator.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;
namespace yrt
{
void py_setup_biniterator(py::module& m)
{
	auto c = py::class_<BinIterator>(m, "BinIterator");
	c.def("get", &BinIterator::get, "idx"_a);
	c.def("begin", &BinIterator::begin);
	c.def("end", &BinIterator::end);
	c.def("size", &BinIterator::size);

	auto c_range =
	    py::class_<BinIteratorRange, BinIterator>(m, "BinIteratorRange");
	c_range.def(py::init<bin_t>(), "num"_a);
	c_range.def(py::init<bin_t, bin_t, bin_t>(), "idxStart"_a, "idxEnd"_a,
	            "idxStride"_a = 1);

	auto c_vector =
	    py::class_<BinIteratorVector, BinIterator>(m, "BinIteratorVector");
	c_vector.def(py::init(
	                 [](const std::vector<bin_t>& vec)
	                 {
		                 auto idxs = std::make_unique<std::vector<bin_t>>(vec);
		                 return BinIteratorVector(idxs);
	                 }),
	             "indices"_a);

	auto c_chronological =
	    py::class_<BinIteratorChronological, BinIteratorRange>(
	        m, "BinIteratorChronological");
	c_chronological.def(py::init<bin_t, bin_t, bin_t>(), "numSubsets"_a,
	                    "numEvents"_a, "idxSubset"_a);

	auto c_chronological_interleaved =
	    py::class_<BinIteratorChronologicalInterleaved, BinIteratorRange>(
	        m, "BinIteratorChronologicalInterleaved");
	c_chronological_interleaved.def(py::init<bin_t, bin_t, bin_t>(),
	                                "numSubsets"_a, "numEvents"_a,
	                                "idxSubset"_a);
}
}  // namespace yrt
#endif

namespace yrt
{

bin_t BinIterator::get(bin_t idx) const
{
	if (idx >= size())
	{
		throw std::range_error(
		    "The idx given does not exist in the range of the BinIterator");
	}
	return getSafe(idx);
}

BinIteratorRange::BinIteratorRange(bin_t num)
    : m_idxStart(0), m_idxEnd(num - 1), m_idxStride(1)
{
}

BinIteratorRange::BinIteratorRange(bin_t p_idxStart, bin_t p_idxEnd,
                                   bin_t p_idxStride)
    : m_idxStart(p_idxStart),
      m_idxEnd(getIdxEnd(p_idxStart, p_idxEnd, p_idxStride)),
      m_idxStride(p_idxStride)
{
}

BinIteratorRange::BinIteratorRange(std::tuple<bin_t, bin_t, bin_t> info)
    : m_idxStart(std::get<0>(info)),
      m_idxEnd(
          getIdxEnd(std::get<0>(info), std::get<1>(info), std::get<2>(info))),
      m_idxStride(std::get<2>(info))
{
}

bin_t BinIteratorRange::getIdxEnd(bin_t idxStart, bin_t idxEnd, bin_t stride)
{
	return idxStart + stride * ((idxEnd - idxStart) / stride);
}

bin_t BinIteratorRange::begin() const
{
	return m_idxStart;
}

bin_t BinIteratorRange::end() const
{
	return m_idxEnd;
}

bin_t BinIteratorRange::getSafe(bin_t idx) const
{
	return m_idxStart + m_idxStride * idx;
}

size_t BinIteratorRange::size() const
{
	return (m_idxEnd - m_idxStart) / m_idxStride + 1;
}

BinIteratorRange2D::BinIteratorRange2D(bin_t p_idxStart, bin_t p_numSlices,
                                       bin_t p_sliceSize, bin_t p_idxStride)
    : m_idxStart(p_idxStart),
      m_numSlices(p_numSlices),
      m_sliceSize(p_sliceSize),
      m_idxStride(p_idxStride)
{
}

bin_t BinIteratorRange2D::begin() const
{
	return m_idxStart;
}

bin_t BinIteratorRange2D::end() const
{
	return m_idxStart + m_numSlices * m_idxStride;
}

size_t BinIteratorRange2D::size() const
{
	return m_numSlices * m_sliceSize - 1;
}

bin_t BinIteratorRange2D::getSafe(bin_t idx) const
{
	bin_t sliceIdx = idx / m_sliceSize;
	bin_t idxOffset = idx % m_sliceSize;
	return m_idxStart + m_idxStride * sliceIdx + idxOffset;
}

BinIteratorRangeHistogram3D::BinIteratorRangeHistogram3D(size_t p_numZBin,
                                                         size_t p_numPhi,
                                                         size_t p_numR,
                                                         int p_numSubsets,
                                                         int p_idxSubset)
    : m_numZBin(p_numZBin),
      m_numPhi(p_numPhi),
      m_numR(p_numR),
      m_numSubsets(p_numSubsets),
      m_idxSubset(p_idxSubset)
{
	m_phiStride = m_numSubsets;
	m_phi0 = m_idxSubset;
	m_numPhiSubset = m_numPhi / m_numSubsets;  // Number of rs in the subset
	// In the case that we would miss some bins because of the "floor" division
	// above
	if (m_phi0 + m_numPhiSubset * m_phiStride < m_numPhi)
	{
		m_numPhiSubset += 1;
	}
	m_histoSize = m_numR * m_numPhiSubset * m_numZBin;
}

bin_t BinIteratorRangeHistogram3D::begin() const
{
	bin_t r = 0;
	bin_t phi = m_phi0;
	bin_t z_bin = 0;
	return z_bin * m_numPhi * m_numR + phi * m_numR + r;
}

bin_t BinIteratorRangeHistogram3D::end() const
{
	bin_t r = m_numR - 1;
	bin_t phi = (m_phiStride * (m_numPhiSubset - 1)) + m_phi0;
	bin_t z_bin = (m_numZBin - 1);
	return z_bin * m_numPhi * m_numR + phi * m_numR + r;
}

size_t BinIteratorRangeHistogram3D::size() const
{
	return m_histoSize;
}

bin_t BinIteratorRangeHistogram3D::getSafe(bin_t idx) const
{
	bin_t z_bin = idx / (m_numPhiSubset * m_numR);
	bin_t phi = (idx % (m_numPhiSubset * m_numR)) / m_numR;
	bin_t r = (idx % (m_numPhiSubset * m_numR)) % m_numR;
	phi = m_phiStride * phi + m_phi0;  // scale and shift the phi coordinate
	return z_bin * m_numPhi * m_numR + phi * m_numR + r;
}


BinIteratorVector::BinIteratorVector(
    std::unique_ptr<std::vector<bin_t>>& p_idxList)
{
	m_idxList = std::move(p_idxList);
}

bin_t BinIteratorVector::begin() const
{
	return (*m_idxList.get())[0];
}

bin_t BinIteratorVector::end() const
{
	return (*m_idxList.get())[m_idxList->size() - 1];
}

bin_t BinIteratorVector::getSafe(bin_t idx) const
{
	return (*m_idxList.get())[idx];
}

size_t BinIteratorVector::size() const
{
	return m_idxList->size();
}

BinIteratorChronological::BinIteratorChronological(bin_t p_numSubsets,
                                                   bin_t p_numEvents,
                                                   bin_t p_idxSubset)
    : BinIteratorRange(getSubsetRange(p_numSubsets, p_numEvents, p_idxSubset))
{
}

std::tuple<bin_t, bin_t, bin_t>
    BinIteratorChronological::getSubsetRange(bin_t numSubsets, bin_t numEvents,
                                             bin_t idxSubset)
{
	if (idxSubset > numSubsets)
	{
		throw std::invalid_argument("The number of subsets has to be higher "
		                            "than the desired subset index.");
	}
	const bin_t rest = numEvents % numSubsets;

	bin_t idxStart = ((numEvents - rest) * idxSubset) / numSubsets;
	bin_t idxEnd;

	if (idxSubset == numSubsets - 1)
	{
		// the last numBins % numSubsets are added here
		idxEnd =
		    (((numEvents - rest) * (idxSubset + 1)) / numSubsets + rest) - 1;
	}
	else
	{
		idxEnd = (((numEvents - rest) * (idxSubset + 1)) / numSubsets) - 1;
	}
	return std::make_tuple(idxStart, idxEnd, 1);
}

BinIteratorChronologicalInterleaved::BinIteratorChronologicalInterleaved(
    bin_t p_numSubsets, bin_t p_numEvents, bin_t p_idxSubset)
    : BinIteratorRange(p_idxSubset, p_numEvents - 1, p_numSubsets)
{
}

}  // namespace yrt
