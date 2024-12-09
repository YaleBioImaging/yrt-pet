/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ListMode.hpp"
#include "utils/Assert.hpp"
#include "utils/Types.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_listmode(py::module& m)
{
	auto c = py::class_<ListMode, ProjectionData>(m, "ListMode");
	c.def("getProjectionValue", &ListMode::getProjectionValue);
	c.def("setProjectionValue", &ListMode::setProjectionValue);
	c.def("getBinIter", &ListMode::getBinIter);
}

#endif  // if BUILD_PYBIND11

ListMode::ListMode(const Scanner& pr_scanner) : ProjectionData{pr_scanner} {}

float ListMode::getProjectionValue(bin_t id) const
{
	(void)id;
	return 1.0f;
}

void ListMode::setProjectionValue(bin_t id, float val)
{
	(void)id;
	(void)val;
	throw std::logic_error("setProjectionValue unimplemented");
}

timestamp_t ListMode::getScanDuration() const
{
	// By default, return timestamp of the last event - timestamp of first event
	return getTimestamp(count() - 1) - getTimestamp(0);
}

std::unique_ptr<BinIterator> ListMode::getBinIter(int numSubsets,
                                                  int idxSubset) const
{
	ASSERT_MSG(idxSubset < numSubsets,
	           "The subset index has to be smaller than the number of subsets");
	ASSERT_MSG(
	    idxSubset >= 0 && numSubsets > 0,
	    "The subset index cannot be negative, the number of subsets cannot "
	    "be less than or equal to zero");

	size_t numEvents = count();
	return std::make_unique<BinIteratorChronological>(numSubsets, numEvents,
	                                                  idxSubset);
}
