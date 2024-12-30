/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/RangeList.hpp"
#include <sstream>
#include <algorithm>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

void py_setup_utilities_rangelist(py::module& m)
{
	auto m_utils = m.def_submodule("Utilities");
	auto c_range = py::class_<Util::RangeList>(m_utils, "RangeList");
	c_range.def(py::init<>());
	c_range.def(py::init<const std::string&>());
	c_range.def("isIn", &Util::RangeList::isIn, py::arg("idx"));
	c_range.def("empty", &Util::RangeList::empty);
	c_range.def("insertSorted", [](Util::RangeList & self, int begin, int end)
	{ self.insertSorted(begin, end); }, py::arg("begin"), py::arg("end"));
	c_range.def("__repr__",
	            [](const Util::RangeList& self)
	            {
		            std::stringstream ss;
		            ss << self;
		            return ss.str();
	            });
	c_range.def("__getitem__",
	            [](const Util::RangeList& self, const int idx)
	            {
		            const std::pair<int, int>& range = self.get().at(idx);
		            return py::make_tuple(range.first, range.second);
	            });
}
#endif

namespace Util
{
	RangeList::RangeList(const std::string& p_Ranges)
	{
		readFromString(p_Ranges);
	}
	void RangeList::readFromString(const std::string& p_Ranges)
	{
		std::vector<std::string> ranges = split(p_Ranges, ",");
		for (std::string range : ranges)
		{
			std::vector<std::string> limits = split(range, "-");
			int begin, end;
			switch (limits.size())
			{
			case 1: begin = end = std::stoi(limits[0]); break;
			case 2:
				begin = std::stoi(limits[0]);
				end = std::stoi(limits[1]);
				break;
			default: std::cerr << "Could not parse range" << std::endl; return;
			}
			insertSorted(begin, end);
		}
	}
	void RangeList::sort()
	{
		std::vector<std::pair<int, int>> newRanges;
		for (auto range : m_Ranges)
		{
			RangeList::insertSorted(newRanges, range.first, range.second);
		}
		m_Ranges = newRanges;
	}
	void RangeList::insertSorted(const int begin, const int end)
	{
		insertSorted(m_Ranges, begin, end);
	}
	void RangeList::insertSorted(std::vector<std::pair<int, int>>& ranges,
	                             const int begin, const int end)
	{
		// Sweep-line
		// Step 1. Build list of events (+1 begin, -1 end)
		std::vector<std::pair<int, bool>> events;
		for (auto range : ranges)
		{
			events.push_back({range.first, true});
			events.push_back({range.second, false});
		}
		events.push_back({begin, true});
		events.push_back({end, false});
		// Step 2. Sort events
		std::sort(events.begin(), events.end(),
		          [](std::pair<int, bool>& a, std::pair<int, bool>& b)
		          {
			          return (a.first < b.first) ||
			                 (a.first == b.first && a.second && !b.second);
		          });
		// Step 3. Build list
		ranges.clear();
		if (!events.empty())
		{
			auto it = events.begin();
			auto itNext = it;
			itNext++;
			assert(it->second);
			int begin = it->first;
			int end = -1;
			int count = 1;
			it++;
			bool newInterval = false;
			while (it != events.end())
			{
				itNext++;
				count += it->second? 1: -1;
				if (newInterval)
				{
					begin = it->first;
					newInterval = false;
				}
				if (count == 0 && (itNext == events.end() ||
				                   itNext->first > it->first + 1))
				{
					end = it->first;
					ranges.push_back({begin, end});
					newInterval = true;
				}
				it++;
			}
		}
	}

	const std::vector<std::pair<int, int>>& RangeList::get() const
	{
		return m_Ranges;
	}

	size_t RangeList::getSizeTotal() const
	{
		size_t size = 0;
		for (auto range : m_Ranges)
		{
			size += range.second - range.first + 1;
		}
		return size;
	}

	bool RangeList::isIn(int idx) const
	{
		for (auto range : m_Ranges)
		{
			if (idx >= range.first and idx <= range.second)
			{
				return true;
			}
		}
		return false;
	}
	bool RangeList::empty() const
	{
		return m_Ranges.size() == 0;
	}
}
