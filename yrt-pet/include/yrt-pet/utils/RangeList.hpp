/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Utilities.hpp"
#include <iostream>
#include <vector>

namespace yrt
{
namespace util
{
class RangeList
{
protected:
	std::vector<std::pair<int, int>> m_Ranges;

public:
	RangeList() {}
	explicit RangeList(const std::string& p_Ranges);
	void readFromString(const std::string& p_Ranges);
	static void insertSorted(std::vector<std::pair<int, int>>& ranges,
	                         const int begin, const int end);
	void insertSorted(const int begin, const int end);
	void sort();
	const std::vector<std::pair<int, int>>& get() const;
	size_t getSizeTotal() const;
	bool isIn(int idx) const;
	bool empty() const;
	friend std::ostream& operator<<(std::ostream& os, const RangeList& ranges)
	{
		for (auto& range : ranges.get())
		{
			if (&range != &ranges.get().front())
			{
				os << ", ";
			}
			os << "[" << range.first << "," << range.second << "]";
		}
		return os;
	}
};
}  // namespace util
}  // namespace yrt
