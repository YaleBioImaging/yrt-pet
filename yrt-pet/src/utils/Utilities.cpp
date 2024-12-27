/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/Utilities.hpp"

#include "utils/Assert.hpp"
#include "utils/Types.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

void py_setup_utilities(py::module& m)
{
	m.def("compiledWithCuda", &Util::compiledWithCuda);

	auto c_transform = py::class_<transform_t>(m, "transform_t");
	c_transform.def(py::init(
	                    [](const py::tuple& transformTuple)
	                    {
		                    ASSERT_MSG(transformTuple.size() == 2,
		                               "Transform tuple misformed");
		                    const auto rotationTuple =
		                        py::cast<py::tuple>(transformTuple[0]);
		                    ASSERT_MSG(rotationTuple.size() == 9,
		                               "Transform tuple misformed in rotation");
		                    const auto translationTuple =
		                        py::cast<py::tuple>(transformTuple[1]);
		                    ASSERT_MSG(
		                        translationTuple.size() == 3,
		                        "Transform tuple misformed in translation");

		                    transform_t transform{};
		                    transform.r00 = py::cast<float>(rotationTuple[0]);
		                    transform.r01 = py::cast<float>(rotationTuple[1]);
		                    transform.r02 = py::cast<float>(rotationTuple[2]);
		                    transform.r10 = py::cast<float>(rotationTuple[3]);
		                    transform.r11 = py::cast<float>(rotationTuple[4]);
		                    transform.r12 = py::cast<float>(rotationTuple[5]);
		                    transform.r20 = py::cast<float>(rotationTuple[6]);
		                    transform.r21 = py::cast<float>(rotationTuple[7]);
		                    transform.r22 = py::cast<float>(rotationTuple[8]);
		                    transform.tx = py::cast<float>(translationTuple[0]);
		                    transform.ty = py::cast<float>(translationTuple[1]);
		                    transform.tz = py::cast<float>(translationTuple[2]);

		                    return transform;
	                    }),
	                "transformTuple"_a);
	c_transform.def("toTuple",
	                [](const transform_t& self)
	                {
		                return py::make_tuple(
		                    py::make_tuple(self.r00, self.r01, self.r02,
		                                   self.r10, self.r11, self.r12,
		                                   self.r20, self.r21, self.r22),
		                    py::make_tuple(self.tx, self.ty, self.tz));
	                });
	c_transform.def("getTranslation", [](const transform_t& self)
	                { return py::make_tuple(self.tx, self.ty, self.tz); });
	c_transform.def("getRotationMatrix",
	                [](const transform_t& self)
	                {
		                return py::make_tuple(self.r00, self.r01, self.r02,
		                                      self.r10, self.r11, self.r12,
		                                      self.r20, self.r21, self.r22);
	                });

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
	bool beginWithNonWhitespace(const std::string& input)
	{
		const std::string whiteSpace(" \t");
		return (input.find_first_not_of(whiteSpace) == 0);
	}

	std::string getDatetime()
	{
		const auto now = std::chrono::system_clock::now();

		// Convert the time point to a time_t, which represents the calendar
		// time
		const std::time_t now_time_t =
		    std::chrono::system_clock::to_time_t(now);

		// Convert the time_t to a tm structure, which represents the calendar
		// time broken down into components
		const std::tm now_tm = *std::localtime(&now_time_t);

		// Print the current date and time in a human-readable format
		char buffer[80];
		std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", &now_tm);
		return std::string{buffer};
	}

	/*
	  remove both the leading and trailing whitespaces
	*/
	std::string stripWhitespaces(const std::string& input)
	{  // From Jinsong Ouyang's class : MISoftware.cpp

		const std::string whiteSpace(" \t");

		int idxBeg = input.find_first_not_of(whiteSpace);
		int idxEnd = input.find_last_not_of(whiteSpace);

		int len;

		if (idxEnd > 0)
			len = idxEnd - idxBeg + 1;
		else
			len = input.size() - idxBeg;

		std::string output;

		if (idxBeg >= 0)
			output = input.substr(idxBeg, len);
		else
			output = "";

		return output;
	}

	bool equalsIgnoreCase(const char s1[], const char s2[])
	{
		return equalsIgnoreCase(std::string(s1), std::string(s2));
	}

	bool equalsIgnoreCase(const std::string& s1, const std::string& s2)
	{
		// convert s1 and s2 into lower case strings
		std::string str1 = s1;
		std::string str2 = s2;
		std::transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
		std::transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
		// std::cout << "str1: " << str1 << ", str2: " << str2 << std::endl;
		if (str1.compare(str2) == 0)
			return true;  // The strings are same
		return false;     // not matched
	}

	std::string getSizeWithSuffix(double size, int precision)
	{
		int i = 0;
		const std::string units[] = {"B",  "kB", "MB", "GB", "TB",
		                             "PB", "EB", "ZB", "YB"};
		while (size > 1024)
		{
			size /= 1024;
			i++;
		}
		std::stringstream ss;
		ss << std::setprecision(precision) << std::fixed << size << " "
		   << units[i];
		return ss.str();
	}

	std::string toLower(const std::string& s)
	{
		std::string newString = s;
		std::transform(s.begin(), s.end(), newString.begin(), ::tolower);
		return newString;
	}

	std::string toUpper(const std::string& s)
	{
		std::string newString = s;
		std::transform(s.begin(), s.end(), newString.begin(), ::toupper);
		return newString;
	}

	std::vector<std::string> split(const std::string str,
	                               const std::string regex_str)
	{
		std::regex regexz(regex_str);
		return {std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
		        std::sregex_token_iterator()};
	}

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
		// Case 1: If the vector is empty, just insert the new range
		if (ranges.empty())
		{
			ranges.push_back({begin, end});
			return;
		}

		// Step 1: Insert the new range in sorted order
		auto it = ranges.begin();
		while (it != ranges.end() && it->first < begin)
		{
			++it;
		}
		ranges.insert(it, {begin, end});

		// Step 2: Merge overlapping or adjacent ranges
		std::vector<std::pair<int, int>> mergedRanges;
		mergedRanges.push_back(ranges[0]);

		for (size_t i = 1; i < ranges.size(); ++i)
		{
			auto& last = mergedRanges.back();
			auto& current = ranges[i];

			// If the current range overlaps with or is adjacent to the last
			// range, merge them
			if (current.first <= last.second + 1)
			{
				last.second = std::max(last.second, current.second);
			}
			else
			{
				mergedRanges.push_back(current);
			}
		}

		// Replace the original vector with the merged ranges
		ranges = mergedRanges;
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

	bool compiledWithCuda()
	{
#if BUILD_CUDA
		return true;
#else
		return false;
#endif
	}

	/* clang-format off */

template<>
uint8_t generateMask<uint8_t>(unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint16_t generateMask<uint16_t>(unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint32_t generateMask<uint32_t>(unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint64_t generateMask<uint64_t>(unsigned int pMSBLimit, unsigned int pLSBLimit);

template<>
uint8_t truncateBits<uint8_t,uint8_t>(uint8_t pCode, unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint16_t truncateBits<uint16_t,uint16_t>(uint16_t pCode, unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint32_t truncateBits<uint32_t,uint32_t>(uint32_t pCode, unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint64_t truncateBits<uint32_t,uint64_t>(uint32_t pCode, unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint32_t truncateBits<uint64_t,uint32_t>(uint64_t pCode, unsigned int pMSBLimit, unsigned int pLSBLimit);
template<>
uint64_t truncateBits<uint64_t,uint64_t>(uint64_t pCode, unsigned int pMSBLimit, unsigned int pLSBLimit);

template<>
void setBits<uint8_t>(uint8_t& pCode, unsigned int pInsertionMSBLimit, unsigned int pInsertionLSBLimit, uint8_t pToInsert);
template<>
void setBits<uint16_t>(uint16_t& pCode, unsigned int pInsertionMSBLimit, unsigned int pInsertionLSBLimit, uint16_t pToInsert);
template<>
void setBits<uint32_t>(uint32_t& pCode, unsigned int pInsertionMSBLimit, unsigned int pInsertionLSBLimit, uint32_t pToInsert);
template<>
void setBits<uint64_t>(uint64_t& pCode, unsigned int pInsertionMSBLimit, unsigned int pInsertionLSBLimit, uint64_t pToInsert);

	/* clang-format on */

}  // namespace Util
