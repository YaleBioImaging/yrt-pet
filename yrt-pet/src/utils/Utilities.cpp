/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/Utilities.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_utilities(py::module& m)
{
	m.def("compiledWithCuda", &Util::compiledWithCuda);
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
		const std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

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
