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

	auto c_detpair = py::class_<det_pair_t>(m, "det_pair_t");
	c_detpair.def(py::init(
	                  [](const py::tuple& pair)
	                  {
		                  ASSERT_MSG(pair.size() == 2,
		                             "detector pair misformed");
		                  det_pair_t detPair{};
		                  detPair.d1 = py::cast<det_id_t>(pair[0]);
		                  detPair.d2 = py::cast<det_id_t>(pair[1]);
		                  return detPair;
	                  }),
	              "detector_pair"_a);

	c_detpair.def("toTuple", [](const det_pair_t& self)
	              { return py::make_tuple(self.d1, self.d2); });
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
