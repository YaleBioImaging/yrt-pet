/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>
#include <iostream>
#include <string>

namespace Util
{
	// ----------------- String manipulation -----------------
	bool beginWithNonWhitespace(const std::string& input);
	std::string stripWhitespaces(const std::string& input);
	bool equalsIgnoreCase(const char s1[], const char s2[]);
	bool equalsIgnoreCase(const std::string& s1, const std::string& s2);
	std::string getSizeWithSuffix(double size, int precision);
	std::string toLower(const std::string& s);
	std::string toUpper(const std::string& s);
	std::string getDatetime();


	std::vector<std::string> split(const std::string str,
	                               const std::string regex_str);

	class RangeList
	{
	protected:
		std::vector<std::pair<int, int>> m_Ranges;

	public:
		RangeList() {}
		RangeList(const std::string& p_Ranges);
		void readFromString(const std::string& p_Ranges);
		static void insertSorted(std::vector<std::pair<int, int>>& ranges,
		                         const int begin, const int end);
		void insertSorted(const int begin, const int end);
		void sort();
		const std::vector<std::pair<int, int>>& get() const;
		size_t getSizeTotal() const;
		bool isIn(int idx) const;
		bool empty() const;
		friend std::ostream& operator<<(std::ostream& os,
		                                const RangeList& ranges)
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

	// ----------------- Bit manipulation -----------------

	// This generates a mask of all 1s from bits at position pLSBLimit to
	// pLSBLimit (inclusive)
	template <typename T>
	T generateMask(unsigned int pMSBLimit, unsigned int pLSBLimit)
	{
		return ((1ull << (pMSBLimit + 1ull)) - 1ull) -
		       ((1ull << pLSBLimit) - 1ull);
	}

	// This returns the bits in pCode from pMSBLimit to pLSBLimit (inclusive)
	template <typename TInput, typename TOutput>
	TOutput truncateBits(TInput pCode, unsigned int pMSBLimit,
	                     unsigned int pLSBLimit)
	{
		return (pCode >> static_cast<TInput>(pLSBLimit)) &
		       ((1ull << (pMSBLimit - pLSBLimit + 1)) - 1);
	}

	// This Sets the bits of pCode from pInsertionMSBLimit to pInsertionLSBLimit
	// (inclusive) to the value stored by the LSB bits of pToInsert
	template <typename T>
	void setBits(T& pCode, unsigned int pInsertionMSBLimit,
	             unsigned int pInsertionLSBLimit, T pToInsert)
	{
		// Create the insertion mask
		T lInsertionMask =
		    generateMask<T>(pInsertionMSBLimit, pInsertionLSBLimit);
		// Bitshift and truncate the insertion
		T lToInsert_shifted =
		    (pToInsert << pInsertionLSBLimit) & lInsertionMask;
		// Reset the bits in the insertion area
		T lCode_masked = pCode & (~lInsertionMask);
		// Insert the bits
		pCode = pCode | lToInsert_shifted;
	}

	template <typename TSrc, typename TDst>
	TDst reinterpretAndCast(void* src, int offset = 0)
	{
		return static_cast<TDst>(*(reinterpret_cast<TSrc*>(src) + offset));
	}

	// ----------------- System -----------------
	bool compiledWithCuda();

}  // namespace Util
