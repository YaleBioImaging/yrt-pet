/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/ProgressDisplay.hpp"
#include "utils/Assert.hpp"

#include <cassert>
#include <iostream>

namespace Util
{
	ProgressDisplay::ProgressDisplay(int64_t p_total, int64_t p_increment)
	    : m_total(p_total),
	      m_lastDisplayedPercentage(0),
	      m_increment(p_increment)
	{
	}

	void ProgressDisplay::progress(int64_t newProgress)
	{
		if (m_total > 0)
		{
			const int8_t newPercentage = getNewPercentage(
			    newProgress, m_total, m_lastDisplayedPercentage, m_increment);
			if (newPercentage > 0)
			{
				m_lastDisplayedPercentage = newPercentage;
				std::cout << "Progress: " << static_cast<int>(newPercentage)
				          << "%" << std::endl;
			}
		}
	}

	void ProgressDisplay::setTotal(int64_t p_total)
	{
		m_total = p_total;
	}

	void ProgressDisplay::reset()
	{
		m_lastDisplayedPercentage = -1;
	}

	int8_t ProgressDisplay::getNewPercentage(int64_t newProgress,
	                                         int64_t totalProgress,
	                                         int8_t lastDisplayedPercentage,
	                                         int64_t increment)
	{
		ASSERT(newProgress <= totalProgress);
		const int8_t newPercentage = (100 * newProgress) / totalProgress;
		if (newPercentage >= lastDisplayedPercentage + increment ||
		    lastDisplayedPercentage < 0)
		{
			return newPercentage;
		}
		return -1;
	}

}  // namespace Util
