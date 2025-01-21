/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/ProgressDisplayMultiThread.hpp"
#include "utils/Assert.hpp"
#include "utils/ProgressDisplay.hpp"

#include <iostream>

namespace Util
{
	ProgressDisplayMultiThread::ProgressDisplayMultiThread(int p_numThreads,
	                                                       int64_t p_totalWork,
	                                                       int64_t p_increment)
	    : m_lastDisplayedPercentage{-1},
	      m_totalWork{p_totalWork},
	      m_increment{p_increment}
	{
		setNumThreads(p_numThreads);
	}

	void ProgressDisplayMultiThread::setTotalWork(int64_t p_totalWork)
	{
		m_totalWork = p_totalWork;
	}

	void ProgressDisplayMultiThread::setNumThreads(int numThreads)
	{
		m_progressPerThread.resize(numThreads, 0);
	}

	int ProgressDisplayMultiThread::getNumThreads() const
	{
		return static_cast<int>(m_progressPerThread.size());
	}

	void ProgressDisplayMultiThread::reset()
	{
		std::fill(m_progressPerThread.begin(), m_progressPerThread.end(), 0);
		m_lastDisplayedPercentage = -1;
	}

	void ProgressDisplayMultiThread::progress(int threadId,
	                                          int64_t progressStep)
	{
		ASSERT(m_totalWork > 0);
		m_progressPerThread[threadId] += progressStep;

		// Only use thread 0 to compute progress bar increment
		if (threadId == 0)
		{
			int64_t currentWork = 0;
			for (int thread = 0; thread < getNumThreads(); ++thread)
			{
				currentWork += m_progressPerThread[thread];
			}

			const int8_t newPercentage = ProgressDisplay::getNewPercentage(
			    currentWork, m_totalWork, m_lastDisplayedPercentage,
			    m_increment);
			if (newPercentage >= 0)
			{
				m_lastDisplayedPercentage = newPercentage;
				std::cout << "Progress: " << static_cast<int>(newPercentage)
				          << "%" << std::endl;
			}
		}
	}

}  // namespace Util