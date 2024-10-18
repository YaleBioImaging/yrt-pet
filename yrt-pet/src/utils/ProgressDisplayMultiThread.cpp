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
	ProgressDisplayMultiThread::ProgressDisplayMultiThread(int64_t p_numThreads,
	                                                       int64_t p_totalWork,
	                                                       int64_t p_increment)
	    : m_totalWork{p_totalWork}, m_increment{p_increment}
	{
		setNumThreads(p_numThreads);
	}

	void ProgressDisplayMultiThread::setTotalWork(int64_t p_totalWork)
	{
		m_totalWork = p_totalWork;
	}

	void ProgressDisplayMultiThread::setNumThreads(int numThreads)
	{
		m_lastDisplayedPercentages.resize(numThreads, 0);
		m_progressPerThread.resize(numThreads, 0);
	}

	int ProgressDisplayMultiThread::getNumThreads() const
	{
		return static_cast<int>(m_lastDisplayedPercentages.size());
	}

	void ProgressDisplayMultiThread::start() const
	{
		std::cout << "\033[2J";
	}

	void ProgressDisplayMultiThread::finish() const
	{
		const int numThreads = getNumThreads();
		for (int i = 0; i < numThreads; i++)
		{
			displayPercentage(i, 100);
		}
		std::cout << moveCursor(numThreads + 2);
	}

	void ProgressDisplayMultiThread::progress(int threadId,
	                                          int64_t progressStep)
	{
		ASSERT(m_totalWork > 0);
		m_progressPerThread[threadId] += progressStep;

		const int8_t newPercentage = ProgressDisplay::getNewPercentage(
		    m_progressPerThread[threadId], getWorkPerThread(),
		    m_lastDisplayedPercentages[threadId], m_increment);
		if (newPercentage > 0)
		{
			m_lastDisplayedPercentages[threadId] = newPercentage;

			displayPercentage(threadId, newPercentage);
		}
	}

	int64_t ProgressDisplayMultiThread::getWorkPerThread() const
	{
		return m_totalWork / getNumThreads() + 1;
	}

	void ProgressDisplayMultiThread::displayPercentage(int threadId,
	                                                   int8_t percentage) const
	{
		constexpr int barWidth = 50;
		const int pos = percentage * barWidth / 100;
		std::string bar;
		for (int i = 0; i < barWidth; ++i)
		{
			if (i < pos)
				bar += "=";
			else if (i == pos)
				bar += ">";
			else
				bar += " ";
		}

		// Print the progress bar for this thread (thread-specific line)
		const std::string line = moveCursor(threadId + 1) + "\rThread " +
		                         std::to_string(threadId) + ": [" + bar + "] " +
		                         std::to_string(static_cast<int>(percentage)) +
		                         " %   ";
		std::cout << line << std::flush;
	}

	std::string ProgressDisplayMultiThread::moveCursor(int row)
	{
		return "\033[" + std::to_string(row) + ";0H";
	}

}  // namespace Util