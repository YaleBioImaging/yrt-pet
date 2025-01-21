/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Util
{
	/*Warning: This class is only made to work for command-line output, not for
	 * accumulating the output in a file (ex: SLURM)*/
	class ProgressDisplayMultiThread
	{
	public:
		explicit ProgressDisplayMultiThread(int p_numThreads,
		                                    int64_t p_totalWork = -1,
		                                    int64_t p_increment = 10);

		void reset();
		void setTotalWork(int64_t p_totalWork);
		void setNumThreads(int numThreads);
		int getNumThreads() const;
		void progress(int threadId, int64_t progressStep);

	private:
		void displayPercentageIfNeeded() const;

		// In percentages, so never higher than 100
		int8_t m_lastDisplayedPercentage;

		// In the same units as m_totalWork
		std::vector<int64_t> m_progressPerThread;

		// Total work to do
		int64_t m_totalWork;

		// Increment in percentage
		int64_t m_increment;
	};

}  // namespace Util
