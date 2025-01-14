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
		explicit ProgressDisplayMultiThread(int64_t p_numThreads,
		                                    int64_t p_totalWork = -1,
		                                    int64_t p_increment = 10);

		void setTotalWork(int64_t p_totalWork);
		void setNumThreads(int numThreads);
		int getNumThreads() const;
		void start() const;
		void finish() const;
		void progress(int threadId, int64_t progressStep);
		int64_t getWorkPerThread() const;

	private:
		// TODO: Make this display compatible with log files
		void displayPercentage(int threadId, int8_t percentage) const;
		static int getCurrentCursorRow();
		static std::string moveCursor(int row);

		// In percentages, so never higher than 100
		std::vector<int8_t> m_lastDisplayedPercentages;

		// In the same units as m_totalWork
		std::vector<int64_t> m_progressPerThread;
		int64_t m_totalWork;

		// Increment in percentage
		int64_t m_increment;
	};

}  // namespace Util
