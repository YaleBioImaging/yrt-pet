/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/Types.hpp"

namespace Util
{
	class ProgressDisplay
	{
	public:
		explicit ProgressDisplay(int64_t p_total = -1,
		                         int64_t p_increment = 20);
		void progress(int64_t newProgress);
		void setTotal(int64_t p_total);
		void reset();

		static int8_t getNewPercentage(int64_t newProgress,
		                               int64_t totalProgress,
		                               int8_t lastDisplayedPercentage,
		                               int64_t increment);

	private:
		int64_t m_total;
		int8_t m_lastDisplayedPercentage; // Should never be higher than 100
		int64_t m_increment;
	};
}  // namespace Util
