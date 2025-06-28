/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/Timer.hpp"

namespace Util
{
	Timer::Timer() : m_running{false}
	{
		reset();
	}

	void Timer::reset()
	{
		m_elapsedTime = duration_t::zero();
	}

	void Timer::start()
	{
		reset();
		resume();
	}

	void Timer::pause()
	{
		m_elapsedTime +=
		    std::chrono::high_resolution_clock::now() - m_latestTime;
		m_running = false;
	}

	void Timer::resume()
	{
		m_latestTime = std::chrono::high_resolution_clock::now();
		m_running = true;
	}

	Timer::duration_t Timer::getElapsedTime() const
	{
		if (m_running)
		{
			return std::chrono::high_resolution_clock::now() - m_latestTime;
		}
		return m_elapsedTime;
	}

}  // namespace Util
