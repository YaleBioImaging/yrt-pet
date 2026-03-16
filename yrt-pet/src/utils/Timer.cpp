/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Timer.hpp"
#include "yrt-pet/utils/Assert.hpp"

namespace yrt
{
namespace util
{

Timer::Timer()
{
	reset();
}

void Timer::reset()
{
	m_elapsedTime = duration_t::zero();
	m_running = false;
}

void Timer::run()
{
	ASSERT_MSG(!m_running, "Timer already running");
	m_latestTime = std::chrono::high_resolution_clock::now();
	m_running = true;
}

void Timer::pause()
{
	ASSERT_MSG(m_running, "Timer already paused");
	m_elapsedTime += std::chrono::high_resolution_clock::now() - m_latestTime;
	m_running = false;
}

bool Timer::isRunning() const
{
	return m_running;
}

Timer::duration_t Timer::getElapsedTime() const
{
	if (m_running)
	{
		return std::chrono::high_resolution_clock::now() - m_latestTime;
	}
	return m_elapsedTime;
}

double Timer::getElapsedMilliseconds() const
{
	return std::chrono::duration_cast<
	           std::chrono::duration<double, std::milli>>(getElapsedTime())
	    .count();
}

double Timer::getElapsedSeconds() const
{
	return std::chrono::duration_cast<
	           std::chrono::duration<double, std::ratio<1, 1>>>(
	           getElapsedTime())
	    .count();
}

}  // namespace util
}  // namespace yrt
