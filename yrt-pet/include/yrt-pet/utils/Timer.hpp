/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <chrono>
#include <ctime>
#include <iomanip>

namespace yrt
{
namespace util
{

class Timer
{
	using walltime_t = std::chrono::time_point<std::chrono::system_clock>;

	using duration_t =
	    std::chrono::duration<signed long int, std::ratio<1, 1000000000>>;

public:
	Timer();

	void reset();
	void run();
	void pause();

	duration_t getElapsedTime() const;
	double getElapsedMilliseconds() const;
	bool isRunning() const;

private:
	bool m_running;
	duration_t m_elapsedTime;
	walltime_t m_latestTime;
};

}  // namespace util
}  // namespace yrt
