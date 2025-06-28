/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <chrono>
#include <ctime>
#include <iomanip>

namespace Util
{
	class Timer
	{
		using walltime_t = std::chrono::time_point<std::chrono::system_clock>;

		using duration_t =
		    std::chrono::duration<signed long int, std::ratio<1, 1000000000>>;

	public:
		Timer();

		void reset();
		void start();
		void pause();
		void resume();

		duration_t getElapsedTime() const;

	private:
		bool m_running;
		duration_t m_elapsedTime;
		walltime_t m_latestTime;
	};
}  // namespace Util
