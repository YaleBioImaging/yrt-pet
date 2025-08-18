/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/Logger.hpp"

namespace yrt
{

template <int LEVEL>
Logger<LEVEL>& Logger<LEVEL>::operator<<(std::ostream& (*manip)(std::ostream&))
{
	if (globals::getVerbosityLevel() >= LEVEL)
	{
		std::cout << manip;
	}
	return *this;
}

template class Logger<0>;  // Not to be used. Only for complete silence
template class Logger<1>;  // Standard
template class Logger<2>;
template class Logger<3>;
template class Logger<4>;
template class Logger<5>;

}  // namespace yrt
