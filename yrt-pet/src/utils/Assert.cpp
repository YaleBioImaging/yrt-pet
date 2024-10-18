/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <iostream>

#include "utils/Assert.hpp"

void assertion_failed(char const* expr, char const* function, char const* file,
                      long line, bool critical)
{
	std::string fullErrorMessage;
	fullErrorMessage += "Assertion failed: (" + std::string{expr} + ") in " +
	                    file + ":" + std::to_string(line) + ": " + function;

	if (critical)
	{
		throw std::runtime_error("Critial error: " + fullErrorMessage);
	}
	else
	{
		std::cerr << "Warning: " << fullErrorMessage << std::endl;
	}
}
void assertion_failed_msg(char const* expr, char const* msg,
                          char const* function, char const* file, long line,
                          bool critical)
{
	std::string fullErrorMessage;
	if (critical)
	{
		fullErrorMessage += "Critial error: ";
	}
	else
	{
		fullErrorMessage += "Warning: ";
	}
	fullErrorMessage += msg;

	fullErrorMessage += "\n\nDetails: assertion failed: (" + std::string{expr} +
	                    ") in " + file + ":" + std::to_string(line) + ": " +
	                    function;

	if (critical)
	{
		throw std::runtime_error(fullErrorMessage);
	}
	else
	{
		std::cerr << fullErrorMessage << std::endl;
	}
}

void Util::printExceptionMessage(const std::exception& e)
{
	constexpr auto delim = "===================";
	std::cerr << "Exception caught:\n"
	          << delim << "\n"
	          << e.what() << "\n"
	          << delim << std::endl;
}
