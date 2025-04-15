/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <string>
#include <variant>
#include <vector>

namespace IO
{
	using ArgumentValue =
	    std::variant<std::string, int, float, bool, std::vector<std::string>>;

	// Argument type definitions
	struct ArgumentDefinition
	{
		std::string name;
		std::string description;
		bool isRequired;
		ArgumentValue defaultValue;
		std::string group;            // For command line arg only
		std::string shortOptionName;  // For command line arg only
	};
}  // namespace IO
