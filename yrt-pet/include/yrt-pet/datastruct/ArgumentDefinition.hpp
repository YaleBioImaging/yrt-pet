/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace yrt
{
namespace io
{

enum class TypeOfArgument
{
	NONE,  // Placeholder, should be illegal to use
	STRING,
	INT,
	FLOAT,
	BOOL,
	VECTOR_OF_STRINGS
};

using ArgumentValue = std::variant<std::monostate, std::string, int, float,
                                   bool, std::vector<std::string>>;

// Map: argument name, argument value
using OptionsResult = std::unordered_map<std::string, ArgumentValue>;

// Argument type definitions
struct ArgumentDefinition
{
	std::string name;
	std::string description;
	bool isRequired;
	TypeOfArgument dtype;
	ArgumentValue defaultValue;
	std::string group;            // For command line arg only
	std::string shortOptionName;  // For command line arg only
};

}  // namespace io
}  // namespace yrt
