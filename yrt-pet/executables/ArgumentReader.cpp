/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "ArgumentReader.hpp"
#include "datastruct/IO.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace IO
{
	std::shared_ptr<cxxopts::Value>
	    ArgumentRegistry::valueTypeToCxxoptsValue(ArgumentValue v)
	{
		if (std::holds_alternative<std::string>(v))
		{
			return cxxopts::value<std::string>();
		}
		if (std::holds_alternative<int>(v))
		{
			return cxxopts::value<int>();
		}
		if (std::holds_alternative<float>(v))
		{
			return cxxopts::value<float>();
		}
		if (std::holds_alternative<bool>(v))
		{
			return cxxopts::value<bool>();
		}
		if (std::holds_alternative<std::vector<std::string>>(v))
		{
			return cxxopts::value<std::vector<std::string>>();
		}
		return nullptr;
	}

	ArgumentValue
	    ArgumentRegistry::castCxxoptsOptionValue(const cxxopts::OptionValue& o,
	                                             TypeOfArgument t)
	{
		if (t == TypeOfArgument::STRING)
		{
			return o.as<std::string>();
		}
		if (t == TypeOfArgument::INT)
		{
			return o.as<int>();
		}
		if (t == TypeOfArgument::FLOAT)
		{
			return o.as<float>();
		}
		if (t == TypeOfArgument::BOOL)
		{
			return o.as<bool>();
		}
		if (t == TypeOfArgument::VECTOR_OF_STRINGS)
		{
			return o.as<std::vector<std::string>>();
		}
		return nullptr;
	}

	std::shared_ptr<cxxopts::Value>
	    ArgumentRegistry::argumentTypeToCxxoptsValue(TypeOfArgument t)
	{
		if (t == TypeOfArgument::STRING)
		{
			return cxxopts::value<std::string>();
		}
		if (t == TypeOfArgument::INT)
		{
			return cxxopts::value<int>();
		}
		if (t == TypeOfArgument::FLOAT)
		{
			return cxxopts::value<float>();
		}
		if (t == TypeOfArgument::BOOL)
		{
			return cxxopts::value<bool>();
		}
		if (t == TypeOfArgument::VECTOR_OF_STRINGS)
		{
			return cxxopts::value<std::vector<std::string>>();
		}
		return nullptr;
	}

	void ArgumentRegistry::registerArgumentInternal(
	    const ArgumentDefinition& arg)
	{
		arguments[arg.name] = arg;
		if (std::find(groups.begin(), groups.end(), arg.group) == groups.end())
		{
			groups.push_back(arg.group);
		}
	}

	void ArgumentRegistry::registerArgument(const std::string& name,
	                                        const std::string& description,
	                                        bool required,
	                                        TypeOfArgument argumentType,
	                                        const std::string& group)
	{
		ASSERT_MSG(argumentType != TypeOfArgument::NONE,
		           "Cannot use None for argument type");

		registerArgumentInternal({name, description, required, argumentType,
		                          std::monostate{}, group, ""});
	}

	const std::map<std::string, ArgumentDefinition>&
	    ArgumentRegistry::getArguments() const
	{
		return arguments;
	}

	const std::vector<std::string>& ArgumentRegistry::getGroups() const
	{
		return groups;
	}

	ArgumentReader::ArgumentReader(const ArgumentRegistry& pr_registry,
	                               const std::string& p_executableName)
	    : mr_registry(pr_registry), m_executableName(p_executableName)
	{
		// Initialize with default values
		for (const auto& [name, arg] : mr_registry.getArguments())
		{
			m_values[name] = arg.defaultValue;
		}
	}

	bool ArgumentReader::loadFromCommandLine(int argc, char** argv)
	{
		cxxopts::Options options(argv[0], m_executableName);
		options.positional_help("[optional args]").show_positional_help();

		setupCommandLineOptions(options);

		const auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return false;
		}

		parseCommandLineResult(result);
		validateRequiredParameters();

		return true;
	}

	void ArgumentReader::loadFromJSON(const std::string& filename)
	{
		std::ifstream file(filename);
		if (!file.is_open())
		{
			throw std::runtime_error("Could not open JSON file: " + filename);
		}

		nlohmann::json j;
		file >> j;
		fromJSON(j);
		validateRequiredParameters();
	}

	void ArgumentReader::saveToJSON(const std::string& filename) const
	{
		std::ofstream file(filename);
		if (!file.is_open())
		{
			throw std::runtime_error("Could not open file for writing: " +
			                         filename);
		}

		file << toJSON().dump(4);
	}

	bool ArgumentReader::validate() const
	{
		try
		{
			validateRequiredParameters();
			return true;
		}
		catch (const std::exception&)
		{
			return false;
		}
	}

	nlohmann::json ArgumentReader::toJSON() const
	{
		nlohmann::json j;
		for (const auto& [name, value] : m_values)
		{
			const auto& arg = mr_registry.getArguments().at(name);
			std::visit(
			    [&](const auto& v)
			    {
				    using T = std::decay_t<decltype(v)>;
				    if constexpr (!std::is_same_v<T, std::monostate>)
				    {
					    j[arg.name] = v;
				    }
			    },
			    value);
		}
		return j;
	}

	void ArgumentReader::fromJSON(const nlohmann::json& j)
	{
		for (const auto& entry : mr_registry.getArguments())
		{
			std::string name = entry.first;
			ArgumentDefinition arg = entry.second;
			if (j.contains(name))
			{
				std::visit(
				    [&](auto& v)
				    {
					    using T = std::decay_t<decltype(v)>;
					    if constexpr (std::is_same_v<T, std::string>)
					    {
						    v = j[name].get<std::string>();
					    }
					    else if constexpr (std::is_same_v<T, int>)
					    {
						    v = j[name].get<int>();
					    }
					    else if constexpr (std::is_same_v<T, float>)
					    {
						    v = j[name].get<float>();
					    }
					    else if constexpr (std::is_same_v<T, bool>)
					    {
						    v = j[name].get<bool>();
					    }
					    else if constexpr (std::is_same_v<
					                           T, std::vector<std::string>>)
					    {
						    v = j[name].get<std::vector<std::string>>();
					    }
				    },
				    m_values[name]);
			}
		}
	}

	const OptionsResult& ArgumentReader::getAllArguments() const
	{
		return m_values;
	}

	void
	    ArgumentReader::setupCommandLineOptions(cxxopts::Options& options) const
	{
		for (const auto& group : mr_registry.getGroups())
		{
			auto groupOptions = options.add_options(group);
			for (auto& [name, arg] : mr_registry.getArguments())
			{
				if (arg.group == group)
				{
					std::string cxxoptsParamName;
					if (!arg.shortOptionName.empty())
					{
						cxxoptsParamName.append(arg.shortOptionName + ",");
					}
					cxxoptsParamName.append(arg.name);

					auto cxxoptsValue =
					    ArgumentRegistry::argumentTypeToCxxoptsValue(arg.dtype);
					groupOptions(cxxoptsParamName, arg.description,
					             cxxoptsValue);
				}
			}
		}

		options.add_options()("h,help", "Print help");
	}

	void ArgumentReader::parseCommandLineResult(
	    const cxxopts::ParseResult& result)
	{
		for (const auto& entry : mr_registry.getArguments())
		{
			std::string name = entry.first;
			ArgumentDefinition arg = entry.second;
			if (result.count(arg.name))
			{
				if (arg.dtype == TypeOfArgument::STRING)
				{
					m_values[name] = result[arg.name].as<std::string>();
				}
				else if (arg.dtype == TypeOfArgument::INT)
				{
					m_values[name] = result[arg.name].as<int>();
				}
				else if (arg.dtype == TypeOfArgument::FLOAT)
				{
					m_values[name] = result[arg.name].as<float>();
				}
				else if (arg.dtype == TypeOfArgument::BOOL)
				{
					m_values[name] = result[arg.name].as<bool>();
				}
				else if (arg.dtype == TypeOfArgument::VECTOR_OF_STRINGS)
				{
					m_values[name] =
					    result[arg.name].as<std::vector<std::string>>();
				}
			}
			else
			{
				if (arg.isRequired)
				{
					// Explicitly put a blank when the argument is required but
					// unspecified
					m_values[name] = std::monostate{};
				}
			}
		}
	}

	void ArgumentReader::validateRequiredParameters() const
	{
		std::vector<std::string> missingParams;

		for (const auto& [name, arg] : mr_registry.getArguments())
		{
			if (arg.isRequired)
			{
				const auto argumentValueVariant = m_values.at(name);
				if (std::holds_alternative<std::monostate>(
				        argumentValueVariant))
				{
					missingParams.push_back(name);
				}
			}
		}

		if (!missingParams.empty())
		{
			std::stringstream ss;
			ss << "Missing required parameters: ";
			ss << Util::join(missingParams, ", ");
			throw std::invalid_argument(ss.str());
		}
	}
}  // namespace IO
