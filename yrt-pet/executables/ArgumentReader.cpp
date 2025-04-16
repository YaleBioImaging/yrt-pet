/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "ArgumentReader.hpp"
#include "datastruct/IO.hpp"
#include "utils/Assert.hpp"
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
	    ArgumentRegistry::cxxoptsOptionValue(const cxxopts::OptionValue& o,
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
	                                        IO::TypeOfArgument argumentType,
	                                        const std::string& group)
	{
		ASSERT_MSG(argumentType != TypeOfArgument::NONE,
		           "Cannot use None for argument type");

		registerArgumentInternal(
		    {name, description, required, std::monostate{}, group, ""});
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

	// ReconstructionConfig implementation
	ArgumentReader::ArgumentReader(const ArgumentRegistry& pr_registry)
	    : mr_registry(pr_registry)
	{
		// Initialize with default values
		for (const auto& [name, arg] : mr_registry.getArguments())
		{
			m_values[name] = arg.defaultValue;
		}
	}

	bool ArgumentReader::loadFromCommandLine(int argc, char** argv)
	{
		cxxopts::Options options(argv[0], "Reconstruction executable");
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
		for (const auto& [name, arg] : mr_registry.getArguments())
		{
			if (j.contains(arg.name))
			{
				std::visit(
				    [&](auto& v)
				    {
					    using T = std::decay_t<decltype(v)>;
					    if constexpr (std::is_same_v<T, std::string>)
					    {
						    v = j[arg.name].get<std::string>();
					    }
					    else if constexpr (std::is_same_v<T, int>)
					    {
						    v = j[arg.name].get<int>();
					    }
					    else if constexpr (std::is_same_v<T, float>)
					    {
						    v = j[arg.name].get<float>();
					    }
					    else if constexpr (std::is_same_v<T, bool>)
					    {
						    v = j[arg.name].get<bool>();
					    }
					    else if constexpr (std::is_same_v<
					                           T, std::vector<std::string>>)
					    {
						    v = j[arg.name].get<std::vector<std::string>>();
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

	void ArgumentReader::setupCommandLineOptions(cxxopts::Options& options)
	{
		for (const auto& group : mr_registry.getGroups())
		{
			auto groupOptions = options.add_options(group);
			for (auto& [name, arg] : mr_registry.getArguments())
			{
				if (arg.group == group)
				{
					std::visit(
					    [&](const ArgumentValue& v)
					    {
						    auto cxxoptsValue =
						        ArgumentRegistry::valueTypeToCxxoptsValue(v);
						    std::string cxxoptsParamName;
						    if (!arg.shortOptionName.empty())
						    {
							    cxxoptsParamName.append(arg.shortOptionName +
							                            ",");
						    }
						    cxxoptsParamName.append(arg.name);

						    groupOptions(cxxoptsParamName, arg.description,
						                 cxxoptsValue);
					    },
					    arg.defaultValue);
				}
			}
		}

		options.add_options()("h,help", "Print help");
	}

	void ArgumentReader::parseCommandLineResult(
	    const cxxopts::ParseResult& result)
	{
		for (const auto& [name, arg] : mr_registry.getArguments())
		{
			if (result.count(arg.name))
			{
				std::visit(
				    [&](auto& v)
				    {
					    using T = std::decay_t<decltype(v)>;
					    if constexpr (std::is_same_v<T, std::string>)
					    {
						    v = result[arg.name].as<std::string>();
					    }
					    else if constexpr (std::is_same_v<T, int>)
					    {
						    v = result[arg.name].as<int>();
					    }
					    else if constexpr (std::is_same_v<T, float>)
					    {
						    v = result[arg.name].as<float>();
					    }
					    else if constexpr (std::is_same_v<T, bool>)
					    {
						    v = result[arg.name].as<bool>();
					    }
					    else if constexpr (std::is_same_v<
					                           T, std::vector<std::string>>)
					    {
						    v = result[arg.name].as<std::vector<std::string>>();
					    }
				    },
				    m_values[name]);
			}
			else
			{
				m_values[name] = arg.defaultValue;
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
				std::visit(
				    [&](const auto& v)
				    {
					    using T = std::decay_t<decltype(v)>;
					    if constexpr (std::is_same_v<T, std::string>)
					    {
						    if (v.empty())
							    missingParams.push_back(name);
					    }
					    else if constexpr (std::is_same_v<
					                           T, std::vector<std::string>>)
					    {
						    if (v.empty())
							    missingParams.push_back(name);
					    }
				    },
				    m_values.at(name));
			}
		}

		if (!missingParams.empty())
		{
			std::stringstream ss;
			ss << "Missing required parameters: ";
			for (const auto& param : missingParams)
			{
				ss << param << " ";
			}
			throw std::runtime_error(ss.str());
		}
	}
}  // namespace IO
