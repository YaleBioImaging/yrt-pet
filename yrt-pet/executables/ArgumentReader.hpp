/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "PluginOptionsHelper.hpp"
#include "datastruct/ArgumentDefinition.hpp"
#include "utils/JSONUtils.hpp"

#include <cxxopts.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace IO
{
	class ArgumentRegistry
	{
	public:
		ArgumentRegistry() = default;
		void registerArgumentInternal(const ArgumentDefinition& arg);

		template <typename ArgumentType>
		void registerArgument(std::string name, std::string description,
		                      bool required)
		{
			registerArgumentInternal(
			    {name, description, required, ArgumentType{}, "", ""});
		}
		template <typename ArgumentType>
		void registerArgument(std::string name, std::string description,
		                      bool required, ArgumentType defaultValue,
		                      std::string group = "",
		                      std::string shortOptionName = "")
		{
			registerArgumentInternal({name, description, required, defaultValue,
			                          group, shortOptionName});
		}

		const std::map<std::string, ArgumentDefinition>& getArguments() const;
		const std::vector<std::string>& getGroups() const;

		static std::shared_ptr<cxxopts::Value>
		    argumentTypeToCxxoptsValue(TypeOfArgument t);
		static std::shared_ptr<cxxopts::Value>
		    valueTypeToCxxoptsValue(ArgumentValue v);

	private:
		std::map<std::string, ArgumentDefinition> arguments;
		std::vector<std::string> groups;
	};

	class ArgumentReader
	{
	public:
		explicit ArgumentReader(const ArgumentRegistry& pr_registry);

		// Load configuration from command line
		//  Return value: true -> parsing complete false -> Parsing incomplete
		//  due to "help" requested
		bool loadFromCommandLine(int argc, char** argv);

		// Load configuration from JSON file
		void loadFromJSON(const std::string& filename);

		// Save configuration to JSON file
		void saveToJSON(const std::string& filename) const;

		// Validate configuration
		bool validate() const;

		// Access configuration values
		template <typename T>
		T getValue(const std::string& name) const;

		// Set configuration values
		template <typename T>
		void setValue(const std::string& name, const T& value);

		// Get all values as JSON
		nlohmann::json toJSON() const;

		// Set all values from JSON
		void fromJSON(const nlohmann::json& j);

		const Plugin::OptionsResult& getPluginResults() const;

	private:
		std::map<std::string, ArgumentValue> m_values;
		Plugin::OptionsResult m_pluginOptionsResults;
		const ArgumentRegistry& mr_registry;

		void setupCommandLineOptions(cxxopts::Options& options);
		void parseCommandLineResult(const cxxopts::ParseResult& result);
		void validateRequiredParameters() const;
	};

	// Template implementations
	template <typename T>
	T ArgumentReader::getValue(const std::string& name) const
	{
		auto it = m_values.find(name);
		if (it == m_values.end())
		{
			throw std::runtime_error("Configuration value not found: " + name);
		}
		return std::get<T>(it->second);
	}

	template <typename T>
	void ArgumentReader::setValue(const std::string& name, const T& value)
	{
		m_values[name] = value;
	}
}  // namespace IO
