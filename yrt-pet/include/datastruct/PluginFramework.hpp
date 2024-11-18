/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/ProjectionData.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Utilities.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class Scanner;
class ProjectionData;

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#endif

namespace Plugin
{
	enum class InputFormatsChoice
	{
		ALL,
		ONLYLISTMODES,
		ONLYHISTOGRAMS
	};

	// Help string, Whether it's bool value (false: no, true: yes)
	using OptionInfo = std::tuple<std::string, bool>;
	// Option name, Option info
	using OptionPerPlugin = std::pair<std::string, OptionInfo>;
	// Each format will have a list of options
	using OptionsListPerPlugin = std::vector<OptionPerPlugin>;
	// Key: Format name, Value: List of options
	using OptionsList = std::unordered_map<std::string, OptionsListPerPlugin>;
	// Map: format name, value
	using OptionsResult = std::unordered_map<std::string, std::string>;

	using ProjectionDataFactory =
	    std::function<std::unique_ptr<ProjectionData>(
	        const Scanner&, const std::string&, const OptionsResult&)>;
	using OptionsAdder = std::function<OptionsListPerPlugin()>;
#if BUILD_PYBIND11
	using Pybind11ModuleAdder = std::function<void(pybind11::module&)>;
#endif

#pragma GCC visibility push(hidden)

	class PluginRegistry
	{
	public:
		static PluginRegistry& instance()
		{
			static PluginRegistry instance;
			return instance;
		}

		bool registerFormat(const std::string& formatName,
		                    const ProjectionDataFactory& factory,
		                    const OptionsAdder& optionsAdder, bool isListMode)
		{
			m_factoriesMap[formatName] = factory;
			m_optionsAddersMap[formatName] = optionsAdder;
			m_isListModeMap[formatName] = isListMode;
			return true;
		}

#if BUILD_PYBIND11
		bool registerPybind11Module(const std::string& formatName,
		                            const Pybind11ModuleAdder& moduleAdder)
		{
			m_pybind11ModuleAddersMap[formatName] = moduleAdder;
			return true;
		}
		void addAllPybind11Modules(pybind11::module& m) const
		{
			for (const auto& moduleAdder : m_pybind11ModuleAddersMap)
			{
				auto sub = m.def_submodule(moduleAdder.first.c_str());
				moduleAdder.second(sub);
			}
		}
#endif

		bool isFormatListMode(const std::string& formatName) const
		{
			const auto it = m_isListModeMap.find(formatName);
			if (it != m_isListModeMap.end())
			{
				return it->second;
			}
			throw std::invalid_argument("Unknown format: " + formatName);
		}

		std::vector<std::string> getAllFormats(
		    InputFormatsChoice choice = InputFormatsChoice::ALL) const
		{
			std::vector<std::string> keys;
			keys.reserve(m_factoriesMap.size());
			for (const auto& pair : m_factoriesMap)
			{
				if (choice == InputFormatsChoice::ALL ||
				    (choice == InputFormatsChoice::ONLYLISTMODES &&
				     m_isListModeMap.at(pair.first)) ||
				    (choice == InputFormatsChoice::ONLYHISTOGRAMS &&
				     !m_isListModeMap.at(pair.first)))
				{
					keys.push_back(pair.first);
				}
			}
			return keys;
		}

		std::unique_ptr<ProjectionData> create(const std::string& formatName,
		                                        const Scanner& scanner,
		                                        const std::string& filename,
		                                        const OptionsResult& args) const
		{
			const auto it = m_factoriesMap.find(formatName);
			if (it != m_factoriesMap.end())
			{
				return it->second(scanner, filename, args);
			}
			throw std::invalid_argument("Unknown format: " + formatName);
		}

		OptionsList getAllOptions(
		    InputFormatsChoice choice = InputFormatsChoice::ALL) const
		{
			OptionsList options;
			for (const auto& pair : m_optionsAddersMap)
			{
				if (choice == InputFormatsChoice::ALL ||
				    (choice == InputFormatsChoice::ONLYLISTMODES &&
				     m_isListModeMap.at(pair.first)) ||
				    (choice == InputFormatsChoice::ONLYHISTOGRAMS &&
				     !m_isListModeMap.at(pair.first)))
				{
					options[pair.first] = pair.second();
				}
			}
			return options;
		}

	private:
		std::unordered_map<std::string, ProjectionDataFactory> m_factoriesMap;
		std::unordered_map<std::string, OptionsAdder> m_optionsAddersMap;
#if BUILD_PYBIND11
		std::unordered_map<std::string, Pybind11ModuleAdder>
		    m_pybind11ModuleAddersMap;
#endif
		std::unordered_map<std::string, bool> m_isListModeMap;
		PluginRegistry() = default;
	};
#pragma GCC visibility pop
}  // namespace Plugin


#define REGISTER_PROJDATA_PLUGIN(formatName, className, factoryFunc, \
                                 optionsAdder)                       \
	namespace AddedPlugins                                           \
	{                                                                \
		struct className##Register                                   \
		{                                                            \
			className##Register()                                    \
			{                                                        \
				Plugin::PluginRegistry::instance().registerFormat(   \
				    Util::toUpper(formatName),                       \
+                   factoryFunc, optionsAdder,                       \
				    className::IsListMode());                        \
			}                                                        \
		};                                                           \
		static className##Register global_##className##Register;     \
	}

#if BUILD_PYBIND11

#define REGISTER_PROJDATA_PYBIND11(formatName, className, moduleAdderFunc) \
	namespace AddedPlugins                                                 \
	{                                                                      \
		struct className##Pybind11Register                                 \
		{                                                                  \
			className##Pybind11Register()                                  \
			{                                                              \
				Plugin::PluginRegistry::instance().registerPybind11Module( \
				    formatName, moduleAdderFunc);                          \
			}                                                              \
		};                                                                 \
		static className##Pybind11Register                                 \
		    global_##className##Pybind11Register;                          \
	}

#endif
