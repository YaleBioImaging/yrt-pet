/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "PluginOptionsHelper.hpp"

#include "ArgumentReader.hpp"
#include "yrt-pet/utils/Assert.hpp"

namespace yrt::plugin
{
void addOptionsFromPlugins(io::ArgumentRegistry& registry,
                           plugin::InputFormatsChoice choice)
{
	const plugin::OptionsList pluginOptions =
	    plugin::PluginRegistry::instance().getAllOptions(choice);

	// Group the plugin options (in case multiple plugins have the
	// same options)

	// Key: Option name, vector of pairs {format name, corresponding option
	// info}
	using OptionsListGrouped = std::unordered_map<
	    std::string, std::vector<std::pair<std::string, plugin::OptionInfo>>>;
	OptionsListGrouped pluginOptionsGrouped;
	for (auto& pluginOption : pluginOptions)
	{
		const plugin::OptionsListPerPlugin& optionsListInCurrentPlugin =
		    pluginOption.second;
		for (const plugin::OptionPerPlugin& option : optionsListInCurrentPlugin)
		{
			const plugin::OptionInfo& optionInfo = option.second;
			const std::string& optionName = option.first;
			auto pluginOption_it = pluginOptionsGrouped.find(optionName);

			if (pluginOption_it == pluginOptionsGrouped.end())
			{
				// non existant, create
				pluginOptionsGrouped[optionName] = {
				    {pluginOption.first, optionInfo}};
			}
			else
			{
				// preexistant, append to it
				pluginOptionsGrouped[optionName].emplace_back(
				    pluginOption.first, optionInfo);
			}
		}
	}

	std::string inputFormatOptionsGroup = "Input format";
	for (auto& pluginOptionGrouped : pluginOptionsGrouped)
	{
		const auto& listOfPluginsThatHaveCurrentOption =
		    pluginOptionGrouped.second;
		std::string optionHelp;
		auto argType = io::TypeOfArgument::NONE;
		const size_t numPluginsThatHaveCurrentOptions =
		    listOfPluginsThatHaveCurrentOption.size();
		for (size_t i = 0; i < numPluginsThatHaveCurrentOptions; ++i)
		{
			const auto& [pluginName, helpForPlugin] =
			    listOfPluginsThatHaveCurrentOption[i];
			const bool isLastPlugin = i == numPluginsThatHaveCurrentOptions - 1;

			optionHelp +=
			    "For " + pluginName + ": " + std::get<0>(helpForPlugin);
			if (!isLastPlugin)
			{
				optionHelp += "\n";
			}

			// Due to a cxxopts limitation, it should not be allowed to
			//  provide two plugins that have different argument types
			//  (OptionInfo::second). Send an error here if that happens
			const io::TypeOfArgument currentArgType =
			    std::get<1>(helpForPlugin);

			if (argType == io::TypeOfArgument::NONE)
			{
				// First init
				argType = currentArgType;
			}
			else
			{
				const auto errorMsg = "A plugin already uses option " +
				                      pluginOptionGrouped.first +
				                      " with a different argument type";
				ASSERT_MSG(argType == currentArgType, errorMsg.c_str());
			}
			ASSERT_MSG(argType != io::TypeOfArgument::NONE,
			           "Unspecified argument type in plugin definition");
		}

		registry.registerArgument(pluginOptionGrouped.first, optionHelp, false,
		                          argType, inputFormatOptionsGroup);
	}
}
}  // namespace yrt::plugin
