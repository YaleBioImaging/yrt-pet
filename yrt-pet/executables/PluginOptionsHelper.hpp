/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "ArgumentReader.hpp"
#include "datastruct/PluginFramework.hpp"

namespace PluginOptionsHelper
{
	void addOptionsFromPlugins(
	    IO::ArgumentRegistry& registry,
	    Plugin::InputFormatsChoice choice = Plugin::InputFormatsChoice::ALL);
}  // namespace PluginOptionsHelper
