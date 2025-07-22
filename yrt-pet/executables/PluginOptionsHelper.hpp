/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "ArgumentReader.hpp"
#include "yrt-pet/datastruct/PluginFramework.hpp"

namespace yrt
{
namespace plugin
{
void addOptionsFromPlugins(io::ArgumentRegistry& registry,
                           InputFormatsChoice choice = InputFormatsChoice::ALL);
}  // namespace plugin
}  // namespace yrt
