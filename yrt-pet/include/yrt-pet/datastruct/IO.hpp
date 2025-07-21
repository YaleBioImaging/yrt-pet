/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"

#include <memory>
#include <string>


namespace yrt
{

class Scanner;

namespace io
{

std::unique_ptr<ProjectionData>
    openProjectionData(const std::string& input_fname,
                       const std::string& input_format, const Scanner& scanner,
                       const OptionsResult&);

std::string possibleFormats(
    plugin::InputFormatsChoice choice = plugin::InputFormatsChoice::ALL);

bool isFormatListMode(const std::string& format);

// Projector-related
OperatorProjector::ProjectorType getProjector(const std::string& projectorName);

}  // namespace io
}  // namespace yrt
