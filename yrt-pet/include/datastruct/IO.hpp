/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/ProjectionData.hpp"
#include "operators/OperatorProjector.hpp"

#include <memory>
#include <string>

class Scanner;

namespace IO
{
	std::unique_ptr<ProjectionData> openProjectionData(
	    const std::string& input_fname, const std::string& input_format,
	    const Scanner& scanner, const Plugin::OptionsResult&);

	std::string possibleFormats(Plugin::InputFormatsChoice choice = Plugin::InputFormatsChoice::ALL);

	bool isFormatListMode(const std::string& format);

	// Projector-related
	OperatorProjector::ProjectorType
	    getProjector(const std::string& projectorName);

}  // namespace IO
