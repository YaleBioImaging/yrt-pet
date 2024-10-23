/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/IO.hpp"

#include "datastruct/projection/ProjectionData.hpp"
#include "utils/Utilities.hpp"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_io(py::module& m)
{
	m.def("openProjectionData", &IO::openProjectionData, "input_fname"_a,
	      "input_format"_a, "scanner"_a, "pluginOptions"_a);
	m.def("getProjector", IO::getProjector, "projector_name"_a);
	m.def("requiresGPU", IO::requiresGPU, "projector_type"_a);
	m.def("possibleFormats", IO::possibleFormats);
}
#endif

std::unique_ptr<ProjectionData> IO::openProjectionData(
    const std::string& input_fname, const std::string& input_format,
    const Scanner& scanner, const Plugin::OptionsResult& pluginOptions)
{
	return Plugin::PluginRegistry::instance().create(
	    input_format, scanner, input_fname, pluginOptions);
}

std::string IO::possibleFormats(Plugin::InputFormatsChoice choice)
{
	const std::vector<std::string> formats =
	    Plugin::PluginRegistry::instance().getAllFormats(choice);
	std::string stringList;
	size_t i;
	for (i = 0; i < formats.size() - 1; ++i)
	{
		stringList += formats[i] + ", ";
	}
	stringList += "and " + formats[i] + ".";
	return stringList;
}

bool IO::isFormatListMode(const std::string& format)
{
	const std::string format_upper = Util::toUpper(format);
	return Plugin::PluginRegistry::instance().isFormatListMode(format_upper);
}

OperatorProjector::ProjectorType
    IO::getProjector(const std::string& projectorName)
{
	const std::string projectorName_upper = Util::toUpper(projectorName);

	// Projector type
	if (projectorName_upper == "DD_GPU")
	{
		return OperatorProjector::ProjectorType::DD_GPU;
	}
	if (projectorName_upper == "S" || projectorName_upper == "SIDDON")
	{
		return OperatorProjector::ProjectorType::SIDDON;
	}
	if (projectorName_upper == "D" || projectorName_upper == "DD" ||
	    projectorName_upper == "DD_CPU")
	{
		return OperatorProjector::ProjectorType::DD;
	}
	throw std::invalid_argument(
	    "Invalid Projector name, choices are Siddon (S), "
	    "Distance-Driven cpu (D) and Distance-Driven gpu (DD_GPU)");
}

bool IO::requiresGPU(OperatorProjector::ProjectorType projector)
{
	if (projector == OperatorProjector::DD_GPU)
	{
		return true;
	}
	return false;
}
