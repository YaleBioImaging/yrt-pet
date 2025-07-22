/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/IO.hpp"

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Utilities.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_io(py::module& m)
{
	m.def("openProjectionData", &io::openProjectionData, "input_fname"_a,
	      "input_format"_a, "scanner"_a, "pluginOptions"_a);
	m.def("getProjector", io::getProjector, "projector_name"_a);
	m.def("possibleFormats", io::possibleFormats);
}
}  // namespace yrt

#endif

namespace yrt
{
std::unique_ptr<ProjectionData> io::openProjectionData(
    const std::string& input_fname, const std::string& input_format,
    const Scanner& scanner, const OptionsResult& pluginOptions)
{
	const std::string format_upper = util::toUpper(input_format);
	return plugin::PluginRegistry::instance().create(
	    format_upper, scanner, input_fname, pluginOptions);
}

std::string io::possibleFormats(plugin::InputFormatsChoice choice)
{
	const std::vector<std::string> formats =
	    plugin::PluginRegistry::instance().getAllFormats(choice);
	std::string stringList;
	size_t i;
	for (i = 0; i < formats.size() - 1; ++i)
	{
		stringList += formats[i] + ", ";
	}
	stringList += "and " + formats[i] + ".";
	return stringList;
}

bool io::isFormatListMode(const std::string& format)
{
	ASSERT_MSG(!format.empty(), "No format specified");
	const std::string format_upper = util::toUpper(format);
	return plugin::PluginRegistry::instance().isFormatListMode(format_upper);
}

OperatorProjector::ProjectorType
    io::getProjector(const std::string& projectorName)
{
	const std::string projectorName_upper = util::toUpper(projectorName);

	// Projector type
	if (projectorName_upper == "S" || projectorName_upper == "SIDDON")
	{
		return OperatorProjector::ProjectorType::SIDDON;
	}
	if (projectorName_upper == "D" || projectorName_upper == "DD")
	{
		return OperatorProjector::ProjectorType::DD;
	}
	throw std::invalid_argument(
	    "Invalid Projector name, choices are Siddon (S), "
	    "Distance-Driven cpu (D)");
}
}  // namespace yrt