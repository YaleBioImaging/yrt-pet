/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/UniformHistogram.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_uniformhistogram(py::module& m)
{
	auto c = py::class_<UniformHistogram, Histogram3D>(m, "UniformHistogram");
	c.def(py::init<const Scanner&, float>(), py::arg("scanner"),
	      py::arg("value") = 1.0f);
}
#endif


UniformHistogram::UniformHistogram(const Scanner& pr_scanner, float p_value)
    : Histogram3D(pr_scanner), m_value(p_value)
{
}

void UniformHistogram::writeToFile(const std::string& filename) const
{
	std::ofstream file;
	file.open(filename.c_str(), std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "The file given \"" + filename + "\" could not be opened",
		    std::make_error_code(std::errc::io_error));
	}
	int magic = MAGIC_NUMBER;
	int num_dims = 3;
	const size_t shape[3]{Histogram3D::numZBin, Histogram3D::numPhi,
	                      Histogram3D::numR};
	file.write((char*)&magic, sizeof(int));
	file.write((char*)&num_dims, sizeof(int));
	file.write((char*)shape, 3 * sizeof(size_t));

	for (size_t i = 0; i < Histogram3D::count(); i++)
	{
		file.write((char*)&m_value, sizeof(float));
	}
}

float UniformHistogram::getProjectionValue(bin_t binId) const
{
	(void)binId;
	return m_value;
}

void UniformHistogram::setProjectionValue(bin_t binId, float val)
{
	(void)binId;
	(void)val;
}

void UniformHistogram::incrementProjection(bin_t binId, float val)
{
	(void)binId;
	(void)val;
}

void UniformHistogram::clearProjections(float p_value)
{
	setValue(p_value);
}

void UniformHistogram::setValue(float p_value)
{
	m_value = p_value;
}

bool UniformHistogram::isUniform() const
{
	return true;
}
