/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"

#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_detectormask(pybind11::module& m)
{
	auto c = py::class_<DetectorMask>(m, "DetectorMask", py::buffer_protocol());
	c.def(py::init<const std::string&>(), "fname"_a);
	c.def(py::init<const Array1DBase<bool>&>(), "maskArray"_a);
	c.def("checkAgainstScanner", &DetectorMask::checkAgainstScanner);
	c.def_buffer(
	    [](DetectorMask& self) -> py::buffer_info
	    {
		    Array1D<bool>& d = self.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(bool),
		                           py::format_descriptor<bool>::format(), 1,
		                           d.getDims(), d.getStrides());
	    });
	c.def("getData",
	      static_cast<const Array1D<bool>& (DetectorMask::*)() const>(
	          &DetectorMask::getData));
	c.def("checkDetector", &DetectorMask::checkDetector, "detId"_a);
}

}  // namespace yrt
#endif

namespace yrt
{

DetectorMask::DetectorMask(const std::string& pr_fname)
{
	readFromFile(pr_fname);
}

DetectorMask::DetectorMask(const Array1DBase<bool>& pr_data)
{
	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->copy(pr_data);
}

DetectorMask::DetectorMask(const Array3DBase<float>& pr_data)
{
	mp_data = std::make_unique<Array1D<bool>>();

	const size_t size = pr_data.getSizeTotal();
	mp_data->allocate(size);

	for (size_t i = 0; i < size; ++i)
	{
		mp_data->setFlat(i, pr_data.getFlat(i) > 0.0f);
	}
}

DetectorMask::DetectorMask(const DetectorMask& other)
{
	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->copy(other.getData());
}

void DetectorMask::readFromFile(const std::string& fname)
{
	mp_data = std::make_unique<Array1D<bool>>();
	mp_data->readFromFile(fname);
}

Array1D<bool>& DetectorMask::getData()
{
	ASSERT(mp_data != nullptr);
	return *mp_data;
}

const Array1D<bool>& DetectorMask::getData() const
{
	ASSERT(mp_data != nullptr);
	return *mp_data;
}

bool DetectorMask::checkAgainstScanner(const Scanner& scanner) const
{
	return scanner.getNumDets() == mp_data->getSizeTotal();
}

size_t DetectorMask::getNumDets() const
{
	return mp_data->getSizeTotal();
}

bool DetectorMask::checkDetector(size_t detId) const
{
	// Assumes that mp_maskArray != nullptr
	return mp_data->getFlat(detId);
}

void DetectorMask::writeToFile(const std::string& fname) const
{
	mp_data->writeToFile(fname);
}

}  // namespace yrt