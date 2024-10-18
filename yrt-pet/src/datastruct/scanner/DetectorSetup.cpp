/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/DetectorSetup.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
void py_setup_detectorsetup(pybind11::module& m)
{
	auto c = py::class_<DetectorSetup>(m, "DetectorSetup");
	c.def("getNumDets", &DetectorSetup::getNumDets);
	c.def("getXpos", &DetectorSetup::getXpos);
	c.def("getYpos", &DetectorSetup::getYpos);
	c.def("getZpos", &DetectorSetup::getZpos);
	c.def("getXorient", &DetectorSetup::getXorient);
	c.def("getYorient", &DetectorSetup::getYorient);
	c.def("getZorient", &DetectorSetup::getZorient);
	c.def("getPos", &DetectorSetup::getPos);
	c.def("getOrient", &DetectorSetup::getOrient);
	c.def("writeToFile", &DetectorSetup::writeToFile);
}
#endif

Vector3D DetectorSetup::getPos(det_id_t id) const
{
	return {getXpos(id), getYpos(id), getZpos(id)};
}

Vector3D DetectorSetup::getOrient(det_id_t id) const
{
	return {getXorient(id), getYorient(id), getZorient(id)};
}
