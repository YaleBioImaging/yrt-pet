/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/DetectorSetup.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_detectorsetup(pybind11::module& m)
{
	auto c = py::class_<DetectorSetup, std::shared_ptr<DetectorSetup>>(
	    m, "DetectorSetup");
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
}  // namespace yrt
#endif

namespace yrt
{
Vector3D DetectorSetup::getPos(det_id_t id) const
{
	return {getXpos(id), getYpos(id), getZpos(id)};
}

Vector3D DetectorSetup::getOrient(det_id_t id) const
{
	return {getXorient(id), getYorient(id), getZorient(id)};
}

bool DetectorSetup::isLORAllowed(det_id_t det1, det_id_t det2) const
{
	(void)det1;
	(void)det2;
	return true;
}

}  // namespace yrt
