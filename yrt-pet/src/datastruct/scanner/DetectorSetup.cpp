/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/DetectorSetup.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_detectorsetup(pybind11::module& m)
{
	auto c = py::class_<DetectorSetup, std::shared_ptr<DetectorSetup>>(
	    m, "DetectorSetup");
	c.def("getNumDets", &DetectorSetup::getNumDets);
	c.def("getXpos", &DetectorSetup::getXpos, "detId"_a);
	c.def("getYpos", &DetectorSetup::getYpos, "detId"_a);
	c.def("getZpos", &DetectorSetup::getZpos, "detId"_a);
	c.def("getXorient", &DetectorSetup::getXorient, "detId"_a);
	c.def("getYorient", &DetectorSetup::getYorient, "detId"_a);
	c.def("getZorient", &DetectorSetup::getZorient, "detId"_a);
	c.def("getPos", &DetectorSetup::getPos, "detId"_a);
	c.def("getOrient", &DetectorSetup::getOrient, "detId"_a);
	c.def("writeToFile", &DetectorSetup::writeToFile,"fname"_a);
	c.def("hasMask", &DetectorSetup::hasMask);
	c.def("addMask", static_cast<void(DetectorSetup::*)(const std::string&)>(&DetectorSetup::addMask), "mask_fname"_a);
	c.def("addMask", static_cast<void(DetectorSetup::*)(const DetectorMask& mask)>(&DetectorSetup::addMask), "mask"_a);
	c.def("getMask", &DetectorSetup::getMask);
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

bool DetectorSetup::isDetectorAllowed(det_id_t id) const
{
	if (hasMask())
	{
		return mp_mask->isDetectorEnabled(id);
	}
	// Allow everything by default
	return true;
}

bool DetectorSetup::hasMask() const
{
	return mp_mask != nullptr;
}

DetectorMask& DetectorSetup::getMask()
{
	ASSERT(mp_mask != nullptr);
	return *mp_mask;
}

void DetectorSetup::addMask(const std::string& mask_fname)
{
	mp_mask = std::make_unique<DetectorMask>(mask_fname);
}
void DetectorSetup::addMask(const DetectorMask& mask)
{
	// Calls the copy constructor
	mp_mask = std::make_unique<DetectorMask>(mask);
}

}  // namespace yrt
