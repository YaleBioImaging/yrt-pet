/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/ImageBase.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_imagebase(py::module& m)
{
	auto c = py::class_<ImageBase, Variable>(m, "ImageBase");

	c.def("getParams", &ImageBase::getParams);
	c.def("getRadius", &ImageBase::getRadius);
	c.def("setParams", &ImageBase::setParams, "params"_a);

	c.def("fill", &ImageBase::fill, "value"_a,
	      "Set all voxels to the given value");
	c.def("setValue", &ImageBase::fill, "value"_a,
	      "Set all voxels to the given value (legacy)");
	c.def("addFirstImageToSecond", &ImageBase::addFirstImageToSecond,
	      "second"_a);
	c.def("applyThreshold", &ImageBase::applyThreshold, "maskImage"_a,
	      "threshold"_a, "val_le_scale"_a, "val_le_off"_a, "val_gt_scale"_a,
	      "val_gt_off"_a);
	c.def("writeToFile", &ImageBase::writeToFile, "filename"_a);

	// EM update multiplication
	c.def("updateEMThresholdStatic", &ImageBase::updateEMThresholdStatic,
	      "update_img"_a, "sens_img"_a, "threshold"_a);
	c.def("updateEMThresholdDynamic",
	      static_cast<void (ImageBase::*)(ImageBase*, const ImageBase*, float)>(
	          &ImageBase::updateEMThresholdDynamic),
	      "update_img"_a, "sens_img"_a, "threshold"_a);
	c.def("updateEMThresholdDynamic",
	      static_cast<void (ImageBase::*)(
	          ImageBase* updateImg, const ImageBase* sensImg,
	          const std::vector<float>& c_r, float threshold)>(
	          &ImageBase::updateEMThresholdDynamic),
	      "update_img"_a, "sens_img"_a, "sens_scaling"_a, "threshold"_a);
}

}  // namespace yrt
#endif

namespace yrt
{

ImageBase::ImageBase(const ImageParams& imgParams) : m_params(imgParams) {}

const ImageParams& ImageBase::getParams() const
{
	return m_params;
}

void ImageBase::setParams(const ImageParams& newParams)
{
	m_params = newParams;
}

size_t ImageBase::unravel(int iz, int iy, int ix, frame_t it) const
{
	return ix + (iy + (iz + m_params.nz * it) * m_params.ny) * m_params.nx;
}

float ImageBase::getRadius() const
{
	return m_params.fovRadius;
}

}  // namespace yrt
