/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdater.hpp"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/utils/Types.hpp"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <utility>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorprojectorupdater(py::module& m)
{
	auto c = py::class_<OperatorProjectorUpdater, std::shared_ptr<OperatorProjectorUpdater>>(m, "OperatorProjectorUpdater");
	c.def("forwardUpdate",
		   [](OperatorProjectorUpdater& self,
			  float weight,
			  Image* in_image,
			  int offset_x, int vz,
			  frame_t dynamicFrame,
			  size_t numVoxelPerFrame) -> float {
			 return self.forwardUpdate(weight, in_image->getRawPointer(),
		                              offset_x, vz, dynamicFrame,
		                              numVoxelPerFrame);
		   },
		   py::arg("weight"), py::arg("in_image"),
		   py::arg("offset_x"), py::arg("z"), py::arg("dynamicFrame") = 0,
		   py::arg("numVoxelPerFrame") = 0);
	c.def("backUpdate",
			   [](OperatorProjectorUpdater& self,
				  float value,
				  float weight,
				  Image* in_image,
				  int offset_x, int vz,
				  frame_t dynamicFrame,
				  size_t numVoxelPerFrame) -> void {
				 self.backUpdate(value, weight, in_image->getRawPointer(), offset_x,
		                    vz, dynamicFrame, numVoxelPerFrame);
				 },
			   py::arg("value"), py::arg("weight"), py::arg("in_image"),
			   py::arg("offset_x"), py::arg("z"), py::arg("dynamicFrame") = 0,
			   py::arg("numVoxelPerFrame") = 0);

	// Default updater
	auto def_upd = py::class_<OperatorProjectorUpdaterDefault3D,
                            OperatorProjectorUpdater,
                            std::shared_ptr<OperatorProjectorUpdaterDefault3D>>(
      m, "OperatorProjectorUpdaterDefault3D");
	def_upd.def(py::init<>());

	auto def_upd_4d = py::class_<OperatorProjectorUpdaterDefault4D,
	                          OperatorProjectorUpdater,
	                          std::shared_ptr<OperatorProjectorUpdaterDefault4D>>(
	    m, "OperatorProjectorUpdaterDefault4D");
	def_upd_4d.def(py::init<>());

	// LR updater
	auto lr_upd = py::class_<OperatorProjectorUpdaterLR,
                            OperatorProjectorUpdater,
                            std::shared_ptr<OperatorProjectorUpdaterLR>>(
      m, "OperatorProjectorUpdaterLR");

	lr_upd.def(py::init<const Array3D<float>&>());
	lr_upd.def("getHBasisCopy", &OperatorProjectorUpdaterLR::getHBasisCopy);


	lr_upd.def("setHBasis", &OperatorProjectorUpdaterLR::setHBasis);

	lr_upd.def("setUpdateH", [](OperatorProjectorUpdaterLR& self,
							   bool updateH) {
				   self.setUpdateH(updateH);
			   });

	lr_upd.def("getUpdateH", &OperatorProjectorUpdaterLR::getUpdateH);

}
}  // namespace yrt

#endif


namespace yrt
{

float OperatorProjectorUpdaterDefault3D::forwardUpdate(
    float weight, float* cur_img_ptr, size_t offset, int vz,
    frame_t dynamicFrame, size_t numVoxelPerFrame) const
{
	(void) dynamicFrame;
	(void) numVoxelPerFrame;
	return weight * cur_img_ptr[offset];
}

void OperatorProjectorUpdaterDefault3D::backUpdate(float value, float weight,
                                                   float* cur_img_ptr,
                                                   size_t offset, int vz,
                                                   frame_t dynamicFrame,
                                                   size_t numVoxelPerFrame)
{
	(void) dynamicFrame;
	(void) numVoxelPerFrame;
	float output = value * weight;
	std::atomic_ref<float> atomic_elem(cur_img_ptr[offset]);
	atomic_elem.fetch_add(output);
}


float OperatorProjectorUpdaterDefault4D::forwardUpdate(
    float weight, float* cur_img_ptr, size_t offset, int vz,
    frame_t dynamicFrame, size_t numVoxelPerFrame) const
{
	return weight * cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset];
}

void OperatorProjectorUpdaterDefault4D::backUpdate(float value, float weight,
                                                   float* cur_img_ptr,
                                                   size_t offset, int vz,
                                                   frame_t dynamicFrame,
                                                   size_t numVoxelPerFrame)
{
	float output = value * weight;
	std::atomic_ref<float> atomic_elem(cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset]);
	atomic_elem.fetch_add(output);
}

OperatorProjectorUpdaterLR::OperatorProjectorUpdaterLR(
    const Array3DBase<float>& pr_HBasis)
{
	setHBasis(pr_HBasis);
}

const Array3DAlias<float>& OperatorProjectorUpdaterLR::getHBasis() const
{
	return mp_HBasis;
}

std::unique_ptr<Array3D<float>>
    OperatorProjectorUpdaterLR::getHBasisCopy() const
{
	auto dims = mp_HBasis.getDims();
	auto out  = std::make_unique<Array3D<float>>();
	out->allocate(dims[0], dims[1], dims[2]);
	out->copy(mp_HBasis);
	return out;

}

void OperatorProjectorUpdaterLR::setHBasis(const Array3DBase<float>& pr_HBasis) {
	mp_HBasis.bind(pr_HBasis);
	auto dims = mp_HBasis.getDims();
	m_rank = static_cast<int>(dims[0]);
	m_nz = static_cast<int>(dims[1]);
	m_numDynamicFrames = static_cast<int>(dims[2]);
}

void OperatorProjectorUpdaterLR::setHBasisWrite(
    const Array3DBase<float>& pr_HWrite)
{
	mp_HWrite.bind(pr_HWrite);
}

const Array3DAlias<float>& OperatorProjectorUpdaterLR::getHBasisWrite() {
	return mp_HWrite;
}

void OperatorProjectorUpdaterLR::setCurrentImgBuffer(ImageBase* img)
{
	if (auto* imgCPU = dynamic_cast<ImageOwned*>(img))
		m_currentImg = imgCPU->getRawPointer();
	else if (auto* imgGPU = dynamic_cast<ImageDevice*>(img))
		m_currentImg = imgGPU->getDevicePointer();
	else
		throw std::runtime_error("Unsupported image type");

	ASSERT_MSG(m_currentImg != nullptr, "Null image data pointer");
}

const float* OperatorProjectorUpdaterLR::getCurrentImgBuffer() const
{
	return m_currentImg;
}

void OperatorProjectorUpdaterLR::setUpdateH(bool updateH) {
	m_updateH = updateH;
}

bool OperatorProjectorUpdaterLR::getUpdateH() const {
	return m_updateH;
}


float OperatorProjectorUpdaterLR::forwardUpdate(float weight,
                                                float* cur_img_ptr,
                                                size_t offset, int vz,
                                                frame_t dynamicFrame,
                                                size_t numVoxelPerFrame) const
{
	float cur_img_lr_val = 0.0f;
	const float* H_ptr = mp_HBasis.getRawPointer();

	for (int l = 0; l < m_rank; ++l)
	{
		float cur_H_ptr = *(H_ptr + l * (m_nz * m_numDynamicFrames) + vz * m_numDynamicFrames + dynamicFrame);
		const size_t offset_rank = l * numVoxelPerFrame;
		cur_img_lr_val += cur_img_ptr[offset + offset_rank] * cur_H_ptr;
	}
	return weight * cur_img_lr_val;
}

void OperatorProjectorUpdaterLR::backUpdate(float value, float weight,
                                            float* cur_img_ptr, size_t offset,
                                            int vz, frame_t dynamicFrame,
                                            size_t numVoxelPerFrame)
{
	const float Ay = value * weight;

	if (!m_updateH)
	{
		const float* H_ptr = mp_HBasis.getRawPointer();
		for (int l = 0; l < m_rank; ++l)
		{
			const float cur_H_ptr = *(H_ptr + l * (m_nz * m_numDynamicFrames) + vz * m_numDynamicFrames + dynamicFrame);
			const size_t offset_rank = l * numVoxelPerFrame;
			const float output = Ay * cur_H_ptr;
			std::atomic_ref<float> atomic_elem(cur_img_ptr[offset + offset_rank]);
			atomic_elem.fetch_add(output);
		}
	}
	else {
		float* H_ptr = mp_HWrite.getRawPointer();
		for (int l = 0; l < m_rank; ++l) {
			const size_t offset_rank = l * numVoxelPerFrame;
			const float output = Ay * cur_img_ptr[offset + offset_rank];
			std::atomic_ref<float> atomic_elem(H_ptr[l * (m_nz * m_numDynamicFrames) + vz * m_numDynamicFrames + dynamicFrame]);
			atomic_elem.fetch_add(output);
		}
	}
}


void OperatorProjectorUpdaterLRDualUpdate::backUpdate(float value, float weight,
                                                      float* raw_img_ptr,
                                                      size_t offset, int vz,
                                                      frame_t dynamicFrame,
                                                      size_t numVoxelPerFrame)
{
	const float Ay = value * weight;
	const float* H_ptr_read = mp_HBasis.getRawPointer();
	const float* W_ptr_read = this->getCurrentImgBuffer();
	float* H_ptr_write = mp_HWrite.getRawPointer();

	for (int l = 0; l < m_rank; ++l)
	{

		const float cur_H_ptr = *(H_ptr_read + l * m_numDynamicFrames + dynamicFrame);
		const size_t offset_rank = l * numVoxelPerFrame;
		const float outputWUpdate = Ay * cur_H_ptr;
		const float outputHUpdate = Ay * W_ptr_read[offset + offset_rank];
		std::atomic_ref<float> atomic_elemW(raw_img_ptr[offset + offset_rank]);
		atomic_elemW.fetch_add(outputWUpdate);
		std::atomic_ref<float> atomic_elemH(H_ptr_write[l * m_numDynamicFrames + dynamicFrame]);
		atomic_elemH.fetch_add(outputHUpdate);
	}
}



} // namespace yrt
