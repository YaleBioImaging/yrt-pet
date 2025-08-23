/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdater.hpp"
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
			  int offset_x,
			  frame_t dynamicFrame,
			  size_t numVoxelPerFrame) -> float {
			 return self.forwardUpdate(weight, in_image->getRawPointer(),
			 offset_x, dynamicFrame, numVoxelPerFrame);
		   },
		   py::arg("weight"), py::arg("in_image"),
		   py::arg("offset_x"), py::arg("dynamicFrame") = 0,
		   py::arg("numVoxelPerFrame") = 0);
	c.def("backUpdate",
			   [](OperatorProjectorUpdater& self,
				  float value,
				  float weight,
				  Image* in_image,
				  int offset_x,
				  frame_t dynamicFrame,
				  size_t numVoxelPerFrame) -> void {
				 self.backUpdate(value, weight, in_image->getRawPointer(),
				 offset_x, dynamicFrame, numVoxelPerFrame);
				 },
			   py::arg("value"), py::arg("weight"), py::arg("in_image"),
			   py::arg("offset_x"), py::arg("dynamicFrame") = 0,
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

	lr_upd.def(py::init<const Array2D<float>&>());
	lr_upd.def("getHBasisCopy", &OperatorProjectorUpdaterLR::getHBasisCopy);


	lr_upd.def("setHBasis", &OperatorProjectorUpdaterLR::setHBasis);

	// new numpy version (2D: [rank, time])
//	lr_upd.def("setHBasisFromNumpy",
//	     [](OperatorProjectorUpdaterLR& self, py::buffer& np_data) {
//	        py::buffer_info buffer = np_data.request();
//		    if (buffer.ndim != 2) {
//			     throw std::invalid_argument("HBasis numpy array must be 2D (rank x time).");
//		     }
//		     if (buffer.format != py::format_descriptor<float>::format())
//		     {
//			     throw std::invalid_argument(
//			         "HBasis buffer given has to have a float32 format");
//		     }
//
//		    const int rank = static_cast<int>(buffer.shape[0]);
//		    const int numTimeFrames = static_cast<int>(buffer.shape[1]);
//		    // Create an alias into the numpy data
//		    Array2DAlias<float> alias;
//		    alias.bind(reinterpret_cast<float*>(buffer.ptr), rank, numTimeFrames);
//		    self.setHBasis(alias);
//	     },
//	     py::arg("HBasis"), py::keep_alive<1, 2>());

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
    float weight, float* cur_img_ptr,
    int offset_x, frame_t dynamicFrame,
    size_t numVoxelPerFrame) const
{
	(void) dynamicFrame;
	(void) numVoxelPerFrame;
	return weight * cur_img_ptr[offset_x];
}

void OperatorProjectorUpdaterDefault3D::backUpdate(
    float value, float weight, float* cur_img_ptr,
    int offset_x, frame_t dynamicFrame,
    size_t numVoxelPerFrame)
{
	(void) dynamicFrame;
	(void) numVoxelPerFrame;
	float output = value * weight;
	std::atomic_ref<float> atomic_elem(cur_img_ptr[offset_x]);
	atomic_elem.fetch_add(output);
}


float OperatorProjectorUpdaterDefault4D::forwardUpdate(
    float weight, float* cur_img_ptr,
    int offset_x, frame_t dynamicFrame,
    size_t numVoxelPerFrame) const
{
	return weight * cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset_x];
}

void OperatorProjectorUpdaterDefault4D::backUpdate(
    float value, float weight, float* cur_img_ptr,
    int offset_x, frame_t dynamicFrame,
    size_t numVoxelPerFrame)
{
	float output = value * weight;
	std::atomic_ref<float> atomic_elem(cur_img_ptr[dynamicFrame * numVoxelPerFrame + offset_x]);
	atomic_elem.fetch_add(output);
}


//OperatorProjectorUpdaterLR::OperatorProjectorUpdaterLR(
//    const Array2D<float>& HBasis,
//    bool           updateH
//    )
//    : OperatorProjectorUpdater()
//      , m_HBasis()
//      , m_updateH(updateH)
//{
//	// infer dimensions from HBasis [rank x numTimeFrames]
//	auto dims = m_HBasis.getDims();
//
//	if (dims[0] < 0 || dims[1] < 0) {
//		throw std::invalid_argument("HBasis must have nonzero dimensions");
//	}
//
//	m_rank = static_cast<int>(dims[0]);
//	dynamicFrames = static_cast<int>(dims[1]);
//
//	m_HBasis.allocate(dims[0], dims[1]);
//	m_HBasis.copy(HBasis);  // copies data; m_HBasis is mutable after this
//}

OperatorProjectorUpdaterLR::OperatorProjectorUpdaterLR(
    const Array2DBase<float>& pr_HBasis)
{
	setHBasis(pr_HBasis);
}

const Array2DAlias<float>& OperatorProjectorUpdaterLR::getHBasis() const
{
	return mp_HBasis;
}

std::unique_ptr<Array2D<float>> OperatorProjectorUpdaterLR::getHBasisCopy() const
{
	auto dims = mp_HBasis.getDims();
	auto out  = std::make_unique<Array2D<float>>();
	out->allocate(dims[0], dims[1]);
	out->copy(mp_HBasis);
	return out;

}

void OperatorProjectorUpdaterLR::setHBasis(const Array2DBase<float>& pr_HBasis) {
	mp_HBasis.bind(pr_HBasis);
	auto dims = mp_HBasis.getDims();
	m_rank = static_cast<int>(dims[0]);
	m_numDynamicFrames = static_cast<int>(dims[1]);
}

void OperatorProjectorUpdaterLR::setHBasisWrite(const Array2DBase<float>& pr_HWrite) {
	mp_HWrite.bind(pr_HWrite);
}

const Array2DAlias<float>& OperatorProjectorUpdaterLR::getHBasisWrite() {
	return mp_HWrite;
}

//void OperatorProjectorUpdaterLR::setHBasis(const Array2D<float>& HBasis) {
//	auto dims = HBasis.getDims(); // [rank, numTimeFrames]
//	if (dims[0] == 0 || dims[1] == 0) {
//		throw std::invalid_argument("HBasis must have nonzero dimensions");
//	}
//	m_rank = static_cast<int>(dims[0]);
//	dynamicFrames = static_cast<int>(dims[1]);
//
//	// Bind the alias to the provided backing storage
//	m_HBasis.bind(HBasis);  // Array2DAlias::bind(const Array2DBase<T>&)
//}

//void OperatorProjectorUpdaterLR::setHBasis(const Array2DAlias<float>& HBasisAlias) {
//	auto dims = HBasisAlias.getDims();
//	m_rank = static_cast<int>(dims[0]);
//	dynamicFrames = static_cast<int>(dims[1]);
//	m_HBasis.bind(HBasisAlias); // alias-of-an-alias ??
//}

void OperatorProjectorUpdaterLR::setUpdateH(bool updateH) {
	m_updateH = updateH;
}

bool OperatorProjectorUpdaterLR::getUpdateH() const {
	return m_updateH;
}


float OperatorProjectorUpdaterLR::forwardUpdate(
    float weight, float* cur_img_ptr,
    int offset_x, frame_t dynamicFrame, size_t numVoxelPerFrame) const
{
	float cur_img_lr_val = 0.0f;
	const float* H_ptr = mp_HBasis.getRawPointer();

	for (int l = 0; l < m_rank; ++l)
	{
		float cur_H_ptr = *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
		const size_t offset_rank = l * numVoxelPerFrame;
		cur_img_lr_val += cur_img_ptr[offset_x + offset_rank] * cur_H_ptr;
	}
	return weight * cur_img_lr_val;
}

// void OperatorProjectorUpdaterLR::backUpdate(
//     float value, float weight, float* cur_img_ptr,
//     int offset_x, frame_t dynamicFrame, size_t numVoxelPerFrame)
// {
// 	const float Ay = value * weight;
// 	float* H_ptr = mp_HBasis.getRawPointer();
//
// 	if (! m_updateH)
// 	{
// 		for (int l = 0; l < m_rank; ++l)
// 		{
// 			const float cur_H_ptr = *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
// 			const size_t offset_rank = l * numVoxelPerFrame;
// 			const float output = Ay * cur_H_ptr;
// 			float* ptr = &cur_img_ptr[offset_x + offset_rank];
// #pragma omp atomic
// 			*ptr += output;
// 		}
// 	}
// 	else {
// 		for (int l = 0; l < m_rank; ++l) {
// 			const size_t offset_rank = l * numVoxelPerFrame;
// 			const float output = Ay * cur_img_ptr[offset_x + offset_rank];
// 			float* ptr = H_ptr + l * m_numDynamicFrames + dynamicFrame;
// #pragma omp atomic
// 			*ptr += output;
// 		}
// 	}
//
// }


void OperatorProjectorUpdaterLR::backUpdate(
	float value, float weight, float* cur_img_ptr,
	int offset_x, frame_t dynamicFrame, size_t numVoxelPerFrame)
{
	const float Ay = value * weight;

	if (!m_updateH)
	{
		const float* H_ptr = mp_HBasis.getRawPointer();
		for (int l = 0; l < m_rank; ++l)
		{
			const float cur_H_ptr = *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
			const size_t offset_rank = l * numVoxelPerFrame;
			const float output = Ay * cur_H_ptr;
			std::atomic_ref<float> atomic_elem(cur_img_ptr[offset_x + offset_rank]);
			atomic_elem.fetch_add(output);
		}
	}
	else {
		float* H_ptr = mp_HWrite.getRawPointer();
		for (int l = 0; l < m_rank; ++l) {
			const size_t offset_rank = l * numVoxelPerFrame;
			const float output = Ay * cur_img_ptr[offset_x + offset_rank];
			std::atomic_ref<float> atomic_elem(H_ptr[l * m_numDynamicFrames + dynamicFrame]);
			atomic_elem.fetch_add(output);
		}
	}
}


} // namespace yrt
