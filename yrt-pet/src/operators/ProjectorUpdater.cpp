/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/ProjectorUpdater.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/utils/Types.hpp"


#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>
namespace py = pybind11;

namespace yrt
{
void py_setup_projectorupdater(py::module& m)
{
	auto c = py::class_<ProjectorUpdater, std::shared_ptr<ProjectorUpdater>>(
	    m, "ProjectorUpdater");
	c.def(
	    "forwardUpdate",
	    [](ProjectorUpdater& self, float weight, Image* in_image, int offset_x,
	       frame_t dynamicFrame, size_t numVoxelsPerFrame) -> float
	    {
		    return self.forwardUpdate(weight, in_image->getRawPointer(),
		                              offset_x, dynamicFrame,
		                              numVoxelsPerFrame);
	    },
	    py::arg("weight"), py::arg("in_image"), py::arg("offset_x"),
	    py::arg("dynamicFrame") = 0, py::arg("numVoxelsPerFrame") = 0);
	c.def(
	    "backUpdate",
	    [](ProjectorUpdater& self, float value, float weight, Image* in_image,
	       int offset_x, frame_t dynamicFrame, size_t numVoxelsPerFrame,
	       int threadId) -> void
	    {
		    self.backUpdate(value, weight, in_image->getRawPointer(), offset_x,
		                    dynamicFrame, numVoxelsPerFrame, threadId);
	    },
	    py::arg("value"), py::arg("weight"), py::arg("in_image"),
	    py::arg("offset_x"), py::arg("dynamicFrame") = 0,
	    py::arg("numVoxelsPerFrame") = 0, py::arg("threadId") = 0);

	// Default updater
	auto def_upd_4d = py::class_<ProjectorUpdaterDefault4D, ProjectorUpdater,
	                             std::shared_ptr<ProjectorUpdaterDefault4D>>(
	    m, "ProjectorUpdaterDefault4D");
	def_upd_4d.def(py::init<>());

	// LR updater
	auto lr_upd = py::class_<ProjectorUpdaterLR, ProjectorUpdater,
	                         std::shared_ptr<ProjectorUpdaterLR>>(
	    m, "ProjectorUpdaterLR");

	lr_upd.def(py::init<Array2DOwned<float>&>());
	lr_upd.def("getHBasisCopy", &ProjectorUpdaterLR::getHBasisCopy);


	lr_upd.def("setHBasis", &ProjectorUpdaterLR::setHBasis);

	// new numpy version (2D: [rank, time])
	//	lr_upd.def("setHBasisFromNumpy",
	//	     [](ProjectorUpdaterLR& self, py::buffer& np_data) {
	//	        py::buffer_info buffer = np_data.request();
	//		    if (buffer.ndim != 2) {
	//			     throw std::invalid_argument("HBasis numpy array must be 2D
	//(rank x time).");
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
	//		    alias.bind(reinterpret_cast<float*>(buffer.ptr), rank,
	// numTimeFrames); 		    self.setHBasis(alias);
	//	     },
	//	     py::arg("HBasis"), py::keep_alive<1, 2>());

	lr_upd.def("setUpdateH", [](ProjectorUpdaterLR& self, bool updateH)
	           { self.setUpdateH(updateH); });

	lr_upd.def("getUpdateH", &ProjectorUpdaterLR::getUpdateH);
}
}  // namespace yrt

#endif


namespace yrt
{

float ProjectorUpdaterDefault4D::forwardUpdate(float weight, float* cur_img_ptr,
                                               size_t offset,
                                               frame_t dynamicFrame,
                                               size_t numVoxelsPerFrame) const
{
	return weight * cur_img_ptr[dynamicFrame * numVoxelsPerFrame + offset];
}

void ProjectorUpdaterDefault4D::backUpdate(float value, float weight,
                                           float* cur_img_ptr, size_t offset,
                                           frame_t dynamicFrame,
                                           size_t numVoxelsPerFrame,
                                           int /*tid*/)
{
	const float output = value * weight;
	const std::atomic_ref<float> atomic_elem(
	    cur_img_ptr[dynamicFrame * numVoxelsPerFrame + offset]);
	atomic_elem += output;
}


// ProjectorUpdaterLR::ProjectorUpdaterLR(
//     const Array2D<float>& HBasis,
//     bool           updateH
//     )
//     : ProjectorUpdater()
//       , m_HBasis()
//       , m_updateH(updateH)
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
// }

ProjectorUpdaterLR::ProjectorUpdaterLR(const Array2DBase<float>& pr_HBasis)
{
	setHBasis(pr_HBasis);
}

Array2DAlias<float> ProjectorUpdaterLR::getHBasis()
{
	return Array2DAlias<float>(mp_HBasis);
}

std::unique_ptr<Array2DOwned<float>> ProjectorUpdaterLR::getHBasisCopy() const
{
	const auto dims = mp_HBasis.getDims();
	auto out = std::make_unique<Array2DOwned<float>>();
	out->allocate(dims[0], dims[1]);
	out->copy(mp_HBasis);
	return out;
}

void ProjectorUpdaterLR::setHBasis(const Array2DBase<float>& pr_HBasis)
{
	mp_HBasis.bind(pr_HBasis);
	const auto dims = mp_HBasis.getDims();
	m_rank = static_cast<int>(dims[0]);
	m_numDynamicFrames = static_cast<int>(dims[1]);
}

void ProjectorUpdaterLR::setHBasisWrite(const Array2DBase<float>& pr_HWrite)
{
	mp_HWrite.bind(pr_HWrite);
	initializeWriteThread();
}

const Array2DAlias<float>& ProjectorUpdaterLR::getHBasisWrite()
{
	return mp_HWrite;
}

const Array3DOwned<double>& ProjectorUpdaterLR::getHBasisWriteThread()
{
	return m_HWriteThread;
}

void ProjectorUpdaterLR::setCurrentImgBuffer(ImageBase* img)
{
	if (auto* imgCPU = dynamic_cast<ImageOwned*>(img))
	{
		m_currentImg = imgCPU->getRawPointer();
	}
#if BUILD_CUDA
	else if (auto* imgGPU = dynamic_cast<ImageDevice*>(img))
	{
		m_currentImg = imgGPU->getDevicePointer();
	}
#endif
	else
	{
		throw std::runtime_error("Unsupported image type");
	}

	ASSERT_MSG(m_currentImg != nullptr, "Null image data pointer");
}

const float* ProjectorUpdaterLR::getCurrentImgBuffer() const
{
	return m_currentImg;
}

void ProjectorUpdaterLR::initializeWriteThread()
{
	const int numThreads = globals::getNumThreads();
	m_HWriteThread.allocate(numThreads, m_rank, m_numDynamicFrames);
	m_HWriteThread.fill(0.0);
}


void ProjectorUpdaterLR::accumulateH()
{
	const int numThreads = m_HWriteThread.getDims()[0];
	for (int t = 0; t < m_numDynamicFrames; ++t)
	{
		for (int r = 0; r < m_rank; ++r)
		{
			double sum = 0.0;
			for (int i = 0; i < numThreads; ++i)
			{
				sum += m_HWriteThread[i][r][t];
			}
			mp_HWrite[r][t] = sum;
		}
	}
	m_HWriteThread.fill(0.0);
}


void ProjectorUpdaterLR::setUpdateH(bool updateH)
{
	m_updateH = updateH;
}

bool ProjectorUpdaterLR::getUpdateH() const
{
	return m_updateH;
}


float ProjectorUpdaterLR::forwardUpdate(float weight, float* cur_img_ptr,
                                        size_t offset, frame_t dynamicFrame,
                                        size_t numVoxelsPerFrame) const
{
	float cur_img_lr_val = 0.0f;
	const float* H_ptr = mp_HBasis.getRawPointer();

	for (int l = 0; l < m_rank; ++l)
	{
		const float cur_H_ptr =
		    *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
		const size_t offset_rank = l * numVoxelsPerFrame;
		cur_img_lr_val += cur_img_ptr[offset + offset_rank] * cur_H_ptr;
	}
	return weight * cur_img_lr_val;
}

void ProjectorUpdaterLR::backUpdate(float value, float weight,
                                    float* cur_img_ptr, size_t offset,
                                    frame_t dynamicFrame,
                                    size_t numVoxelsPerFrame, int tid)
{
	const float Ay = value * weight;

	if (!m_updateH)
	{
		const float* H_ptr = mp_HBasis.getRawPointer();
		for (int l = 0; l < m_rank; ++l)
		{
			const float cur_H_ptr =
			    *(H_ptr + l * m_numDynamicFrames + dynamicFrame);
			const size_t offset_rank = l * numVoxelsPerFrame;
			const float output = Ay * cur_H_ptr;
			const std::atomic_ref<float> atomic_elem(
			    cur_img_ptr[offset + offset_rank]);
			atomic_elem += output;
		}
	}
	else
	{
		auto* H_ptr = m_HWriteThread[tid].getRawPointer();
		for (int l = 0; l < m_rank; ++l)
		{
			const size_t offset_rank = l * numVoxelsPerFrame;
			const double output = Ay * cur_img_ptr[offset + offset_rank];
			H_ptr[l * m_numDynamicFrames + dynamicFrame] += output;
		}
	}
}

ProjectorUpdaterLRDualUpdate::ProjectorUpdaterLRDualUpdate(
    const Array2DBase<float>& pr_HBasis)
    : ProjectorUpdaterLR(pr_HBasis)
{
	setUpdateH(true);
}

void ProjectorUpdaterLRDualUpdate::backUpdate(float value, float weight,
                                              float* cur_img_ptr, size_t offset,
                                              frame_t dynamicFrame,
                                              size_t numVoxelsPerFrame, int tid)
{
	const float Ay = value * weight;
	const float* H_ptr_read = mp_HBasis.getRawPointer();
	const float* W_ptr_read = this->getCurrentImgBuffer();
	double* H_ptr_write = m_HWriteThread[tid].getRawPointer();

	for (int l = 0; l < m_rank; ++l)
	{

		const float cur_H_ptr =
		    *(H_ptr_read + l * m_numDynamicFrames + dynamicFrame);
		const size_t offset_rank = l * numVoxelsPerFrame;
		const float outputWUpdate = Ay * cur_H_ptr;
		const float outputHUpdate = Ay * W_ptr_read[offset + offset_rank];
		H_ptr_write[l * m_numDynamicFrames + dynamicFrame] += outputHUpdate;
		const std::atomic_ref<float> atomic_elemW(
		    cur_img_ptr[offset + offset_rank]);
		atomic_elemW += outputWUpdate;
	}
}


}  // namespace yrt
