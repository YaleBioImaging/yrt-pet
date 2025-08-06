/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdater.hpp"


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
			  int event_timeframe,
			  size_t numVoxelPerFrame) -> float {
			 return self.forwardUpdate(weight, in_image->getRawPointer(),
			 offset_x, event_timeframe, numVoxelPerFrame);
		   },
		   py::arg("weight"), py::arg("in_image"),
		   py::arg("offset_x"), py::arg("event_timeframe") = 0,
		   py::arg("numVoxelPerFrame") = 0);
	c.def("backUpdate",
			   [](OperatorProjectorUpdater& self,
				  float value,
				  float weight,
				  Image* in_image,
				  int offset_x,
				  int event_timeframe,
				  size_t numVoxelPerFrame) -> void {
				 self.backUpdate(value, weight, in_image->getRawPointer(),
				 offset_x, event_timeframe, numVoxelPerFrame);
				 },
			   py::arg("value"), py::arg("weight"), py::arg("in_image"),
			   py::arg("offset_x"), py::arg("event_timeframe") = 0,
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

	lr_upd.def(py::init<>());

	lr_upd.def(
	    "getHBasisArray",
	    [](OperatorProjectorUpdaterLR& self) {
		    const Array2DAlias<float>& H = self.getHBasis();
		    auto dims = H.getDims();
		    size_t rank = dims[0];
		    size_t numTimeFrames = dims[1];

		    // Allocate a new NumPy array (row-major: [rank, time])
		    py::array_t<float> arr({static_cast<ssize_t>(rank),
		                            static_cast<ssize_t>(numTimeFrames)});
		    auto buf = arr.request();
		    float* out = static_cast<float*>(buf.ptr);

		    // Copy data from H into the numpy array
		    for (size_t l = 0; l < rank; ++l) {
			    for (size_t t = 0; t < numTimeFrames; ++t) {
				    out[l * numTimeFrames + t] = H[l][t];
			    }
		    }
		    return arr;
	    }  // no py::arg() here
	);

//	lr_upd.def("getHBasis", &OperatorProjectorUpdaterLR::getHBasis,
//	           py::return_value_policy::reference_internal);

	lr_upd.def("setHBasis", [](OperatorProjectorUpdaterLR& self,
	                           const Array2DAlias<float>& HBasis) {
		     self.setHBasis(HBasis);
	     });

	// new numpy version (2D: [rank, time])
	lr_upd.def("setHBasisFromNumpy",
	     [](OperatorProjectorUpdaterLR& self, py::buffer& np_data) {
	        py::buffer_info buffer = np_data.request();
		    if (buffer.ndim != 2) {
			     throw std::invalid_argument("HBasis numpy array must be 2D (rank x time).");
		     }
		     if (buffer.format != py::format_descriptor<float>::format())
		     {
			     throw std::invalid_argument(
			         "HBasis buffer given has to have a float32 format");
		     }

		    const int rank = static_cast<int>(buffer.shape[0]);
		    const int numTimeFrames = static_cast<int>(buffer.shape[1]);
		    // Create an alias into the numpy data
		    Array2DAlias<float> alias;
		    alias.bind(reinterpret_cast<float*>(buffer.ptr), rank, numTimeFrames);
		    self.setHBasis(alias);
		    // Keep numpy array alive by attaching it to the Python wrapper object
//		    py::object py_self = py::cast(self);
//		    py_self.attr("_hbasis_backing") = np_data;
	     },
	     py::arg("HBasis"), py::keep_alive<1, 2>());

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
    int offset_x, int event_timeframe,
    size_t numVoxelPerFrame) const
{
	(void) event_timeframe;
	(void) numVoxelPerFrame;
	return weight * cur_img_ptr[offset_x];
}

void OperatorProjectorUpdaterDefault3D::backUpdate(
    float value, float weight, float* cur_img_ptr,
    int offset_x, int event_timeframe,
    size_t numVoxelPerFrame)
{
	(void) event_timeframe;
	(void) numVoxelPerFrame;
	float output = value * weight;
	float* ptr = &cur_img_ptr[offset_x];
#pragma omp atomic
	*ptr += output;
}


float OperatorProjectorUpdaterDefault4D::forwardUpdate(
    float weight, float* cur_img_ptr,
    int offset_x, int event_timeframe,
    size_t numVoxelPerFrame) const
{
	return weight * cur_img_ptr[event_timeframe * numVoxelPerFrame + offset_x];
}

void OperatorProjectorUpdaterDefault4D::backUpdate(
    float value, float weight, float* cur_img_ptr,
    int offset_x, int event_timeframe,
    size_t numVoxelPerFrame)
{
	float output = value * weight;
	float* ptr = &cur_img_ptr[event_timeframe * numVoxelPerFrame + offset_x];
#pragma omp atomic
	*ptr += output;
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
//	m_numTimeFrames = static_cast<int>(dims[1]);
//
//	m_HBasis.allocate(dims[0], dims[1]);
//	m_HBasis.copy(HBasis);  // copies data; m_HBasis is mutable after this
//}

const Array2DAlias<float>& OperatorProjectorUpdaterLR::getHBasis() const
{
	return m_HBasis;
}

void OperatorProjectorUpdaterLR::setHBasis(const Array2D<float>& HBasis) {
	auto dims = HBasis.getDims(); // [rank, numTimeFrames]
	if (dims[0] == 0 || dims[1] == 0) {
		throw std::invalid_argument("HBasis must have nonzero dimensions");
	}
	m_rank = static_cast<int>(dims[0]);
	m_numTimeFrames = static_cast<int>(dims[1]);

	// Bind the alias to the provided backing storage
	m_HBasis.bind(HBasis);  // Array2DAlias::bind(const Array2DBase<T>&)
}

void OperatorProjectorUpdaterLR::setHBasis(const Array2DAlias<float>& HBasisAlias) {
	auto dims = HBasisAlias.getDims();
	m_rank = static_cast<int>(dims[0]);
	m_numTimeFrames = static_cast<int>(dims[1]);
	m_HBasis.bind(HBasisAlias); // alias-of-an-alias ??
}

void OperatorProjectorUpdaterLR::setUpdateH(bool updateH) {
	m_updateH = updateH;
}

bool OperatorProjectorUpdaterLR::getUpdateH() const {
	return m_updateH;
}


float OperatorProjectorUpdaterLR::forwardUpdate(
    float weight, float* cur_img_ptr,
    int offset_x, int event_timeframe, size_t numVoxelPerFrame) const
{
	float cur_img_lr_val = 0.0f;
	const float* H_ptr = m_HBasis.getRawPointer();
//	const int nt = static_cast<int>(m_numTimeFrames);

	for (int l = 0; l < m_rank; ++l)
	{
		float cur_H_ptr = *(H_ptr + l * m_numTimeFrames + event_timeframe);
		const size_t offset_rank = l * numVoxelPerFrame;
		cur_img_lr_val += cur_img_ptr[offset_x + offset_rank] * cur_H_ptr;
	}
	return weight * cur_img_lr_val;
}

void OperatorProjectorUpdaterLR::backUpdate(
    float value, float weight, float* cur_img_ptr,
    int offset_x, int event_timeframe, size_t numVoxelPerFrame)
{
	float Ay = value * weight;
	float* H_ptr = m_HBasis.getRawPointer();

	if (! m_updateH)
	{
		for (int l = 0; l < m_rank; ++l)
		{
			float cur_H_ptr = *(H_ptr + l * m_numTimeFrames + event_timeframe);
			const size_t offset_rank = l * numVoxelPerFrame;
			float output = Ay * cur_H_ptr;
			float* ptr = &cur_img_ptr[offset_x + offset_rank];
#pragma omp atomic
			*ptr += output;
		}
	}
	else {
		for (int l = 0; l < m_rank; ++l) {
			const size_t offset_rank = l * numVoxelPerFrame;
			float output = Ay * cur_img_ptr[offset_x + offset_rank];
			float* ptr = H_ptr + l * m_numTimeFrames + event_timeframe;
#pragma omp atomic
			*ptr += output;
		}
	}

}


} // namespace yrt