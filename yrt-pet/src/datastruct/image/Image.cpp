/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/Image.hpp"

#include "datastruct/image/ImageBase.hpp"
#include "geometry/Constants.hpp"
#include "utils/Assert.hpp"
#include "utils/Tools.hpp"
#include "utils/Types.hpp"
#include "utils/Utilities.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_image(py::module& m)
{
	auto c = py::class_<Image, ImageBase>(m, "Image", py::buffer_protocol());
	c.def("setValue", &Image::setValue, py::arg("initValue"));
	c.def_buffer(
	    [](Image& img) -> py::buffer_info
	    {
		    Array3DBase<float>& d = img.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 3,
		                           d.getDims(), d.getStrides());
	    });
	c.def("isMemoryValid", &Image::isMemoryValid);
	c.def("copyFromImage", &Image::copyFromImage, py::arg("sourceImage"));
	c.def("multWithScalar", &Image::multWithScalar, py::arg("scalar"));
	c.def("addFirstImageToSecond", &Image::addFirstImageToSecond,
	      py::arg("secondImage"));
	c.def("applyThreshold", &Image::applyThreshold, py::arg("maskImage"),
	      py::arg("threshold"), py::arg("val_le_scale"), py::arg("val_le_off"),
	      py::arg("val_gt_scale"), py::arg("val_gt_off"));
	c.def("updateEMThreshold", &Image::updateEMThreshold, py::arg("updateImg"),
	      py::arg("normImage"), py::arg("threshold"));
	c.def("dotProduct", &Image::dotProduct, py::arg("y"));
	c.def("getRadius", &Image::getRadius);
	c.def("getParams", &Image::getParams);
	c.def("interpolImage",
	      static_cast<float (Image::*)(const Vector3D& pt, const Image& sens)
	                      const>(&Image::interpolateImage),
	      py::arg("pt"), py::arg("sens"));
	c.def("interpolImage",
	      static_cast<float (Image::*)(const Vector3D& pt) const>(
	          &Image::interpolateImage),
	      py::arg("pt"));
	c.def(
	    "nearestNeighbor",
	    [](const Image& img, const Vector3D& pt) -> py::tuple
	    {
		    int pi, pj, pk;
		    float val = img.nearestNeighbor(pt, &pi, &pj, &pk);
		    return py::make_tuple(val, pi, pj, pk);
	    },
	    py::arg("pt"));
	c.def(
	    "getNearestNeighborIdx",
	    [](const Image& img, const Vector3D& pt) -> py::tuple
	    {
		    int pi, pj, pk;
		    img.getNearestNeighborIdx(pt, &pi, &pj, &pk);
		    return py::make_tuple(pi, pj, pk);
	    },
	    py::arg("pt"));
	c.def("getArray", &Image::getArray);
	c.def("transformImage",
	      static_cast<std::unique_ptr<Image> (Image::*)(
	          const Vector3D& rotation, const Vector3D& translation) const>(
	          &Image::transformImage),
	      py::arg("rotation"), py::arg("translation"));
	c.def("transformImage",
	      static_cast<std::unique_ptr<Image> (Image::*)(const transform_t& t)
	                      const>(&Image::transformImage),
	      py::arg("transform"));
	c.def("updateImageNearestNeighbor", &Image::updateImageNearestNeighbor,
	      py::arg("pt"), py::arg("value"), py::arg("doMultiplication"));
	c.def("assignImageNearestNeighbor", &Image::assignImageNearestNeighbor,
	      py::arg("pt"), py::arg("value"));
	c.def("updateImageInterpolate", &Image::updateImageInterpolate,
	      py::arg("pt"), py::arg("value"), py::arg("doMultiplication") = false);
	c.def("assignImageInterpolate", &Image::assignImageInterpolate,
	      py::arg("pt"), py::arg("value"));
	c.def("writeToFile", &Image::writeToFile, py::arg("filename"));

	auto c_alias = py::class_<ImageAlias, Image>(m, "ImageAlias");
	c_alias.def(py::init<const ImageParams&>(), py::arg("img_params"));
	c_alias.def(
	    "bind",
	    [](ImageAlias& self, py::buffer& np_data)
	    {
		    py::buffer_info buffer = np_data.request();
		    if (buffer.ndim != 3)
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have 3 dimensions");
		    }
		    if (buffer.format != py::format_descriptor<float>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float32 format");
		    }
		    std::vector<int> dims = {self.getParams().nz, self.getParams().ny,
		                             self.getParams().nx};
		    for (int i = 0; i < 3; i++)
		    {
			    if (buffer.shape[i] != dims[i])
			    {
				    throw std::invalid_argument(
				        "The buffer shape does not match with the image "
				        "parameters");
			    }
		    }
		    static_cast<Array3DAlias<float>&>(self.getData())
		        .bind(reinterpret_cast<float*>(buffer.ptr), dims[0], dims[1],
		              dims[2]);
	    },
	    py::arg("numpy_data"));

	auto c_owned = py::class_<ImageOwned, Image>(m, "ImageOwned");
	c_owned.def(py::init<const ImageParams&>(), py::arg("img_params"));
	c_owned.def(py::init<const ImageParams&, std::string>(),
	            py::arg("img_params"), py::arg("filename"));
	c_owned.def(py::init<std::string>(), py::arg("filename"));
	c_owned.def("allocate", &ImageOwned::allocate);
	c_owned.def("readFromFile", &ImageOwned::readFromFile, py::arg("filename"));
}

#endif  // if BUILD_PYBIND11

Image::Image() : ImageBase{} {}

Image::Image(const ImageParams& imgParams) : ImageBase(imgParams) {}

void Image::setValue(float initValue)
{
	mp_array->fill(initValue);
}

void Image::copyFromImage(const ImageBase* imSrc)
{
	const auto imSrc_ptr = dynamic_cast<const Image*>(imSrc);
	ASSERT_MSG(imSrc_ptr != nullptr, "Image not in host");
	ASSERT_MSG(mp_array != nullptr, "Image not allocated");
	mp_array->copy(imSrc_ptr->getData());
	setParams(imSrc_ptr->getParams());
}

Array3DBase<float>& Image::getData()
{
	return *mp_array;
}

const Array3DBase<float>& Image::getData() const
{
	return *mp_array;
}

float* Image::getRawPointer()
{
	return mp_array->getRawPointer();
}

const float* Image::getRawPointer() const
{
	return mp_array->getRawPointer();
}

bool Image::isMemoryValid() const
{
	return mp_array != nullptr && mp_array->getRawPointer() != nullptr;
}

void Image::addFirstImageToSecond(ImageBase* secondImage) const
{
	auto* second_Image = dynamic_cast<Image*>(secondImage);

	ASSERT(second_Image != nullptr);
	ASSERT_MSG(secondImage->getParams().isSameDimensionsAs(getParams()),
	           "The two images do not share the same image space");

	second_Image->getData() += *mp_array;
}

void Image::multWithScalar(float scalar)
{
	*mp_array *= scalar;
}

// return the value of the voxel the nearest to "point":
float Image::nearestNeighbor(const Vector3D& pt) const
{
	int ix, iy, iz;

	if (getNearestNeighborIdx(pt, &ix, &iy, &iz))
	{
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		return mp_array->getFlat(iz * num_xy + iy * num_x + ix);
	}
	return 0;
}

// return the value of the voxel the nearest to "point":
float Image::nearestNeighbor(const Vector3D& pt, int* pi, int* pj,
                             int* pk) const
{
	if (getNearestNeighborIdx(pt, pi, pj, pk))
	{
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		return mp_array->getFlat(*pk * num_xy + *pj * num_x + *pi);
	}
	return 0.0;
}


// update image with "value" using nearest neighbor method:
void Image::updateImageNearestNeighbor(const Vector3D& pt, float value,
                                       bool mult_flag)
{
	int ix, iy, iz;
	if (getNearestNeighborIdx(pt, &ix, &iy, &iz))
	{
		// update multiplicatively or additively:
		float* ptr = mp_array->getRawPointer();
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		const size_t idx = iz * num_xy + iy * num_x + ix;
		if (mult_flag)
		{
			ptr[idx] *= value;
		}
		else
		{
			ptr[idx] += value;
		}
	}
}

// assign image with "value" using nearest neighbor method:
void Image::assignImageNearestNeighbor(const Vector3D& pt, float value)
{
	int ix, iy, iz;
	if (getNearestNeighborIdx(pt, &ix, &iy, &iz))
	{
		// update multiplicatively or additively:
		float* ptr = mp_array->getRawPointer();
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		ptr[iz * num_xy + iy * num_x + ix] = value;
	}
}

// Returns true if the point `pt` is inside the image
bool Image::getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj,
                                  int* pk) const
{
	const ImageParams& params = getParams();
	const float x = pt.x - params.off_x;
	const float y = pt.y - params.off_y;
	const float z = pt.z - params.off_z;

	const float dx = (x + params.length_x / 2.0) / params.length_x *
	                 static_cast<float>(params.nx);
	const float dy = (y + params.length_y / 2.0) / params.length_y *
	                 static_cast<float>(params.ny);
	const float dz = (z + params.length_z / 2.0) / params.length_z *
	                 static_cast<float>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

	if (ix < 0 || ix >= params.nx || iy < 0 || iy >= params.ny || iz < 0 ||
	    iz >= params.nz)
	{
		// Point outside grid
		return false;
	}

	*pi = ix;
	*pj = iy;
	*pk = iz;

	return true;
}


// interpolation operation. It does not account for the offset values.
float Image::interpolateImage(const Vector3D& pt) const
{
	const ImageParams& params = getParams();
	const float x = pt.x - params.off_x;
	const float y = pt.y - params.off_y;
	const float z = pt.z - params.off_z;

	const float dx = (x + params.length_x / 2.0f) / params.length_x *
	                 static_cast<float>(params.nx);
	const float dy = (y + params.length_y / 2.0f) / params.length_y *
	                 static_cast<float>(params.ny);
	const float dz = (z + params.length_z / 2.0f) / params.length_z *
	                 static_cast<float>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

	if (ix < 0 || ix >= params.nx || iy < 0 || iy >= params.ny || iz < 0 ||
	    iz >= params.nz)
	{
		// Point outside grid
		return 0.0f;
	}

	const float delta_x = dx - static_cast<float>(ix);
	const float delta_y = dy - static_cast<float>(iy);
	const float delta_z = dz - static_cast<float>(iz);

	// parameters of the x interpolation:
	int ix1, ix2, iy1, iy2, iz1, iz2;
	float dx1, dy1, dz1;
	if (delta_x < 0.5)
	{
		ix1 = ix;
		dx1 = 0.5 - delta_x;
		if (ix != 0)
			ix2 = ix - 1;
		else
			ix2 = ix1;
	}
	else
	{
		ix1 = ix;
		dx1 = delta_x - 0.5;
		if (ix != (params.nx - 1))
			ix2 = ix + 1;
		else
			ix2 = ix1;
	}
	// parameters of the y interpolation:
	if (delta_y < 0.5)
	{
		iy1 = iy;
		dy1 = 0.5 - delta_y;
		if (iy != 0)
			iy2 = iy - 1;
		else
			iy2 = iy1;
	}
	else
	{
		iy1 = iy;
		dy1 = delta_y - 0.5;
		if (iy != (params.ny - 1))
			iy2 = iy + 1;
		else
			iy2 = iy1;
	}
	// parameters of the z interpolation:
	if (delta_z < 0.5)
	{
		iz1 = iz;
		dz1 = 0.5 - delta_z;
		if (iz != 0)
			iz2 = iz - 1;
		else
			iz2 = iz1;
	}
	else
	{
		iz1 = iz;
		dz1 = delta_z - 0.5;
		if (iz != (params.nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}
	// interpolate in z:
	const float* ptr = mp_array->getRawPointer();
	const size_t num_x = params.nx;
	const size_t num_xy = params.nx * params.ny;
	const float* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	const float* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	const float* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	const float* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
	const float v1 = ptr_11[ix1] * (1 - dz1) + ptr_21[ix1] * dz1;
	const float v2 = ptr_12[ix1] * (1 - dz1) + ptr_22[ix1] * dz1;
	const float v3 = ptr_11[ix2] * (1 - dz1) + ptr_21[ix2] * dz1;
	const float v4 = ptr_12[ix2] * (1 - dz1) + ptr_22[ix2] * dz1;
	// interpolate in y:
	const float vv1 = v1 * (1 - dy1) + v2 * dy1;
	const float vv2 = v3 * (1 - dy1) + v4 * dy1;
	// interpolate in the x direction:
	const float vvv = vv1 * (1 - dx1) + vv2 * dx1;

	return vvv;
}

// calculate the value of a point on the image matrix
// using tri-linear interpolation and weighting with image "sens":
float Image::interpolateImage(const Vector3D& pt, const Image& sens) const
{
	const ImageParams& params = getParams();
	const float x = pt.x - params.off_x;
	const float y = pt.y - params.off_y;
	const float z = pt.z - params.off_z;

	const float dx = (x + params.length_x / 2.0f) / params.length_x *
	                 static_cast<float>(params.nx);
	const float dy = (y + params.length_y / 2.0f) / params.length_y *
	                 static_cast<float>(params.ny);
	const float dz = (z + params.length_z / 2.0f) / params.length_z *
	                 static_cast<float>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

	if (ix < 0 || ix >= params.nx || iy < 0 || iy >= params.ny || iz < 0 ||
	    iz >= params.nz)
	{
		// Point outside grid
		return 0.0f;
	}

	const float delta_x = dx - static_cast<float>(ix);
	const float delta_y = dy - static_cast<float>(iy);
	const float delta_z = dz - static_cast<float>(iz);

	// parameters of the x interpolation:
	int ix1, ix2, iy1, iy2, iz1, iz2;
	float dx1, dy1, dz1;
	if (delta_x < 0.5)
	{
		ix1 = ix;
		dx1 = 0.5 - delta_x;
		if (ix != 0)
			ix2 = ix - 1;
		else
			ix2 = ix1;
	}
	else
	{
		ix1 = ix;
		dx1 = delta_x - 0.5;
		if (ix != (params.nx - 1))
			ix2 = ix + 1;
		else
			ix2 = ix1;
	}
	// parameters of the y interpolation:
	if (delta_y < 0.5)
	{
		iy1 = iy;
		dy1 = 0.5 - delta_y;
		if (iy != 0)
			iy2 = iy - 1;
		else
			iy2 = iy1;
	}
	else
	{
		iy1 = iy;
		dy1 = delta_y - 0.5;
		if (iy != (params.ny - 1))
			iy2 = iy + 1;
		else
			iy2 = iy1;
	}
	// parameters of the z interpolation:
	if (delta_z < 0.5)
	{
		iz1 = iz;
		dz1 = 0.5 - delta_z;
		if (iz != 0)
			iz2 = iz - 1;
		else
			iz2 = iz1;
	}
	else
	{
		iz1 = iz;
		dz1 = delta_z - 0.5;
		if (iz != (params.nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}
	// interpolate in z:
	const float* ptr = getRawPointer();
	const float* sptr = sens.getRawPointer();
	const size_t num_x = params.nx;
	const size_t num_xy = params.nx * params.ny;
	const float* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	const float* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	const float* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	const float* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
	const float* sptr_11 = sptr + iz1 * num_xy + iy1 * num_x;
	const float* sptr_21 = sptr + iz2 * num_xy + iy1 * num_x;
	const float* sptr_12 = sptr + iz1 * num_xy + iy2 * num_x;
	const float* sptr_22 = sptr + iz2 * num_xy + iy2 * num_x;
	const float v1 = ptr_11[ix1] * sptr_11[ix1] * (1 - dz1) +
	                 ptr_21[ix1] * sptr_21[ix1] * dz1;
	const float v2 = ptr_12[ix1] * sptr_12[ix1] * (1 - dz1) +
	                 ptr_22[ix1] * sptr_22[ix1] * dz1;
	const float v3 = ptr_11[ix2] * sptr_11[ix2] * (1 - dz1) +
	                 ptr_21[ix2] * sptr_21[ix2] * dz1;
	const float v4 = ptr_12[ix2] * sptr_12[ix2] * (1 - dz1) +
	                 ptr_22[ix2] * sptr_22[ix2] * dz1;
	// interpolate in y:
	const float vv1 = v1 * (1 - dy1) + v2 * dy1;
	const float vv2 = v3 * (1 - dy1) + v4 * dy1;
	// interpolate in the x direction:
	const float vvv = vv1 * (1 - dx1) + vv2 * dx1;

	return vvv;
}

// update image with "value" using trilinear interpolation:
void Image::updateImageInterpolate(const Vector3D& point, float value,
                                   bool mult_flag)
{
	const ImageParams& params = getParams();
	const float x = point.x - params.off_x;
	const float y = point.y - params.off_y;
	const float z = point.z - params.off_z;

	float dx = (x + params.length_x / 2) / params.length_x * ((float)params.nx);
	float dy = (y + params.length_y / 2) / params.length_y * ((float)params.ny);
	float dz = (z + params.length_z / 2) / params.length_z * ((float)params.nz);

	int ix = (int)dx;
	int iy = (int)dy;
	int iz = (int)dz;

	if (ix < 0 || ix >= params.nx || iy < 0 || iy >= params.ny || iz < 0 ||
	    iz >= params.nz)
	{
		// Point outside grid
		return;
	}

	float delta_x = dx - (float)ix;
	float delta_y = dy - (float)iy;
	float delta_z = dz - (float)iz;

	// parameters of the x interpolation:
	int ix1, ix2, iy1, iy2, iz1, iz2;
	float dx1, dy1, dz1;
	if (delta_x < 0.5)
	{
		ix1 = ix;
		dx1 = 0.5 - delta_x;
		if (ix != 0)
			ix2 = ix - 1;
		else
			ix2 = ix1;
	}
	else
	{
		ix1 = ix;
		dx1 = delta_x - 0.5;
		if (ix != (params.nx - 1))
			ix2 = ix + 1;
		else
			ix2 = ix1;
	}

	// parameters of the y interpolation:
	if (delta_y < 0.5)
	{
		iy1 = iy;
		dy1 = 0.5 - delta_y;
		if (iy != 0)
			iy2 = iy - 1;
		else
			iy2 = iy1;
	}
	else
	{
		iy1 = iy;
		dy1 = delta_y - 0.5;
		if (iy != (params.ny - 1))
			iy2 = iy + 1;
		else
			iy2 = iy1;
	}

	// parameters of the z interpolation:
	if (delta_z < 0.5)
	{
		iz1 = iz;
		dz1 = 0.5 - delta_z;
		if (iz != 0)
			iz2 = iz - 1;
		else
			iz2 = iz1;
	}
	else
	{
		iz1 = iz;
		dz1 = delta_z - 0.5;
		if (iz != (params.nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}

	// interpolate multiplicatively or additively:
	float* ptr = mp_array->getRawPointer();
	size_t num_x = params.nx;
	size_t num_xy = params.nx * params.ny;
	float* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	float* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	float* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	float* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
	if (mult_flag)
	{
		ptr_11[ix1] *= value * (1 - dz1) * (1 - dy1) * (1 - dx1);
		ptr_21[ix1] *= value * dz1 * (1 - dy1) * (1 - dx1);
		ptr_11[ix2] *= value * (1 - dz1) * (1 - dy1) * dx1;
		ptr_21[ix2] *= value * dz1 * (1 - dy1) * dx1;
		ptr_12[ix1] *= value * (1 - dz1) * dy1 * (1 - dx1);
		ptr_22[ix1] *= value * dz1 * dy1 * (1 - dx1);
		ptr_12[ix2] *= value * (1 - dz1) * dy1 * dx1;
		ptr_22[ix2] *= value * dz1 * dy1 * dx1;
	}
	else
	{
		ptr_11[ix1] += value * (1 - dz1) * (1 - dy1) * (1 - dx1);
		ptr_21[ix1] += value * dz1 * (1 - dy1) * (1 - dx1);
		ptr_11[ix2] += value * (1 - dz1) * (1 - dy1) * dx1;
		ptr_21[ix2] += value * dz1 * (1 - dy1) * dx1;
		ptr_12[ix1] += value * (1 - dz1) * dy1 * (1 - dx1);
		ptr_22[ix1] += value * dz1 * dy1 * (1 - dx1);
		ptr_12[ix2] += value * (1 - dz1) * dy1 * dx1;
		ptr_22[ix2] += value * dz1 * dy1 * dx1;
	}
}

// assign image with "value" using trilinear interpolation:
void Image::assignImageInterpolate(const Vector3D& point, float value)
{
	const ImageParams& params = getParams();
	const float x = point.x - params.off_x;
	const float y = point.y - params.off_y;
	const float z = point.z - params.off_z;

	float dx = (x + params.length_x / 2) / params.length_x * ((float)params.nx);
	float dy = (y + params.length_y / 2) / params.length_y * ((float)params.ny);
	float dz = (z + params.length_z / 2) / params.length_z * ((float)params.nz);

	int ix = (int)dx;
	int iy = (int)dy;
	int iz = (int)dz;

	if (ix < 0 || ix >= params.nx || iy < 0 || iy >= params.ny || iz < 0 ||
	    iz >= params.nz)
	{
		// Point outside grid
		return;
	}

	float delta_x = dx - (float)ix;
	float delta_y = dy - (float)iy;
	float delta_z = dz - (float)iz;

	// parameters of the x interpolation:
	float dx1, dy1, dz1;
	int ix1, ix2, iy1, iy2, iz1, iz2;
	if (delta_x < 0.5)
	{
		ix1 = ix;
		dx1 = 0.5 - delta_x;
		if (ix != 0)
			ix2 = ix - 1;
		else
			ix2 = ix1;
	}
	else
	{
		ix1 = ix;
		dx1 = delta_x - 0.5;
		if (ix != (params.nx - 1))
			ix2 = ix + 1;
		else
			ix2 = ix1;
	}

	// parameters of the y interpolation:
	if (delta_y < 0.5)
	{
		iy1 = iy;
		dy1 = 0.5 - delta_y;
		if (iy != 0)
			iy2 = iy - 1;
		else
			iy2 = iy1;
	}
	else
	{
		iy1 = iy;
		dy1 = delta_y - 0.5;
		if (iy != (params.ny - 1))
			iy2 = iy + 1;
		else
			iy2 = iy1;
	}

	// parameters of the z interpolation:
	if (delta_z < 0.5)
	{
		iz1 = iz;
		dz1 = 0.5 - delta_z;
		if (iz != 0)
			iz2 = iz - 1;
		else
			iz2 = iz1;
	}
	else
	{
		iz1 = iz;
		dz1 = delta_z - 0.5;
		if (iz != (params.nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}

	// assign:
	float* ptr = mp_array->getRawPointer();
	const size_t num_x = params.nx;
	const size_t num_xy = params.nx * params.ny;
	float* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	float* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	float* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	float* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
	ptr_11[ix1] = value * (1 - dz1) * (1 - dy1) * (1 - dx1);
	ptr_21[ix1] = value * dz1 * (1 - dy1) * (1 - dx1);
	ptr_11[ix2] = value * (1 - dz1) * (1 - dy1) * dx1;
	ptr_21[ix2] = value * dz1 * (1 - dy1) * dx1;
	ptr_12[ix1] = value * (1 - dz1) * dy1 * (1 - dx1);
	ptr_22[ix1] = value * dz1 * dy1 * (1 - dx1);
	ptr_12[ix2] = value * (1 - dz1) * dy1 * dx1;
	ptr_22[ix2] = value * dz1 * dy1 * dx1;
}

// this function writes "image" on disk @ "image_fname"
void Image::writeToFile(const std::string& fname) const
{
	ASSERT(!fname.empty());
	ASSERT_MSG_WARNING(
	    Util::endsWith(fname, ".nii") || Util::endsWith(fname, ".nii.gz"),
	    "The NIfTI image file extension should be either .nii or .nii.gz");

	const ImageParams& params = getParams();
	const int dims[] = {3, params.nx, params.ny, params.nz};
	nifti_image* nim = nifti_make_new_nim(dims, NIFTI_TYPE_FLOAT32, 0);
	nim->nx = params.nx;
	nim->ny = params.ny;
	nim->nz = params.nz;
	nim->nbyper = sizeof(float);
	nim->datatype = NIFTI_TYPE_FLOAT32;
	nim->pixdim[0] = 0.0f;
	nim->dx = params.vx;
	nim->dy = params.vy;
	nim->dz = params.vz;
	nim->pixdim[1] = params.vx;
	nim->pixdim[2] = params.vy;
	nim->pixdim[3] = params.vz;
	nim->scl_slope = 1.0f;
	nim->scl_inter = 0.0f;
	nim->data =
	    const_cast<void*>(reinterpret_cast<const void*>(getRawPointer()));
	nim->qform_code = 0;
	nim->sform_code = NIFTI_XFORM_SCANNER_ANAT;
	nim->slice_dim = 3;
	nim->sto_xyz.m[0][0] = -params.vx;
	nim->sto_xyz.m[1][1] = -params.vy;
	nim->sto_xyz.m[2][2] = params.vz;
	nim->sto_xyz.m[0][3] =
	    -offsetToOrigin(params.off_x, params.vx, params.length_x);
	nim->sto_xyz.m[1][3] =
	    -offsetToOrigin(params.off_y, params.vy, params.length_y);
	nim->sto_xyz.m[2][3] =
	    offsetToOrigin(params.off_z, params.vz, params.length_z);
	nim->xyz_units = NIFTI_UNITS_MM;
	nim->time_units = NIFTI_UNITS_SEC;
	nim->nifti_type = NIFTI_FTYPE_NIFTI1_1;
	// Write something here in nim->descrip;

	nim->fname = strdup(fname.c_str());

	nifti_image_write(nim);

	nim->data = nullptr;
	nifti_image_free(nim);
}

void Image::applyThreshold(const ImageBase* maskImg, float threshold,
                           float val_le_scale, float val_le_off,
                           float val_gt_scale, float val_gt_off)
{
	const Image* maskImg_Image = dynamic_cast<const Image*>(maskImg);
	ASSERT_MSG(maskImg_Image != nullptr, "Input image has the wrong type");

	float* ptr = mp_array->getRawPointer();
	const float* mask_ptr = maskImg_Image->getRawPointer();
	for (size_t k = 0; k < mp_array->getSizeTotal(); k++, ptr++, mask_ptr++)
	{
		if (*mask_ptr <= threshold)
		{
			*ptr = *ptr * val_le_scale + val_le_off;
		}
		else
		{
			*ptr = *ptr * val_gt_scale + val_gt_off;
		}
	}
}

void Image::updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
                              float threshold)
{
	Image* updateImg_Image = dynamic_cast<Image*>(updateImg);
	const Image* normImg_Image = dynamic_cast<const Image*>(normImg);

	ASSERT_MSG(updateImg_Image != nullptr, "Update image has the wrong type");
	ASSERT_MSG(normImg_Image != nullptr, "Norm image has the wrong type");
	ASSERT_MSG(normImg_Image->getParams().isSameAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(updateImg_Image->getParams().isSameAs(getParams()),
	           "Image dimensions mismatch");

	float* ptr = mp_array->getRawPointer();
	float* up_ptr = updateImg_Image->getRawPointer();
	const float* norm_ptr = normImg_Image->getRawPointer();

	for (size_t k = 0; k < mp_array->getSizeTotal();
	     k++, ptr++, up_ptr++, norm_ptr++)
	{
		if (*norm_ptr > threshold)
		{
			*ptr *= *up_ptr / *norm_ptr;
		}
	}
}

float Image::dotProduct(const Image& y) const
{
	float out = 0.0;
	const float* x_ptr = getRawPointer();
	const float* y_ptr = y.getRawPointer();
	for (size_t k = 0; k < mp_array->getSizeTotal(); k++, x_ptr++, y_ptr++)
	{
		out += (*x_ptr) * (*y_ptr);
	}
	return out;
}

Array3DAlias<float> Image::getArray() const
{
	return {mp_array.get()};
}

void Image::transformImage(const Vector3D& rotation,
                           const Vector3D& translation, Image& dest,
                           float weight) const
{
	ImageParams params = getParams();
	const float* rawPtr = getRawPointer();
	const int num_xy = params.nx * params.ny;
	const float alpha = rotation.z;
	const float beta = rotation.y;
	const float gamma = rotation.x;

	for (int i = 0; i < params.nz; i++)
	{
		const float z = indexToPositionInDimension<0>(i);

		for (int j = 0; j < params.ny; j++)
		{
			const float y = indexToPositionInDimension<1>(j);

			for (int k = 0; k < params.nx; k++)
			{
				const float x = indexToPositionInDimension<2>(k);

				float newX = x * cos(alpha) * cos(beta) +
				             y * (-sin(alpha) * cos(gamma) +
				                  sin(beta) * sin(gamma) * cos(alpha)) +
				             z * (sin(alpha) * sin(gamma) +
				                  sin(beta) * cos(alpha) * cos(gamma));
				newX += translation.x;
				float newY = x * sin(alpha) * cos(beta) +
				             y * (sin(alpha) * sin(beta) * sin(gamma) +
				                  cos(alpha) * cos(gamma)) +
				             z * (sin(alpha) * sin(beta) * cos(gamma) -
				                  sin(gamma) * cos(alpha));
				newY += translation.y;
				float newZ = -x * sin(beta) + y * sin(gamma) * cos(beta) +
				             z * cos(beta) * cos(gamma);
				newZ += translation.z;

				const float currentValue =
				    rawPtr[i * num_xy + j * params.nx + k];
				dest.updateImageInterpolate({newX, newY, newZ},
				                            weight * currentValue, false);
			}
		}
	}
}

std::unique_ptr<Image> Image::transformImage(const Vector3D& rotation,
                                             const Vector3D& translation) const
{
	auto newImg = std::make_unique<ImageOwned>(getParams());
	newImg->allocate();
	newImg->setValue(0.0);
	transformImage(rotation, translation, *newImg, 1.0f);
	return newImg;
}

void Image::transformImage(const transform_t& t, Image& dest,
                           float weight) const
{
	const ImageParams params = getParams();
	const float* rawPtr = getRawPointer();
	const int num_xy = params.nx * params.ny;
	for (int i = 0; i < params.nz; i++)
	{
		const float z = indexToPositionInDimension<0>(i);

		for (int j = 0; j < params.ny; j++)
		{
			const float y = indexToPositionInDimension<1>(j);

			for (int k = 0; k < params.nx; k++)
			{
				const float x = indexToPositionInDimension<2>(k);

				float newX = x * t.r00 + y * t.r01 + z * t.r02;
				newX += t.tx;
				float newY = x * t.r10 + y * t.r11 + z * t.r12;
				newY += t.ty;
				float newZ = x * t.r20 + y * t.r21 + z * t.r22;
				newZ += t.tz;

				const float currentValue =
				    rawPtr[i * num_xy + j * params.nx + k];
				dest.updateImageInterpolate({newX, newY, newZ},
				                            weight * currentValue, false);
			}
		}
	}
}

std::unique_ptr<Image> Image::transformImage(const transform_t& t) const
{
	auto newImg = std::make_unique<ImageOwned>(getParams());
	newImg->allocate();
	newImg->setValue(0.0);
	transformImage(t, *newImg, 1.0f);
	return newImg;
}

ImageOwned::ImageOwned(const ImageParams& imgParams) : Image{imgParams}
{
	mp_array = std::make_unique<Array3D<float>>();
}

ImageOwned::ImageOwned(const ImageParams& imgParams,
                       const std::string& filename)
    : ImageOwned{imgParams}
{
	// Compare given image parameters against given file
	readFromFile(filename);
}

ImageOwned::ImageOwned(const std::string& filename) : Image{}
{
	mp_array = std::make_unique<Array3D<float>>();

	// Deduce image parameters from given file
	readFromFile(filename);
}

void ImageOwned::allocate()
{
	ASSERT(mp_array != nullptr);
	const ImageParams& params = getParams();
	reinterpret_cast<Array3D<float>*>(mp_array.get())
	    ->allocate(params.nz, params.ny, params.nx);
}

mat44 ImageOwned::adjustAffineMatrix(mat44 matrix)
{
	// Flip X-axis if diagonal element is negative
	if (matrix.m[0][0] < 0)
	{
		matrix.m[0][0] *= -1;
		matrix.m[0][3] *= -1;  // Adjust translation
	}

	// Flip Y-axis if diagonal element is negative
	if (matrix.m[1][1] < 0)
	{
		matrix.m[1][1] *= -1;
		matrix.m[1][3] *= -1;  // Adjust translation
	}

	// Flip Z-axis if diagonal element is negative (optional)
	if (matrix.m[2][2] < 0)
	{
		matrix.m[2][2] *= -1;
		matrix.m[2][3] *= -1;  // Adjust translation
	}

	return matrix;
}

void ImageOwned::readFromFile(const std::string& fname)
{
	nifti_image* niftiImage = nifti_image_read(fname.c_str(), 1);

	if (niftiImage == nullptr)
	{
		throw std::invalid_argument("An error occured while reading file" +
		                            fname);
	}

	mat44 transformMatrix;
	if (niftiImage->sform_code > 0)
	{
		transformMatrix = niftiImage->sto_xyz;  // Use sform matrix
	}
	else if (niftiImage->qform_code > 0)
	{
		transformMatrix = niftiImage->qto_xyz;  // Use qform matrix
	}
	else
	{
		std::cout << "Warning: The NIfTI image file given does not have a "
		             "qform or an sform."
		          << std::endl;
		std::cout << "This mapping method is not recommended, and is present "
		             "mainly for compatibility with ANALYZE 7.5 files."
		          << std::endl;
		std::memset(transformMatrix.m, 0, 16 * sizeof(float));
		transformMatrix.m[0][0] = 1.0f;
		transformMatrix.m[1][1] = 1.0f;
		transformMatrix.m[2][2] = 1.0f;
		transformMatrix.m[3][3] = 1.0f;
	}
	transformMatrix = adjustAffineMatrix(transformMatrix);

	// TODO: Check Image direction matrix and do the resampling if needed

	float voxelSpacing[3];
	voxelSpacing[0] = niftiImage->dx;  // Spacing along x
	voxelSpacing[1] = niftiImage->dy;  // Spacing along y
	voxelSpacing[2] = niftiImage->dz;  // Spacing along z

	const int spaceUnits = niftiImage->xyz_units;
	if (spaceUnits == NIFTI_UNITS_METER)
	{
		for (int i = 0; i < 3; i++)
		{
			voxelSpacing[i] = voxelSpacing[i] / 1000.0f;
		}
	}
	else if (spaceUnits == NIFTI_UNITS_MICRON)
	{
		for (int i = 0; i < 3; i++)
		{
			voxelSpacing[i] = voxelSpacing[i] * 1000.0f;
		}
	}

	float imgOrigin[3];
	imgOrigin[0] = transformMatrix.m[0][3];  // x-origin
	imgOrigin[1] = transformMatrix.m[1][3];  // y-origin
	imgOrigin[2] = transformMatrix.m[2][3];  // z-origin

	const ImageParams& params = getParams();

	if (params.isValid())
	{
		checkImageParamsWithGivenImage(voxelSpacing, imgOrigin,
		                               niftiImage->dim);
	}
	else
	{
		ImageParams newParams;
		newParams.vx = voxelSpacing[0];
		newParams.vy = voxelSpacing[1];
		newParams.vz = voxelSpacing[2];
		ASSERT_MSG(niftiImage->dim[0] == 3, "NIfTI Image's dim[0] is not 3");
		newParams.nx = niftiImage->dim[1];
		newParams.ny = niftiImage->dim[2];
		newParams.nz = niftiImage->dim[3];
		newParams.off_x = originToOffset(imgOrigin[0], newParams.vx,
		                                 newParams.vx * newParams.nx);
		newParams.off_y = originToOffset(imgOrigin[1], newParams.vy,
		                                 newParams.vy * newParams.ny);
		newParams.off_z = originToOffset(imgOrigin[2], newParams.vz,
		                                 newParams.vz * newParams.nz);
		newParams.setup();
		setParams(newParams);
	}

	allocate();

	readNIfTIData(niftiImage->datatype, niftiImage->data, niftiImage->scl_slope,
	              niftiImage->scl_inter);

	nifti_image_free(niftiImage);
}

void ImageOwned::readNIfTIData(int datatype, void* data, float slope,
                               float intercept)
{
	const ImageParams& params = getParams();

	float* imgData = getRawPointer();
	const int numVoxels = params.nx * params.ny * params.nz;

	if (datatype == NIFTI_TYPE_FLOAT32)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (*(reinterpret_cast<float*>(data) + i) * slope) + intercept;
	}
	else if (datatype == NIFTI_TYPE_FLOAT64)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<double, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT8)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<int8_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT16)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<int16_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT32)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<int32_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT64)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<int64_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT8)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<uint8_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT16)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<uint16_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT32)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<uint32_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT64)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (Util::reinterpretAndCast<uint64_t, float>(data, i) * slope) +
			    intercept;
	}
}


void ImageOwned::checkImageParamsWithGivenImage(float voxelSpacing[3],
                                                float imgOrigin[3],
                                                const int dim[8]) const
{
	const ImageParams& params = getParams();

	ASSERT(dim[0] == 3);

	if (!(APPROX_EQ_THRESH(voxelSpacing[0], params.vx, 1e-3) &&
	      APPROX_EQ_THRESH(voxelSpacing[1], params.vy, 1e-3) &&
	      APPROX_EQ_THRESH(voxelSpacing[2], params.vz, 1e-3)))
	{
		std::string errorString = "Spacing mismatch "
		                          "between given image and the "
		                          "image parameters provided:\n";
		errorString += "Given image: vx=" + std::to_string(voxelSpacing[0]) +
		               " vy=" + std::to_string(voxelSpacing[1]) +
		               " vz=" + std::to_string(voxelSpacing[2]) + "\n";
		errorString += "Image parameters: vx=" + std::to_string(params.vx) +
		               " vy=" + std::to_string(params.vy) +
		               " vz=" + std::to_string(params.vz);
		throw std::invalid_argument(errorString);
	}

	ASSERT_MSG(dim[1] == params.nx, "Size mismatch in X dimension");
	ASSERT_MSG(dim[2] == params.ny, "Size mismatch in Y dimension");
	ASSERT_MSG(dim[3] == params.nz, "Size mismatch in Z dimension");

	const float expectedOffsetX =
	    originToOffset(imgOrigin[0], params.vx, params.length_x);
	const float expectedOffsetY =
	    originToOffset(imgOrigin[1], params.vy, params.length_y);
	const float expectedOffsetZ =
	    originToOffset(imgOrigin[2], params.vz, params.length_z);

	if (!(APPROX_EQ_THRESH(expectedOffsetX, params.off_x, 1e-3) &&
	      APPROX_EQ_THRESH(expectedOffsetY, params.off_y, 1e-3) &&
	      APPROX_EQ_THRESH(expectedOffsetZ, params.off_z, 1e-3)))
	{
		std::string errorString = "Volume offsets mismatch "
		                          "between given image and the "
		                          "image parameters provided:\n";
		errorString += "Given image: off_x=" + std::to_string(expectedOffsetX) +
		               " off_y=" + std::to_string(expectedOffsetY) +
		               " off_z=" + std::to_string(expectedOffsetZ) + "\n";
		errorString +=
		    "Image parameters: off_x=" + std::to_string(params.off_x) +
		    " off_y=" + std::to_string(params.off_y) +
		    " off_z=" + std::to_string(params.off_z);
		throw std::invalid_argument(errorString);
	}
}

float Image::originToOffset(float origin, float voxelSize, float length)
{
	return origin + 0.5f * length - 0.5f * voxelSize;
}

float Image::offsetToOrigin(float off, float voxelSize, float length)
{
	return off - 0.5f * length + 0.5f * voxelSize;
}

template <int Dimension>
float Image::indexToPositionInDimension(int index) const
{
	static_assert(Dimension >= 0 && Dimension < 3);
	const ImageParams& params = getParams();
	float voxelSize, length, offset;
	if constexpr (Dimension == 0)
	{
		voxelSize = params.vz;
		length = params.length_z;
		offset = params.off_z;
	}
	else if constexpr (Dimension == 1)
	{
		voxelSize = params.vy;
		length = params.length_y;
		offset = params.off_y;
	}
	else if constexpr (Dimension == 2)
	{
		voxelSize = params.vx;
		length = params.length_x;
		offset = params.off_x;
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}
	return static_cast<float>(index) * voxelSize - 0.5f * length + offset +
	       0.5f * voxelSize;
}
template float Image::indexToPositionInDimension<0>(int index) const;
template float Image::indexToPositionInDimension<1>(int index) const;
template float Image::indexToPositionInDimension<2>(int index) const;

ImageAlias::ImageAlias(const ImageParams& imgParams) : Image(imgParams)
{
	mp_array = std::make_unique<Array3DAlias<float>>();
}

void ImageAlias::bind(Array3DBase<float>& p_data)
{
	static_cast<Array3DAlias<float>*>(mp_array.get())->bind(p_data);
	if (mp_array->getRawPointer() != p_data.getRawPointer())
	{
		throw std::runtime_error("An error occured during Image binding");
	}
}
