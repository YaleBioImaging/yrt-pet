/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageBase.hpp"
#include "geometry/Constants.hpp"
#include "utils/Assert.hpp"

#include <sitkCastImageFilter.h>
#include <sitkImageFileReader.h>
#include <sitkImageFileWriter.h>
#include <sitkImportImageFilter.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace sitk = itk::simple;

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
	c.def("transformImage", &Image::transformImage, py::arg("rotation"),
	      py::arg("translation"));
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
	ASSERT_MSG(imSrc_ptr!= nullptr, "Image not in host");
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
	return mp_array->getRawPointer() != nullptr;
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

	// if point is outside of the grid, return false
	if ((std::abs(x) >= (params.length_x / 2.0)) ||
	    (std::abs(y) >= (params.length_y / 2.0)) ||
	    (std::abs(z) >= (params.length_z / 2.0)))
	{
		return false;
	}

	const float dx = (x + params.length_x / 2.0) / params.length_x *
	                 static_cast<float>(params.nx);
	const float dy = (y + params.length_y / 2.0) / params.length_y *
	                 static_cast<float>(params.ny);
	const float dz = (z + params.length_z / 2.0) / params.length_z *
	                 static_cast<float>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

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

	// if point outside of the image, return 0:
	if ((std::abs(x) >= (params.length_x / 2)) ||
	    (std::abs(y) >= (params.length_y / 2)) ||
	    (std::abs(z) >= (params.length_z / 2)))
	{
		return 0.0;
	}
	const float dx = (x + params.length_x / 2) / params.length_x *
	                 static_cast<float>(params.nx);
	const float dy = (y + params.length_y / 2) / params.length_y *
	                 static_cast<float>(params.ny);
	const float dz = (z + params.length_z / 2) / params.length_z *
	                 static_cast<float>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

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

	// if point outside of the image, return 0:
	if ((std::abs(x) >= (params.length_x / 2)) ||
	    (std::abs(y) >= (params.length_y / 2)) ||
	    (std::abs(z) >= (params.length_z / 2)))
	{
		return 0.;
	}

	const float dx = (x + params.length_x / 2) / params.length_x *
	                 static_cast<float>(params.nx);
	const float dy = (y + params.length_y / 2) / params.length_y *
	                 static_cast<float>(params.ny);
	const float dz = (z + params.length_z / 2) / params.length_z *
	                 static_cast<float>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

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

	// if point is outside of the grid do nothing:
	if ((std::abs(x) >= (params.length_x / 2)) ||
	    (std::abs(y) >= (params.length_y / 2)) ||
	    (std::abs(z) >= (params.length_z / 2)))
	{
		return;
	}

	float dx = (x + params.length_x / 2) / params.length_x * ((float)params.nx);
	float dy = (y + params.length_y / 2) / params.length_y * ((float)params.ny);
	float dz = (z + params.length_z / 2) / params.length_z * ((float)params.nz);

	int ix = (int)dx;
	int iy = (int)dy;
	int iz = (int)dz;

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

	// if point is outside of the grid do nothing:
	if ((std::abs(x) >= (params.length_x / 2)) ||
	    (std::abs(y) >= (params.length_y / 2)) ||
	    (std::abs(z) >= (params.length_z / 2)))
	{
		return;
	}

	float dx = (x + params.length_x / 2) / params.length_x * ((float)params.nx);
	float dy = (y + params.length_y / 2) / params.length_y * ((float)params.ny);
	float dz = (z + params.length_z / 2) / params.length_z * ((float)params.nz);

	int ix = (int)dx;
	int iy = (int)dy;
	int iz = (int)dz;

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
	const ImageParams& params = getParams();

	std::vector<unsigned int> sitkSize{{static_cast<unsigned int>(params.nx),
	                                    static_cast<unsigned int>(params.ny),
	                                    static_cast<unsigned int>(params.nz)}};
	sitk::Image sitkImage{sitkSize[0], sitkSize[1], sitkSize[2],
	                      sitk::sitkFloat32};

	updateSitkImageFromParameters(sitkImage, params);

	const float* rawPtr = getRawPointer();
	for (unsigned int z = 0; z < static_cast<unsigned int>(params.nz); ++z)
	{
		const unsigned int z_offset = z * (params.nx * params.ny);
		for (unsigned int y = 0; y < static_cast<unsigned int>(params.ny); ++y)
		{
			const unsigned int y_offset = y * params.nx;
			for (unsigned int x = 0; x < static_cast<unsigned int>(params.nx);
			     ++x)
			{
				const unsigned int index = z_offset + y_offset + x;
				sitkImage.SetPixelAsFloat(std::vector<unsigned int>{x, y, z},
				                          rawPtr[index]);
			}
		}
	}

	sitk::WriteImage(sitkImage, fname);
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

std::unique_ptr<Image> Image::transformImage(const Vector3D& rotation,
                                             const Vector3D& translation) const
{
	ImageParams params = getParams();
	const float* rawPtr = getRawPointer();
	const int num_xy = params.nx * params.ny;
	auto newImg = std::make_unique<ImageOwned>(params);
	newImg->allocate();
	newImg->setValue(0.0);
	const float alpha = rotation.z;
	const float beta = rotation.y;
	const float gamma = rotation.x;
	for (int i = 0; i < params.nz; i++)
	{
		const float z = static_cast<float>(i) * params.vz -
		                params.length_z / 2.0 + params.off_z + params.vz / 2.0;
		for (int j = 0; j < params.ny; j++)
		{
			const float y = static_cast<float>(j) * params.vy -
			                params.length_y / 2.0 + params.off_y +
			                params.vy / 2.0;
			for (int k = 0; k < params.nx; k++)
			{
				const float x = static_cast<float>(k) * params.vx -
				                params.length_x / 2.0 + params.off_x +
				                params.vx / 2.0;

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
				newImg->updateImageInterpolate({newX, newY, newZ}, currentValue,
				                               false);
			}
		}
	}
	return newImg;
}

ImageOwned::ImageOwned(const ImageParams& imgParams) : Image{imgParams} {}

ImageOwned::ImageOwned(const ImageParams& imgParams,
                       const std::string& filename)
    : Image{imgParams}
{
	// Compare given image parameters against given file
	readFromFile(filename);
}

ImageOwned::ImageOwned(const std::string& filename) : Image{}
{
	// Deduct image parameters from given file
	readFromFile(filename);
}

void ImageOwned::allocate()
{
	const ImageParams& params = getParams();
	mp_sitkImage = std::make_unique<sitk::Image>(params.nx, params.ny,
	                                             params.nz, sitk::sitkFloat32);

	updateSitkImageFromParameters(*mp_sitkImage, params);

	auto arrayAlias = std::make_unique<Array3DAlias<float>>();
	arrayAlias->bind(mp_sitkImage->GetBufferAsFloat(), params.nz, params.ny,
	                 params.nx);
	mp_array = std::move(arrayAlias);
}

void ImageOwned::readFromFile(const std::string& fname)
{
	const ImageParams& params = getParams();
	if (params.isValid())
	{
		mp_sitkImage = std::make_unique<sitk::Image>(sitk::ReadImage(fname));
		if (mp_sitkImage->GetPixelID() != sitk::sitkFloat32)
		{
			*mp_sitkImage = sitk::Cast(*mp_sitkImage, sitk::sitkFloat32);
		}

		checkImageParamsWithSitkImage();

		// TODO: Check the Image direction matrix and do the resampling if
		// needed

		auto arrayAlias = std::make_unique<Array3DAlias<float>>();
		arrayAlias->bind(mp_sitkImage->GetBufferAsFloat(), params.nz, params.ny,
		                 params.nx);
		mp_array = std::move(arrayAlias);
	}
	else
	{
		mp_sitkImage = std::make_unique<sitk::Image>(sitk::ReadImage(fname));
		if (mp_sitkImage->GetPixelID() != sitk::sitkFloat32)
		{
			*mp_sitkImage = sitk::Cast(*mp_sitkImage, sitk::sitkFloat32);
		}

		// TODO: Check the Image direction matrix and do the resampling if
		// needed

		const ImageParams newParams =
		    createImageParamsFromSitkImage(*mp_sitkImage);
		setParams(newParams);

		auto arrayAlias = std::make_unique<Array3DAlias<float>>();
		arrayAlias->bind(mp_sitkImage->GetBufferAsFloat(), newParams.nz,
		                 newParams.ny, params.nx);
		mp_array = std::move(arrayAlias);
	}
}

void ImageOwned::checkImageParamsWithSitkImage() const
{
	const ImageParams& params = getParams();

	ASSERT(mp_sitkImage->GetDimension() == 3);
	const auto sitkSpacing = mp_sitkImage->GetSpacing();

	if (!(APPROX_EQ_THRESH(static_cast<float>(sitkSpacing[0]), params.vx,
	                       1e-3) &&
	      APPROX_EQ_THRESH(static_cast<float>(sitkSpacing[1]), params.vy,
	                       1e-3) &&
	      APPROX_EQ_THRESH(static_cast<float>(sitkSpacing[2]), params.vz,
	                       1e-3)))
	{
		std::string errorString = "Spacing mismatch "
		                          "between given image and the "
		                          "image parameters provided:\n";
		errorString += "Given image: vx=" + std::to_string(sitkSpacing[0]) +
		               " vy=" + std::to_string(sitkSpacing[1]) +
		               " vz=" + std::to_string(sitkSpacing[2]) + "\n";
		errorString += "Image parameters: vx=" + std::to_string(params.vx) +
		               " vy=" + std::to_string(params.vy) +
		               " vz=" + std::to_string(params.vz);
		throw std::invalid_argument(errorString);
	}

	const auto sitkSize = mp_sitkImage->GetSize();
	ASSERT_MSG(sitkSize[0] == static_cast<unsigned int>(params.nx),
	           "Size mismatch in X dimension");
	ASSERT_MSG(sitkSize[1] == static_cast<unsigned int>(params.ny),
	           "Size mismatch in Y dimension");
	ASSERT_MSG(sitkSize[2] == static_cast<unsigned int>(params.nz),
	           "Size mismatch in Z dimension");

	const auto sitkOrigin = mp_sitkImage->GetOrigin();
	const float expectedOffsetX = sitkOriginToImageParamsOffset(
	    sitkOrigin[0], params.vx, params.length_x);
	const float expectedOffsetY = sitkOriginToImageParamsOffset(
	    sitkOrigin[1], params.vy, params.length_y);
	const float expectedOffsetZ = sitkOriginToImageParamsOffset(
	    sitkOrigin[2], params.vz, params.length_z);

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

void ImageOwned::writeToFile(const std::string& fname) const
{
	updateSitkImageFromParameters(*mp_sitkImage, getParams());
	sitk::WriteImage(*mp_sitkImage, fname);
}

void Image::updateSitkImageFromParameters(itk::simple::Image& sitkImage,
                                          const ImageParams& params)
{
	const std::vector<unsigned int> sitkSize{
	    {static_cast<unsigned int>(params.nx),
	     static_cast<unsigned int>(params.ny),
	     static_cast<unsigned int>(params.nz)}};

	const std::vector<double> sitkSpacing{{params.vx, params.vy, params.vz}};
	sitkImage.SetSpacing(sitkSpacing);

	const std::vector<double> sitkDirection{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
	sitkImage.SetDirection(sitkDirection);

	std::vector<double> sitkOrigin;
	sitkOrigin.resize(3);
	sitkOrigin[0] =
	    imageParamsOffsetToSitkOrigin(params.off_x, params.vx, params.length_x);
	sitkOrigin[1] =
	    imageParamsOffsetToSitkOrigin(params.off_y, params.vy, params.length_y);
	sitkOrigin[2] =
	    imageParamsOffsetToSitkOrigin(params.off_z, params.vz, params.length_z);
	sitkImage.SetOrigin(sitkOrigin);
}

float Image::sitkOriginToImageParamsOffset(double sitkOrigin, float voxelSize,
                                           float length)
{
	return static_cast<float>(sitkOrigin) + 0.5f * length - 0.5f * voxelSize;
}

double Image::imageParamsOffsetToSitkOrigin(float off, float voxelSize,
                                            float length)
{
	return off - 0.5 * length + 0.5 * voxelSize;
}

ImageParams Image::createImageParamsFromSitkImage(const sitk::Image& sitkImage)
{
	ImageParams newParams;

	auto sitkSize = sitkImage.GetSize();
	newParams.nx = sitkSize[0];
	newParams.ny = sitkSize[1];
	newParams.nz = sitkSize[2];

	auto sitkSpacing = sitkImage.GetSpacing();
	newParams.vx = sitkSpacing[0];
	newParams.vy = sitkSpacing[1];
	newParams.vz = sitkSpacing[2];
	newParams.length_x = newParams.nx * newParams.vx;
	newParams.length_y = newParams.ny * newParams.vy;
	newParams.length_z = newParams.nz * newParams.vz;

	auto sitkOrigin = sitkImage.GetOrigin();
	newParams.off_x = sitkOriginToImageParamsOffset(sitkOrigin[0], newParams.vx,
	                                                newParams.length_x);
	newParams.off_y = sitkOriginToImageParamsOffset(sitkOrigin[1], newParams.vy,
	                                                newParams.length_y);
	newParams.off_z = sitkOriginToImageParamsOffset(sitkOrigin[2], newParams.vz,
	                                                newParams.length_z);

	newParams.setup();

	return newParams;
}

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
