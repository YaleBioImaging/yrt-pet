/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageBase.hpp"
#include "utils/Assert.hpp"

#include <cmath>
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
		    Array3DBase<double>& d = img.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(double),
		                           py::format_descriptor<double>::format(), 3,
		                           d.getDims(), d.getStrides());
	    });
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
	      static_cast<double (Image::*)(const Vector3D& pt, const Image& sens)
	                      const>(&Image::interpolateImage),
	      py::arg("pt"), py::arg("sens"));
	c.def("interpolImage",
	      static_cast<double (Image::*)(const Vector3D& pt) const>(
	          &Image::interpolateImage),
	      py::arg("pt"));
	c.def(
	    "nearestNeighbor",
	    [](const Image& img, const Vector3D& pt) -> py::tuple
	    {
		    int pi, pj, pk;
		    double val = img.nearestNeighbor(pt, &pi, &pj, &pk);
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
		    if (buffer.format != py::format_descriptor<double>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float64 format");
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
		    static_cast<Array3DAlias<double>&>(self.getData())
		        .bind(reinterpret_cast<double*>(buffer.ptr), dims[0], dims[1],
		              dims[2]);
	    },
	    py::arg("numpy_data"));

	auto c_owned = py::class_<ImageOwned, Image>(m, "ImageOwned");
	c_owned.def(py::init<const ImageParams&>(), py::arg("img_params"));
	c_owned.def(py::init<const ImageParams&, std::string>(),
	            py::arg("img_params"), py::arg("filename"));
	c_owned.def("allocate", &ImageOwned::allocate);
	c_owned.def("isAllocated", &ImageOwned::isAllocated);
	c_owned.def("readFromFile", &ImageOwned::readFromFile, py::arg("filename"));
}

#endif  // if BUILD_PYBIND11


Image::Image(const ImageParams& img_params) : ImageBase(img_params) {}

void Image::setValue(double initValue)
{
	m_dataPtr->fill(initValue);
}

void Image::copyFromImage(const Image* imSrc)
{
	ASSERT(m_dataPtr != nullptr);
	m_dataPtr->copy(imSrc->getData());
	setParams(imSrc->getParams());
}

Array3DBase<double>& Image::getData()
{
	return *m_dataPtr;
}

const Array3DBase<double>& Image::getData() const
{
	return *m_dataPtr;
}

void Image::addFirstImageToSecond(ImageBase* secondImage) const
{
	auto* second_Image = dynamic_cast<Image*>(secondImage);

	ASSERT(second_Image != nullptr);
	ASSERT_MSG(secondImage->getParams().isSameDimensionsAs(getParams()),
	           "The two images do not share the same image space");

	second_Image->getData() += *m_dataPtr;
}

void Image::multWithScalar(double scalar)
{
	*m_dataPtr *= scalar;
}

// return the value of the voxel the nearest to "point":
double Image::nearestNeighbor(const Vector3D& pt) const
{
	int ix, iy, iz;

	if (getNearestNeighborIdx(pt, &ix, &iy, &iz))
	{
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		return m_dataPtr->get_flat(iz * num_xy + iy * num_x + ix);
	}
	return 0;
}

// return the value of the voxel the nearest to "point":
double Image::nearestNeighbor(const Vector3D& pt, int* pi, int* pj,
                              int* pk) const
{
	if (getNearestNeighborIdx(pt, pi, pj, pk))
	{
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		return m_dataPtr->get_flat(*pk * num_xy + *pj * num_x + *pi);
	}
	return 0.0;
}


// update image with "value" using nearest neighbor method:
void Image::updateImageNearestNeighbor(const Vector3D& pt, double value,
                                       bool mult_flag)
{
	int ix, iy, iz;
	if (getNearestNeighborIdx(pt, &ix, &iy, &iz))
	{
		// update multiplicatively or additively:
		double* ptr = m_dataPtr->getRawPointer();
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
void Image::assignImageNearestNeighbor(const Vector3D& pt, double value)
{
	int ix, iy, iz;
	if (getNearestNeighborIdx(pt, &ix, &iy, &iz))
	{
		// update multiplicatively or additively:
		double* ptr = m_dataPtr->getRawPointer();
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		ptr[iz * num_xy + iy * num_x + ix] = value;
	}
}

// Returns true if the point `pt` is inside the image
bool Image::getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj,
                                  int* pk) const
{
	const double x = pt.x;
	const double y = pt.y;
	const double z = pt.z;
	const ImageParams& params = getParams();

	// if point is outside of the grid, return false
	if ((fabs(x) >= (params.length_x / 2.0)) ||
	    (fabs(y) >= (params.length_y / 2.0)) ||
	    (fabs(z) >= (params.length_z / 2.0)))
	{
		return false;
	}

	const double dx = (x + params.length_x / 2.0) / params.length_x *
	                  static_cast<double>(params.nx);
	const double dy = (y + params.length_y / 2.0) / params.length_y *
	                  static_cast<double>(params.ny);
	const double dz = (z + params.length_z / 2.0) / params.length_z *
	                  static_cast<double>(params.nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

	*pi = ix;
	*pj = iy;
	*pk = iz;

	return true;
}


// interpolation operation. It does not account for the offset values.
double Image::interpolateImage(const Vector3D& pt) const
{
	const double x = pt.x;
	const double y = pt.y;
	const double z = pt.z;
	// if point outside of the image, return 0:
	if ((fabs(x) >= (getParams().length_x / 2)) ||
	    (fabs(y) >= (getParams().length_y / 2)) ||
	    (fabs(z) >= (getParams().length_z / 2)))
	{
		return 0.0;
	}
	const double dx = (x + getParams().length_x / 2) / getParams().length_x *
	                  static_cast<double>(getParams().nx);
	const double dy = (y + getParams().length_y / 2) / getParams().length_y *
	                  static_cast<double>(getParams().ny);
	const double dz = (z + getParams().length_z / 2) / getParams().length_z *
	                  static_cast<double>(getParams().nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

	const double delta_x = dx - static_cast<double>(ix);
	const double delta_y = dy - static_cast<double>(iy);
	const double delta_z = dz - static_cast<double>(iz);

	// parameters of the x interpolation:
	int ix1, ix2, iy1, iy2, iz1, iz2;
	double dx1, dy1, dz1;
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
		if (ix != (getParams().nx - 1))
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
		if (iy != (getParams().ny - 1))
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
		if (iz != (getParams().nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}
	// interpolate in z:
	const double* ptr = m_dataPtr->getRawPointer();
	const size_t num_x = getParams().nx;
	const size_t num_xy = getParams().nx * getParams().ny;
	const double* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	const double* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	const double* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	const double* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
	const double v1 = ptr_11[ix1] * (1 - dz1) + ptr_21[ix1] * dz1;
	const double v2 = ptr_12[ix1] * (1 - dz1) + ptr_22[ix1] * dz1;
	const double v3 = ptr_11[ix2] * (1 - dz1) + ptr_21[ix2] * dz1;
	const double v4 = ptr_12[ix2] * (1 - dz1) + ptr_22[ix2] * dz1;
	// interpolate in y:
	const double vv1 = v1 * (1 - dy1) + v2 * dy1;
	const double vv2 = v3 * (1 - dy1) + v4 * dy1;
	// interpolate in the x direction:
	const double vvv = vv1 * (1 - dx1) + vv2 * dx1;

	return vvv;
}

// calculate the value of a point on the image matrix
// using tri-linear interpolation and weighting with image "sens":
double Image::interpolateImage(const Vector3D& pt, const Image& sens) const
{
	const double x = pt.x;
	const double y = pt.y;
	const double z = pt.z;

	// if point outside of the image, return 0:
	if ((fabs(x) >= (getParams().length_x / 2)) ||
	    (fabs(y) >= (getParams().length_y / 2)) ||
	    (fabs(z) >= (getParams().length_z / 2)))
	{
		return 0.;
	}

	const double dx = (x + getParams().length_x / 2) / getParams().length_x *
	                  static_cast<double>(getParams().nx);
	const double dy = (y + getParams().length_y / 2) / getParams().length_y *
	                  static_cast<double>(getParams().ny);
	const double dz = (z + getParams().length_z / 2) / getParams().length_z *
	                  static_cast<double>(getParams().nz);

	const int ix = static_cast<int>(dx);
	const int iy = static_cast<int>(dy);
	const int iz = static_cast<int>(dz);

	const double delta_x = dx - static_cast<double>(ix);
	const double delta_y = dy - static_cast<double>(iy);
	const double delta_z = dz - static_cast<double>(iz);

	// parameters of the x interpolation:
	int ix1, ix2, iy1, iy2, iz1, iz2;
	double dx1, dy1, dz1;
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
		if (ix != (getParams().nx - 1))
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
		if (iy != (getParams().ny - 1))
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
		if (iz != (getParams().nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}
	// interpolate in z:
	const double* ptr = m_dataPtr->getRawPointer();
	const double* sptr = sens.getData().getRawPointer();
	const size_t num_x = getParams().nx;
	const size_t num_xy = getParams().nx * getParams().ny;
	const double* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	const double* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	const double* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	const double* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
	const double* sptr_11 = sptr + iz1 * num_xy + iy1 * num_x;
	const double* sptr_21 = sptr + iz2 * num_xy + iy1 * num_x;
	const double* sptr_12 = sptr + iz1 * num_xy + iy2 * num_x;
	const double* sptr_22 = sptr + iz2 * num_xy + iy2 * num_x;
	const double v1 = ptr_11[ix1] * sptr_11[ix1] * (1 - dz1) +
	                  ptr_21[ix1] * sptr_21[ix1] * dz1;
	const double v2 = ptr_12[ix1] * sptr_12[ix1] * (1 - dz1) +
	                  ptr_22[ix1] * sptr_22[ix1] * dz1;
	const double v3 = ptr_11[ix2] * sptr_11[ix2] * (1 - dz1) +
	                  ptr_21[ix2] * sptr_21[ix2] * dz1;
	const double v4 = ptr_12[ix2] * sptr_12[ix2] * (1 - dz1) +
	                  ptr_22[ix2] * sptr_22[ix2] * dz1;
	// interpolate in y:
	const double vv1 = v1 * (1 - dy1) + v2 * dy1;
	const double vv2 = v3 * (1 - dy1) + v4 * dy1;
	// interpolate in the x direction:
	const double vvv = vv1 * (1 - dx1) + vv2 * dx1;

	return vvv;
}

// update image with "value" using trilinear interpolation:
void Image::updateImageInterpolate(const Vector3D& point, double value,
                                   bool mult_flag)
{
	double x = point.x;
	double y = point.y;
	double z = point.z;

	// if point is outside of the grid do nothing:
	if ((fabs(x) >= (getParams().length_x / 2)) ||
	    (fabs(y) >= (getParams().length_y / 2)) ||
	    (fabs(z) >= (getParams().length_z / 2)))
	{
		return;
	}

	double dx = (x + getParams().length_x / 2) / getParams().length_x *
	            ((double)getParams().nx);
	double dy = (y + getParams().length_y / 2) / getParams().length_y *
	            ((double)getParams().ny);
	double dz = (z + getParams().length_z / 2) / getParams().length_z *
	            ((double)getParams().nz);

	int ix = (int)dx;
	int iy = (int)dy;
	int iz = (int)dz;

	double delta_x = dx - (double)ix;
	double delta_y = dy - (double)iy;
	double delta_z = dz - (double)iz;

	// parameters of the x interpolation:
	int ix1, ix2, iy1, iy2, iz1, iz2;
	double dx1, dy1, dz1;
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
		if (ix != (getParams().nx - 1))
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
		if (iy != (getParams().ny - 1))
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
		if (iz != (getParams().nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}

	// interpolate multiplicatively or additively:
	double* ptr = m_dataPtr->getRawPointer();
	size_t num_x = getParams().nx;
	size_t num_xy = getParams().nx * getParams().ny;
	double* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	double* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	double* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	double* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
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
void Image::assignImageInterpolate(const Vector3D& point, double value)
{
	double x = point.x;
	double y = point.y;
	double z = point.z;

	// if point is outside of the grid do nothing:
	if ((fabs(x) >= (getParams().length_x / 2)) ||
	    (fabs(y) >= (getParams().length_y / 2)) ||
	    (fabs(z) >= (getParams().length_z / 2)))
	{
		return;
	}

	double dx = (x + getParams().length_x / 2) / getParams().length_x *
	            ((double)getParams().nx);
	double dy = (y + getParams().length_y / 2) / getParams().length_y *
	            ((double)getParams().ny);
	double dz = (z + getParams().length_z / 2) / getParams().length_z *
	            ((double)getParams().nz);

	int ix = (int)dx;
	int iy = (int)dy;
	int iz = (int)dz;

	double delta_x = dx - (double)ix;
	double delta_y = dy - (double)iy;
	double delta_z = dz - (double)iz;

	// parameters of the x interpolation:
	double dx1, dy1, dz1;
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
		if (ix != (getParams().nx - 1))
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
		if (iy != (getParams().ny - 1))
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
		if (iz != (getParams().nz - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}

	// assign:
	double* ptr = m_dataPtr->getRawPointer();
	const size_t num_x = getParams().nx;
	const size_t num_xy = getParams().nx * getParams().ny;
	double* ptr_11 = ptr + iz1 * num_xy + iy1 * num_x;
	double* ptr_21 = ptr + iz2 * num_xy + iy1 * num_x;
	double* ptr_12 = ptr + iz1 * num_xy + iy2 * num_x;
	double* ptr_22 = ptr + iz2 * num_xy + iy2 * num_x;
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
void Image::writeToFile(const std::string& image_fname) const
{
	m_dataPtr->writeToFile(image_fname);
}


// this function copy "image_file_name" to image:
void ImageOwned::readFromFile(const std::string& image_file_name)
{
	std::array<size_t, 3> dims{static_cast<size_t>(getParams().nz),
	                           static_cast<size_t>(getParams().ny),
	                           static_cast<size_t>(getParams().nx)};
	m_dataPtr->readFromFile(image_file_name, dims);
}

void Image::applyThreshold(const ImageBase* maskImg, double threshold,
                           double val_le_scale, double val_le_off,
                           double val_gt_scale, double val_gt_off)
{
	const Image* maskImg_Image = dynamic_cast<const Image*>(maskImg);
	ASSERT_MSG(maskImg_Image != nullptr, "Input image has the wrong type");

	double* ptr = m_dataPtr->getRawPointer();
	const double* mask_ptr = maskImg_Image->getData().getRawPointer();
	for (size_t k = 0; k < m_dataPtr->getSizeTotal(); k++, ptr++, mask_ptr++)
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
                              double threshold)
{
	Image* updateImg_Image = dynamic_cast<Image*>(updateImg);
	const Image* normImg_Image = dynamic_cast<const Image*>(normImg);

	ASSERT_MSG(updateImg_Image != nullptr, "Update image has the wrong type");
	ASSERT_MSG(normImg_Image != nullptr, "Norm image has the wrong type");
	ASSERT_MSG(normImg_Image->getParams().isSameAs(getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(updateImg_Image->getParams().isSameAs(getParams()),
	           "Image dimensions mismatch");

	double* ptr = m_dataPtr->getRawPointer();
	double* up_ptr = updateImg_Image->getData().getRawPointer();
	const double* norm_ptr = normImg_Image->getData().getRawPointer();

	for (size_t k = 0; k < m_dataPtr->getSizeTotal();
	     k++, ptr++, up_ptr++, norm_ptr++)
	{
		if (*norm_ptr > threshold)
		{
			*ptr *= *up_ptr / *norm_ptr;
		}
	}
}

double Image::dotProduct(const Image& y) const
{
	double out = 0.0;
	const double* x_ptr = m_dataPtr->getRawPointer();
	const double* y_ptr = y.getData().getRawPointer();
	for (size_t k = 0; k < m_dataPtr->getSizeTotal(); k++, x_ptr++, y_ptr++)
	{
		out += (*x_ptr) * (*y_ptr);
	}
	return out;
}

Array3DAlias<double> Image::getArray() const
{
	return {m_dataPtr.get()};
}

std::unique_ptr<Image> Image::transformImage(const Vector3D& rotation,
                                             const Vector3D& translation) const
{
	ImageParams params = getParams();
	const double* rawPtr = getData().getRawPointer();
	const int num_xy = params.nx * params.ny;
	auto newImg = std::make_unique<ImageOwned>(params);
	newImg->allocate();
	newImg->setValue(0.0);
	const double alpha = rotation.z;
	const double beta = rotation.y;
	const double gamma = rotation.x;
	for (int i = 0; i < params.nz; i++)
	{
		const double z = static_cast<double>(i) * params.vz -
		                 params.length_z / 2.0 + params.off_z + params.vz / 2.0;
		for (int j = 0; j < params.ny; j++)
		{
			const double y = static_cast<double>(j) * params.vy -
			                 params.length_y / 2.0 + params.off_y +
			                 params.vy / 2.0;
			for (int k = 0; k < params.nx; k++)
			{
				const double x = static_cast<double>(k) * params.vx -
				                 params.length_x / 2.0 + params.off_x +
				                 params.vx / 2.0;

				double newX = x * cos(alpha) * cos(beta) +
				              y * (-sin(alpha) * cos(gamma) +
				                   sin(beta) * sin(gamma) * cos(alpha)) +
				              z * (sin(alpha) * sin(gamma) +
				                   sin(beta) * cos(alpha) * cos(gamma));
				newX += translation.x;
				double newY = x * sin(alpha) * cos(beta) +
				              y * (sin(alpha) * sin(beta) * sin(gamma) +
				                   cos(alpha) * cos(gamma)) +
				              z * (sin(alpha) * sin(beta) * cos(gamma) -
				                   sin(gamma) * cos(alpha));
				newY += translation.y;
				double newZ = -x * sin(beta) + y * sin(gamma) * cos(beta) +
				              z * cos(beta) * cos(gamma);
				newZ += translation.z;

				const double currentValue =
				    rawPtr[i * num_xy + j * params.nx + k];
				newImg->updateImageInterpolate({newX, newY, newZ}, currentValue,
				                               false);
			}
		}
	}
	return newImg;
}

ImageOwned::ImageOwned(const ImageParams& img_params) : Image(img_params)
{
	m_dataPtr = std::make_unique<Array3D<double>>();
}

ImageOwned::ImageOwned(const ImageParams& img_params,
                       const std::string& filename)
    : ImageOwned(img_params)
{
	readFromFile(filename);
}

void ImageOwned::allocate()
{
	static_cast<Array3D<double>*>(m_dataPtr.get())
	    ->allocate(getParams().nz, getParams().ny, getParams().nx);
}

bool ImageOwned::isAllocated() const
{
	return m_dataPtr.get()->getRawPointer() != nullptr;
}

ImageAlias::ImageAlias(const ImageParams& img_params) : Image(img_params)
{
	m_dataPtr = std::make_unique<Array3DAlias<double>>();
}

void ImageAlias::bind(Array3DBase<double>& p_data)
{
	static_cast<Array3DAlias<double>*>(m_dataPtr.get())->bind(p_data);
	if (m_dataPtr->getRawPointer() != p_data.getRawPointer())
	{
		throw std::runtime_error("An error occured during Image binding");
	}
}