/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/image/Image.hpp"

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/datastruct/image/ImageUtils.cuh"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/geometry/Matrix.hpp"
#include "yrt-pet/geometry/TransformUtils.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include "yrt-pet/utils/Types.hpp"
#include "yrt-pet/utils/Utilities.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_image(py::module& m)
{
	auto c = py::class_<Image, ImageBase>(m, "Image", py::buffer_protocol());
	c.def("setValue", &Image::setValue, py::arg("initValue"));
	c.def_buffer(
	    [](Image& img) -> py::buffer_info
	    {
		    Array4DBase<float>& d = img.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 4,
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
	c.def("interpolateImage",
	      static_cast<float (Image::*)(const Vector3D& pt, const Image& sens,
	                                   int frame) const>(
	          &Image::interpolateImage),
	      py::arg("pt"), py::arg("sens"), py::arg("frame") = 0);
	c.def("interpolateImage",
	      static_cast<float (Image::*)(const Vector3D& pt, int frame) const>(
	          &Image::interpolateImage),
	      py::arg("pt"), py::arg("frame") = 0);
	c.def(
	    "updateImageInterpolate",
	    [](Image& img, const Vector3D& pt, float value, frame_t frame, bool mult_flag)
	    {
		    if (mult_flag)
		    {
			    img.updateImageInterpolate<true>(pt, value, frame);
		    }
		    else
		    {
			    img.updateImageInterpolate<false>(pt, value, frame);
		    }
	    },
	    py::arg("pt"), py::arg("value"), py::arg("frame") = 0, py::arg("mult_flag") = false);
	c.def("assignImageInterpolate", &Image::assignImageInterpolate,
	      py::arg("pt"), py::arg("value"), py::arg("frame") = 0);
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
	    "updateImageNearestNeighbor",
	    [](Image& img, const Vector3D& pt, float value, frame_t frame, bool mult_flag)
	    {
		    if (mult_flag)
		    {
			    img.updateImageNearestNeighbor<true>(pt, value, frame);
		    }
		    else
		    {
			    img.updateImageNearestNeighbor<false>(pt, value, frame);
		    }
	    },
	    py::arg("pt"), py::arg("value"), py::arg("frame") = 0, py::arg("mult_flag") = false);
	c.def("assignImageNearestNeighbor", &Image::assignImageNearestNeighbor,
	      py::arg("pt"), py::arg("value"), py::arg("frame") = 0);
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
	      static_cast<std::unique_ptr<ImageOwned> (Image::*)(
	          const Vector3D& rotation, const Vector3D& translation) const>(
	          &Image::transformImage),
	      py::arg("rotation"), py::arg("translation"));
	c.def("transformImage",
	      static_cast<std::unique_ptr<ImageOwned> (Image::*)(
	          const transform_t& t) const>(&Image::transformImage),
	      py::arg("transform"));
	c.def("writeToFile", &Image::writeToFile, py::arg("filename"));
	c.def("getNumFrames", &Image::getNumFrames);

	auto c_alias = py::class_<ImageAlias, Image>(m, "ImageAlias");
	c_alias.def(py::init<const ImageParams&>(), py::arg("img_params"));
	c_alias.def(
	    "bind",
	    [](ImageAlias& self, py::buffer& np_data)
	    {
		    py::buffer_info buffer = np_data.request();
		    if (buffer.ndim != 4 && buffer.ndim != 3)
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have 3 or 4 dimensions");
		    }
		    if (buffer.format != py::format_descriptor<float>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float32 format");
		    }
		    std::vector<int> dims = {self.getNumFrames(), self.getParams().nz,
		                             self.getParams().ny, self.getParams().nx};
		    for (int i = 0; i < buffer.ndim; i++)
		    {
			    if (buffer.shape[i] != dims[buffer.ndim == 4 ? i : i + 1])
			    {
				    throw std::invalid_argument(
				        "The buffer shape does not match with the image "
				        "parameters");
			    }
		    }
		    static_cast<Array4DAlias<float>&>(self.getData())
		        .bind(reinterpret_cast<float*>(buffer.ptr), dims[0], dims[1],
		              dims[2], dims[3]);
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
}  // namespace yrt
#endif  // if BUILD_PYBIND11

namespace yrt
{

Image::Image() : ImageBase{} {}

Image::Image(const ImageParams& imgParams)
    : ImageBase(imgParams)
{
}

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

Array4DBase<float>& Image::getData()
{
	return *mp_array;
}

const Array4DBase<float>& Image::getData() const
{
	return *mp_array;
}

int Image::getNumFrames() const
{
	return getParams().num_frames;
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

float Image::voxelSum() const
{
	// Use double to avoid precision loss
	const ImageParams& params = getParams();
	const size_t numVoxels = getNumFrames() * params.nx * params.ny * params.nz;
	const float* rawPtr = mp_array->getRawPointer();
	std::function<double(double, double)> func_sum = [](double a, double b)
	{ return a + b; };

	return util::simpleReduceArray(rawPtr, numVoxels, func_sum, 0.0,
	                               globals::getNumThreads());
}

void Image::multWithScalar(float scalar)
{
	*mp_array *= scalar;
}

// return the value of the voxel the nearest to "point":
float Image::nearestNeighbor(const Vector3D& pt, int frame) const
{
	int ix, iy, iz;

	if (getNearestNeighborIdx(pt, &ix, &iy, &iz, frame))
	{
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		const size_t num_xyz = getParams().nz * getParams().nx * getParams().ny;
		return mp_array->getFlat(frame * num_xyz + iz * num_xy + iy * num_x +
		                         ix);
	}
	return 0;
}

// return the value of the voxel the nearest to "point":
float Image::nearestNeighbor(const Vector3D& pt, int* pi, int* pj, int* pk,
                             int frame) const
{
	if (getNearestNeighborIdx(pt, pi, pj, pk, frame))
	{
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		const size_t num_xyz = getParams().nx * getParams().ny * getParams().nz;
		return mp_array->getFlat(frame * num_xyz + *pk * num_xy + *pj * num_x +
		                         *pi);
	}
	return 0.0;
}

// update image with "value" using nearest neighbor method:
template <bool MULT_FLAG>
void Image::updateImageNearestNeighbor(const Vector3D& pt, float value,
                                      int frame)
{
	int ix, iy, iz;
	if (getNearestNeighborIdx(pt, &ix, &iy, &iz, frame))
	{
		// update multiplicatively or additively:
		float* ptr = mp_array->getRawPointer();
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		const size_t num_xyz = getParams().nx * getParams().ny * getParams().nz;
		const size_t idx = frame * num_xyz + iz * num_xy + iy * num_x + ix;
		if constexpr (MULT_FLAG)
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
void Image::assignImageNearestNeighbor(const Vector3D& pt, float value,
                                       int frame)
{
	int ix, iy, iz;
	if (getNearestNeighborIdx(pt, &ix, &iy, &iz, frame))
	{
		// update multiplicatively or additively:
		float* ptr = mp_array->getRawPointer();
		const size_t num_x = getParams().nx;
		const size_t num_xy = getParams().nx * getParams().ny;
		const size_t num_xyz = getParams().nx * getParams().ny * getParams().nz;
		ptr[frame * num_xyz + iz * num_xy + iy * num_x + ix] = value;
	}
}

// Returns true if the point `pt` is inside the image
bool Image::getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj, int* pk,
                                  int frame) const
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
	    iz >= params.nz || frame < 0 || frame >= getNumFrames())
	{
		// Point outside grid
		return false;
	}

	*pi = ix;
	*pj = iy;
	*pk = iz;

	return true;
}

// interpolation operation.
float Image::interpolateImage(const Vector3D& pt, int frame) const
{
	const ImageParams& params = getParams();

	float weights[8];
	int indices[8];

	util::trilinearInterpolate(pt.x, pt.y, pt.z, params.nx, params.ny,
	                           params.nz, params.length_x, params.length_y,
	                           params.length_z, params.off_x, params.off_y,
	                           params.off_z, indices, weights);

	const float* rawPtr = getRawPointer() + frame * params.nx * params.ny * params.nz;
	float total = 0.0f;

	for (size_t i = 0; i < 8; i++)
	{
		total += rawPtr[indices[i]] * weights[i];
	}

	return total;
}

// calculate the value of a point on the image matrix
// using tri-linear interpolation and weighting with image "sens":
float Image::interpolateImage(const Vector3D& pt, const Image& sens,
                              int frame) const
{
	const ImageParams& params = getParams();

	float weights[8];
	int indices[8];

	util::trilinearInterpolate(pt.x, pt.y, pt.z, params.nx, params.ny,
	                           params.nz, params.length_x, params.length_y,
	                           params.length_z, params.off_x, params.off_y,
	                           params.off_z, indices, weights);

	const float* rawPtr = getRawPointer() + frame * params.nx * params.ny * params.nz;;
	const float* sensRawPtr = sens.getRawPointer() + frame * params.nx * params.ny * params.nz;
	float total = 0.0f;

	for (size_t i = 0; i < 8; i++)
	{
		total += rawPtr[indices[i]] * weights[i] * sensRawPtr[indices[i]];
	}

	return total;
}

// update image with "value" using trilinear interpolation:
template <int OPERATION>  // 0: assign, 1: multiply, 2: add
void Image::operationImageInterpolate(const Vector3D& pt, float value, int frame)
{
	// Only allow defined operations
	static_assert(OPERATION >= 0 && OPERATION <= 2);

	const ImageParams& params = getParams();

	float weights[8];
	int indices[8];

	util::trilinearInterpolate(pt.x, pt.y, pt.z, params.nx, params.ny,
	                           params.nz, params.length_x, params.length_y,
	                           params.length_z, params.off_x, params.off_y,
	                           params.off_z, indices, weights);

	float* rawPtr = getRawPointer() + frame * params.nx * params.ny * params.nz;;

	for (size_t i = 0; i < 8; i++)
	{
		if (OPERATION == 0)
		{
			rawPtr[indices[i]] = weights[i] * value;
		}
		else if constexpr (OPERATION == 1)
		{
			rawPtr[indices[i]] *= weights[i] * value;
		}
		else
		{
			rawPtr[indices[i]] += weights[i] * value;
		}
	}
}

// assign image with "value" using trilinear interpolation:
template <bool MULT_FLAG>
void Image::updateImageInterpolate(const Vector3D& pt, float value, int frame)
{
	if (MULT_FLAG)
	{
		operationImageInterpolate<1>(pt, value, frame);
	}
	else
	{
		operationImageInterpolate<2>(pt, value, frame);
	}
}

// assign image with "value" using trilinear interpolation:
void Image::assignImageInterpolate(const Vector3D& pt, float value, int frame)
{
	operationImageInterpolate<0>(pt, value, frame);
}

void Image::operationOnEachVoxel(const std::function<float(size_t)>& func)
{
	const ImageParams& params = getParams();
	float* flatPtr = mp_array->getRawPointer();
	const size_t numVoxels = getNumFrames() * params.nx * params.ny * params.nz;
	for (size_t i = 0; i < numVoxels; i++)
	{
		flatPtr[i] = func(i);
	}
}

void Image::operationOnEachVoxelParallel(
    const std::function<float(size_t)>& func)
{
	const ImageParams& params = getParams();
	float* flatPtr = mp_array->getRawPointer();
	const size_t numVoxels = getNumFrames() * params.nx * params.ny * params.nz;

	util::parallelForChunked(numVoxels, globals::getNumThreads(),
	                         [flatPtr, func](size_t i, size_t /*tid*/)
	                         { flatPtr[i] = func(i); });
}

// this function writes "image" on disk @ "image_fname"
void Image::writeToFile(const std::string& fname) const
{
	ASSERT(!fname.empty());
	ASSERT_MSG_WARNING(
	    util::endsWith(fname, ".nii") || util::endsWith(fname, ".nii.gz"),
	    "The NIfTI image file extension should be either .nii or .nii.gz");

	const ImageParams& params = getParams();
	const int dims[] = {4, params.nx, params.ny, params.nz, getNumFrames()};
	nifti_image* nim = nifti_make_new_nim(dims, NIFTI_TYPE_FLOAT32, 0);
	nim->nx = params.nx;
	nim->ny = params.ny;
	nim->nz = params.nz;
	nim->nt = getNumFrames();
	nim->nbyper = sizeof(float);
	nim->datatype = NIFTI_TYPE_FLOAT32;
	nim->pixdim[0] = 0.0f;
	nim->dx = params.vx;
	nim->dy = params.vy;
	nim->dz = params.vz;
	// todo: is pixel spacing along time always 1 by default?
	nim->dt = 1;
	nim->pixdim[1] = params.vx;
	nim->pixdim[2] = params.vy;
	nim->pixdim[3] = params.vz;
	// todo: is pixel spacing along time always 1 by default?
	nim->pixdim[4] = 1;
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

void Image::updateEMThresholdRankScaled(ImageBase* updateImg,
                                 const ImageBase* normImg,
                                 const float* c_r, int rank,
                                 float threshold)
{
	Image* updateImg_Image = dynamic_cast<Image*>(updateImg);
	const Image* normImg_Image = dynamic_cast<const Image*>(normImg);

	float* ptr = mp_array->getRawPointer();
	float* up_ptr = updateImg_Image->getRawPointer();
	const float* norm_ptr = normImg_Image->getRawPointer();

	// number of voxels per rank slab (nz*ny*nx)
	auto params = updateImg_Image->getParams();
	const size_t J = params.nx * params.ny * params.nz;

// Parallelize across rank slabs (optional)
//#pragma omp parallel for if (rank * J > (1u << 16))
	for (int r = 0; r < rank; ++r) {
		const float cr    = c_r[r];
		const float invcr = 1.0f / cr;               // micro-opt
		const float thr_r = threshold * invcr;       // s[j] > threshold/cr

		const size_t base = r * J;
		float* ptr_r = ptr + base;
		const float* up_ptr_r  = up_ptr + base;

		for (size_t j = 0; j < J; ++j) {
			if (norm_ptr[j] > thr_r) {
				// up / (sj * cr)  ==  (up/sj) * (1/cr)
				ptr_r[j] *= up_ptr_r[j] / (norm_ptr[j] * cr);
			}
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

Array4DAlias<float> Image::getArray() const
{
	return {mp_array.get()};
}

void Image::transformImage(const Vector3D& rotation,
                           const Vector3D& translation, Image& dest,
                           float weight) const
{
	const auto transform =
	    util::fromRotationAndTranslationVectors(rotation, translation);
	return transformImage(transform, dest, weight);
}

std::unique_ptr<ImageOwned>
    Image::transformImage(const Vector3D& rotation,
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
	float* destRawPtr = dest.getRawPointer();
	const ImageParams* paramsPtr = &params;
	const int num_xy = params.nx * params.ny;
	const int nx = params.nx;
	const int ny = params.ny;
	const int nz = params.nz;
	// todo: Update to loop through frames here too?

	const transform_t inv = util::invertTransform(t);

	util::parallelForChunked(
	    nz, globals::getNumThreads(),
	    [paramsPtr, nx, ny, num_xy, inv, weight, destRawPtr,
	     this](size_t i, size_t /*tid*/)
	    {
		    const float z = paramsPtr->indexToPositionInDimension<0>(i);

		    for (int j = 0; j < ny; j++)
		    {
			    const float y = paramsPtr->indexToPositionInDimension<1>(j);

			    for (int k = 0; k < nx; k++)
			    {
				    const float x = paramsPtr->indexToPositionInDimension<2>(k);

				    float newX = x * inv.r00 + y * inv.r01 + z * inv.r02;
				    newX += inv.tx;
				    float newY = x * inv.r10 + y * inv.r11 + z * inv.r12;
				    newY += inv.ty;
				    float newZ = x * inv.r20 + y * inv.r21 + z * inv.r22;
				    newZ += inv.tz;

				    const float valueFromOriginalImage =
				        weight * interpolateImage({newX, newY, newZ});

				    destRawPtr[i * num_xy + j * nx + k] +=
				        valueFromOriginalImage;
			    }
		    }
	    });
}

std::unique_ptr<ImageOwned> Image::transformImage(const transform_t& t) const
{
	auto newImg = std::make_unique<ImageOwned>(getParams());
	newImg->allocate();
	newImg->setValue(0.0);
	transformImage(t, *newImg, 1.0f);
	return newImg;
}

ImageOwned::ImageOwned(const ImageParams& imgParams) : Image{imgParams}
{
	mp_array = std::make_unique<Array4D<float>>();
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
	mp_array = std::make_unique<Array4D<float>>();

	// Deduce image parameters from given file
	readFromFile(filename);
}

void ImageOwned::allocate()
{
	ASSERT(mp_array != nullptr);
	const ImageParams& params = getParams();
	reinterpret_cast<Array4D<float>*>(mp_array.get())
	    ->allocate(getNumFrames(), params.nz, params.ny, params.nx);
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
		throw std::invalid_argument("An error occurred while reading file " +
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
	//	todo: Spacing along time (?)

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
		ASSERT_MSG(niftiImage->dim[0] == 4, "NIfTI Image's dim[0] is not 4");
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
	const int numVoxels = getNumFrames() * params.nx * params.ny * params.nz;

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
			    (util::reinterpretAndCast<double, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT8)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<int8_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT16)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<int16_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT32)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<int32_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_INT64)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<int64_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT8)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<uint8_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT16)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<uint16_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT32)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<uint32_t, float>(data, i) * slope) +
			    intercept;
	}
	else if (datatype == NIFTI_TYPE_UINT64)
	{
		for (int i = 0; i < numVoxels; i++)
			imgData[i] =
			    (util::reinterpretAndCast<uint64_t, float>(data, i) * slope) +
			    intercept;
	}
}


void ImageOwned::checkImageParamsWithGivenImage(float voxelSpacing[3],
                                                float imgOrigin[3],
                                                const int dim[8]) const
{
	const ImageParams& params = getParams();

	ASSERT(dim[0] == 4 || dim[0] == 3);

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
	return util::indexToPosition(index, voxelSize, length, offset);
}

template float Image::indexToPositionInDimension<0>(int index) const;
template float Image::indexToPositionInDimension<1>(int index) const;
template float Image::indexToPositionInDimension<2>(int index) const;

ImageAlias::ImageAlias(const ImageParams& imgParams) : Image{imgParams}
{
	mp_array = std::make_unique<Array4DAlias<float>>();
}

void ImageAlias::bind(Array4DBase<float>& p_data)
{
	static_cast<Array4DAlias<float>*>(mp_array.get())->bind(p_data);
	if (mp_array->getRawPointer() != p_data.getRawPointer())
	{
		throw std::runtime_error("An error occurred during Image binding");
	}
}
}  // namespace yrt
