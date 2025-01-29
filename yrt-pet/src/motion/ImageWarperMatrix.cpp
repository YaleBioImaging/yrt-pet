/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "motion/ImageWarperMatrix.hpp"
#include "utils/Types.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_imagewarpermatrix(py::module& m)
{
	auto c = py::class_<ImageWarperMatrix, ImageWarperTemplate>(
	    m, "ImageWarperMatrix");
	c.def(py::init<>());

	c.def("getTransformation", &ImageWarperMatrix::getTransformation);
	c.def("getInvTransformation", &ImageWarperMatrix::getInvTransformation);
}
#endif

ImageWarperMatrix::ImageWarperMatrix()
{
	m_warpMode = "Matrix";
}


ImageWarperMatrix::~ImageWarperMatrix() {}


/* **************************************************************************************
 * Def.: Clean all the variable of this class.
 * *************************************************************************************/
void ImageWarperMatrix::reset()
{
	ImageWarperTemplate::reset();
	// TODO: Clean the children specific variables.
}


/* **************************************************************************************
 * Def.: Set the transformation parameters that define the transition between
 *the
 *       reference frame and the specified frame.
 * Note: We want the interpolation of the warping to be from the moved image to
 *the
 *		target thus we will apply the inverse transformation.
 * @frameId: The frame Id corresponding to the given transformation parameters.
 * @warpParam: Parameters of the transformation. The parameters expected are a
 *		quaternion [qw, qx, qy, qz] followed by a translation [tx, ty, tz].
 * *************************************************************************************/
void ImageWarperMatrix::setFrameWarpParameters(
    int frameId, const std::vector<double>& warpParam)
{
	// Invert the rotation.
	std::vector<double> invQuaternion;
	invQuaternion.resize(warpParam.size());
	invQuaternion[0] = warpParam[0];
	for (int n = 1; n < 4; n++)
	{
		invQuaternion[n] = -warpParam[n];
	}

	// Create the corresponding rotation matrix.
	std::vector<double> rotMatrix;
	m_rotMatrix[frameId] = convertQuaternionToRotationMatrix(invQuaternion);

	// Invert the translation.
	for (int i = 0; i < 3; i++)
	{
		// YN: Why is this inverted ???
		m_translation[frameId][i] = -warpParam[i + 4];
	}
}


/* **************************************************************************************
 * Def.: Deform the image in the reference frame toward the selected frame id.
 * @_image : Pointer to where we want to save the warped image.
 * @_frameId : Id of the frame to which we want to deform toward.
 * *************************************************************************************/
void ImageWarperMatrix::warp(Image* _image, int _frameId) const
{
	std::vector<double> voxPos;
	voxPos.resize(3);
	Vector3D movVoxPos{0.0, 0.0, 0.0};
	Array3DAlias<float> data = _image->getArray();
	Array2DAlias<float> slice;

	for (int k = 0; k < m_imNbVoxel[2]; k++)
	{
		slice.bind(data.getSlicePtr(k), m_imNbVoxel[1], m_imNbVoxel[0]);
		voxPos[2] = getVoxelPhysPos(k, 2);
		for (int j = 0; j < m_imNbVoxel[1]; j++)
		{
			voxPos[1] = getVoxelPhysPos(j, 1);
			float* ptr = slice[j];
			for (int i = 0; i < m_imNbVoxel[0]; i++)
			{
				voxPos[0] = getVoxelPhysPos(i, 0);

				applyTransformation(voxPos, movVoxPos, _frameId);
				ptr[i] = mp_refImage->interpolateImage(movVoxPos);
			}
		}
	}
}


/* **************************************************************************************
 * Def.: Warp the provided image with the transpose of the warping matrix of the
 *       reference frame to the selected frame Id.
 * @_image : Pointer to the image to warp and where the result of the warp will
 * be saved.
 * @_frameId : Id of the frame to which we want to deform from.
 * *************************************************************************************/
void ImageWarperMatrix::inverseWarp(Image* _image, int _frameId) const
{
	const ImageParams& img_params = _image->getParams();
	auto tmpCopy = std::make_unique<ImageOwned>(img_params);
	tmpCopy->allocate();
	tmpCopy->copyFromImage(_image);
	_image->setValue(0.0);

	std::vector<double> voxPos;
	voxPos.resize(3);
	Vector3D movVoxPos{0.0, 0.0, 0.0};

	std::vector<std::vector<int>> voxCompIndex;
	voxCompIndex.resize(8);
	for (size_t i = 0; i < voxCompIndex.size(); i++)
	{
		voxCompIndex[i].resize(3);
	}
	std::vector<double> voxCompInterpWeight;
	voxCompInterpWeight.resize(8);

	double currVoxelVal;
	bool pointValid;

	Array3DAlias<float> data_copy = tmpCopy->getArray();
	Array2DAlias<float> slice_copy;
	Array3DAlias<float> data = _image->getArray();
	float* raw_img_ptr = &data.getFlat(0);
	size_t num_x = img_params.nx;
	size_t num_xy = img_params.nx * img_params.ny;

	for (int k = 0; k < m_imNbVoxel[2]; k++)
	{
		slice_copy.bind(data_copy.getSlicePtr(k), m_imNbVoxel[1],
		                m_imNbVoxel[0]);
		voxPos[2] = getVoxelPhysPos(k, 2);
		for (int j = 0; j < m_imNbVoxel[1]; j++)
		{
			voxPos[1] = getVoxelPhysPos(j, 1);
			float* row_ptr = slice_copy[j];
			for (int i = 0; i < m_imNbVoxel[0]; i++)
			{
				voxPos[0] = getVoxelPhysPos(i, 0);

				applyTransformation(voxPos, movVoxPos, _frameId);
				pointValid = invInterpolComponent(movVoxPos, voxCompIndex,
				                                  voxCompInterpWeight);
				currVoxelVal = row_ptr[i];

				if (pointValid == true)
				{
					for (size_t l = 0; l < voxCompIndex.size(); l++)
					{
						float* cur_img_ptr =
						    raw_img_ptr + voxCompIndex[l][2] * num_xy +
						    voxCompIndex[l][1] * num_x + voxCompIndex[l][0];
						*cur_img_ptr += currVoxelVal * voxCompInterpWeight[l];
					}
				}
			}
		}
	}
}


/* **************************************************************************************
 * Def.: Initialize the variables required to acomplish it's tasks.
 * Note: Currently, even the reference frame transform is initialized. Since it
 * is not
 *       used currently, it is kind of a waste. However, it could be usefull if
 * we
 *       attempt reconstruction in another voxel basis.
 * *************************************************************************************/
void ImageWarperMatrix::initWarpModeSpecificParameters()
{

	m_rotMatrix.resize(m_numberOfFrame);
	m_translation.resize(m_numberOfFrame);
	for (int m = 0; m < m_numberOfFrame; m++)
	{
		m_rotMatrix[m].resize(9);
		m_translation[m].resize(3);
	}
}


/* **************************************************************************************
 * Def.:  Evaluate the physical position of a voxel in the three dimensions.
 * @voxelId: The voxel Id in each dimension.
 * *************************************************************************************/
std::vector<double>
    ImageWarperMatrix::getVoxelPhysPos(const std::vector<int>& voxelId)
{
	std::vector<double> voxPos;
	voxPos.resize(3);
	for (int i = 0; i < 3; i++)
	{
		voxPos[i] = getVoxelPhysPos(voxelId[i], i);
	}
	return voxPos;
}


/* **************************************************************************************
 * Def.: Evaluate the physical position of a voxel center for the specified
 * dimension.
 * @voxelId: The voxel Id in the specified dimension.
 * @voxelDim: The dimension of interest.
 * Note: It assume that the image is centered at 0 in all dimension.
 * *************************************************************************************/
double ImageWarperMatrix::getVoxelPhysPos(int voxelId, int voxelDim) const
{
	return ((double)voxelId + 0.5) *
	           (m_imSize[voxelDim] / (double)m_imNbVoxel[voxelDim]) -
	       0.5 * m_imSize[voxelDim];
}


/* **************************************************************************************
 * Def.: Evaluate the corresponding coordinates of a given spatial position of
 * the
 *       reference frame into the image space of the specified frame.
 * @pos: Coordinates in the reference frame.
 * @result: Corresponding coordinates in the specified frame.
 * @frameId: The frame of interest.
 * *************************************************************************************/
void ImageWarperMatrix::applyTransformation(const std::vector<double>& pos,
                                            Vector3D& result, int frameId) const
{
	result.x = m_rotMatrix[frameId][0] * pos[0] +
	           m_rotMatrix[frameId][1] * pos[1] +
	           m_rotMatrix[frameId][2] * pos[2];
	result.y = m_rotMatrix[frameId][3] * pos[0] +
	           m_rotMatrix[frameId][4] * pos[1] +
	           m_rotMatrix[frameId][5] * pos[2];
	result.z = m_rotMatrix[frameId][6] * pos[0] +
	           m_rotMatrix[frameId][7] * pos[1] +
	           m_rotMatrix[frameId][8] * pos[2];

	result.x += m_translation[frameId][0];
	result.y += m_translation[frameId][1];
	result.z += m_translation[frameId][2];
}


/* **************************************************************************************
 * Def.: Evaluate the corresponding coordinates of a given spatial position of
 * the
 *       specified frame into the image space of the reference frame.
 * @pos: Coordinates in the reference frame.
 * @result: Corresponding coordinates in the specified frame.
 * @frameId: The frame of origin.
 * *************************************************************************************/
void ImageWarperMatrix::applyInvTransformation(const std::vector<double>& pos,
                                               Vector3D& result,
                                               int frameId) const
{
	Vector3D result_tmp{0.0, 0.0, 0.0};

	result_tmp.x = pos[0] - m_translation[frameId][0];
	result_tmp.y = pos[1] - m_translation[frameId][1];
	result_tmp.z = pos[2] - m_translation[frameId][2];

	result.x = m_rotMatrix[frameId][0] * result_tmp.x +
	           m_rotMatrix[frameId][3] * result_tmp.y +
	           m_rotMatrix[frameId][6] * result_tmp.z;
	result.y = m_rotMatrix[frameId][1] * result_tmp.x +
	           m_rotMatrix[frameId][4] * result_tmp.y +
	           m_rotMatrix[frameId][7] * result_tmp.z;
	result.z = m_rotMatrix[frameId][2] * result_tmp.x +
	           m_rotMatrix[frameId][5] * result_tmp.y +
	           m_rotMatrix[frameId][8] * result_tmp.z;
}


/* **************************************************************************************
 * Def.: Find the Id-s and values of the voxel that were used to interpolate the
 *value
 *       of the warped voxel with the given position.
 * @pt: The warped position from which we want to extract the interplation
 *componnent.
 * @voxIndex: The Id of the voxels used in the interpolation for the given
 *position.
 * @voxValue: Interpolation weight of the voxels used in the interpolation for
 *the given
 *		position.
 * Note:
 *		- voxIndex and voxValue are filled with the same order.
 * *************************************************************************************/
bool ImageWarperMatrix::invInterpolComponent(
    const Vector3D& pt, std::vector<std::vector<int>>& voxIndex,
    std::vector<double>& voxValue) const
{
	int ix, iy, iz, ix1, ix2, iy1, iy2, iz1, iz2;
	double dx, dy, dz, dx1, dy1, dz1, delta_x, delta_y, delta_z;

	// if point outside of the image, return 0:
	if ((std::abs(pt.x) >= (m_imSize[0] / 2)) ||
	    (std::abs(pt.y) >= (m_imSize[1] / 2)) ||
	    (std::abs(pt.z) >= (m_imSize[2] / 2)))
	{
		return false;
	}
	dx = (pt.x + m_imSize[0] / 2) / m_imSize[0] * ((double)m_imNbVoxel[0]);
	dy = (pt.y + m_imSize[1] / 2) / m_imSize[1] * ((double)m_imNbVoxel[1]);
	dz = (pt.z + m_imSize[2] / 2) / m_imSize[2] * ((double)m_imNbVoxel[2]);

	ix = (int)dx;
	iy = (int)dy;
	iz = (int)dz;

	delta_x = dx - (double)ix;
	delta_y = dy - (double)iy;
	delta_z = dz - (double)iz;

	// parameters of the x interpolation:
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
		if (ix != (m_imNbVoxel[0] - 1))
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
		if (iy != (m_imNbVoxel[1] - 1))
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
		if (iz != (m_imNbVoxel[2] - 1))
			iz2 = iz + 1;
		else
			iz2 = iz1;
	}

	// Extract the contributing voxels Id-s and weights.
	voxIndex[0] = {ix1, iy1, iz1};
	voxIndex[1] = {ix2, iy1, iz1};
	voxIndex[2] = {ix1, iy2, iz1};
	voxIndex[3] = {ix2, iy2, iz1};
	voxIndex[4] = {ix1, iy1, iz2};
	voxIndex[5] = {ix2, iy1, iz2};
	voxIndex[6] = {ix1, iy2, iz2};
	voxIndex[7] = {ix2, iy2, iz2};

	voxValue[0] = (1 - dz1) * (1 - dy1) * (1 - dx1);
	voxValue[1] = (1 - dz1) * (1 - dy1) * dx1;
	voxValue[2] = (1 - dz1) * dy1 * (1 - dx1);
	voxValue[3] = (1 - dz1) * dy1 * dx1;
	voxValue[4] = dz1 * (1 - dy1) * (1 - dx1);
	voxValue[5] = dz1 * (1 - dy1) * dx1;
	voxValue[6] = dz1 * dy1 * (1 - dx1);
	voxValue[7] = dz1 * dy1 * dx1;

	return true;
}

transform_t ImageWarperMatrix::getTransformation(int frameId) const
{
	return transform_t{static_cast<float>(m_rotMatrix[frameId][0]),
	                   static_cast<float>(m_rotMatrix[frameId][1]),
	                   static_cast<float>(m_rotMatrix[frameId][2]),
	                   static_cast<float>(m_rotMatrix[frameId][3]),
	                   static_cast<float>(m_rotMatrix[frameId][4]),
	                   static_cast<float>(m_rotMatrix[frameId][5]),
	                   static_cast<float>(m_rotMatrix[frameId][6]),
	                   static_cast<float>(m_rotMatrix[frameId][7]),
	                   static_cast<float>(m_rotMatrix[frameId][8]),
	                   static_cast<float>(m_translation[frameId][0]),
	                   static_cast<float>(m_translation[frameId][1]),
	                   static_cast<float>(m_translation[frameId][2])};
}

transform_t ImageWarperMatrix::getInvTransformation(int frameId) const
{
	return transform_t{static_cast<float>(m_rotMatrix[frameId][0]),
	                   static_cast<float>(m_rotMatrix[frameId][3]),
	                   static_cast<float>(m_rotMatrix[frameId][6]),
	                   static_cast<float>(m_rotMatrix[frameId][1]),
	                   static_cast<float>(m_rotMatrix[frameId][4]),
	                   static_cast<float>(m_rotMatrix[frameId][7]),
	                   static_cast<float>(m_rotMatrix[frameId][2]),
	                   static_cast<float>(m_rotMatrix[frameId][5]),
	                   static_cast<float>(m_rotMatrix[frameId][8]),
	                   static_cast<float>(-m_translation[frameId][0]),
	                   static_cast<float>(-m_translation[frameId][1]),
	                   static_cast<float>(-m_translation[frameId][2])};
}
