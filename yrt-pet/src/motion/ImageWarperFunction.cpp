/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "motion/ImageWarperFunction.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_imagewarperfunction(py::module& m)
{
	auto c = py::class_<ImageWarperFunction, ImageWarperTemplate>(
	    m, "ImageWarperFunction");
	c.def(py::init<>());
}
#endif

ImageWarperFunction::ImageWarperFunction()
{
	m_warpMode = "Function";
}


ImageWarperFunction::~ImageWarperFunction() {}


/* **************************************************************************************
 * Def.: Clean all the variable of this class.
 * *************************************************************************************/
void ImageWarperFunction::reset()
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
 *		target thus we will store the inverse transformation.
 * @frameId: The frame Id corresponding to the given transformation parameters.
 * @warpParam: Parameters of the transformation. The parameters expected are a
 *		quaternion [qw, qx, qy, qz] followed by a translation [tx, ty, tz].
 * *************************************************************************************/
void ImageWarperFunction::setFrameWarpParameters(
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
		m_translation[frameId][i] = -warpParam[i + 4];
	}
}


/* **************************************************************************************
 * Def.: Deform the image in the reference frame toward the selected frame.
 * @image : Pointer to where we want to store the warped image.
 * @frameId : Id of the frame to which we want to deform toward.
 * *************************************************************************************/
void ImageWarperFunction::warp(Image* image, int frameId) const
{
	std::vector<double> voxPos;
	voxPos.resize(3);
	Vector3D movVoxPos{0.0, 0.0, 0.0};
	Array3DAlias<float> data = image->getArray();
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
				applyTransformation(voxPos, movVoxPos, frameId);
				ptr[i] = mp_refImage->interpolateImage(movVoxPos);
			}
		}
	}
}


/* **************************************************************************************
 * Def.: Warp the provided image with the inverse transform of warping the
 * reference
 *       frame to the selected frame Id.
 * @image: Pointer to the image to warp and where the result of the warp will be
 * saved.
 * @frameId: Id of the frame to which we want to deform from.
 * *************************************************************************************/
void ImageWarperFunction::inverseWarp(Image* image, int frameId) const
{
	auto tmpCopy = std::make_unique<ImageOwned>(image->getParams());
	tmpCopy->allocate();
	tmpCopy->copyFromImage(image);
	image->setValue(0.0);
	std::vector<double> voxPos;
	voxPos.resize(3);
	auto movVoxPos = Vector3D{0.0, 0.0, 0.0};
	Array3DAlias<float> data = image->getArray();
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
				applyInvTransformation(voxPos, movVoxPos, frameId);
				ptr[i] = tmpCopy->interpolateImage(movVoxPos);
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
void ImageWarperFunction::initWarpModeSpecificParameters()
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
std::vector<double> ImageWarperFunction::getVoxelPhysPos(std::vector<int> voxelId)
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
double ImageWarperFunction::getVoxelPhysPos(int voxelId, int voxelDim) const
{
	return ((double)voxelId + 0.5) *
	           (m_imSize[voxelDim] / (double)m_imNbVoxel[voxelDim]) -
	       0.5 * m_imSize[voxelDim];
}


/* **************************************************************************************
 * Def.: Evaluate the corresponding coordinates of a given spatial position of
 * the
 *       reference frame into the image space of the specified framed.
 * @pos: Coordinates in the reference frame.
 * @result: Corresponding coordinates in the specified frame.
 * @frameId: The frame of interest.
 * *************************************************************************************/
void ImageWarperFunction::applyTransformation(const std::vector<double>& pos,
                                              Vector3D& result,
                                              int frameId) const
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
 * @pos: Coordinates in the specified frame.
 * @result: Corresponding coordinates in the reference frame.
 * @frameId: The frame of origin.
 * *************************************************************************************/
void ImageWarperFunction::applyInvTransformation(const std::vector<double>& pos,
                                                 Vector3D& result,
                                                 int frameId) const
{
	auto result_tmp = Vector3D{0.0, 0.0, 0.0};

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
