/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "motion/ImageWarperTemplate.hpp"

/* **************************************************************************************
 * Def : Class that implement the warping tool for image motion correction in
 *       tomographic reconstruction using the function approach.
 *
 * Note:
 *		- Create a parent for this class and Matrix? They share a good deal of
 *code.
 * *************************************************************************************/

class ImageWarperFunction : public ImageWarperTemplate
{
public:
	// List of public methods.
	ImageWarperFunction();
	~ImageWarperFunction() override;

private:
	// List of private parameters
	// The rotation matrix of each frame.
	std::vector<std::vector<double>> m_rotMatrix;
	// The translation vector of each frame.
	std::vector<std::vector<double>> m_translation;

	// List of private function
	// Implementation of the parent virtual methods.
	void initWarpModeSpecificParameters() override;
	void reset() override;
	void warp(Image* image, int motionFrameId) const override;
	void inverseWarp(Image* image, int motionFrameId) const override;
	void setFrameWarpParameters(int motionFrameId,
	                            const std::vector<double>& warpParam) override;
	// Specific methods used in this class.
	double getVoxelPhysPos(int voxelId, int voxelDim) const;
	std::vector<double> getVoxelPhysPos(std::vector<int> voxelId);
	void applyTransformation(const std::vector<double>& pos, Vector3D& result,
	                         int frameId) const;
	void applyInvTransformation(const std::vector<double>& pos,
	                            Vector3D& result, int frameId) const;
};
