/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "motion/ImageWarperTemplate.hpp"

/* **************************************************************************************
 * Def : Class that implement the warping tool for image motion correction in
 *       tomographic reconstruction using the matrix approach.
 *
 * Note:
 *		- Create a parent for this class and Matrix? They share a good deal of
 *code.
 * *************************************************************************************/

class ImageWarperMatrix : public ImageWarperTemplate
{
public:
    // List of public methods.
    ImageWarperMatrix();
    ~ImageWarperMatrix() override;

    transform_t getTransformation(int frameId) const;
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
    double getVoxelPhysPos(int voxelId, int voxelDim) const;
    std::vector<double> getVoxelPhysPos(const std::vector<int>& voxelId);
    void applyTransformation(const std::vector<double>& pos, Vector3D& result,
                             int frameId) const;
    void applyInvTransformation(const std::vector<double>& pos, Vector3D& result,
                                int frameId) const;
    bool invInterpolComponent(const Vector3D& pt, std::vector<std::vector<int>>& voxIndex,
                              std::vector<double>& voxValue) const;
};
