/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageBase.hpp"
#include "yrt-pet/datastruct/image/nifti/nifti1_io.h"
#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <functional>
#include <string>

namespace yrt
{
struct transform_t;
class ImageOwned;

class Image : public ImageBase
{
public:
	~Image() override = default;

	Array4DBase<float>& getData();
	const Array4DBase<float>& getData() const;
	float* getRawPointer();
	const float* getRawPointer() const;
	bool isMemoryValid() const;
	int getNumFrames() const;

	void copyFromImage(const ImageBase* imSrc) override;
	void multWithScalar(float scalar);
	void addFirstImageToSecond(ImageBase* secondImage) const override;

	float voxelSum() const;
	void setValue(float initValue) override;
	void applyThreshold(const ImageBase* maskImg, float threshold,
	                    float val_le_scale, float val_le_off,
	                    float val_gt_scale, float val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       float threshold) override;
	void updateEMThresholdRankScaled(ImageBase* updateImg,
	                                 const ImageBase* normImg,
	                                 const float* c_r, int rank,
	                                 float threshold) override;

	void writeToFile(const std::string& fname) const override;

	Array4DAlias<float> getArray() const;

	void transformImage(const Vector3D& rotation, const Vector3D& translation,
	                    Image& dest, float weight) const;
	std::unique_ptr<ImageOwned>
	    transformImage(const Vector3D& rotation,
	                   const Vector3D& translation) const;
	void transformImage(const transform_t& t, Image& dest, float weight) const;
	std::unique_ptr<ImageOwned> transformImage(const transform_t& t) const;

	float dotProduct(const Image& y) const;
	float nearestNeighbor(const Vector3D& pt, int frame = 0) const;
	float nearestNeighbor(const Vector3D& pt, int* pi, int* pj, int* pk,
	                      int frame = 0) const;
	template<bool MULT_FLAG>
	void updateImageNearestNeighbor(const Vector3D& pt, float value,
	                                int frame = 0);
	void assignImageNearestNeighbor(const Vector3D& pt, float value, int frame = 0);
	bool getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj,
	                           int* pk, int frame = 0) const;

	float interpolateImage(const Vector3D& pt, int frame = 0) const;
	float interpolateImage(const Vector3D& pt, const Image& sens,
	                       int frame = 0) const;
	template<bool MULT_FLAG>
	void updateImageInterpolate(const Vector3D& point, float value,
	                            int frame = 0);
	void assignImageInterpolate(const Vector3D& point, float value, int frame = 0);

	void operationOnEachVoxel(const std::function<float(size_t)>& func);
	// Note: The function given as argument should be able to be called in
	// parallel without race conditions for different bins.
	// In other words, two different bins shouldn't point
	// to the same memory location.
	void operationOnEachVoxelParallel(const std::function<float(size_t)>& func);

protected:
	static float originToOffset(float origin, float voxelSize, float length);
	static float offsetToOrigin(float off, float voxelSize, float length);

	Image();
	explicit Image(const ImageParams& imgParams);
	std::unique_ptr<Array4DBase<float>> mp_array;

private:
	// Helper
	template <int OPERATION> // operations 0: assign, 1: multiply, 2: add
	void operationImageInterpolate(const Vector3D& pt, float value, int frame);
};

class ImageOwned : public Image
{
public:
	explicit ImageOwned(const ImageParams& imgParams);
	ImageOwned(const ImageParams& imgParams, const std::string& filename);
	explicit ImageOwned(const std::string& filename);
	void allocate();
	void readFromFile(const std::string& fname);

private:
	void checkImageParamsWithGivenImage(float voxelSpacing[3],
	                                    float imgOrigin[3],
	                                    const int dim[8]) const;
	void readNIfTIData(int datatype, void* data, float slope, float intercept);
	static mat44 adjustAffineMatrix(mat44 matrix);
};

class ImageAlias : public Image
{
public:
	explicit ImageAlias(const ImageParams& imgParams);
	void bind(Array4DBase<float>& p_data);
};
}  // namespace yrt
