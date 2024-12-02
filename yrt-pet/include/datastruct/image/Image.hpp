/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/image/nifti/nifti1_io.h"
#include "geometry/Vector3D.hpp"
#include "utils/Array.hpp"

#include <string>

struct transform_t;

class Image : public ImageBase
{
public:
	~Image() override = default;

	Array3DBase<float>& getData();
	const Array3DBase<float>& getData() const;
	float* getRawPointer();
	const float* getRawPointer() const;
	bool isMemoryValid() const;

	void copyFromImage(const ImageBase* imSrc) override;
	void multWithScalar(float scalar);
	void addFirstImageToSecond(ImageBase* secondImage) const override;

	void setValue(float initValue) override;
	void applyThreshold(const ImageBase* maskImg, float threshold,
	                    float val_le_scale, float val_le_off,
	                    float val_gt_scale, float val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       float threshold) override;
	void writeToFile(const std::string& fname) const override;

	Array3DAlias<float> getArray() const;

	void transformImage(const Vector3D& rotation, const Vector3D& translation,
	                    Image& dest, float weight) const;
	std::unique_ptr<Image> transformImage(const Vector3D& rotation,
	                                      const Vector3D& translation) const;
	void transformImage(const transform_t& t, Image& dest, float weight) const;
	std::unique_ptr<Image> transformImage(const transform_t& t) const;

	float dotProduct(const Image& y) const;
	float nearestNeighbor(const Vector3D& pt) const;
	float nearestNeighbor(const Vector3D& pt, int* pi, int* pj, int* pk) const;
	void updateImageNearestNeighbor(const Vector3D& pt, float value,
	                                bool mult_flag);
	void assignImageNearestNeighbor(const Vector3D& pt, float value);
	bool getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj,
	                           int* pk) const;

	float interpolateImage(const Vector3D& pt) const;
	float interpolateImage(const Vector3D& pt, const Image& sens) const;
	void updateImageInterpolate(const Vector3D& point, float value,
	                            bool mult_flag);
	void assignImageInterpolate(const Vector3D& point, float value);

	template <int Dimension>
	float indexToPositionInDimension(int index) const;

protected:
	static float originToOffset(float origin, float voxelSize, float length);
	static float offsetToOrigin(float off, float voxelSize, float length);

	Image();
	explicit Image(const ImageParams& imgParams);
	std::unique_ptr<Array3DBase<float>> mp_array;
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
	void bind(Array3DBase<float>& p_data);
};
