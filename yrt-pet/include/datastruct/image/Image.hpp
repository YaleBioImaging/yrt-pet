/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "geometry/Vector3D.hpp"
#include "utils/Array.hpp"

#include <string>

class Image : public ImageBase
{
public:
	Image(const ImageParams& img_params);
	~Image() override = default;

	Array3DBase<double>& getData();
	const Array3DBase<double>& getData() const;
	void copyFromImage(const Image* imSrc);
	void multWithScalar(double scalar);
	void addFirstImageToSecond(ImageBase* secondImage) const override;

	void setValue(double initValue) override;
	void applyThreshold(const ImageBase* maskImg, double threshold,
	                    double val_le_scale, double val_le_off,
	                    double val_gt_scale, double val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       double threshold) override;
	void writeToFile(const std::string& image_fname) const override;

	Array3DAlias<double> getArray() const;
	std::unique_ptr<Image> transformImage(const Vector3D& rotation,
	                                      const Vector3D& translation) const;

	double dotProduct(const Image& y) const;
	double nearestNeighbor(const Vector3D& pt) const;
	double nearestNeighbor(const Vector3D& pt, int* pi, int* pj, int* pk) const;
	void updateImageNearestNeighbor(const Vector3D& pt, double value,
	                                bool mult_flag);
	void assignImageNearestNeighbor(const Vector3D& pt, double value);
	bool getNearestNeighborIdx(const Vector3D& pt, int* pi, int* pj,
	                           int* pk) const;

	double interpolateImage(const Vector3D& pt) const;
	double interpolateImage(const Vector3D& pt, const Image& sens) const;
	void updateImageInterpolate(const Vector3D& point, double value,
	                            bool mult_flag);
	void assignImageInterpolate(const Vector3D& point, double value);

protected:
	std::unique_ptr<Array3DBase<double>> m_dataPtr;
};

class ImageOwned : public Image
{
public:
	ImageOwned(const ImageParams& img_params);
	ImageOwned(const ImageParams& img_params, const std::string& filename);
	void allocate();
	bool isAllocated() const;
	void readFromFile(const std::string& image_file_name);
};

class ImageAlias : public Image
{
public:
	ImageAlias(const ImageParams& img_params);
	void bind(Array3DBase<double>& p_data);
};
