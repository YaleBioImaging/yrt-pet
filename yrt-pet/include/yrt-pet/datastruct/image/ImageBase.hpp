/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/operators/Variable.hpp"
#include "yrt-pet/utils/Types.hpp"

#include "nlohmann/json_fwd.hpp"
#include <string>

#define IMAGEPARAMS_FILE_VERSION 1.1

namespace yrt
{

class ImageParams
{
public:
	int nx;
	int ny;
	int nz;
	float length_x;
	float length_y;
	float length_z;
	float vx;
	float vy;
	float vz;
	float off_x;
	float off_y;
	float off_z;
	frame_t num_frames;

	// Automatically populated fields
	float fovRadius;

	static constexpr float PositioningPrecision = 1e-4f;

	ImageParams();
	ImageParams(int nxi, int nyi, int nzi, float length_xi, float length_yi,
	            float length_zi, float offset_xi = 0.f, float offset_yi = 0.f,
	            float offset_zi = 0.f, frame_t num_framesi = 1);
	ImageParams(const ImageParams& in);
	ImageParams& operator=(const ImageParams& in);
	explicit ImageParams(const std::string& fname);
	bool isSameDimensionsAs(const ImageParams& other) const;
	bool isSameLengthsAs(const ImageParams& other) const;
	bool isSameOffsetsAs(const ImageParams& other) const;
	bool isSameAs(const ImageParams& other) const;

	// Dimensions: 0: Z, 1: Y, 2: X
	template <int Dimension>
	float indexToPositionInDimension(int index) const;
	Vector3D indexToPosition(int ix, int iy, int iz) const;

	void copy(const ImageParams& in);
	void setup();
	void serialize(const std::string& fname) const;
	void writeToJSON(nlohmann::json& j) const;
	void deserialize(const std::string& fname);
	void readFromJSON(nlohmann::json& j);
	bool isValid() const;

private:
	static float readLengthFromJSON(nlohmann::json& j,
	                                const std::string& length_name,
	                                const std::string& v_name, int n);
	template <int Dim>
	void completeDimInfo();
};

class ImageBase : public Variable
{
public:
	ImageBase() = default;
	explicit ImageBase(const ImageParams& imgParams);
	~ImageBase() override = default;

	// Common functions
	float getRadius() const;
	const ImageParams& getParams() const;
	void setParams(const ImageParams& newParams);
	size_t unravel(int iz, int iy, int ix, frame_t it = 0) const;

	virtual void setValue(float initValue) = 0;
	virtual void copyFromImage(const ImageBase* imSrc) = 0;
	virtual void addFirstImageToSecond(ImageBase* second) const = 0;
	virtual void applyThreshold(const ImageBase* mask_img, float threshold,
	                            float val_le_scale, float val_le_off,
	                            float val_gt_scale, float val_gt_off) = 0;
	virtual void writeToFile(const std::string& image_fname) const = 0;
	virtual void updateEMThreshold(ImageBase* update_img,
	                               const ImageBase* norm_img,
	                               float threshold) = 0;
	virtual void updateEMThresholdRankScaled(ImageBase* updateImg,
	                                         const ImageBase* normImg,
	                                         const float* c_r, int rank,
	                                         float threshold) = 0;

private:
	ImageParams m_params;
};

}  // namespace yrt
