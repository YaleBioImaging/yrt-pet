/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Vector3D.hpp"
#include "yrt-pet/utils/Types.hpp"

#include "nlohmann/json_fwd.hpp"
#include <string>


namespace yrt
{

constexpr float IMAGEPARAMS_FILE_VERSION = 1.2f;

class ImageParams
{
public:
	ssize_t nx;
	ssize_t ny;
	ssize_t nz;
	ssize_t nt;
	float length_x;
	float length_y;
	float length_z;
	float vx;
	float vy;
	float vz;
	float off_x;
	float off_y;
	float off_z;

	// Automatically populated fields
	float fovRadius;

	static constexpr float PositioningPrecision = 1e-4f;  // 0.1 micron

	ImageParams();
	ImageParams(ssize_t p_nx, ssize_t p_ny, ssize_t p_nz, float p_length_x,
	            float p_length_y, float p_length_z, float p_offset_x = 0.f,
	            float p_offset_y = 0.f, float p_offset_z = 0.f,
	            ssize_t p_nt = 1);
	ImageParams(const ImageParams& in);
	ImageParams& operator=(const ImageParams& in);
	static ImageParams fromParams(ssize_t nx, ssize_t ny, ssize_t nz, float vx, float vy,
	                              float vz, float originx, float originy,
	                              float originz, ssize_t p_nt = 1);
	static ImageParams fromParams(ssize_t nx, ssize_t ny, ssize_t nz, float vx, float vy,
	                              float vz, ssize_t p_nt = 1);
	explicit ImageParams(const std::string& fname);
	bool isSameNumFramesAs(const ImageParams& other) const;
	bool isSameDimensionsAs(const ImageParams& other) const;
	bool isSameLengthsAs(const ImageParams& other) const;
	bool isSameOffsetAs(const ImageParams& other) const;
	bool isSameAs(const ImageParams& other) const;
	bool isSameAsIgnoreFrames(const ImageParams& other) const;

	// Dimensions: 0: Z, 1: Y, 2: X
	template <int Dimension>
	float indexToPositionInDimension(ssize_t index) const;
	Vector3D indexToPosition(ssize_t ix, ssize_t iy, ssize_t iz) const;
	Vector3D getOrigin() const;
	Vector3D getOffset() const;

	void copy(const ImageParams& in);
	void setup();
	void writeToFile(const std::string& fname) const;
	void serialize(const std::string& fname) const;
	std::string toString() const;
	void writeToJSON(nlohmann::json& j) const;
	void readFromFile(const std::string& fname);
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

}  // namespace yrt