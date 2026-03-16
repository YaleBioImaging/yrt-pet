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
	int nx;
	int ny;
	int nz;
	frame_t nt;
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

	static constexpr float PositioningPrecision = 1e-4f; // 0.1 micron

	ImageParams();
	ImageParams(int p_nx, int p_ny, int p_nz, float p_length_x,
				float p_length_y, float p_length_z, float p_offset_x = 0.f,
				float p_offset_y = 0.f, float p_offset_z = 0.f,
				frame_t p_nt = 1);
	ImageParams(const ImageParams& in);
	ImageParams& operator=(const ImageParams& in);
	explicit ImageParams(const std::string& fname);
	bool isSameNumFramesAs(const ImageParams& other) const;
	bool isSameDimensionsAs(const ImageParams& other) const;
	bool isSameLengthsAs(const ImageParams& other) const;
	bool isSameOffsetAs(const ImageParams& other) const;
	bool isSameAs(const ImageParams& other) const;
	bool isSameAsIgnoreFrames(const ImageParams& other) const;

	// Dimensions: 0: Z, 1: Y, 2: X
	template <int Dimension>
	float indexToPositionInDimension(int index) const;
	Vector3D indexToPosition(int ix, int iy, int iz) const;
	Vector3D getOrigin() const;
	Vector3D getOffset() const;

	void copy(const ImageParams& in);
	void setup();
	void writeToFile(const std::string& fname) const;
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

}