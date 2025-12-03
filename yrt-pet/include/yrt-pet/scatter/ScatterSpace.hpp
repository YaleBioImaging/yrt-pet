/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Histogram.hpp"

namespace yrt
{
/*
 * This class defines an undersampled space where to store values in an array
 * This assumes that the given scanner is centered in (0,0,0)
 */
class ScatterSpace : public Histogram
{
public:
	struct ScatterSpaceIndex
	{

	};

	// Read a scatter-space file to create this object
	ScatterSpace(const Scanner& pr_scanner, const std::string& fname);
	// Create a virgin scatter-space
	ScatterSpace(const Scanner& pr_scanner, size_t p_numTOFBins,
	             size_t p_numPlanes, size_t p_numAngles);

	void readFromFile(const std::string& fname);
	void writeToFile(const std::string& fname) const;

	float getAxialFOV() const;
	size_t getNumPlanes() const;
	size_t getNumAngles() const;
	float getRadius() const;

	float getTOF_mm(size_t TOFBin) const;
	float getPlanePosition(size_t planeIndex) const;
	float getAngle(size_t angleIndex) const;  // In radians

	float getTOFBinStep_mm() const;
	float getAngleStep() const;
	float getPlaneStep() const;

	void symmetrize();

	static float wrapAngle(float angle);
	float clampPlanePosition(float planePosition) const;

private:
	void initStepSizes();

	size_t m_numTOFBins;
	size_t m_numPlanes;
	size_t m_numAngles;

	float m_TOFBinStep_mm;
	float m_angleStep;
	float m_planeStep;

	std::unique_ptr<Array5D<float>> mp_values;
};

}  // namespace yrt
