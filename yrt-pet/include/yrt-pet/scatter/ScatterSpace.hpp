/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/Histogram.hpp"

namespace yrt
{
/*
 * This class defines an undersampled space where values are stored in a 5D
 * array.
 * This assumes that the given scanner is centered in (0,0,0).
 * The scatter space assumes virtual "detectors" uniformly distributed in a
 * perfect cylinder. These virtual detectors form the endpoints of the LORs
 * sampled by this space.
 * The objective is to use this space to compute and store scatter estimates in
 * this undersampled collection of LORs rather than a fully-sampled Histogram3D.
 *
 * The array structures the lines of responses using a pair of cylindrical
 * coordinates (with constant radius) and a TOF value.
 *
 * The continuous indices are as follows:
 *  - TOF value: Difference of arrival time between virtual detector 2 (t2) and
 *    virtual detector 1 (t1), or "t2 - t1", in ps
 *    - The TOF values are always positive and go from 0 to
 *      "scanner diameter"/"speed of light"
 *  - Plane position of virtual detector 1
 *    - Goes from "-axialFOV/2" to "axialFOV/2"
 *  - Angular position of virtual detector 1
 *    - Goes from 0 to 2pi (radians)
 *  - Plane position of virtual detector 2
 *    - Same interval as for virtual detector 1
 *  - Angular position of virtual detector 2
 *    - Same interval as for virtual detector 1
 *
 * The logical indices of this array sample the continuous indices in a regular
 * grid, similar to image voxels.
 *
 * Note that, for cases without TOF, this space would have duplicates, since an
 * LOR representing d2->d1 would be equivalent to an LOR representing d1->d2.
 *
 * Also, this space is not designed to only hold scatter estimate values, it can
 * be used to store any collection of float32 values.
 */
class ScatterSpace : public Histogram
{
public:
	struct ScatterSpaceIndex
	{
		size_t tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2;
	};

	struct ScatterSpacePosition
	{
		float tof_ps, planePosition1, angle1, planePosition2, angle2;
	};

	// Read a scatter-space file to create this object
	ScatterSpace(const Scanner& pr_scanner, const std::string& fname);
	// Create a virgin scatter-space
	ScatterSpace(const Scanner& pr_scanner, size_t p_numTOFBins,
	             size_t p_numPlanes, size_t p_numAngles);

	// IO
	void readFromFile(const std::string& fname);
	void writeToFile(const std::string& fname) const;

	// Do nearest neighbor
	ScatterSpaceIndex
	    getNearestNeighborIndex(const ScatterSpacePosition& pos) const;
	float
		getNearestNeighborValue(const ScatterSpacePosition& pos) const;

	// Do linear interpolation
	float getLinearInterpolationValue(const ScatterSpacePosition& pos) const;

	float getAxialFOV() const;
	float getRadius() const;
	float getDiameter() const;
	float getMaxTOF_ps() const;
	size_t getNumPlanes() const;
	size_t getNumAngles() const;

	// Get the continuous position from logical indices
	float getTOF_ps(size_t TOFBin) const;             // in picoseconds
	float getPlanePosition(size_t planeIndex) const;  // in mm
	float getAngle(size_t angleIndex) const;          // In radians

	// get the logical indices given the continuous position
	size_t getTOFBin(float tof_ps) const;
	size_t getPlaneIndex(float planePosition) const;
	size_t getAngleIndex(float angle) const;

	float getTOFBinStep_ps() const;
	float getAngleStep() const;
	float getPlaneStep() const;

	float getValue(const ScatterSpaceIndex& idx) const;
	float getValue(size_t tofBin, size_t planeIndex1, size_t angleIndex1,
	               size_t planeIndex2, size_t angleIndex2) const;

	void symmetrize();

	static float wrapAngle(float angle);
	size_t wrapAngleIndex(int angleIndex) const;
	float clampPlanePosition(float planePosition) const;
	float clampTOF(float tof_ps) const;

private:
	void initStepSizes();

	size_t m_numTOFBins;
	size_t m_numPlanes;
	size_t m_numAngles;

	float m_TOFBinStep_ps;
	float m_angleStep;
	float m_planeStep;

	// Dimensions:
	//  - TOF bin
	//  - plane index 1
	//  - angle index 1
	//  - plane index 2
	//  - angle index 2
	std::unique_ptr<Array5D<float>> mp_values;
};

}  // namespace yrt
