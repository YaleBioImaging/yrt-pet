/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/projection/Histogram.hpp"

namespace yrt
{

// This class defines an undersampled space where values are stored in a 5D
// array.
// This assumes that the given scanner is centered in (0,0,0).
// The scatter-space assumes virtual "detectors" uniformly distributed in a
// perfect cylinder. These virtual detectors form the endpoints of the LORs
// sampled by this space.
// The objective is to use this space to compute and store scatter estimates in
// this undersampled collection of LORs rather than a fully-sampled Histogram3D.
//
// The array structures the lines of responses using a pair of cylindrical
// coordinates (with constant radius) and a TOF value.
//
// The continuous indices are as follows:
//  - TOF value: Difference of arrival time between virtual detector 2 (t2) and
//    virtual detector 1 (t1), or "t2 - t1", in ps
//    - The TOF values are always positive and go from 0 to
//      "scanner diameter"/"speed of light"
//  - Plane position of virtual detector 1
//    - Goes from "-axialFOV/2" to "axialFOV/2"
//  - Angular position of virtual detector 1
//    - Goes from 0 to 2pi (radians)
//  - Plane position of virtual detector 2
//    - Same interval as for virtual detector 1
//  - Angular position of virtual detector 2
//    - Same interval as for virtual detector 1
//
// The logical indices of this array sample the continuous indices in a regular
// grid, similar to image voxels. This means that the sampled position is in the
// center of its assigned region.
//
// Note that, for cases without TOF, this space would have duplicates, since an
// LOR representing d2->d1 would be equivalent to an LOR representing d1->d2.
//
// Also, this space is not designed to only hold scatter estimate values, it can
// be used to store any collection of float32 values.
class ScatterSpace : public Histogram
{
public:
	struct ScatterSpaceIndex
	{
		size_t tofBin, planeIndex1, angleIndex1, planeIndex2, angleIndex2;
		ScatterSpaceIndex()
		    : tofBin(0),
		      planeIndex1(0),
		      angleIndex1(0),
		      planeIndex2(0),
		      angleIndex2(0)
		{
		}
		ScatterSpaceIndex(size_t t, size_t p1, size_t a1, size_t p2, size_t a2)
		    : tofBin(t),
		      planeIndex1(p1),
		      angleIndex1(a1),
		      planeIndex2(p2),
		      angleIndex2(a2)
		{
		}
	};

	struct ScatterSpacePosition
	{
		float tof_ps, planePosition1, angle1, planePosition2, angle2;
		ScatterSpacePosition()
		    : tof_ps(0),
		      planePosition1(0),
		      angle1(0),
		      planePosition2(0),
		      angle2(0)
		{
		}
		ScatterSpacePosition(float t, float p1, float a1, float p2, float a2)
		    : tof_ps(t),
		      planePosition1(p1),
		      angle1(a1),
		      planePosition2(p2),
		      angle2(a2)
		{
		}
	};

	static constexpr size_t MinNumAngles = 4ull;
	static constexpr size_t RecommendedNumPlanes = 20ull;
	static constexpr size_t RecommendedNumAngles = 100ull;

	// Read a scatter-space file to create this object
	ScatterSpace(const Scanner& pr_scanner, const std::string& fname);
	// Create a virgin scatter-space
	ScatterSpace(const Scanner& pr_scanner, size_t p_numTOFBins,
	             size_t p_numPlanes, size_t p_numAngles);

	// Memory management
	void allocate();
	bool isMemoryValid() const;

	// I/O
	void readFromFile(const std::string& fname);
	void writeToFile(const std::string& fname) const;

	// Do nearest neighbor
	ScatterSpaceIndex
	    getNearestNeighborIndex(const ScatterSpacePosition& pos) const;
	float getNearestNeighborValue(const ScatterSpacePosition& pos) const;

	// Do linear interpolation
	float getLinearInterpolationValue(const ScatterSpacePosition& pos) const;

	// Helpers
	static void computeCylindricalCoordinates(const Line3D& lor,
	                                          float& planePosition1,
	                                          float& angle1,
	                                          float& planePosition2,
	                                          float& angle2);
	Vector3D getVirtualDetectorPos(size_t planeIndex, size_t angleIndex) const;
	Vector3D cylindricalCoordinatesToPoint(float planePosition, float angle) const;
	Line3D lineFromCylindricalCoordinates(float planePosition1, float angle1,
	                                      float planePosition2,
	                                      float angle2) const;

	// Getters
	ScatterSpacePosition getPosition(const ScatterSpaceIndex& idx) const;
	std::tuple<float, Line3D>
	    getTOFAndLORFromIndex(const ScatterSpaceIndex& idx) const;
	Line3D getLORFromIndex(const ScatterSpaceIndex& idx) const;  // Ignore TOF
	ScatterSpacePosition
	    histogramBinToScatterSpacePosition(const histo_bin_t& histoBinId) const;
	Array5DBase<float>& getData() { return *mp_values; }
	const Array5DBase<float>& getData() const { return *mp_values; }

	// Get the continuous position from the logical index
	float getTOF_ps(size_t TOFBin) const;             // in picoseconds
	float getPlanePosition(size_t planeIndex) const;  // in mm
	float getAngle(size_t angleIndex) const;          // In radians

	// Get the logical index given the continuous position (nearest neighbor)
	size_t getTOFBin(float tof_ps) const;
	size_t getPlaneIndex(float planePosition) const;
	size_t getAngleIndex(float angle) const;

	// Read/write in the array
	float getValue(const ScatterSpaceIndex& idx) const;
	float getValue(size_t tofBin, size_t planeIndex1, size_t angleIndex1,
	               size_t planeIndex2, size_t angleIndex2) const;
	float getValueFlat(size_t flatIdx) const;
	void setValue(const ScatterSpaceIndex& idx, float value);
	void setValue(size_t tofBin, size_t planeIndex1, size_t angleIndex1,
	              size_t planeIndex2, size_t angleIndex2, float value);
	void setValueFlat(size_t flatIdx, float value);
	void incrementValue(const ScatterSpaceIndex& idx, float value);
	void incrementValueAtomic(const ScatterSpaceIndex& idx, float value);
	void incrementValue(size_t tofBin, size_t planeIndex1, size_t angleIndex1,
	                    size_t planeIndex2, size_t angleIndex2, float value);
	void incrementValueFlat(size_t flatIdx, float value);
	void scaleValues(float scale);
	void fill(float value);

	// To avoid d1-d2 vs d2-d1 problems (for no TOF)
	void symmetrizeIfNeeded();

	// Fill non-direct planes from the nearest direct plane
	// estimate(p1, p2) := estimate(avg, avg) where avg = round((p1+p2)/2)
	void fillNonDirectPlanes();

	// Clamp and wrap
	float clampTOF(float tof_ps) const;
	float clampPlanePosition(float planePosition) const;
	static float wrapAngle(float angle);
	size_t wrapAngleIndex(int angleIndex) const;

	// Size of the array
	size_t getNumTOFBins() const;
	size_t getNumPlanes() const;
	size_t getNumAngles() const;
	size_t getSizeTotal() const;
	ScatterSpaceIndex unravelIndex(size_t flatIdx) const;
	size_t getFlatIdx(const ScatterSpaceIndex& idx) const;

	// Scatter-space properties
	float getTOFBinStep_ps() const;
	float getPlaneStep() const;
	float getAngleStep() const;
	float getMinSampledPlanePosition() const;
	float getMaxSampledPlanePosition() const;
	float getMinSampledAngle() const;
	float getMaxSampledAngle() const;

	// From the scanner properties
	float getAxialFOV() const;  // Cylinder length
	float getRadius() const;    // Cylinder radius
	float getDiameter() const;  // Cylinder diameter

	// This the TOF for when a positron annihilation happened on the face of one
	// of the two virtual detectors. One photon was immediately detected and
	// the other had to travel the entirety of the scanner diameter
	float
	    getMaxTOF_ps() const;  // From cylinder diameter and the speed of light

	// Required functions for usage within reconstruction
	size_t count() const override;
	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;
	void clearProjections(float value) override;
	float getProjectionValueFromHistogramBin(
	    histo_bin_t histoBinId) const override;

	// Should not be available to use for this class
	det_id_t getDetector1(bin_t id) const override;
	det_id_t getDetector2(bin_t id) const override;
	det_pair_t getDetectorPair(bin_t id) const override;
	bool hasArbitraryLORs() const override;
	Line3D getArbitraryLOR(bin_t id) const override;
	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                        int idxSubset) const override;

	// For registering the plugin
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const io::OptionsResult& options);
	static plugin::OptionsListPerPlugin getOptions();

private:
	void initStepSizes();

	// YN: Maybe we should make it so that the TOF bins are not centered in
	//  their sample but rather centered towards the start (center of the LOR).
	//  This way, we always sample the central TOF bin.
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
	std::unique_ptr<Array5DOwned<float>> mp_values;
};

}  // namespace yrt
