/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/projection/Histogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <vector>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace yrt
{
void py_setup_sinogram3d(pybind11::module& m);
}
#endif

namespace yrt
{

// coord_t is also defined in Histogram3D.hpp; redeclared here to keep
// Sinogram3D.hpp independent of Histogram3D.hpp.
using coord_t = uint32_t;

// Parameters fully describing a Sinogram3D's geometry.
// All derived fields (numZ, numSegSide, phiStep, rStep, rMin) are populated
// by validate().
struct SinogramParams
{
	size_t numPhi;  // view-angle bins covering [0, π)
	size_t numR;    // radial-offset bins covering (−R, R)
	size_t numTOF;  // TOF bins; 1 for non-TOF data
	size_t span;    // axial compression: rings diffs grouped in [s·span …
	                // (s+1)·span−1]
	size_t maxRingDiff;  // maximum ring difference included (≤
	                     // scanner.maxRingDiff)

	// Derived — populated by validate()
	size_t numZ;        // total axial bins across all segments
	size_t numSegSide;  // number of non-negative segments (incl. s=0)
	float phiStep;      // π / numPhi  (rad)
	float rStep;        // 2R / numR   (mm)
	float rMin;         // −R          (mm)

	// Compute derived fields and assert inputs are coherent with scanner.
	void validate(const Scanner& scanner);

	// Convenience constructor.  maxRingDiff=0 uses scanner.maxRingDiff.
	static SinogramParams fromScanner(const Scanner& scanner, size_t numPhi,
	                                  size_t numR, size_t span,
	                                  size_t maxRingDiff = 0,
	                                  size_t numTOF = 1);

	// Read SinogramParams from the header of a Sinogram3D binary file.
	static SinogramParams fromFile(const std::string& filename,
	                               const Scanner& scanner);
};

// Abstract base for 3D sinograms in physical (phi, r, segment) coordinates.
// Storage layout: [numTOF, numZ, numPhi, numR] (C row-major).
// Segment ordering in the z axis: 0, +1, −1, +2, −2, …
class Sinogram3D : public Histogram
{
public:
	static constexpr bool IsListMode() { return false; }
	// Distinct magic number — different from the plain Array MAGIC_NUMBER so
	// that the file reader can detect the extra sinogram-params fields.
	static constexpr int SINOGRAM_MAGIC = 732174001;

	Array4DBase<float>& getData() { return *mp_data; }
	const Array4DBase<float>& getData() const { return *mp_data; }

	virtual void writeToFile(const std::string& filename) const;
	~Sinogram3D() override = 0;

	// ---- 4D coordinate <-> flat bin ----------------------------------------
	bin_t getBinId(size_t tof, size_t z, size_t phi, size_t r) const;
	void getCoords(bin_t id, size_t& tof, size_t& z, size_t& phi,
	               size_t& r) const;

	// ---- Axial organisation ------------------------------------------------
	// Signed segment index for global z index iZ.
	int getSegment(size_t iZ) const;
	// Plane within the segment (0-indexed) for global z index iZ.
	size_t getPlaneInSegment(size_t iZ) const;
	// Inverse: global z index from (signed segment, plane).
	size_t getZFromSegmentPlane(int segment, size_t plane) const;
	// Representative ring pair (z1, z2) for global z index iZ.
	// For segment s>0: z2−z1 = s·span.  For s<0: z1−z2 = |s|·span.
	void getRingPair(size_t iZ, coord_t& z1, coord_t& z2) const;

	// ---- Physical coordinate values (bin centres) --------------------------
	float getPhi(size_t iPhi) const;  // rad, in [0, π)
	float getR(size_t iR) const;      // mm, signed
	float getZMid(size_t iZ) const;   // mm, axial midpoint of the ring pair

	// ---- Sinogram <-> detector pair ----------------------------------------
	// Exact reverse: uses scanner LUT positions + cross-product formula.
	bin_t getBinIdFromDetPair(det_id_t d1, det_id_t d2) const;

	// ---- ProjectionData interface ------------------------------------------
	// getDetector1/2: approximate forward map using nominal scanner radius.
	// doi is set to 0; TOF bin is ignored (central LOR returned).
	det_id_t getDetector1(bin_t id) const override;
	det_id_t getDetector2(bin_t id) const override;
	det_pair_t getDetectorPair(bin_t id) const override;

	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float v) override;
	size_t count() const override;
	void clearProjections(float value) override;
	void incrementProjection(bin_t id, float incVal);

	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                        int idxSubset) const override;

	// ---- Histogram interface -----------------------------------------------
	// Returns the detector pair so cross-format lookups use det_pair_t.
	histo_bin_t getHistogramBin(bin_t bin) const override;
	// Handles det_pair_t (exact lookup) and bin_t (direct index for
	// sino-to-sino case).
	float getProjectionValueFromHistogramBin(histo_bin_t h) const override;

	bool isMemoryValid() const;
	const SinogramParams& getParams() const { return m_params; }

	// Public dimensions — analogous to Histogram3D's numR / numPhi / numZBin.
	size_t numTOF, numZ, numPhi, numR;
	size_t sinoSize;  // numTOF * numZ * numPhi * numR

protected:
	explicit Sinogram3D(const Scanner& scanner, const SinogramParams& params);

	void setupSegments();

	// Approximate bin → detector pair using nominal scanner radius R and doi=0.
	void getDetPairApprox(size_t iPhi, size_t iR, coord_t z1, coord_t z2,
	                      det_id_t& d1, det_id_t& d2) const;

	SinogramParams m_params;
	std::unique_ptr<Array4DBase<float>> mp_data;

	struct SegmentInfo
	{
		int segment;     // signed segment index
		size_t offset;   // starting global-z index for this segment
		size_t nPlanes;  // number of planes in this segment
	};
	// Ordered: s=0, s=+1, s=−1, s=+2, s=−2, …
	std::vector<SegmentInfo> m_segments;
};

class Sinogram3DOwned : public Sinogram3D
{
public:
	Sinogram3DOwned(const Scanner& scanner, const SinogramParams& params);
	// Construct and immediately load from file (params must match file).
	Sinogram3DOwned(const Scanner& scanner, const SinogramParams& params,
	                const std::string& filename);

	void allocate();
	void readFromFile(const std::string& filename);

	// Plugin factory — reads params from the file header, then loads data.
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const io::OptionsResult& options);
	static plugin::OptionsListPerPlugin getOptions();
};

class Sinogram3DAlias : public Sinogram3D
{
public:
	Sinogram3DAlias(const Scanner& scanner, const SinogramParams& params);
	void bind(Array4DBase<float>& data);
};

}  // namespace yrt
