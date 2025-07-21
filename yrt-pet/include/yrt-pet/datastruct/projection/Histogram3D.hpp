/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/parallel_hashmap/phmap.h"
#include "yrt-pet/datastruct/projection/Histogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <array>
#include <vector>

namespace yrt
{

struct HashDetPair
{
	int operator()(det_pair_t p) const
	{
		const auto hash1 = std::hash<det_id_t>{}(p.d1);
		const auto hash2 = std::hash<det_id_t>{}(p.d2);
		return hash1 ^ hash2;
	}
};
struct EqualDetPair
{
	int operator()(det_pair_t p1, det_pair_t p2) const
	{
		return p1.d1 == p2.d1 && p1.d2 == p2.d2;
	}
};

typedef uint32_t coord_t;
typedef std::array<coord_t, 3> DetCoordinates;      // r, phi, z
typedef std::array<coord_t, 2> DetRingCoordinates;  // r, phi

class Histogram3D : public Histogram
{
public:
	Array3DBase<float>& getData() { return *mp_data; }
	const Array3DBase<float>& getData() const { return *mp_data; }
	virtual void writeToFile(const std::string& filename) const;
	~Histogram3D() override = 0;

	// binId
	bin_t getBinIdFromCoords(coord_t r, coord_t phi, coord_t z_bin) const;
	void getCoordsFromBinId(bin_t binId, coord_t& r, coord_t& phi,
	                        coord_t& z_bin) const;

	// Where all the magic happens
	void getDetPairFromCoords(coord_t r, coord_t phi, coord_t z_bin,
	                          det_id_t& d1, det_id_t& d2) const;
	det_pair_t getDetPairFromBinId(bin_t binId) const;  // Uses latter
	void getCoordsFromDetPair(det_id_t d1, det_id_t d2, coord_t& r,
	                          coord_t& phi, coord_t& z_bin) const;
	bin_t getBinIdFromDetPair(det_id_t d1, det_id_t d2) const;  // Uses latter
	histo_bin_t getHistogramBin(bin_t bin) const override;

	// Functions needed for ProjectionData object and used by the operators
	float getProjectionValue(bin_t binId) const override;
	size_t count() const override;
	void setProjectionValue(bin_t binId, float val) override;
	virtual void incrementProjection(bin_t binId, float incVal);
	det_id_t getDetector1(bin_t id) const override;
	det_id_t getDetector2(bin_t id) const override;
	det_pair_t getDetectorPair(bin_t id) const override;
	void clearProjections();
	void clearProjections(float value) override;
	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                        int idxSubset) const override;

	float getProjectionValueFromHistogramBin(
	    histo_bin_t histoBinId) const override;
	void getZ1Z2(coord_t z_bin, coord_t& z1, coord_t& z2) const;

	bool isMemoryValid() const;

protected:
	explicit Histogram3D(const Scanner& pr_scanner);
	void getDetPairInSameRing(coord_t r_ring, coord_t phi, det_id_t& d1,
	                          det_id_t& d2) const;
	void getCoordsInSameRing(det_id_t d1_ring, det_id_t d2_ring,
	                         coord_t& r_ring, coord_t& phi) const;
	bool getCoordsInSameRing_safe(det_id_t d1_ring, det_id_t d2_ring,
	                              coord_t& r_ring, coord_t& phi) const;
	// Sets up the LUT for reverse compute
	void setupHistogram();

public:
	size_t numR, numPhi, numZBin;
	// size_t numR, numPhi, numZBin
	size_t histoSize;
	// domains:
	// 	r: {0, 1, ..., nr/2-mad-2}
	// 	phi: {0, 1, ..., nr-1}
	// 	z: {0, 1, ..., (mrd+1)*nr-mrd*(mrd+1)/2 - 1}
	// 	where:
	// 		nr: number of detectors in a ring
	// 		mad: minimum angle difference
	// 		mrd: maximum ring difference

protected:
	std::unique_ptr<Array3DBase<float>> mp_data;
	::phmap::flat_hash_map<det_pair_t, DetRingCoordinates, HashDetPair,
	                       EqualDetPair>
	    m_ringMap;
	size_t m_rCut;
	size_t m_numDOIPoss;  // Number of DOI combinations (ex: 2 doi -> 4 lor
	// possibilities)
	size_t m_numZBinDiff;  // Number of z_bins that have z1 < z2
};

class Histogram3DAlias : public Histogram3D
{
public:
	explicit Histogram3DAlias(const Scanner& pr_scanner);
	void bind(Array3DBase<float>& pr_data);
};

class Histogram3DOwned : public Histogram3D
{
public:
	explicit Histogram3DOwned(const Scanner& pr_scanner);
	Histogram3DOwned(const Scanner& pr_scanner, const std::string& filename);
	void allocate();
	void readFromFile(const std::string& filename);

	// For registering the plugin
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const io::OptionsResult& options);
	static plugin::OptionsListPerPlugin getOptions();
};

}  // namespace yrt
