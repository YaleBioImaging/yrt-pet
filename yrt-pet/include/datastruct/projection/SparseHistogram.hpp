/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/Histogram.hpp"
#include "datastruct/scanner/Scanner.hpp"

#include <unordered_map>

class SparseHistogram final : public Histogram
{
public:
	explicit SparseHistogram(const Scanner& pr_scanner);
	SparseHistogram(const Scanner& pr_scanner, const std::string& filename);
	SparseHistogram(const Scanner& pr_scanner,
	                const ProjectionData& pr_projData,
	                const BinIterator* pp_binIter = nullptr);

	void allocate(size_t numBins);

	// Insertion
	template <bool IgnoreZeros = false>
	void accumulate(const ProjectionData& projData,
	                const BinIterator* binIter = nullptr);
	void accumulate(det_pair_t detPair, float projValue);

	// Getters
	float getProjectionValueFromDetPair(det_pair_t detPair) const;

	// Mandatory functions
	size_t count() const override;
	det_id_t getDetector1(bin_t id) const override;
	det_id_t getDetector2(bin_t id) const override;
	det_pair_t getDetectorPair(bin_t id) const override;
	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                        int idxSubset) const override;
	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;
	float getProjectionValueFromHistogramBin(
	    histo_bin_t histoBinId) const override;

	void writeToFile(const std::string& filename) const;
	void readFromFile(const std::string& filename);

	float* getProjectionValuesBuffer();
	det_pair_t* getDetectorPairBuffer();
	const float* getProjectionValuesBuffer() const;
	const det_pair_t* getDetectorPairBuffer() const;

	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const Plugin::OptionsResult& pluginOptions);
	static Plugin::OptionsListPerPlugin getOptions();

private:
	// Comparator for std::unordered_map
	struct det_pair_hash
	{
		size_t operator()(const det_pair_t& pair) const
		{
			return (static_cast<size_t>(pair.d1) << 32) |
			       static_cast<size_t>(pair.d2);
		}
	};
	struct det_pair_equal
	{
		bool operator()(const det_pair_t& pair1, const det_pair_t& pair2) const
		{
			return pair1.d1 == pair2.d1 && pair1.d2 == pair2.d2;
		}
	};

	static det_pair_t SwapDetectorPairIfNeeded(det_pair_t detPair);

	std::unordered_map<det_pair_t, bin_t, det_pair_hash, det_pair_equal>
	    m_detectorMap;
	std::vector<det_pair_t> m_detPairs;
	std::vector<float> m_projValues;
};
