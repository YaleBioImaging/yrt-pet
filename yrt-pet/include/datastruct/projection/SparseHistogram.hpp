/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/Histogram.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/PairHashMap.hpp"

class SparseHistogram final : public Histogram
{
public:
	explicit SparseHistogram(const Scanner& pr_scanner);
	SparseHistogram(const Scanner& pr_scanner, const std::string& filename);
	SparseHistogram(const Scanner& pr_scanner,
	                const ProjectionData& pr_projData);

	void allocate(size_t numBins);

	// Insertion
	template <bool IgnoreZeros = true>
	void accumulate(const ProjectionData& projData);
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

	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const Plugin::OptionsResult& pluginOptions);
	static Plugin::OptionsListPerPlugin getOptions();

private:
	PairHashMap<float> m_map;
};
