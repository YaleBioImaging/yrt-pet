/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/parallel_hashmap/phmap.h"
#include "yrt-pet/datastruct/projection/Histogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"

namespace yrt
{
class SparseHistogram final : public Histogram
{
public:
	explicit SparseHistogram(const Scanner& pr_scanner);
	SparseHistogram(const Scanner& pr_scanner, const std::string& filename);
	SparseHistogram(const Scanner& pr_scanner,
	                const ProjectionData& pr_projData);

	void allocate(size_t numBins);

	// Insertion
	template <bool IgnoreZeros = true, bool PrintProgress = true>
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
	           const io::OptionsResult& options);
	static plugin::OptionsListPerPlugin getOptions();

private:
	struct det_pair_hash
	{
		size_t operator()(const det_pair_t& pair) const
		{
			return (static_cast<size_t>(pair.d1) << 4) ^
			       static_cast<size_t>(pair.d2);
		}
	};

	static det_pair_t SwapDetectorPairIfNeeded(det_pair_t detPair);

	phmap::flat_hash_map<det_pair_t, bin_t, det_pair_hash> m_detectorMap;
	std::vector<det_pair_t> m_detPairs;
	std::vector<float> m_projValues;
};
}  // namespace yrt
