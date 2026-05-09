/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/projection/Histogram.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Array.hpp"

namespace yrt
{

class RandomsHistogram : public Histogram
{
public:
	explicit RandomsHistogram(const Scanner& pr_scanner, float p_tau = 0.f);
	RandomsHistogram(const Scanner& pr_scanner, const std::string& filename,
	                 float p_tau = 0.f);

	void populateFromListMode(const ListMode& listMode);

	// Histogram interface
	float getProjectionValueFromHistogramBin(
	    histo_bin_t histoBinId) const override;

	// ProjectionData pure virtuals
	size_t count() const override;
	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;
	det_id_t getDetector1(bin_t id) const override;
	det_id_t getDetector2(bin_t id) const override;
	std::unique_ptr<BinIterator>
	    getBinIter(int numSubsets, int idxSubset) const override;

	// I/O
	void writeToFile(const std::string& filename) const;
	void readFromFile(const std::string& filename);

	// Plugin
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const io::OptionsResult& options);
	static plugin::OptionsListPerPlugin getOptions();

private:
	std::unique_ptr<Array1DOwned<float>> mp_singles;
	float m_timeWindow;
};

}  // namespace yrt
