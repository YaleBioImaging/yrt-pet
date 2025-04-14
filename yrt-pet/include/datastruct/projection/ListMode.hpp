/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "datastruct/projection/ProjectionData.hpp"

#include <memory>

class Histogram;

class ListMode : public ProjectionData
{
public:
	~ListMode() override = default;

	static constexpr bool IsListMode() { return true; }

	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;

	// Pre-implemented functions
	timestamp_t getScanDuration() const override;
	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                        int idxSubset) const override;

	void setSensitivityHistogram(const Histogram* pp_sensitivity);
	const Histogram* getSensitivityHistogram() const;
protected:
	explicit ListMode(const Scanner& pr_scanner);
	const Histogram* mp_sensitivity;
};
