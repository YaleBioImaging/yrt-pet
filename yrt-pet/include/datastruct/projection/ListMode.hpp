/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "datastruct/projection/ProjectionData.hpp"

#include <memory>

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

	// TODO NOW:
	//  Unify the code that computes the frames based on the LORMotion
	//   (Use the "ListMode" class)
	//  Make the timeAveraging of the sens image based on LORMotion

protected:
	explicit ListMode(const Scanner& pr_scanner);
};
