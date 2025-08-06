/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"

#include <memory>

namespace yrt
{

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

	void addLORMotion(const std::string& lorMotion_fname);
	virtual void addLORMotion(const std::shared_ptr<LORMotion>& pp_lorMotion);
	virtual void addDynamicFraming(const std::vector<timestamp_t>& dynamicFramingVector);
	virtual void addDynamicFraming(const std::shared_ptr<DynamicFraming>& pp_dynamicFraming);

	bool hasMotion() const override;
	frame_t getDynamicFrame(bin_t id) const override;
	frame_t getMotionFrame(bin_t id) const override;
	size_t getNumDynamicFrames() const override;
	size_t getNumMotionFrames() const override;
	transform_t getTransformOfMotionFrame(frame_t frame) const override;
	float getDurationOfMotionFrame(frame_t frame) const override;

protected:
	explicit ListMode(const Scanner& pr_scanner);

	std::shared_ptr<LORMotion> mp_lorMotion;
	std::unique_ptr<Array1D<frame_t>> mp_motionFrames;
	std::shared_ptr<DynamicFraming> mp_dynamicFraming;
	std::unique_ptr<Array1D<frame_t>> mp_dynamicFrames;
};
}
