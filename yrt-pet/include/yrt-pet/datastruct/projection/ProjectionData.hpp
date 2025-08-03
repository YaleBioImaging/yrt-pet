/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/operators/Variable.hpp"
#include "yrt-pet/utils/Types.hpp"

#include <functional>
#include <memory>
#include <set>

namespace yrt
{
class BinIterator;
enum class ProjectionPropertiesVariable;
enum class ConstraintVariable;
using ConstraintParams = std::unordered_map<ConstraintVariable, size_t>;

struct ProjectionProperties
{
	bin_t bin;
	Line3D lor;
	float tofValue;
	Vector3D det1Orient;
	Vector3D det2Orient;
};

class ProjectionData : public Variable
{
public:
	~ProjectionData() override = default;
	const Scanner& getScanner() const;

	// Mandatory methods
	virtual size_t count() const = 0;
	virtual float getProjectionValue(bin_t id) const = 0;
	virtual void setProjectionValue(bin_t id, float val) = 0;
	virtual det_id_t getDetector1(bin_t id) const = 0;
	virtual det_id_t getDetector2(bin_t id) const = 0;
	virtual det_pair_t getDetectorPair(bin_t id) const;
	virtual histo_bin_t getHistogramBin(bin_t bin) const;
	virtual std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                                int idxSubset) const = 0;

	// Optional methods
	virtual timestamp_t getTimestamp(bin_t id) const;
	virtual frame_t getFrame(bin_t id) const;
	virtual bool isUniform() const;
	virtual bool hasRandomsEstimates() const;
	virtual float getRandomsEstimate(bin_t id) const;
	// Time-of-flight
	virtual bool hasTOF() const;
	virtual float getTOFValue(bin_t id) const;
	// For motion correction
	virtual bool hasMotion() const;
	virtual size_t getNumFrames() const;
	virtual transform_t getTransformOfFrame(frame_t frame) const;
	virtual float getDurationOfFrame(frame_t frame) const;
	virtual timestamp_t getScanDuration() const;
	// Special case when the LOR is not defined directly from the scanner's LUT
	virtual bool hasArbitraryLORs() const;
	virtual Line3D getArbitraryLOR(bin_t id) const;

	// Helper functions
	virtual ProjectionProperties getProjectionProperties(bin_t bin) const;

	Line3D getLOR(bin_t bin) const;
	virtual void clearProjections(float value);
	virtual void divideMeasurements(const ProjectionData* measurements,
	                                const BinIterator* binIter);

	virtual void operationOnEachBin(const std::function<float(bin_t)>& func);
	// Note: The function given as argument should be able to be called in
	// parallel without race conditions for different bins.
	// In other words, two different bins shouldn't point
	// to the same memory location.
	virtual void
	    operationOnEachBinParallel(const std::function<float(bin_t)>& func);

protected:
	explicit ProjectionData(const Scanner& pr_scanner);

	const Scanner& mr_scanner;
};

}  // namespace yrt
