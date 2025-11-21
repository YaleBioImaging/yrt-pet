/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"

#include <vector>

namespace yrt
{

typedef std::vector<int> PositionList;

class Image;
class ProjectionData;

class OperatorProjectorDD : public OperatorProjector
{
public:
	explicit OperatorProjectorDD(
	    const OperatorProjectorParams& pr_projParams,
	    const std::vector<Constraint*>& pr_constraints = {});

	float forwardProjection(
		const Image* in_image, const Line3D& lor, const Vector3D& n1,
		const Vector3D& n2, int tid,
		frame_t dynamicFrame = 0, const TimeOfFlightHelper* tofHelper = nullptr,
		float tofValue = 0.0f, const ProjectionPsfManager* psfManager = nullptr) const;

	void backProjection(Image* in_image, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float proj_value, int tid,
	                    frame_t dynamicFrame = 0,
	                    const TimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.0f, const ProjectionPsfManager* psfManager = nullptr) const;

	float forwardProjection(const Image* img,
	                        const ProjectionProperties& projectionProperties,
	                        size_t pos = 0) const override;

	void backProjection(Image* img,
	                    const ProjectionProperties& projectionProperties,
	                    float projValue, size_t pos = 0) const override;

	static float get_overlap_safe(float p0, float p1, float d0, float d1);
	static float get_overlap_safe(float p0, float p1, float d0, float d1,
	                              const ProjectionPsfManager* psfManager,
	                              const float* psfKernel);
	static float get_overlap(float p0, float p1, float d0, float d1,
	                         const ProjectionPsfManager* psfManager = nullptr,
	                         const float* psfKernel = nullptr);

	std::set<ProjectionPropertyType>
	    getProjectionPropertyTypes() const override;

private:
	template <bool IS_FWD, bool FLAG_TOF>
	void dd_project_ref(Image* in_image, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float& proj_value,
	                    OperatorProjectorUpdater& updater,
	                    frame_t dynamicFrame = 0,
	                    const TimeOfFlightHelper* tofHelper = nullptr, float tofValue = 0.f,
	                    const ProjectionPsfManager* psfManager = nullptr) const;
};

}  // namespace yrt
