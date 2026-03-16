/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/Projector.hpp"

#include <vector>

namespace yrt
{

typedef std::vector<int> PositionList;

class Image;
class ProjectionData;

class ProjectorDD : public Projector
{
public:
	explicit ProjectorDD(
	    const ProjectorParams& pr_projParams);

	float forwardProjection(
	    const Image* in_image, const Line3D& lor, const Vector3D& n1,
	    const Vector3D& n2, int tid, frame_t dynamicFrame = 0,
	    const TimeOfFlightHelper* tofHelper = nullptr, float tofValue = 0.0f,
	    const ProjectionPsfManager* psfManager = nullptr) const;

	void backProjection(Image* in_image, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float proj_value, int tid,
	                    frame_t dynamicFrame = 0,
	                    const TimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.0f,
	                    const ProjectionPsfManager* psfManager = nullptr) const;

	float forwardProjection(const Image* image,
	                        const ProjectionPropertyManager& propManager,
	                        PropertyUnit* projectionProperties,
	                        size_t pos) const override;
	void backProjection(Image* image,
	                    const ProjectionPropertyManager& propManager,
	                    PropertyUnit* projectionProperties, size_t pos,
	                    float projValue) const override;

	static float getOverlap_safe(float p0, float p1, float d0, float d1);
	static float getOverlap_safe(float p0, float p1, float d0, float d1,
	                              const ProjectionPsfManager* psfManager,
	                              const float* psfKernel);
	static float getOverlap(float p0, float p1, float d0, float d1,
	                         const ProjectionPsfManager* psfManager = nullptr,
	                         const float* psfKernel = nullptr);

	static std::set<ProjectionPropertyType> getNeededProperties();

protected:
	std::set<ProjectionPropertyType>
	    getProjectionPropertyTypesInternal() const override;

private:
	template <bool IS_FWD, bool FLAG_TOF, bool USE_UPDATER>
	void dd_project_ref(Image* in_image, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float& projValue,
	                    ProjectorUpdater* updater, frame_t dynamicFrame = 0,
	                    int tid = 0,
	                    const TimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.f,
	                    const ProjectionPsfManager* psfManager = nullptr) const;
};

}  // namespace yrt
