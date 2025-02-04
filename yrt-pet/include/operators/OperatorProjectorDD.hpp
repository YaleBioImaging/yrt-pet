/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/OperatorProjector.hpp"

#include <vector>

typedef std::vector<int> PositionList;

class Image;
class ProjectionData;

class OperatorProjectorDD : public OperatorProjector
{
public:
	explicit OperatorProjectorDD(const Scanner& pr_scanner,
	                             float tofWidth_ps = 0.0f, int tofNumStd = -1,
	                             const std::string& psfProjFilename = "");

	explicit OperatorProjectorDD(const OperatorProjectorParams& p_projParams);

	float forwardProjection(
	    const Image* in_image, const Line3D& lor, const Vector3D& n1,
	    const Vector3D& n2, const TimeOfFlightHelper* tofHelper = nullptr,
	    float tofValue = 0.0f,
	    const ProjectionPsfManager* psfManager = nullptr) const;

	void backProjection(Image* in_image, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float proj_value,
	                    const TimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.0f,
	                    const ProjectionPsfManager* psfManager = nullptr) const;

	float forwardProjection(
	    const Image* img,
	    const ProjectionProperties& projectionProperties) const override;

	void backProjection(Image* img,
	                    const ProjectionProperties& projectionProperties,
	                    float projValue) const override;

	static float get_overlap_safe(float p0, float p1, float d0, float d1);
	static float get_overlap_safe(float p0, float p1, float d0, float d1,
	                              const ProjectionPsfManager* psfManager,
	                              const float* psfKernel);
	static float get_overlap(float p0, float p1, float d0, float d1,
	                         const ProjectionPsfManager* psfManager = nullptr,
	                         const float* psfKernel = nullptr);


private:
	template <bool IS_FWD, bool FLAG_TOF>
	void dd_project_ref(Image* in_image, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float& proj_value,
	                    const TimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.f,
	                    const ProjectionPsfManager* psfManager = nullptr) const;
};
