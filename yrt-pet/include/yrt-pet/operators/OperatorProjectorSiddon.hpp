/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/MultiRayGenerator.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"

#include "omp.h"

namespace yrt
{
class Image;

class OperatorProjectorSiddon : public OperatorProjector
{
public:
	explicit OperatorProjectorSiddon(
	    const OperatorProjectorParams& pr_projParams,
	    const std::vector<Constraint*>& pr_constraints = {});

	float forwardProjection(const Image* img,
	                        const ProjectionProperties& projectionProperties,
	                        int tid = 0) const override;

	void backProjection(Image* img,
	                    const ProjectionProperties& projectionProperties,
	                    float projValue, int tid = 0) const override;

	// Projection
	float forwardProjection(const Image* img, const Line3D& lor,
	                        const Vector3D& n1, const Vector3D& n2,
	                        const TimeOfFlightHelper* tofHelper = nullptr,
	                        float tofValue = 0.f) const;
	void backProjection(Image* img, const Line3D& lor, const Vector3D& n1,
	                    const Vector3D& n2, float projValue,
	                    const TimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.f) const;

	// Without Multi-ray siddon
	static float
	    singleForwardProjection(const Image* img, const Line3D& lor,
	                            const TimeOfFlightHelper* tofHelper = nullptr,
	                            float tofValue = 0.f);
	static void
	    singleBackProjection(Image* img, const Line3D& lor, float projValue,
	                         const TimeOfFlightHelper* tofHelper = nullptr,
	                         float tofValue = 0.f);


	template <bool IS_FWD, bool FLAG_INCR, bool FLAG_TOF>
	static void project_helper(Image* img, const Line3D& lor, float& value,
	                           const TimeOfFlightHelper* tofHelper = nullptr,
	                           float tofValue = 0.f);

	int getNumRays() const;
	void setNumRays(int n);

	std::set<ProjectionPropertyType>
	    getProjectionPropertyTypes() const override;

private:
	int m_numRays;
	std::unique_ptr<std::vector<MultiRayGenerator>> mp_lineGen;
};
}  // namespace yrt
