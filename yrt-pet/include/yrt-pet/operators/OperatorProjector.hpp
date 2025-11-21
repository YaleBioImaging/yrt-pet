/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/Operator.hpp"
#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdater.hpp"
#include "yrt-pet/operators/ProjectionPsfManager.hpp"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/utils/Types.hpp"

namespace yrt
{
class BinIterator;
class Image;
class Scanner;
class ProjectionData;
class Histogram;


class OperatorProjector : public OperatorProjectorBase
{
public:
	explicit OperatorProjector(
	    const OperatorProjectorParams& pr_projParams,
	    const std::vector<Constraint*>& pr_constraints = {});

	// Virtual functions
	virtual float
	    forwardProjection(const Image* image,
	                      const ProjectionProperties& projectionProperties,
	                      size_t pos = 0) const = 0;
	virtual void
	    backProjection(Image* image,
	                   const ProjectionProperties& projectionProperties,
	                   float projValue, size_t pos = 0) const = 0;

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	OperatorProjectorUpdater* getUpdater();
	void setupUpdater(const OperatorProjectorParams& p_projParams);
	void setUpdater(std::unique_ptr<OperatorProjectorUpdater> pp_updater);

	const TimeOfFlightHelper* getTOFHelper() const;
	const ProjectionPsfManager* getProjectionPsfManager() const;

protected:

	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	void setupProjPsfManager(const std::string& projPsf_fname);

	// Time of flight
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManager> mp_projPsfManager;

	// Number of threads
	int m_numThreads;

	// Updater for forward and back-projection
	std::unique_ptr<OperatorProjectorUpdater> mp_updater;
};
}  // namespace yrt
