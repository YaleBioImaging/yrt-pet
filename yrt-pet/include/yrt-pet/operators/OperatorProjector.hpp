/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
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

	explicit OperatorProjector(const Scanner& pr_scanner,
	                           float tofWidth_ps = 0.0f, int tofNumStd = -1,
	                           const std::string& projPsf_fname = "",
	                           ProjectorUpdaterType projectorUpdaterType =
	                               OperatorProjectorBase::DEFAULT3D);

	explicit OperatorProjector(const OperatorProjectorParams& p_projParams,
	                           ProjectorUpdaterType projectorUpdaterType =
	                               OperatorProjectorBase::DEFAULT3D);

	// Virtual functions
	virtual float forwardProjection(
	    const Image* image,
	    const ProjectionProperties& projectionProperties, int tid) const = 0;
	virtual void
	    backProjection(Image* image,
	                   const ProjectionProperties& projectionProperties,
	                   float projValue, int tid) const = 0;

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void addTOF(float tofWidth_ps, int tofNumStd = -1);
	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	void setupProjPsfManager(const std::string& projPsf_fname);
	OperatorProjectorUpdater* getUpdater();
	void setUpdaterType(OperatorProjectorBase::ProjectorUpdaterType p_updaterType);
	void setUpdater(std::unique_ptr<OperatorProjectorUpdater> pp_updater);

	const TimeOfFlightHelper* getTOFHelper() const;
	const ProjectionPsfManager* getProjectionPsfManager() const;

protected:
	// Time of flight
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManager> mp_projPsfManager;

	// Updater for forward and back-projection
	std::unique_ptr<OperatorProjectorUpdater> mp_updater;
};
}  // namespace yrt
