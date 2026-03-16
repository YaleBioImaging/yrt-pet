/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "Projector.hpp"
#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/Operator.hpp"
#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/operators/ProjectionPsfManager.hpp"
#include "yrt-pet/operators/Projector.hpp"
#include "yrt-pet/operators/ProjectorUpdater.hpp"
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
	    const ProjectorParams& pr_projParams,
	    const BinIterator* pp_binIter = nullptr,
	    const std::vector<Constraint*>& pr_constraints = {});

	void initBinLoader(
	    const std::vector<Constraint*>& pr_constraints);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void addTOF(float tofWidth_ps, int tofNumStd = -1);
	void addProjPSF(const std::string& projPsf_fname);
	void setUpdater(std::unique_ptr<ProjectorUpdater> pp_updater);
	std::set<ProjectionPropertyType> getProjectionPropertyTypes() const;
	PropertyUnit* getProjectionProperties() const;
	PropertyUnit* getConstraintVariables() const;

	const TimeOfFlightHelper* getTOFHelper() const;
	const ProjectionPsfManager* getProjectionPsfManager() const;
	ProjectorUpdater* getUpdater();

	const BinLoader* getBinLoader() const;
	unsigned int getElementSize() const;

protected:
	void setupBinLoader(
	    const std::vector<Constraint*>& pr_constraints = {});
	void allocateBuffers();

	// The actual projector used in applyA and applyAH
	std::unique_ptr<Projector> mp_projector;

	// To hold temporary projection-space data (one row per thread)
	std::unique_ptr<BinLoader> mp_binLoader;
};
}  // namespace yrt
