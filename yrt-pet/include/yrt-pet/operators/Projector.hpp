/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/ProjectionPsfManager.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"
#include "yrt-pet/operators/ProjectorUpdater.hpp"
#include "yrt-pet/operators/TimeOfFlight.hpp"

#include <memory>

namespace yrt
{

class Projector
{
public:
	virtual ~Projector() = default;

	explicit Projector(const ProjectorParams& pr_projParams);
	static std::unique_ptr<Projector> create(const ProjectorParams& params);

	// Virtual functions
	virtual float forwardProjection(
	    const Image* image, const ProjectionPropertyManager& propManager,
	    PropertyUnit* projectionProperties, size_t pos) const = 0;
	virtual void backProjection(Image* image,
	                            const ProjectionPropertyManager& propManager,
	                            PropertyUnit* projectionProperties, size_t pos,
	                            float projValue) const = 0;

	void addTOF(float tofWidth_ps, int tofNumStd = -1);
	void addProjPSF(const std::string& projPsf_fname);
	void setUpdater(std::unique_ptr<ProjectorUpdater> pp_updater);

	const Scanner& getScanner() const;
	std::set<ProjectionPropertyType> getProjectionPropertyTypes() const;
	const TimeOfFlightHelper* getTOFHelper() const;
	const ProjectionPsfManager* getProjectionPsfManager() const;
	ProjectorUpdater* getUpdater();
	bool hasUpdater() const;
	UpdaterType getUpdaterType() const;

protected:
	virtual std::set<ProjectionPropertyType>
	    getProjectionPropertyTypesInternal() const;

	const Scanner& mr_scanner;

	// Time of flight
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManager> mp_projPsfManager;

	// Updater for forward and back-projection
	UpdaterType m_updaterType;
	std::unique_ptr<ProjectorUpdater> mp_updater;

	// Properties to be gathered by a PropStruct to use this projector
	std::set<ProjectionPropertyType> projPropertyTypes;

private:
	void setupUpdater(const ProjectorParams& pr_projParams);
	void setupProjPropertyTypes(const ProjectorParams& pr_projParams);
};

}  // namespace yrt
