/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/ProjectionData.hpp"
#include "operators/Operator.hpp"
#include "operators/ProjectionPsfManager.hpp"
#include "operators/TimeOfFlight.hpp"
#include "utils/Types.hpp"

class BinIterator;
class Image;
class Scanner;
class ProjectionData;
class Histogram;

class OperatorProjectorParams
{
public:
	OperatorProjectorParams(const BinIterator* pp_binIter,
	                        const Scanner& pr_scanner,
	                        float p_tofWidth_ps = 0.f, int p_tofNumStd = 0,
	                        std::string p_psfProjFilename = "",
	                        int p_num_rays = 1);

	const BinIterator* binIter;
	const Scanner& scanner;

	// Time of Flight
	float tofWidth_ps;
	int tofNumStd;

	// Projection-domain PSF
	std::string psfProjFilename;

	// Multi-ray siddon only
	int numRays;
};

// Device-agnostic virtual class
class OperatorProjectorBase : public Operator
{
public:
	explicit OperatorProjectorBase(const OperatorProjectorParams& p_projParams);

	const Scanner& getScanner() const;
	const BinIterator* getBinIter() const;

	virtual void setAttImage(const Image* p_attImage);  // alias
	virtual void setAttImageForBackprojection(const Image* p_attImage);
	void setAttenuationImage(const Image* p_attImage);
	virtual void setAddHisto(const Histogram* p_addHisto);
	void setBinIter(const BinIterator* p_binIter);

	const Image* getAttImage() const;
	const Image* getAttImageForBackprojection() const;
	const Histogram* getAddHisto() const;

protected:
	// To take scanner properties into account
	const Scanner& scanner;

	// Bin iterator
	const BinIterator* binIter;

	// Attenuation image for forward projection (when there is motion)
	const Image* attImageForForwardProjection;
	// For generating a sensitivity image with attenuation correction
	const Image* attImageForBackprojection;
	// Additive histogram
	const Histogram* addHisto;
};

class OperatorProjector : public OperatorProjectorBase
{
public:
	enum ProjectorType
	{
		SIDDON = 0,
		DD,
		DD_GPU
	};

	explicit OperatorProjector(const OperatorProjectorParams& p_projParams);

	// Virtual functions
	virtual float forwardProjection(
	    const Image* image,
	    const ProjectionProperties& projectionProperties) const = 0;
	virtual void
	    backProjection(Image* image,
	                   const ProjectionProperties& projectionProperties,
	                   float projValue) const = 0;

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	void setupProjPsfManager(const std::string& psfFilename);

	const TimeOfFlightHelper* getTOFHelper() const;
	const ProjectionPsfManager* getProjectionPsfManager() const;

protected:
	// Time of flight
	std::unique_ptr<TimeOfFlightHelper> mp_tofHelper;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManager> mp_projPsfManager;
};
