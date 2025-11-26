/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorUpdaterBase.hpp"
#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/operators/Operator.hpp"

namespace yrt
{

class BinIterator;
class Image;
class Scanner;
class ProjectionData;
class Histogram;


class OperatorProjectorParams
{
public:
	enum ProjectorUpdaterType
	{
		DEFAULT3D = 0,
		DEFAULT4D,
		LR,
		LRDUALUPDATE
	};

	explicit OperatorProjectorParams(const Scanner& pr_scanner);

	OperatorProjectorParams(const OperatorProjectorParams& other);

	const BinIterator* binIter;
	const Scanner& scanner;

	void addTOF(float tofWidth_ps, int tofNumStd);
	float getTOFWidth_ps() const;
	int getTOFNumStd() const;
	bool hasTOF() const;
	void bindHBasis(float* HBasis_ptr, size_t rank, size_t T);
	Array2DBase<float>& getHBasis();

	// Projection-domain PSF
	std::string projPsf_fname;

	// Multi-ray siddon only
	int numRays;

	// Number of threads
	int numThreads;

	// Projector Updater type (e.g., DEFAULT3D)
	ProjectorUpdaterType projectorUpdaterType;

	// Optional bool for H-update in LR updater
	bool updateH;
	// LR members
	Array2DAlias<float> HBasis;

	// Projection property types (in addition to types needed for projector and
	// included in projection data) - Ignored for now
	std::set<ProjectionPropertyType> projPropertyTypesExtra;

private:
	// Time of Flight
	float m_tofWidth_ps;
	int m_tofNumStd;
};

// Device-agnostic virtual class
class OperatorProjectorBase : public Operator
{
public:

	enum ProjectorType
	{
		SIDDON = 0,
		DD
	};

	explicit OperatorProjectorBase(
	    const OperatorProjectorParams& p_projParams,
	    const std::vector<Constraint*>& pr_constraints = {});

	const Scanner& getScanner() const;
	const BinIterator* getBinIter() const;
	const BinFilter* getBinFilter() const;
	ProjectionProperties getProjectionProperties() const;
	ConstraintParams getConstraintParams() const;

	unsigned int getElementSize() const;

	void setBinIter(const BinIterator* p_binIter);

	virtual std::set<ProjectionPropertyType> getProjectionPropertyTypesInternal() const;
	std::set<ProjectionPropertyType> getProjectionPropertyTypes() const;

	virtual void initBinFilter(
	    const std::set<ProjectionPropertyType>& projPropertyTypesExtra,
	    const int numThreads);

	void setupBinFilter(
	    const std::set<ProjectionPropertyType>& pr_projPropertiesExtra);

	void allocateBuffers(int numThreads);

protected:
	// To take scanner properties into account
	const Scanner& scanner;

	// Bin iterator (note: bin iterators may move from the projector object in
	// the future)
	const BinIterator* binIter;

	OperatorProjectorParams::ProjectorUpdaterType m_projectorUpdaterType;

	// Constraints for bin iterator
	std::vector<Constraint*> m_constraints;
	std::unique_ptr<BinFilter> mp_binFilter;
	std::unique_ptr<char[]> m_projectionProperties;
	std::unique_ptr<char[]> m_constraintParams;
};

}  // namespace yrt
