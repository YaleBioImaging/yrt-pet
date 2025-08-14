/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinIteratorConstrained.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
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
	OperatorProjectorParams(const BinIterator* pp_binIter,
	                        const Scanner& pr_scanner,
	                        float p_tofWidth_ps = 0.f, int p_tofNumStd = 0,
	                        const std::string& pr_projPsf_fname = "",
	                        int p_num_rays = 1);

	const BinIterator* binIter;
	const Scanner& scanner;

	// Time of Flight
	float tofWidth_ps;
	int tofNumStd;

	// Projection-domain PSF
	std::string projPsf_fname;

	// Multi-ray siddon only
	int numRays;
};

// Device-agnostic virtual class
class OperatorProjectorBase : public Operator
{
public:
	explicit OperatorProjectorBase(
	    const Scanner& pr_scanner,
	    const BinIteratorConstrained& pr_binIteratorConstrained);
	explicit OperatorProjectorBase(
	    const OperatorProjectorParams& p_projParams,
	    const BinIteratorConstrained& pr_binIteratorConstrained);

	const Scanner& getScanner() const;
	const BinIterator* getBinIter() const;
	const BinIteratorConstrained* getBinIterContrained() const;

	void setBinIter(const BinIterator* p_binIter);

	virtual std::vector<ProjectionPropertyType>
	    getProjectionPropertyTypes() const;

protected:
	// To take scanner properties into account
	const Scanner& scanner;

	// Bin iterator
	const BinIterator* binIter;
	// Note: In the future, maybe bin iterators should not be a member of the
	//  projector object.
	const BinIteratorConstrained& binIterConstrained;
};

}  // namespace yrt
