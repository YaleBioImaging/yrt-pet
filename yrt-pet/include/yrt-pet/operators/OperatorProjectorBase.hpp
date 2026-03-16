/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/Operator.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"

namespace yrt
{

class BinIterator;
class Image;
class Scanner;
class ProjectionData;
class Histogram;

// Device-agnostic virtual class
class OperatorProjectorBase : public Operator
{
public:
	explicit OperatorProjectorBase(const ProjectorParams& pr_projParams,
	                               const BinIterator* pp_binIter = nullptr);

	const Scanner& getScanner() const;
	const BinIterator* getBinIter() const;
	ProjectorType getProjectorType() const;
	UpdaterType getUpdaterType() const;

	void setBinIter(const BinIterator* p_binIter);

protected:
	// To take scanner properties into account
	const Scanner& scanner;

	// Bin iterator (note: bin iterators may move from the projector object in
	// the future)
	const BinIterator* binIter;

	ProjectorType m_projectorType;
	UpdaterType m_updaterType;
};

}  // namespace yrt
