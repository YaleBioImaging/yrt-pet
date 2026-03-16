/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

namespace yrt
{

HOST_DEVICE_CALLABLE enum class UpdaterType {
	DEFAULT4D = 0,
	LR,
	LRDUALUPDATE
};

enum class ProjectorType
{
	SIDDON = 0,
	DD
};

class ProjectorParams
{
public:
	explicit ProjectorParams(const Scanner& pr_scanner);

	ProjectorParams(const ProjectorParams& other);

	void setProjector(const std::string& projectorName);
	void setProjector(ProjectorType projType);
	void addTOF(float tofWidth_ps, int tofNumStd);
	void removeTOF();
	float getTOFWidth_ps() const;
	int getTOFNumStd() const;
	bool hasTOF() const;
	void bindHBasis(float* HBasis_ptr, size_t rank, size_t T);
	Array2DBase<float>& getHBasis();

	const Scanner& scanner;

	// Siddon or DD
	ProjectorType projectorType;

	// Projection-domain PSF
	std::string projPsf_fname;

	// Multi-ray siddon only
	int numRays;

	// Projector Updater type (e.g., DEFAULT4D)
	UpdaterType updaterType;

	// Optional bool for H-update in LR updater
	bool updateH;
	// LR members
	Array2DAlias<float> HBasis;

	// Projection property types (in addition to types needed for projector and
	// included in projection data) - Unused for now
	std::set<ProjectionPropertyType> projPropertyTypesExtra;

private:
	// Time of Flight
	float m_tofWidth_ps;
	int m_tofNumStd;
};
}  // namespace yrt
