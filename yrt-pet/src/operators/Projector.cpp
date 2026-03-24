/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/Projector.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/ProjectorDD.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_projector(py::module& m)
{
	auto c = py::class_<Projector>(m, "Projector");

	c.def_static("create", &Projector::create, "proj_params"_a);

	c.def("getProjectionPropertyTypes", &Projector::getProjectionPropertyTypes);

	c.def("addTOF", &Projector::addTOF, "tof_width_ps"_a, "tof_num_std"_a);
	c.def("addProjPSF", &Projector::addProjPSF, "proj_psf_fname"_a);
}
}  // namespace yrt

#endif

namespace yrt
{

Projector::Projector(const ProjectorParams& pr_projParams)
    : mr_scanner(pr_projParams.scanner),
      m_updaterType(pr_projParams.updaterType)
{
	if (pr_projParams.hasTOF())
	{
		addTOF(pr_projParams.getTOFWidth_ps(), pr_projParams.getTOFNumStd());
	}
	if (!pr_projParams.projPsf_fname.empty())
	{
		addProjPSF(pr_projParams.projPsf_fname);
	}
	setupUpdater(pr_projParams);
	setupProjPropertyTypes(pr_projParams);
}

std::unique_ptr<Projector> Projector::create(const ProjectorParams& params)
{
	if (params.projectorType == ProjectorType::SIDDON)
	{
		return std::make_unique<ProjectorSiddon>(params);
	}
	if (params.projectorType == ProjectorType::DD)
	{
		return std::make_unique<ProjectorDD>(params);
	}
	throw std::runtime_error("Unsupported projector type");
}

void Projector::addTOF(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<TimeOfFlightHelper>(tofWidth_ps, tofNumStd);
	projPropertyTypes.insert(ProjectionPropertyType::TOF);
	// Note: If the TOF was already added into the properties list by the
	// projector params (from the constructor), it won't be problem because
	// std::set will not have duplicates
}

void Projector::addProjPSF(const std::string& projPsf_fname)
{
	mp_projPsfManager = std::make_unique<ProjectionPsfManager>(projPsf_fname);
}

void Projector::setupUpdater(const ProjectorParams& pr_projParams)
{
	if (pr_projParams.updaterType == UpdaterType::DEFAULT4D)
	{
		// Leave updater to nullptr
	}
	else if (pr_projParams.updaterType == UpdaterType::LR)
	{
		if (pr_projParams.HBasis.getSizeTotal() == 0)
		{
			throw std::invalid_argument(
			    "LR updater was requested but HBasis is empty");
		}
		setUpdater(std::make_unique<ProjectorUpdaterLR>(pr_projParams.HBasis));
		if (auto* updaterLR =
		        dynamic_cast<ProjectorUpdaterLR*>(mp_updater.get()))
		{
			updaterLR->setUpdateH(pr_projParams.updateH);
		}
		else
		{
			throw std::runtime_error("ProjectorUpdater type needs to be "
			                         "ProjectorUpdaterLR to get/set updateH");
		}
	}
	else if (pr_projParams.updaterType == UpdaterType::LRDUALUPDATE)
	{
		if (pr_projParams.HBasis.getSizeTotal() == 0)
		{
			throw std::invalid_argument(
			    "LRDUALUPDATE updater was requested but HBasis is empty");
		}
		setUpdater(std::make_unique<ProjectorUpdaterLRDualUpdate>(
		    pr_projParams.HBasis));
		if (auto* updaterLR =
		        dynamic_cast<ProjectorUpdaterLRDualUpdate*>(mp_updater.get()))
		{
			updaterLR->setUpdateH(pr_projParams.updateH);
		}
		else
		{
			throw std::runtime_error(
			    "ProjectorUpdater type needs to be "
			    "ProjectorUpdaterLRDualUpdate to get/set updateH");
		}
	}
	else
	{
		throw std::invalid_argument("Projector updater type not valid");
	}
}

void Projector::setupProjPropertyTypes(const ProjectorParams& pr_projParams)
{
	// Add the projectorParams's properties
	std::set<ProjectionPropertyType> propertiesToAdd =
	    pr_projParams.projPropertyTypesExtra;
	projPropertyTypes.merge(propertiesToAdd);
}

void Projector::setUpdater(std::unique_ptr<ProjectorUpdater> pp_updater)
{
	mp_updater = std::move(pp_updater);
}

const Scanner& Projector::getScanner() const
{
	return mr_scanner;
}

const TimeOfFlightHelper* Projector::getTOFHelper() const
{
	return mp_tofHelper.get();
}

const ProjectionPsfManager* Projector::getProjectionPsfManager() const
{
	return mp_projPsfManager.get();
}

ProjectorUpdater* Projector::getUpdater()
{
	return mp_updater.get();
}

bool Projector::hasUpdater() const
{
	return mp_updater != nullptr;
}

UpdaterType Projector::getUpdaterType() const
{
	return m_updaterType;
}

std::set<ProjectionPropertyType> Projector::getProjectionPropertyTypes() const
{
	std::set<ProjectionPropertyType> totalProjPropertyTypes(projPropertyTypes);

	// Include the child class's properties
	totalProjPropertyTypes.merge(getProjectionPropertyTypesInternal());

	return totalProjPropertyTypes;
}

std::set<ProjectionPropertyType>
    Projector::getProjectionPropertyTypesInternal() const
{
	// By default, only require the LOR
	return {ProjectionPropertyType::LOR};
}

}  // namespace yrt
