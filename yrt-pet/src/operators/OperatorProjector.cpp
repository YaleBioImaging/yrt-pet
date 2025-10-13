/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjector.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Tools.hpp"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

#include <utility>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorprojector(py::module& m)
{
	auto c = py::class_<OperatorProjector, OperatorProjectorBase>(
		m, "OperatorProjector");
	c.def("setupTOFHelper", &OperatorProjector::setupTOFHelper);
	c.def("getTOFHelper", &OperatorProjector::getTOFHelper);
	c.def("getProjectionPsfManager",
	      &OperatorProjector::getProjectionPsfManager);
	c.def(
		"applyA",
		[](OperatorProjector& self, const Image* img, ProjectionData* proj)
		{
			self.applyA(img, proj);
		}, py::arg("img"), py::arg("proj"));
	c.def(
		"applyAH",
		[](OperatorProjector& self, const ProjectionData* proj, Image* img)
		{
			self.applyAH(proj, img);
		}, py::arg("proj"), py::arg("img"));

	c.def("setUpdaterLRUpdateH",
	      [](OperatorProjector& self, bool updateH)
	      {
		      auto* updaterLR =
			      dynamic_cast<OperatorProjectorUpdaterLR*>(self.getUpdater());
		      if (updaterLR == nullptr)
		      {
			      throw std::bad_cast();
		      }
		      updaterLR->setUpdateH(updateH);
	      });

	c.def("getUpdaterLRUpdateH",
	      [](OperatorProjector& self)
	      {
		      auto* updaterLR =
			      dynamic_cast<OperatorProjectorUpdaterLR*>(self.getUpdater());
		      if (updaterLR == nullptr)
		      {
			      throw std::bad_cast();
		      }

		      return updaterLR->getUpdateH();
	      });

	c.def(
		"setUpdaterLRHBasis",
		[](OperatorProjector& self, py::buffer& np_data)
		{
			py::buffer_info buffer = np_data.request();
			if (buffer.ndim != 2)
			{
				throw std::invalid_argument(
					"The buffer given has to have 2 dimensions");
			}
			if (buffer.format != py::format_descriptor<float>::format())
			{
				throw std::invalid_argument(
					"The buffer given has to have a float32 format");
			}

			auto* updaterLR =
				dynamic_cast<OperatorProjectorUpdaterLR*>(self.getUpdater());
			if (updaterLR == nullptr)
			{
				throw py::cast_error(
					"Projector needs to have a `OperatorProjectorUpdaterLR`");
			}

			Array2DAlias<float> hBasis;
			hBasis.bind(reinterpret_cast<float*>(buffer.ptr),
			            buffer.shape[0], buffer.shape[1]);
			updaterLR->setHBasis(hBasis);

		},
		py::arg("numpy_data"));


	c.def("getUpdaterLRHBasis",
	[](OperatorProjector& self)
	{
		auto* updaterLR =
			dynamic_cast<OperatorProjectorUpdaterLR*>(self.getUpdater());
		if (updaterLR == nullptr)
		{
			throw py::cast_error(
				"Projector needs to have a `OperatorProjectorUpdaterLR`");
		}

		auto dims = updaterLR->getHBasis().getDims();
		auto out  = std::make_unique<Array2D<float>>();
		out->allocate(dims[0], dims[1]);
		out->copy(updaterLR->getHBasis());
		return out;

	});
}
} // namespace yrt

#endif

namespace yrt
{

OperatorProjector::OperatorProjector(
	const Scanner& pr_scanner, float tofWidth_ps, int tofNumStd,
	const std::string& projPsf_fname,
	OperatorProjectorParams::ProjectorUpdaterType projectorUpdaterType)
	: OperatorProjector{
		OperatorProjectorParams{nullptr, pr_scanner, projectorUpdaterType,
		                        tofWidth_ps, tofNumStd, projPsf_fname}} {}

OperatorProjector::OperatorProjector(
	const OperatorProjectorParams& p_projParams)
	: OperatorProjectorBase{p_projParams},
	  mp_tofHelper{nullptr},
	  mp_projPsfManager{nullptr}
{
	if (p_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(p_projParams.tofWidth_ps, p_projParams.tofNumStd);
	}
	if (!p_projParams.projPsf_fname.empty())
	{
		setupProjPsfManager(p_projParams.projPsf_fname);
	}
	setupUpdater(p_projParams);
}

void OperatorProjector::applyA(const Variable* in, Variable* out)
{
	auto* dat = dynamic_cast<ProjectionData*>(out);
	auto* img = dynamic_cast<const Image*>(in);

	ASSERT_MSG(dat != nullptr, "Output variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Input variable has to be an Image");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

	util::parallelForChunked(binIter->size(), globals::getNumThreads(),
	                         [img, dat, this](size_t binIdx, size_t tid)
	                         {
		                         const bin_t bin = binIter->get(binIdx);

		                         ProjectionProperties projectionProperties =
			                         dat->getProjectionProperties(bin);

		                         const float imProj = forwardProjection(
			                         img, projectionProperties, tid);

		                         dat->setProjectionValue(bin, imProj);
	                           });
}

void OperatorProjector::applyAH(const Variable* in, Variable* out)
{
	auto* dat = dynamic_cast<const ProjectionData*>(in);
	auto* img = dynamic_cast<Image*>(out);

	ASSERT_MSG(dat != nullptr, "Input variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Output variable has to be an Image");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

	util::parallelForChunked(
	    binIter->size(), globals::getNumThreads(),
	    [img, dat, this](size_t binIdx, size_t tid)
	    {
		    const bin_t bin = binIter->get(binIdx);

		    ProjectionProperties projectionProperties =
		        dat->getProjectionProperties(bin);

		    float projValue = dat->getProjectionValue(bin);

		    if (std::abs(projValue) == 0.0f)
		    {
			    return;
		    }

		    backProjection(img, projectionProperties, projValue, tid);
	    });
}

void OperatorProjector::addTOF(float tofWidth_ps, int tofNumStd)
{
	setupTOFHelper(tofWidth_ps, tofNumStd);
}

void OperatorProjector::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<TimeOfFlightHelper>(tofWidth_ps, tofNumStd);
	ASSERT_MSG(mp_tofHelper != nullptr,
	           "Error occurred during the setup of TimeOfFlightHelper");
}

void OperatorProjector::setupProjPsfManager(const std::string& projPsf_fname)
{
	mp_projPsfManager = std::make_unique<ProjectionPsfManager>(projPsf_fname);
	ASSERT_MSG(mp_projPsfManager != nullptr,
	           "Error occurred during the setup of ProjectionPsfManager");
}

const TimeOfFlightHelper* OperatorProjector::getTOFHelper() const
{
	return mp_tofHelper.get();
}

const ProjectionPsfManager* OperatorProjector::getProjectionPsfManager() const
{
	return mp_projPsfManager.get();
}

void OperatorProjector::setUpdater(
	std::unique_ptr<OperatorProjectorUpdater> pp_updater)
{
	mp_updater = std::move(pp_updater);
}

OperatorProjectorUpdater* OperatorProjector::getUpdater()
{
	return mp_updater.get();
}

void OperatorProjector::setupUpdater(
	const OperatorProjectorParams& p_projParams)
{
	if (p_projParams.projectorUpdaterType == OperatorProjectorParams::DEFAULT3D)
	{
		setUpdater(std::make_unique<OperatorProjectorUpdaterDefault3D>());
	}
	else if (p_projParams.projectorUpdaterType ==
	         OperatorProjectorParams::DEFAULT4D)
	{
		setUpdater(std::make_unique<OperatorProjectorUpdaterDefault4D>());
	}
	else if (p_projParams.projectorUpdaterType == OperatorProjectorParams::LR)
	{
		if (p_projParams.HBasis.getSizeTotal() == 0)
		{
			throw std::invalid_argument(
				"LR updater was requested but HBasis is empty");
		}
		setUpdater(
			std::make_unique<OperatorProjectorUpdaterLR>(p_projParams.HBasis));
		if (auto* updaterLR = dynamic_cast<OperatorProjectorUpdaterLR*>(mp_updater.get()))
			{
				updaterLR->setUpdateH(p_projParams.updateH);
			}
		else
		{
			throw std::runtime_error("OperatorProjectorUpdater type needs to be "
							"OperatorProjectorUpdaterLR to get/set updateH");
		}
	}
	else if (p_projParams.projectorUpdaterType == OperatorProjectorParams::LRDUALUPDATE)
	{
		if (p_projParams.HBasis.getSizeTotal() == 0)
		{
			throw std::invalid_argument(
				"LRDUALUPDATE updater was requested but HBasis is empty");
		}
		setUpdater(
			std::make_unique<OperatorProjectorUpdaterLRDualUpdate>(p_projParams.HBasis));
		if (auto* updaterLR = dynamic_cast<OperatorProjectorUpdaterLRDualUpdate*>(mp_updater.get()))
		{
			updaterLR->setUpdateH(p_projParams.updateH);
		}
		else
		{
			throw std::runtime_error("OperatorProjectorUpdater type needs to be "
							"OperatorProjectorUpdaterLRDualUpdate to get/set updateH");
		}
	}
	else
	{
		throw std::invalid_argument("ProjectorUpdaterType not valid");
	}
}

} // namespace yrt
