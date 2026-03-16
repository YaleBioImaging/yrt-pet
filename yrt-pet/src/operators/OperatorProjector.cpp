/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjector.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/operators/ProjectorDD.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Tools.hpp"


#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_operatorprojector(py::module& m)
{
	auto c = py::class_<OperatorProjector, OperatorProjectorBase>(
	    m, "OperatorProjector");

	c.def(py::init<const ProjectorParams&, const BinIterator*,
	               const std::vector<Constraint*>&>(),
	      "proj_params"_a, "bin_iter"_a, "constraints"_a);
	c.def(py::init<const ProjectorParams&, const BinIterator*>(),
	      "proj_params"_a, "bin_iter"_a);

	c.def("initBinLoader", &OperatorProjector::initBinLoader, "constraints"_a);

	c.def(
	    "applyA",
	    [](OperatorProjector& self, const Image* img, ProjectionData* proj)
	    { self.applyA(img, proj); }, py::arg("img"), py::arg("proj"));
	c.def(
	    "applyAH",
	    [](OperatorProjector& self, const ProjectionData* proj, Image* img)
	    { self.applyAH(proj, img); }, py::arg("proj"), py::arg("img"));

	c.def("addTOF", &OperatorProjector::addTOF, "tof_width_ps"_a,
	      "tof_num_std"_a);
	c.def("addProjPSF", &OperatorProjector::addProjPSF, "proj_psf_fname"_a);

	c.def("getTOFHelper", &OperatorProjector::getTOFHelper);
	c.def("getProjectionPsfManager",
	      &OperatorProjector::getProjectionPsfManager);

	c.def("getProjectionPropertyTypes",
	      &OperatorProjector::getProjectionPropertyTypes);
	c.def("getBinLoader", &OperatorProjector::getBinLoader);
	c.def("getElementSize", &OperatorProjector::getElementSize);

	c.def("setUpdaterLRUpdateH",
	      [](OperatorProjector& self, bool updateH)
	      {
		      auto* updaterLR =
		          dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
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
		          dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
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
		        dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
		    if (updaterLR == nullptr)
		    {
			    throw py::cast_error(
			        "Projector needs to have a `ProjectorUpdaterLR`");
		    }

		    Array2DAlias<float> hBasis;
		    hBasis.bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0],
		                buffer.shape[1]);
		    updaterLR->setHBasis(hBasis);
	    },
	    py::arg("numpy_data"));

	c.def(
	    "setUpdaterLRHBasisWrite",
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
		        dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
		    if (updaterLR == nullptr)
		    {
			    throw py::cast_error(
			        "Projector needs to have a `ProjectorUpdaterLR`");
		    }

		    Array2DAlias<float> hBasis;
		    hBasis.bind(reinterpret_cast<float*>(buffer.ptr), buffer.shape[0],
		                buffer.shape[1]);
		    updaterLR->setHBasisWrite(hBasis);
	    },
	    py::arg("numpy_data"));


	c.def("getUpdaterLRHBasis",
	      [](OperatorProjector& self)
	      {
		      auto* updaterLR =
		          dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
		      if (updaterLR == nullptr)
		      {
			      throw py::cast_error(
			          "Projector needs to have a `ProjectorUpdaterLR`");
		      }
		      auto H = updaterLR->getHBasis();
		      auto dims = H.getDims();
		      py::array_t<float> arr({dims[0], dims[1]});  // C-contiguous
		      std::memcpy(arr.mutable_data(),              // copy all at once
		                  H.getRawPointer(),
		                  static_cast<size_t>(dims[0] * dims[1]) *
		                      sizeof(float));
		      return arr;  // copy
	      });

	c.def("getUpdaterLRHBasisWrite",
	      [](OperatorProjector& self)
	      {
		      auto* updaterLR =
		          dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
		      if (updaterLR == nullptr)
		      {
			      throw py::cast_error(
			          "Projector needs to have a `ProjectorUpdaterLR`");
		      }
		      auto H = updaterLR->getHBasisWrite();
		      const auto dims = H.getDims();
		      float sum = 0.f;
		      for (size_t d = 0; d < dims[0]; ++d)
		      {
			      for (size_t t = 0; t < dims[1]; ++t)
			      {
				      sum += H[d][t];
			      }
		      }
		      if (sum < EPS_FLT)
		      {
			      updaterLR->accumulateH();
		      }
		      py::array_t<float> arr({dims[0], dims[1]});  // C-contiguous
		      std::memcpy(arr.mutable_data(),              // copy all at once
		                  H.getRawPointer(),
		                  static_cast<size_t>(dims[0] * dims[1]) *
		                      sizeof(float));
		      return arr;  // copy
	      });

	c.def("getUpdaterLRHBasisWriteThread",
	      [](OperatorProjector& self)
	      {
		      auto* updaterLR =
		          dynamic_cast<ProjectorUpdaterLR*>(self.getUpdater());
		      if (updaterLR == nullptr)
		      {
			      throw py::cast_error(
			          "Projector needs to have a `ProjectorUpdaterLR`");
		      }
		      auto& H = updaterLR->getHBasisWriteThread();
		      ASSERT_MSG(H.getRawPointer(), "HBasisWriteThread is nullptr");
		      auto dims = H.getDims();
		      py::array_t<double> arr(
		          {dims[0], dims[1], dims[2]});  // C-contiguous
		      std::memcpy(arr.mutable_data(),    // copy all at once
		                  H.getRawPointer(),
		                  static_cast<size_t>(dims[0] * dims[1] * dims[2]) *
		                      sizeof(double));
		      return arr;  // copy
	      });
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjector::OperatorProjector(
    const ProjectorParams& pr_projParams, const BinIterator* pp_binIter,
    const std::vector<Constraint*>& pr_constraints)
    : OperatorProjectorBase(pr_projParams, pp_binIter)
{
	// Create the specific projector
	mp_projector = Projector::create(pr_projParams);

	// initBinLoader will gather all the properties from the mp_projector
	//  object, which will give the specific Projector's list of needed
	//  properties
	initBinLoader(pr_constraints);
}

void OperatorProjector::initBinLoader(
    const std::vector<Constraint*>& pr_constraints)
{
	setupBinLoader(pr_constraints);
	allocateBuffers();
}

void OperatorProjector::applyA(const Variable* in, Variable* out)
{
	auto* dat = dynamic_cast<ProjectionData*>(out);
	auto* img = dynamic_cast<const Image*>(in);

	ASSERT_MSG(dat != nullptr, "Output variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Input variable has to be an Image");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

	mp_binLoader->parallelDoOnBins(
	    *dat, *binIter,
	    [&img, &dat, this](const ProjectionPropertyManager& propManager,
	                       PropertyUnit* propStruct, size_t pos, bin_t bin)
	    {
		    const float imProj = mp_projector->forwardProjection(
		        img, propManager, propStruct, pos);
		    dat->setProjectionValue(bin, imProj);
	    });
}

void OperatorProjector::applyAH(const Variable* in, Variable* out)
{
	auto* dat = dynamic_cast<const ProjectionData*>(in);
	auto* img = dynamic_cast<Image*>(out);

	ASSERT_MSG(dat != nullptr, "Input variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Output variable has to be an Image");
	ASSERT_MSG(img->isMemoryValid(), "Image array is unallocated");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

	mp_binLoader->parallelDoOnBins(
	    *dat, *binIter,
	    [&img, &dat, this](const ProjectionPropertyManager& propManager,
	                       PropertyUnit* propStruct, size_t pos, bin_t bin)
	    {
		    const float projValue = dat->getProjectionValue(bin);
		    if (std::abs(projValue) == 0.0f)
		    {
			    return;
		    }
		    mp_projector->backProjection(img, propManager, propStruct, pos,
		                                 projValue);
	    });
}

void OperatorProjector::addTOF(float tofWidth_ps, int tofNumStd)
{
	mp_projector->addTOF(tofWidth_ps, tofNumStd);
}

void OperatorProjector::addProjPSF(const std::string& projPsf_fname)
{
	mp_projector->addProjPSF(projPsf_fname);
}

const TimeOfFlightHelper* OperatorProjector::getTOFHelper() const
{
	return mp_projector->getTOFHelper();
}

const ProjectionPsfManager* OperatorProjector::getProjectionPsfManager() const
{
	return mp_projector->getProjectionPsfManager();
}

void OperatorProjector::setUpdater(std::unique_ptr<ProjectorUpdater> pp_updater)
{
	mp_projector->setUpdater(std::move(pp_updater));
}

std::set<ProjectionPropertyType>
    OperatorProjector::getProjectionPropertyTypes() const
{
	auto projProperties = mp_projector->getProjectionPropertyTypes();

	// It is impossible to know yet, and from the PperatorProjector alone, if
	//  dynamic frame will vary or not. It depends on whether the image
	//  parameters that will be used has nt > 1 and whether the data input has
	//  a dynamic framing or not.
	projProperties.insert(ProjectionPropertyType::DYNAMIC_FRAME);

	return projProperties;
}

void OperatorProjector::allocateBuffers()
{
	const int numThreads = globals::getNumThreads();

	// We allocate one row of properties for every thread
	mp_binLoader->allocate(numThreads);
}

ProjectorUpdater* OperatorProjector::getUpdater()
{
	return mp_projector->getUpdater();
}

const BinLoader* OperatorProjector::getBinLoader() const
{
	return mp_binLoader.get();
}

PropertyUnit* OperatorProjector::getProjectionProperties() const
{
	return mp_binLoader->getProjectionPropertiesRawPointer();
}

PropertyUnit* OperatorProjector::getConstraintVariables() const
{
	return mp_binLoader->getConstraintVariablesRawPointer();
}

void OperatorProjector::setupBinLoader(
    const std::vector<Constraint*>& pr_constraints)
{
	// Determine projection property types from projector
	auto projProperties = getProjectionPropertyTypes();

	mp_binLoader = std::make_unique<BinLoader>(pr_constraints, projProperties);
}

unsigned int OperatorProjector::getElementSize() const
{
	const BinLoader* binFilter = getBinLoader();
	ASSERT(binFilter != nullptr);
	return binFilter->getPropertyManager().getElementSize();
}

}  // namespace yrt
