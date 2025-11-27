/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/datastruct/projection/BinFilter.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{

void py_setup_operatorprojectorparams(py::module& m)
{
	auto c = py::class_<OperatorProjectorParams>(m, "OperatorProjectorParams");
	c.def(py::init<Scanner&>(), py::arg("scanner"));

	py::enum_<OperatorProjectorParams::ProjectorUpdaterType>(
	    c, "ProjectorUpdaterType")
	    .value("DEFAULT3D",
	           OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D)
	    .value("DEFAULT4D",
	           OperatorProjectorParams::ProjectorUpdaterType::DEFAULT4D)
	    .value("LR", OperatorProjectorParams::ProjectorUpdaterType::LR)
	    .value("LRDUALUPDATE",
	           OperatorProjectorParams::ProjectorUpdaterType::LRDUALUPDATE)
	    .export_values();

	c.def_property(
	    "HBasis",
	    // getter: const ref to the alias; tie lifetime to parent
	    [](OperatorProjectorParams& p) -> Array2DBase<float>&
	    { return p.getHBasis(); },
	    // setter: accept any 2D array base and bind alias to it
	    [](OperatorProjectorParams& p, Array2DBase<float>& src)
	    {
		    auto dims = src.getDims();
		    p.bindHBasis(src.getRawPointer(), dims[0],
		                 dims[1]);  // NO allocation, just alias the source
	    },
	    py::return_value_policy::reference_internal);

	c.def(
	    "setHBasisFromNumpy",
	    [](OperatorProjectorParams& self, py::buffer& np_data)
	    {
		    py::buffer_info buffer = np_data.request();

		    if (buffer.ndim != 2)
			    throw std::invalid_argument("HBasis must be 2D (rank x time).");

		    if (buffer.format != py::format_descriptor<float>::format())
			    throw std::invalid_argument("HBasis must be float32.");

		    auto* ptr = reinterpret_cast<float*>(buffer.ptr);
		    const size_t rank = static_cast<size_t>(buffer.shape[0]);
		    const size_t T = static_cast<size_t>(buffer.shape[1]);

		    self.bindHBasis(ptr, rank, T);
	    },
	    py::arg("HBasis"),
	    py::keep_alive<1, 2>()  // keep the buffer owner alive
	);

	c.def_readwrite("projectorUpdaterType",
	                &OperatorProjectorParams::projectorUpdaterType);
	c.def_readwrite("updateH", &OperatorProjectorParams::updateH);
	c.def_readwrite("binIter", &OperatorProjectorParams::binIter);
	c.def("addTOF", &OperatorProjectorParams::addTOF, "tofWidth_ps"_a,
	      "tofNumStd"_a);
	c.def_readwrite("projPsf_fname", &OperatorProjectorParams::projPsf_fname);
	c.def_readwrite("num_rays", &OperatorProjectorParams::numRays);
	c.def_readwrite("num_threads", &OperatorProjectorParams::numThreads);
	c.def_readwrite("proj_property_types_extra",
	                &OperatorProjectorParams::projPropertyTypesExtra);
}

void py_setup_operatorprojectorbase(py::module& m)
{
	auto c =
	    py::class_<OperatorProjectorBase, Operator>(m, "OperatorProjectorBase");
	c.def("getBinIter", &OperatorProjectorBase::getBinIter);
	c.def("getScanner", &OperatorProjectorBase::getScanner);

	py::enum_<OperatorProjectorBase::ProjectorType>(c, "ProjectorType")
	    .value("SIDDON", OperatorProjectorBase::ProjectorType::SIDDON)
	    .value("DD", OperatorProjectorBase::ProjectorType::DD)
	    .export_values();

	c.def("getBinFilter", &OperatorProjectorBase::getBinFilter);
	c.def("getElementSize", &OperatorProjectorBase::getElementSize);
}
}  // namespace yrt

#endif

namespace yrt
{

// TODO NOW: Create a setter for the updater (and make the member private)
OperatorProjectorParams::OperatorProjectorParams(const Scanner& pr_scanner)
    : binIter(nullptr),
      scanner(pr_scanner),
      projPsf_fname(""),
      numRays(1),
      numThreads(globals::getNumThreads()),
      projectorUpdaterType(ProjectorUpdaterType::DEFAULT3D),
      updateH(false),
      m_tofWidth_ps(0.f),
      m_tofNumStd(0)
{
}

OperatorProjectorParams::OperatorProjectorParams(
    const OperatorProjectorParams& other)
    : binIter(other.binIter),
      scanner(other.scanner),
      projPsf_fname(other.projPsf_fname),
      numRays(other.numRays),
      numThreads(other.numThreads),
      projectorUpdaterType(other.projectorUpdaterType),
      updateH(other.updateH),
      m_tofWidth_ps(other.m_tofWidth_ps),
      m_tofNumStd(other.m_tofNumStd)
{
	HBasis.bind(other.HBasis);
}

void OperatorProjectorParams::addTOF(float tofWidth_ps, int tofNumStd)
{
	m_tofWidth_ps = tofWidth_ps;
	m_tofNumStd = tofNumStd;
	projPropertyTypesExtra.insert(ProjectionPropertyType::TOF);
}

float OperatorProjectorParams::getTOFWidth_ps() const
{
	return m_tofWidth_ps;
}

int OperatorProjectorParams::getTOFNumStd() const
{
	return m_tofNumStd;
}

bool OperatorProjectorParams::hasTOF() const
{
	return m_tofWidth_ps > 0.f;
}
void OperatorProjectorParams::bindHBasis(float* HBasis_ptr, size_t rank,
                                         size_t T)
{
	HBasis.bind(HBasis_ptr, rank, T);
}
Array2DBase<float>& OperatorProjectorParams::getHBasis()
{
	return HBasis;
}

OperatorProjectorBase::OperatorProjectorBase(
    const OperatorProjectorParams& pr_projParams,
    const std::vector<Constraint*>& pr_constraints)
    : scanner(pr_projParams.scanner),
      binIter{pr_projParams.binIter},
      m_projectorUpdaterType(pr_projParams.projectorUpdaterType),
      m_constraints(pr_constraints)
{
}

void OperatorProjectorBase::initBinFilter(
    const std::set<ProjectionPropertyType>& projPropertyTypesExtra,
    const int numThreads)
{
	setupBinFilter(projPropertyTypesExtra);
	allocateBuffers(numThreads);
}

std::set<ProjectionPropertyType>
    OperatorProjectorBase::getProjectionPropertyTypes() const
{
	std::set<ProjectionPropertyType> projPropTypes;
	if (m_projectorUpdaterType ==
	        OperatorProjectorParams::ProjectorUpdaterType::DEFAULT4D ||
	    m_projectorUpdaterType ==
	        OperatorProjectorParams::ProjectorUpdaterType::LR ||
	    m_projectorUpdaterType ==
	        OperatorProjectorParams::ProjectorUpdaterType::LRDUALUPDATE)
	{
		projPropTypes.insert(ProjectionPropertyType::DYNAMIC_FRAME);
	}
	projPropTypes.merge(getProjectionPropertyTypesInternal());
	return projPropTypes;
}

std::set<ProjectionPropertyType>
    OperatorProjectorBase::getProjectionPropertyTypesInternal() const
{
	return {};
}

const BinIterator* OperatorProjectorBase::getBinIter() const
{
	return binIter;
}

const BinFilter* OperatorProjectorBase::getBinFilter() const
{
	return mp_binFilter.get();
}

const Scanner& OperatorProjectorBase::getScanner() const
{
	return scanner;
}

ProjectionProperties OperatorProjectorBase::getProjectionProperties() const
{
	return m_projectionProperties.get();
}

ConstraintParams OperatorProjectorBase::getConstraintParams() const
{
	return m_constraintParams.get();
}

unsigned int OperatorProjectorBase::getElementSize() const
{
	const auto binFilter = getBinFilter();
	ASSERT(binFilter != nullptr);
	return binFilter->getPropertyManager().getElementSize();
}

void OperatorProjectorBase::setBinIter(const BinIterator* p_binIter)
{
	binIter = p_binIter;
}

void OperatorProjectorBase::setupBinFilter(
    const std::set<ProjectionPropertyType>& pr_projPropertiesExtra)
{
	// Determine projection property types from projector
	auto projProperties = getProjectionPropertyTypes();
	for (auto prop : pr_projPropertiesExtra)
	{
		projProperties.insert(prop);
	}
	// Determine constraints from scanner
	mp_binFilter = std::make_unique<BinFilter>(m_constraints, projProperties);
	mp_binFilter->setupManagers();
}

void OperatorProjectorBase::allocateBuffers(int numThreads)
{
	auto& projPropManager = mp_binFilter->getPropertyManager();
	auto& consManager = mp_binFilter->getConstraintManager();
	if (projPropManager.getElementSize() > 0)
	{
		m_projectionProperties = projPropManager.createDataArray(numThreads);
	}
	if (consManager.getElementSize() > 0)
	{
		m_constraintParams = consManager.createDataArray(numThreads);
	}
}

}  // namespace yrt
