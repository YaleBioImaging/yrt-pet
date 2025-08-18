/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/datastruct/projection/BinIteratorConstrained.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{

void py_setup_operatorprojectorparams(py::module& m)
{
	auto c = py::class_<OperatorProjectorParams>(m, "OperatorProjectorParams");
	c.def(py::init<Scanner&>(), py::arg("scanner"));
	c.def_readwrite("tofWidth_ps", &OperatorProjectorParams::tofWidth_ps);
	c.def_readwrite("tofNumStd", &OperatorProjectorParams::tofNumStd);
	c.def_readwrite("projPsf_fname", &OperatorProjectorParams::projPsf_fname);
	c.def_readwrite("num_rays", &OperatorProjectorParams::numRays);
	c.def_readwrite("num_threads", &OperatorProjectorParams::numThreads);
}

void py_setup_operatorprojectorbase(py::module& m)
{
	auto c =
	    py::class_<OperatorProjectorBase, Operator>(m, "OperatorProjectorBase");
	c.def("getBinIter", &OperatorProjectorBase::getBinIter);
	c.def("getScanner", &OperatorProjectorBase::getScanner);
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorParams::OperatorProjectorParams(const Scanner& pr_scanner)
    : scanner(pr_scanner),
      tofWidth_ps(0.f),
      tofNumStd(0),
      projPsf_fname(""),
      numRays(1),
      numThreads(1)
{
}

OperatorProjectorBase::OperatorProjectorBase(
	const OperatorProjectorParams& pr_projParams,
	const std::vector<Constraint*>& pr_constraints)
    : scanner(pr_projParams.scanner),
      binIter{pr_projParams.binIter},
      m_constraints(pr_constraints)
{
	setupBinIteratorConstrained(pr_projParams.projPropertyTypesExtra);
	allocateBuffers(pr_projParams.numThreads);
}

std::set<ProjectionPropertyType>
    OperatorProjectorBase::getProjectionPropertyTypes() const
{
	return {};
}

const BinIterator* OperatorProjectorBase::getBinIter() const
{
	return binIter;
}

const BinIteratorConstrained*
    OperatorProjectorBase::getBinIterConstrained() const
{
	return m_binIterConstrained.get();
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

void OperatorProjectorBase::setBinIter(const BinIterator* p_binIter)
{
	binIter = p_binIter;
}

void OperatorProjectorBase::setupBinIteratorConstrained(
	const std::set<ProjectionPropertyType>& pr_projPropertiesExtra)
{
	// Determine projection property types from projector
	auto projProperties = getProjectionPropertyTypes();
	for (auto prop : pr_projPropertiesExtra)
	{
		projProperties.insert(prop);
	}
	// Determine constraints from scanner
	m_binIterConstrained = std::make_unique<BinIteratorConstrained>(
		m_constraints, projProperties);
	m_binIterConstrained->setupManagers();
}

void OperatorProjectorBase::allocateBuffers(int numThreads)
{
	auto& projPropManager = m_binIterConstrained->getPropertyManager();
	auto& consManager = m_binIterConstrained->getConstraintManager();
	m_projectionProperties = projPropManager.createDataArray(numThreads);
	m_constraintParams = consManager.createDataArray(numThreads);
}

}  // namespace yrt
