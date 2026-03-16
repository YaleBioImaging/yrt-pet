/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/datastruct/projection/ProjectionProperties.hpp"
#include "yrt-pet/utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
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

OperatorProjectorBase::OperatorProjectorBase(
    const ProjectorParams& pr_projParams, const BinIterator* pp_binIter)
    : scanner(pr_projParams.scanner),
      binIter(pp_binIter),
      m_projectorType(pr_projParams.projectorType),
      m_updaterType(pr_projParams.updaterType)
{
}

const Scanner& OperatorProjectorBase::getScanner() const
{
	return scanner;
}

const BinIterator* OperatorProjectorBase::getBinIter() const
{
	return binIter;
}

ProjectorType OperatorProjectorBase::getProjectorType() const
{
	return m_projectorType;
}

UpdaterType OperatorProjectorBase::getUpdaterType() const
{
	return m_updaterType;
}

void OperatorProjectorBase::setBinIter(const BinIterator* pp_binIter)
{
	binIter = pp_binIter;
}

}  // namespace yrt
