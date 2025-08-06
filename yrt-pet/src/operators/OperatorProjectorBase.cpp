/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorBase.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{

void py_setup_operatorprojectorparams(py::module& m)
{
	auto c = py::class_<OperatorProjectorParams>(m, "OperatorProjectorParams");
	c.def(
	    py::init<BinIterator*, Scanner&, float, int, const std::string&, int>(),
	    py::arg("binIter"), py::arg("scanner"), py::arg("tofWidth_ps") = 0.f,
	    py::arg("tofNumStd") = -1, py::arg("projPsf_fname") = "",
	    py::arg("num_rays") = 1);
	c.def_readwrite("tofWidth_ps", &OperatorProjectorParams::tofWidth_ps);
	c.def_readwrite("tofNumStd", &OperatorProjectorParams::tofNumStd);
	c.def_readwrite("projPsf_fname", &OperatorProjectorParams::projPsf_fname);
	c.def_readwrite("num_rays", &OperatorProjectorParams::numRays);

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

	py::enum_<OperatorProjectorBase::ProjectorUpdaterType>(c, "ProjectorUpdaterType")
	    .value("DEFAULT3D", OperatorProjectorBase::ProjectorUpdaterType::DEFAULT3D)
	    .value("DEFAULT4D", OperatorProjectorBase::ProjectorUpdaterType::DEFAULT4D)
	    .value("LR", OperatorProjectorBase::ProjectorUpdaterType::LR)
	    .export_values();
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorParams::OperatorProjectorParams(
    const BinIterator* pp_binIter, const Scanner& pr_scanner,
    float p_tofWidth_ps, int p_tofNumStd, const std::string& pr_projPsf_fname,
    int p_num_rays)
    : binIter(pp_binIter),
      scanner(pr_scanner),
      tofWidth_ps(p_tofWidth_ps),
      tofNumStd(p_tofNumStd),
      projPsf_fname(pr_projPsf_fname),
      numRays(p_num_rays)
{
}

OperatorProjectorBase::OperatorProjectorBase(
    const OperatorProjectorParams& p_projParams)
    : scanner(p_projParams.scanner), binIter(p_projParams.binIter)
{
}

OperatorProjectorBase::OperatorProjectorBase(const Scanner& pr_scanner)
    : scanner(pr_scanner), binIter{nullptr}
{
}

const BinIterator* OperatorProjectorBase::getBinIter() const
{
	return binIter;
}

const Scanner& OperatorProjectorBase::getScanner() const
{
	return scanner;
}

void OperatorProjectorBase::setBinIter(const BinIterator* p_binIter)
{
	binIter = p_binIter;
}

}  // namespace yrt
