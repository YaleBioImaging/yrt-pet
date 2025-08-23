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

	py::enum_<OperatorProjectorParams::ProjectorUpdaterType>(c, "ProjectorUpdaterType")
	    .value("DEFAULT3D", OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D)
	    .value("DEFAULT4D", OperatorProjectorParams::ProjectorUpdaterType::DEFAULT4D)
	    .value("LR", OperatorProjectorParams::ProjectorUpdaterType::LR)
	    .export_values();

	c.def(
	    py::init<const BinIterator*, const Scanner&, OperatorProjectorParams::ProjectorUpdaterType,
	             float, int, const std::string&, int>(),
	    py::arg("binIter"), py::arg("scanner"),
	    py::arg("ProjectorUpdaterType") = OperatorProjectorParams::ProjectorUpdaterType::DEFAULT3D,
	    py::arg("tofWidth_ps") = 0.f,
	    py::arg("tofNumStd") = -1, py::arg("projPsf_fname") = "",
	    py::arg("num_rays") = 1);

	c.def_property(
	    "HBasis",
	    // getter: const ref to the alias; tie lifetime to parent
	    [](OperatorProjectorParams& p) -> Array2DAlias<float>& {
		    return p.HBasis;
	    },
	    // setter: accept any 2D array base and bind alias to it
	    [](OperatorProjectorParams& p, const Array2DBase<float>& src) {
		    p.HBasis.bind(src);  // NO allocation, just alias the source
	    },
	    py::return_value_policy::reference_internal
	);

	c.def(
	"setHBasisFromNumpy",
	[](OperatorProjectorParams& self, py::buffer& np_data) {
		py::buffer_info buffer = np_data.request();

		if (buffer.ndim != 2)
			throw std::invalid_argument("HBasis must be 2D (rank x time).");

		if (buffer.format != py::format_descriptor<float>::format())
			throw std::invalid_argument("HBasis must be float32.");

		auto* ptr = reinterpret_cast<float*>(buffer.ptr);
		const size_t rank = static_cast<size_t>(buffer.shape[0]);
		const size_t T    = static_cast<size_t>(buffer.shape[1]);

		self.HBasis.bind(ptr, rank, T);
	},
	py::arg("HBasis"),
	py::keep_alive<1, 2>()  // keep the buffer owner alive
);

	c.def_readwrite("updateH", &OperatorProjectorParams::updateH);
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
}
}  // namespace yrt

#endif

namespace yrt
{

OperatorProjectorParams::OperatorProjectorParams(
    const BinIterator* pp_binIter, const Scanner& pr_scanner,
    ProjectorUpdaterType p_projectorUpdaterType,
    float p_tofWidth_ps, int p_tofNumStd, const std::string& pr_projPsf_fname,
    int p_num_rays, bool p_updateH)
    : binIter(pp_binIter),
      scanner(pr_scanner),
      projectorUpdaterType(p_projectorUpdaterType),
      tofWidth_ps(p_tofWidth_ps),
      tofNumStd(p_tofNumStd),
      projPsf_fname(pr_projPsf_fname),
      numRays(p_num_rays),
      updateH(p_updateH)
{
	if (tofWidth_ps > 0.f)
	{
		flagProjTOF = true;
	}
	else
	{
		flagProjTOF = false;
	}
}

OperatorProjectorParams::OperatorProjectorParams(const OperatorProjectorParams& other)
    : binIter(other.binIter)
      , scanner(other.scanner)
      , projectorUpdaterType(other.projectorUpdaterType)
      , flagProjTOF(other.flagProjTOF)
      , tofWidth_ps(other.tofWidth_ps)
      , tofNumStd(other.tofNumStd)
      , projPsf_fname(other.projPsf_fname)
      , numRays(other.numRays)
      , updateH(other.updateH)
{
	HBasis.bind(other.HBasis);
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
