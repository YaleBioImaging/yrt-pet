/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/ProjectorParams.hpp"

#include "yrt-pet/datastruct/IO.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_projectorparams(py::module& m)
{
	auto c = py::class_<ProjectorParams>(m, "ProjectorParams");
	c.def(py::init<Scanner&>(), py::arg("scanner"));

	py::enum_<UpdaterType>(m, "UpdaterType")
	    .value("DEFAULT4D", UpdaterType::DEFAULT4D)
	    .value("LR", UpdaterType::LR)
	    .value("LRDUALUPDATE", UpdaterType::LRDUALUPDATE)
	    .export_values();

	py::enum_<ProjectorType>(c, "ProjectorType")
	    .value("SIDDON", ProjectorType::SIDDON)
	    .value("DD", ProjectorType::DD)
	    .export_values();

	c.def("setProjector", static_cast<void (ProjectorParams::*)(
	                          const std::string& projectorName)>(
	                          &ProjectorParams::setProjector));
	c.def("setProjector",
	      static_cast<void (ProjectorParams::*)(ProjectorType projType)>(
	          &ProjectorParams::setProjector));

	c.def_property(
	    "HBasis",
	    // getter: const ref to the alias; tie lifetime to parent
	    [](ProjectorParams& p) -> Array2DBase<float>& { return p.getHBasis(); },
	    // setter: accept any 2D array base and bind alias to it
	    [](ProjectorParams& p, Array2DBase<float>& src)
	    {
		    auto dims = src.getDims();
		    p.bindHBasis(src.getRawPointer(), dims[0],
		                 dims[1]);  // NO allocation, just alias the source
	    },
	    py::return_value_policy::reference_internal);

	c.def(
	    "setHBasisFromNumpy",
	    [](ProjectorParams& self, py::buffer& np_data)
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

	c.def_readwrite("updaterType", &ProjectorParams::updaterType);
	c.def_readwrite("updateH", &ProjectorParams::updateH);
	c.def("addTOF", &ProjectorParams::addTOF, "tof_width_ps"_a,
	      "tof_num_std"_a);
	c.def_readwrite("projPsf_fname", &ProjectorParams::projPsf_fname);
	c.def_readwrite("numRays", &ProjectorParams::numRays);
	c.def_readwrite("projPropertyTypesExtra",
	                &ProjectorParams::projPropertyTypesExtra);
}
}  // namespace yrt

#endif

namespace yrt
{

ProjectorParams::ProjectorParams(const Scanner& pr_scanner)
    : scanner(pr_scanner),
      projectorType(ProjectorType::SIDDON),
      projPsf_fname(""),
      numRays(1),
      updaterType(UpdaterType::DEFAULT4D),
      updateH(false),
      m_tofWidth_ps(0.f),
      m_tofNumStd(0)
{
}

ProjectorParams::ProjectorParams(const ProjectorParams& other)
    : scanner(other.scanner),
      projectorType(other.projectorType),
      projPsf_fname(other.projPsf_fname),
      numRays(other.numRays),
      updaterType(other.updaterType),
      updateH(other.updateH),
      projPropertyTypesExtra(other.projPropertyTypesExtra),
      m_tofWidth_ps(other.m_tofWidth_ps),
      m_tofNumStd(other.m_tofNumStd)
{
	HBasis.bind(other.HBasis);
}

void ProjectorParams::setProjector(const std::string& projectorName)
{
	const ProjectorType projType = io::getProjector(projectorName);
	setProjector(projType);
}

void ProjectorParams::setProjector(ProjectorType projType)
{
	projectorType = projType;
}

void ProjectorParams::addTOF(float tofWidth_ps, int tofNumStd)
{
	m_tofWidth_ps = tofWidth_ps;
	m_tofNumStd = tofNumStd;
	projPropertyTypesExtra.insert(ProjectionPropertyType::TOF);
}

void ProjectorParams::removeTOF()
{
	m_tofWidth_ps = 0.0f;
	m_tofNumStd = 0;
	projPropertyTypesExtra.erase(ProjectionPropertyType::TOF);
}

float ProjectorParams::getTOFWidth_ps() const
{
	return m_tofWidth_ps;
}

int ProjectorParams::getTOFNumStd() const
{
	return m_tofNumStd;
}

bool ProjectorParams::hasTOF() const
{
	return m_tofWidth_ps > 0.f;
}

void ProjectorParams::bindHBasis(float* HBasis_ptr, size_t rank, size_t T)
{
	HBasis.bind(HBasis_ptr, rank, T);
}

Array2DBase<float>& ProjectorParams::getHBasis()
{
	return HBasis;
}

}  // namespace yrt