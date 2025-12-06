/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/PluginFramework.hpp"

#if BUILD_PYBIND11

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{
void py_setup_array(py::module&);
void py_setup_vector3dall(py::module&);
void py_setup_line3dall(py::module&);
void py_setup_tubeofresponse(py::module& m);
void py_setup_timeofflight(py::module& m);
void py_setup_multiraygenerator(py::module& m);
void py_setup_utilities(py::module& m);
void py_setup_utilities_rangelist(py::module& m);

void py_setup_variable(py::module& m);
void py_setup_imagebase(py::module&);
void py_setup_imageparams(py::module&);
void py_setup_image(py::module&);
void py_setup_projectiondata(py::module& m);
void py_setup_biniterator(py::module& m);
void py_setup_histogram(py::module& m);
void py_setup_histogram3d(py::module& m);
void py_setup_uniformhistogram(py::module& m);
void py_setup_sparsehistogram(py::module& m);
void py_setup_lormotion(py::module& m);
void py_setup_listmode(py::module& m);
void py_setup_listmodelut(py::module& m);
void py_setup_listmodelutdoi(py::module& m);
void py_setup_projectionlist(py::module& m);
void py_setup_detectormask(py::module& m);
void py_setup_detectorsetup(py::module& m);
void py_setup_osem(py::module& m);
void py_setup_reconstructionutils(py::module& m);
void py_setup_scanner(py::module& m);
void py_setup_detcoord(py::module& m);
void py_setup_detregular(py::module& m);
void py_setup_io(py::module& m);

void py_setup_srtm(py::module& m);

void py_setup_operator(py::module& m);
void py_setup_operatorpsf(py::module& m);
void py_setup_operatorvarpsf(py::module& m);
void py_setup_operatorprojectorparams(py::module& m);
void py_setup_projectionpropertytype(py::module& m);
void py_setup_binfilter(py::module& m);
void py_setup_constraints(py::module& m);
void py_setup_operatorprojectorbase(py::module& m);
void py_setup_operatorprojector(py::module& m);
void py_setup_operatorprojectorsiddon(py::module& m);
void py_setup_operatorprojectordd(py::module& m);

void py_setup_globals(py::module& m);
void py_setup_log(py::module& m);

void py_setup_crystal(py::module& m);
void py_setup_singlescattersimulator(py::module& m);
void py_setup_scatterestimator(py::module& m);
void py_setup_scatterspace(py::module& m);

#ifdef BUILD_CUDA
void py_setup_gpuutils(py::module&);
void py_setup_imagedevice(py::module&);
void py_setup_projectiondatadevice(py::module& m);
void py_setup_operatorpsfdevice(py::module& m);
void py_setup_operatorvarpsfdevice(py::module& m);
void py_setup_operatorprojectordevice(py::module& m);
void py_setup_operatorprojectordd_gpu(py::module& m);
void py_setup_operatorprojectorsiddon_gpu(py::module& m);
void py_setup_reconstructionutilsdevice(py::module& m);
#endif


PYBIND11_MODULE(pyyrtpet, m)
{
	py_setup_array(m);
	py_setup_vector3dall(m);
	py_setup_line3dall(m);
	py_setup_tubeofresponse(m);
	py_setup_timeofflight(m);
	py_setup_multiraygenerator(m);

	py_setup_variable(m);
	py_setup_imagebase(m);
	py_setup_imageparams(m);
	py_setup_image(m);
	py_setup_biniterator(m);
	py_setup_projectiondata(m);
	py_setup_histogram(m);
	py_setup_histogram3d(m);
	py_setup_uniformhistogram(m);
	py_setup_sparsehistogram(m);
	py_setup_lormotion(m);
	py_setup_listmode(m);
	py_setup_listmodelut(m);
	py_setup_listmodelutdoi(m);
	py_setup_projectionlist(m);
	py_setup_detectormask(m);
	py_setup_detectorsetup(m);
	py_setup_scanner(m);
	py_setup_detcoord(m);
	py_setup_detregular(m);
	py_setup_io(m);

	py_setup_srtm(m);
	py_setup_utilities(m);
	py_setup_utilities_rangelist(m);

	py_setup_operator(m);
	py_setup_operatorpsf(m);
	py_setup_operatorvarpsf(m);
	py_setup_operatorprojectorbase(m);
	py_setup_operatorprojector(m);
	py_setup_operatorprojectorparams(m);
	py_setup_projectionpropertytype(m);
	py_setup_binfilter(m);
	py_setup_constraints(m);
	py_setup_operatorprojectorsiddon(m);
	py_setup_operatorprojectordd(m);
	py_setup_osem(m);
	py_setup_reconstructionutils(m);

	py_setup_globals(m);
	py_setup_log(m);

	py_setup_crystal(m);
	py_setup_singlescattersimulator(m);
	py_setup_scatterestimator(m);
	py_setup_scatterspace(m);

#ifdef BUILD_CUDA
	py_setup_gpuutils(m);
	py_setup_imagedevice(m);
	py_setup_projectiondatadevice(m);
	py_setup_operatorpsfdevice(m);
	py_setup_operatorvarpsfdevice(m);
	py_setup_operatorprojectordevice(m);
	py_setup_operatorprojectordd_gpu(m);
	py_setup_operatorprojectorsiddon_gpu(m);
	py_setup_reconstructionutilsdevice(m);
#endif

	// Add the plugins
	plugin::PluginRegistry::instance().addAllPybind11Modules(m);
}

}  // namespace yrt

#endif  // if BUILD_PYBIND11
