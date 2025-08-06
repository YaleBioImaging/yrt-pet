/*
* This file is subject to the terms and conditions defined in
* file 'LICENSE.txt', which is part of this source code package.
*/

#include "yrt-pet/recon/OSEMUpdater.hpp"

#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/OperatorProjectorDD.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"
#include "yrt-pet/recon/OSEMUpdater_CPU.hpp"
#include "yrt-pet/recon/OSEM.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_CUDA
#include "yrt-pet/recon/OSEM_GPU.cuh"
#include "yrt-pet/recon/OSEMUpdater_GPU.cuh"
#endif

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace pybind11::literals;

namespace yrt
{
void py_setup_osem_updater(pybind11::module& m)
{
	auto c = py::class_<OSEMUpdater>(m, "OSEMUpdater");

	c.def("computeSensitivityImage",
	      [](OSEMUpdater& self, ImageBase& destImage) {
		      self.computeSensitivityImage(destImage);
	      },
	      py::arg("destImage"));
	c.def("computeEMUpdateImage",
	      [](OSEMUpdater& self, const ImageBase& inputImage, ImageBase& destImage) {
		      self.computeEMUpdateImage(inputImage, destImage);
	      },
	      py::arg("inputImage"), py::arg("destImage"));

	// factory function
	m.def("createOSEMUpdater", &yrt::createOSEMUpdater, py::arg("osem"));

	// Downcast helpers (return raw pointer, may be nullptr if cast fails)
	c.def("as_cpu",
	      [](OSEMUpdater& self) -> OSEMUpdater_CPU* {
		      return dynamic_cast<OSEMUpdater_CPU*>(&self);
	      },
	      py::return_value_policy::reference);

	// CPU subclass
	auto c_cpu = py::class_<OSEMUpdater_CPU, OSEMUpdater>(m, "OSEMUpdaterCPU");
	c_cpu.def(py::init<OSEM_CPU*>(), py::arg("osem_cpu"));
	c_cpu.def("computeSensitivityImage", [](OSEMUpdater_CPU& self, Image& dest)
	          { self.computeSensitivityImage(dest); });
	c_cpu.def("computeEMUpdateImage",
	          [](OSEMUpdater_CPU& self, const Image& in, Image& out)
	          { self.computeEMUpdateImage(in, out); });

#if BUILD_CUDA
	c.def("as_gpu",
	      [](OSEMUpdater& self) -> OSEMUpdater_GPU* {
		      return dynamic_cast<OSEMUpdater_GPU*>(&self);
	      },
	      py::return_value_policy::reference);

	auto c_gpu = py::class_<OSEMUpdater_GPU, OSEMUpdater>(m, "OSEMUpdaterGPU");
	c_gpu.def(py::init<OSEM_GPU*>(), py::arg("osem_gpu"));
	c_gpu.def("computeSensitivityImage",
	              [](OSEMUpdater_GPU& self, ImageDevice& dest)
	              { self.computeSensitivityImage(dest); });
	c_gpu.def("computeEMUpdateImage",
	          [](OSEMUpdater_GPU& self, const ImageDevice& in, ImageDevice& out)
	          { self.computeEMUpdateImage(in, out); });

#endif  // BUILD_CUDA endif
}

}

#endif

namespace yrt
{

std::unique_ptr<OSEMUpdater> createOSEMUpdater(OSEM* pp_osem)
{
	if (!pp_osem) {
		throw std::invalid_argument("OSEM pointer is null");
	}

	if (auto cpu = dynamic_cast<OSEM_CPU*>(pp_osem)) {
		return std::make_unique<OSEMUpdater_CPU>(cpu);
	}
#if BUILD_CUDA
	else if (auto gpu = dynamic_cast<OSEM_GPU*>(pp_osem)) {
		return std::make_unique<OSEMUpdater_GPU>(gpu);
	}
#endif

	throw std::runtime_error("Unsupported OSEM type for updater");
}

}