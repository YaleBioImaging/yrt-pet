/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/ReconstructionUtilsDevice.cuh"

#include "yrt-pet/datastruct/projection/DynamicFraming.hpp"
#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"
#include "yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh"
#include "yrt-pet/recon/LREM_GPU.cuh"
#include "yrt-pet/recon/OSEM_GPU.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_reconstructionutilsdevice(pybind11::module& m)
{
	m.def(
	    "createOperatorProjectorDevice",
	    [](const ProjectorParams& projParams, const BinIterator& binIter,
	       const std::vector<Constraint*>& constraints)
	    {
		    return util::createOperatorProjectorDevice(
		        projParams, binIter, constraints, nullptr, nullptr);
	    },
	    "proj_params"_a, "bin_iter"_a,
	    "constraints"_a = std::vector<Constraint*>());
}
}  // namespace yrt
#endif

namespace yrt::util
{

std::unique_ptr<OSEM> createOSEM_GPU(const Scanner& scanner, bool isLowRank)
{
	std::unique_ptr<OSEM> osem;
	if (!isLowRank)
	{
		osem = std::make_unique<OSEM_GPU>(scanner);
	}
	else
	{
		osem = std::make_unique<LREM_GPU>(scanner);
	}
	return osem;
}

std::unique_ptr<OperatorProjectorBase> createOperatorProjectorDevice(
    const ProjectorParams& projParams, const BinIterator& binIter,
    const std::vector<Constraint*>& constraintsPtr,
    const cudaStream_t* mainStream, const cudaStream_t* auxStream)
{
	const ProjectorType projType = projParams.projectorType;
	if (projType == ProjectorType::SIDDON)
	{
#ifdef BUILD_CUDA
		return std::make_unique<OperatorProjectorSiddon_GPU>(
		    projParams, &binIter, constraintsPtr, mainStream, auxStream);
#else
		throw std::runtime_error("Siddon GPU projector not supported because "
		                         "project was not compiled with CUDA");
#endif
	}
	else if (projType == ProjectorType::DD)
	{
#ifdef BUILD_CUDA
		return std::make_unique<OperatorProjectorDD_GPU>(
		    projParams, &binIter, constraintsPtr, mainStream, auxStream);
#else
		throw std::runtime_error(
		    "Distance-driven GPU projector not supported because "
		    "project was not compiled with CUDA");
#endif
	}
	else
	{
		throw std::runtime_error("Unknown error");
	}
}

}  // namespace yrt::util
