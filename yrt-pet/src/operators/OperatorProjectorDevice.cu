/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorProjectorDevice.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/GPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_operatorprojectordevice(py::module& m)
{
	auto c = py::class_<OperatorProjectorDevice, OperatorProjectorBase>(
	    m, "OperatorProjectorDevice");
}
#endif

OperatorProjectorDevice::OperatorProjectorDevice(
    const OperatorProjectorParams& pr_projParams, bool p_synchronized,
    const cudaStream_t* pp_mainStream, const cudaStream_t* pp_auxStream)
    : OperatorProjectorBase{pr_projParams},
      DeviceSynchronized{p_synchronized, pp_mainStream, pp_auxStream}
{
	if (pr_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(pr_projParams.tofWidth_ps, pr_projParams.tofNumStd);
	}
	if (!pr_projParams.psfProj_fname.empty())
	{
		setupProjPsfManager(pr_projParams.psfProj_fname);
	}

	m_batchSize = 0ull;
}

unsigned int OperatorProjectorDevice::getGridSize() const
{
	return m_launchParams.gridSize;
}

unsigned int OperatorProjectorDevice::getBlockSize() const
{
	return m_launchParams.blockSize;
}

bool OperatorProjectorDevice::isSynchronized() const
{
	return m_synchronized;
}

void OperatorProjectorDevice::setBatchSize(size_t newBatchSize)
{
	m_batchSize = newBatchSize;
	m_launchParams = Util::initiateDeviceParameters(m_batchSize);
}

size_t OperatorProjectorDevice::getBatchSize() const
{
	return m_batchSize;
}

void OperatorProjectorDevice::setupProjPsfManager(
    const std::string& psfFilename)
{
	mp_projPsfManager =
	    std::make_unique<ProjectionPsfManagerDevice>(psfFilename);
	ASSERT_MSG(mp_projPsfManager != nullptr,
	           "Error occured during the setup of ProjectionPsfManagerDevice");
}

void OperatorProjectorDevice::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<DeviceObject<TimeOfFlightHelper>>(
	    tofWidth_ps, tofNumStd);
}

const TimeOfFlightHelper*
    OperatorProjectorDevice::getTOFHelperDevicePointer() const
{
	if (mp_tofHelper != nullptr)
	{
		return mp_tofHelper->getDevicePointer();
	}
	return nullptr;
}

const float*
    OperatorProjectorDevice::getProjPsfKernelsDevicePointer(bool flipped) const
{
	if (mp_projPsfManager != nullptr)
	{
		if (!flipped)
		{
			return mp_projPsfManager->getKernelsDevicePointer();
		}
		return mp_projPsfManager->getFlippedKernelsDevicePointer();
	}
	return nullptr;
}
