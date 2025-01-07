/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorDevice.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/GPUUtils.cuh"
#include "utils/Globals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_operatorprojectordevice(py::module& m)
{
	auto c = py::class_<OperatorProjectorDevice, OperatorProjectorBase>(
	    m, "OperatorProjectorDevice");
}
#endif

namespace Util
{
	GPULaunchParams3D initiateDeviceParameters(const ImageParams& params)
	{
		GPULaunchParams3D launchParams;
		if (params.nz > 1)
		{
			const size_t threadsPerBlockDimImage =
			    GlobalsCuda::ThreadsPerBlockImg3d;
			const auto threadsPerBlockDimImage_float =
			    static_cast<float>(threadsPerBlockDimImage);
			const auto threadsPerBlockDimImage_uint =
			    static_cast<unsigned int>(threadsPerBlockDimImage);

			launchParams.gridSize = {
			    static_cast<unsigned int>(
			        std::ceil(params.nx / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.ny / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.nz / threadsPerBlockDimImage_float))};

			launchParams.blockSize = {threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint};
		}
		else
		{
			const size_t threadsPerBlockDimImage =
			    GlobalsCuda::ThreadsPerBlockImg2d;
			const auto threadsPerBlockDimImage_float =
			    static_cast<float>(threadsPerBlockDimImage);
			const auto threadsPerBlockDimImage_uint =
			    static_cast<unsigned int>(threadsPerBlockDimImage);

			launchParams.gridSize = {
			    static_cast<unsigned int>(
			        std::ceil(params.nx / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.ny / threadsPerBlockDimImage_float)),
			    1};

			launchParams.blockSize = {threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint, 1};
		}
		return launchParams;
	}

	GPULaunchParams initiateDeviceParameters(size_t batchSize)
	{
		GPULaunchParams launchParams{};
		launchParams.gridSize = static_cast<unsigned int>(std::ceil(
		    batchSize / static_cast<float>(GlobalsCuda::ThreadsPerBlockData)));
		launchParams.blockSize = GlobalsCuda::ThreadsPerBlockData;
		return launchParams;
	}
}  // namespace Util

const cudaStream_t* OperatorDevice::getMainStream() const
{
	return mp_mainStream;
}

const cudaStream_t* OperatorDevice::getAuxStream() const
{
	return mp_auxStream;
}

CUScannerParams OperatorDevice::getCUScannerParams(const Scanner& scanner)
{
	CUScannerParams params;
	params.crystalSize_trans = scanner.crystalSize_trans;
	params.crystalSize_z = scanner.crystalSize_z;
	params.numDets = scanner.getNumDets();
	return params;
}

CUImageParams OperatorDevice::getCUImageParams(const ImageParams& imgParams)
{
	CUImageParams params;

	params.voxelNumber[0] = imgParams.nx;
	params.voxelNumber[1] = imgParams.ny;
	params.voxelNumber[2] = imgParams.nz;

	params.imgLength[0] = static_cast<float>(imgParams.length_x);
	params.imgLength[1] = static_cast<float>(imgParams.length_y);
	params.imgLength[2] = static_cast<float>(imgParams.length_z);

	params.voxelSize[0] = static_cast<float>(imgParams.vx);
	params.voxelSize[1] = static_cast<float>(imgParams.vy);
	params.voxelSize[2] = static_cast<float>(imgParams.vz);

	params.offset[0] = static_cast<float>(imgParams.off_x);
	params.offset[1] = static_cast<float>(imgParams.off_y);
	params.offset[2] = static_cast<float>(imgParams.off_z);

	return params;
}

OperatorDevice::OperatorDevice(bool p_synchronized,
                               const cudaStream_t* pp_mainStream,
                               const cudaStream_t* pp_auxStream)
{
	m_synchronized = p_synchronized;
	mp_mainStream = pp_mainStream;
	mp_auxStream = pp_auxStream;
}

OperatorProjectorDevice::OperatorProjectorDevice(
    const OperatorProjectorParams& p_projParams, bool p_synchronized,
    const cudaStream_t* pp_mainStream, const cudaStream_t* pp_auxStream)
    : OperatorProjectorBase{p_projParams},
      OperatorDevice{p_synchronized, pp_mainStream, pp_auxStream}
{
	if (p_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(p_projParams.tofWidth_ps, p_projParams.tofNumStd);
	}
	if (!p_projParams.psfProjFilename.empty())
	{
		setupProjPsfManager(p_projParams.psfProjFilename);
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

ProjectionDataDeviceOwned& OperatorProjectorDevice::getIntermediaryProjData()
{
	ASSERT_MSG(mp_intermediaryProjData != nullptr,
	           "Projection-space GPU Intermediary buffer not initialized");
	return *mp_intermediaryProjData;
}

const ImageDevice& OperatorProjectorDevice::getAttImageDevice() const
{
	ASSERT_MSG(mp_attImageDevice != nullptr,
	           "Device attenuation image not initialized");
	return *mp_attImageDevice;
}

const ImageDevice&
    OperatorProjectorDevice::getAttImageForBackprojectionDevice() const
{
	ASSERT_MSG(mp_attImageForBackprojectionDevice != nullptr,
	           "Device attenuation image for backprojection not initialized");
	return *mp_attImageForBackprojectionDevice;
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

void OperatorProjectorDevice::setAttImageForForwardProjection(const Image* attImage)
{
	OperatorProjectorBase::setAttImageForForwardProjection(attImage);

	mp_attImageDevice = std::make_unique<ImageDeviceOwned>(
	    attImage->getParams(), getAuxStream());
	mp_attImageDevice->allocate(getAuxStream());
	mp_attImageDevice->transferToDeviceMemory(attImage, false);
}

void OperatorProjectorDevice::setAttImageForBackprojection(
    const Image* attImage)
{
	OperatorProjectorBase::setAttImageForBackprojection(attImage);

	mp_attImageForBackprojectionDevice = std::make_unique<ImageDeviceOwned>(
	    attImage->getParams(), getAuxStream());
	mp_attImageForBackprojectionDevice->allocate(getAuxStream());
	mp_attImageForBackprojectionDevice->transferToDeviceMemory(attImage, false);
}

void OperatorProjectorDevice::setAddHisto(const Histogram* p_addHisto)
{
	OperatorProjectorBase::setAddHisto(p_addHisto);
}

void OperatorProjectorDevice::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<DeviceObject<TimeOfFlightHelper>>(
	    tofWidth_ps, tofNumStd);
}

bool OperatorProjectorDevice::requiresIntermediaryProjData() const
{
	// We need an intermediary projectorParam if we'll need to do attenuation
	// correction or additive correction (scatter/randoms)
	return attImageForForwardProjection != nullptr ||
	       attImageForBackprojection != nullptr || addHisto != nullptr;
}

void OperatorProjectorDevice::prepareIntermediaryBufferIfNeeded(
    const ProjectionDataDevice* orig)
{
	if (requiresIntermediaryProjData())
	{
		prepareIntermediaryBuffer(orig);
	}
}

void OperatorProjectorDevice::prepareIntermediaryBuffer(
    const ProjectionDataDevice* orig)
{
	if (mp_intermediaryProjData == nullptr)
	{
		mp_intermediaryProjData =
		    std::make_unique<ProjectionDataDeviceOwned>(orig);
	}
	mp_intermediaryProjData->allocateForProjValues(getAuxStream());
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
