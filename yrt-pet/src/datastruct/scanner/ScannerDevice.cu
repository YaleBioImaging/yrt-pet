/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/scanner/ScannerDevice.cuh"

#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/PageLockedBuffer.cuh"

namespace yrt
{
ScannerDevice::ScannerDevice(const Scanner& pr_scanner,
                             const cudaStream_t* pp_stream)
    : mr_scanner(pr_scanner), isAllocated(false), isLoaded(false)
{
	mpd_detPos = std::make_unique<DeviceArray<float4>>();
	mpd_detOrient = std::make_unique<DeviceArray<float4>>();
	// Constructors should be synchronized
	load({pp_stream, true});
}

void ScannerDevice::load(GPULaunchConfig p_launchConfig)
{
	if (!isAllocated)
	{
		allocate(p_launchConfig);
	}

	// We use a single large buffer to store two sets of data
	const size_t numDets = mr_scanner.getNumDets();
	PageLockedBuffer<float4> tempBuffer(numDets * 2);
	float4* ph_detPos = tempBuffer.getPointer();
	float4* ph_detOrient = ph_detPos + numDets;

	const auto detectorSetup_shared = mr_scanner.getDetectorSetup();
	const DetectorSetup* detectorSetup = detectorSetup_shared.get();

#pragma omp parallel for default(none) \
    firstprivate(ph_detPos, ph_detOrient, numDets, detectorSetup)
	for (size_t id_det = 0; id_det < numDets; id_det++)
	{
		ph_detPos[id_det].x = detectorSetup->getXpos(id_det);
		ph_detPos[id_det].y = detectorSetup->getYpos(id_det);
		ph_detPos[id_det].z = detectorSetup->getZpos(id_det);
		ph_detOrient[id_det].x = detectorSetup->getXorient(id_det);
		ph_detOrient[id_det].y = detectorSetup->getYorient(id_det);
		ph_detOrient[id_det].z = detectorSetup->getZorient(id_det);
	}

	mpd_detOrient->copyFromHost(ph_detOrient, numDets, p_launchConfig);
	mpd_detPos->copyFromHost(ph_detPos, numDets, p_launchConfig);
	isLoaded = true;
}

void ScannerDevice::allocate(GPULaunchConfig p_launchConfig)
{
	const size_t numDets = mr_scanner.getNumDets();
	mpd_detPos->allocate(numDets, p_launchConfig);
	mpd_detOrient->allocate(numDets, p_launchConfig);
	isAllocated = true;
}

const float4* ScannerDevice::getDetPosDevicePointer() const
{
	return mpd_detPos->getDevicePointer();
}
const float4* ScannerDevice::getDetOrientDevicePointer() const
{
	return mpd_detOrient->getDevicePointer();
}

const Scanner& ScannerDevice::getScanner() const
{
	return mr_scanner;
}
}  // namespace yrt
