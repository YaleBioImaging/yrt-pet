/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/operators/OperatorProjectorUpdaterDevice.cuh"
#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/utils/DeviceObject.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"

namespace yrt
{
class OperatorProjectorDevice : public OperatorProjectorBase,
                                public DeviceSynchronized
{
public:
	OperatorProjectorDevice() = delete;

	size_t getBatchSize() const;
	void setupProjPsfManager(const std::string& psfFilename);

	unsigned int getGridSize() const;
	unsigned int getBlockSize() const;

	bool requiresIntermediaryProjData() const;
	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	const DeviceObject<OperatorProjectorUpdaterDevice>* getUpdater();
	const OperatorProjectorUpdaterDevice* getUpdaterDevicePointer();
	void setupUpdater(const OperatorProjectorParams& p_projParams);
	void setUpdater(DeviceObject<OperatorProjectorUpdaterDevice>&& pp_updater);


	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void applyA(const Variable* in, Variable* out, bool synchronize);
	void applyAH(const Variable* in, Variable* out, bool synchronize);

protected:
	explicit OperatorProjectorDevice(
	    const OperatorProjectorParams& pr_projParams,
	    const cudaStream_t* pp_mainStream = nullptr,
	    const cudaStream_t* pp_auxStream = nullptr);

	// These must run on the main stream
	virtual void applyAOnLoadedBatch(ImageDevice& img,
	                                 ProjectionDataDevice& dat,
	                                 bool synchronize) = 0;
	virtual void applyAHOnLoadedBatch(ProjectionDataDevice& dat,
	                                  ImageDevice& img, bool synchronize) = 0;

	void setBatchSize(size_t newBatchSize);

	const TimeOfFlightHelper* getTOFHelperDevicePointer() const;
	const float* getProjPsfKernelsDevicePointer(bool flipped) const;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManagerDevice> mp_projPsfManager;

private:
	size_t m_batchSize;
	GPULaunchParams m_launchParams{};

	// Time of flight
	std::unique_ptr<DeviceObject<TimeOfFlightHelper>> mp_tofHelper;

	// Updater for projection
	DeviceObject<OperatorProjectorUpdaterDevice> m_updater;

	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageDevice;
	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageForBackprojectionDevice;
	// For Attenuation correction or Additive correction
	std::unique_ptr<ProjectionDataDeviceOwned> mp_intermediaryProjData;
};
}  // namespace yrt
