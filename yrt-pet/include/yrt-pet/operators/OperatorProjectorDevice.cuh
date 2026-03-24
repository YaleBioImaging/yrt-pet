/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/operators/DeviceSynchronized.cuh"
#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/operators/ProjectionPsfManagerDevice.cuh"
#include "yrt-pet/operators/Projector.hpp"
#include "yrt-pet/operators/ProjectorUpdaterDevice.cuh"
#include "yrt-pet/operators/TimeOfFlight.hpp"
#include "yrt-pet/utils/DeviceSynchronizedObject.cuh"
#include "yrt-pet/utils/GPUTypes.cuh"

namespace yrt
{
class OperatorProjectorDevice : public OperatorProjectorBase,
                                public DeviceSynchronized
{
public:
	OperatorProjectorDevice() = delete;

	static std::unique_ptr<OperatorProjectorDevice> create(
	    const ProjectorParams& pr_projParams, const BinIterator* pp_binIter,
	    const std::vector<Constraint*>& pr_constraints = {},
	    const cudaStream_t* pp_mainStream = nullptr,
	    const cudaStream_t* pp_auxStream = nullptr, size_t p_memAvailable_bytes = 0);

	// Aliases for uniformity
	void addTOF(float tofWidth_ps, int tofNumStd = -1);
	void addProjPSF(const std::string& projPsf_fname);

	size_t getBatchSize() const;
	void setupProjPsfManager(const std::string& psfFilename);

	std::set<ProjectionPropertyType>
	    getProjectionPropertyTypes(const ProjectionData* dataInput) const;
	unsigned int getGridSize() const;
	unsigned int getBlockSize() const;

	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	UpdaterPointer getUpdaterDevicePointer();
	void setupUpdater(const ProjectorParams& pr_projParams);
	ProjectorUpdaterDeviceWrapper* getUpdaterDeviceWrapper();
	bool hasUpdater() const;
	bool isDEFAULT4D() const;

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void applyA(const Variable* in, Variable* out, bool synchronize);
	void applyAH(const Variable* in, Variable* out, bool synchronize);

	static constexpr float DefaultMemoryShare = 0.9f;

protected:
	explicit OperatorProjectorDevice(
	    const ProjectorParams& pr_projParams,
	    const BinIterator* pp_binIter = nullptr,
	    const std::vector<Constraint*>& pr_constraints = {},
	    const cudaStream_t* pp_mainStream = nullptr,
	    const cudaStream_t* pp_auxStream = nullptr, size_t p_memAvailable_bytes = 0);

	// These must run on the main stream
	virtual void applyAOnLoadedBatch(ImageDevice& img,
	                                 ProjectionListDevice& dat,
	                                 bool synchronize) = 0;
	virtual void applyAHOnLoadedBatch(ProjectionListDevice& dat,
	                                  ImageDevice& img, bool synchronize) = 0;

	void setBatchSize(size_t newBatchSize);

	const TimeOfFlightHelper* getTOFHelperDevicePointer() const;
	const float* getProjPsfKernelsDevicePointer(bool flipped) const;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManagerDevice> mp_projPsfManager;

	// Time of flight
	std::unique_ptr<DeviceSynchronizedObject<TimeOfFlightHelper>> mp_tofHelper;

	// Updater for projection
	ProjectorUpdaterDeviceWrapper m_updaterContainer;

private:
	size_t m_batchSize;
	size_t m_memAvailable_bytes;
	GPULaunchParams m_launchParams{};

	// Bin iterator constraints
	std::vector<Constraint*> m_constraints;

	// This projector is only used as a shortcut to gather the correct
	//  projection properties, not to actually project
	std::unique_ptr<Projector> mp_projector;
};
}  // namespace yrt
