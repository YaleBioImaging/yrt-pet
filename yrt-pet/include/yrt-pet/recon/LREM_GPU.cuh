/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/recon/LREM.hpp"
#include "yrt-pet/recon/OSEM_GPU.cuh"

namespace yrt
{

class LREM_GPU : public OSEM_GPU, public LREM
{
public:
	explicit LREM_GPU(const Scanner& pr_scanner);
	~LREM_GPU() override = default;

	void prepareBuffersForRecon() override;
	void resetEMUpdateImage() override;
	void applyImageUpdate() override;
	void completeSubset() override;
	void setupForDynamicRecon() override;
	void saveForCurrentIteration() override;

	std::unique_ptr<Image>
	    reconstructAlternating(int numWUpdatesPerIteration,
	                           int numHUpdatesPerIteration,
	                           const std::string& out_fname = "") override;
	void prepareForUpdatePhase(bool updateH) override;

protected:
	ProjectorParams& getProjectorParams() override;
	ImageBase* getEMUpdateImageBuffer() override;
	void setupProjectorUpdater(const ProjectorParams& params) override;

private:
	void generateHUpdateSensScalingInternal();
	void sync_cWUpdateDeviceToHost();  // TODO: Uniformize name
	void sync_cWUpdateHostToDevice();
	void syncHostToDeviceHBasis();
	void syncDeviceToHostHBasis();
	void syncHostToDeviceHBasisWrite();
	void syncDeviceToHostHBasisWrite();

	// LR sensitivity matrix factor correction
	DeviceArray<float> m_cWUpdateDevice;
	DeviceArray<float> m_cHUpdateDevice;
};

}  // namespace yrt
