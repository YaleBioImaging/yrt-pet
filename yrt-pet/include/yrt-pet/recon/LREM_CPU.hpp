/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/recon/LREM.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"

#include "yrt-pet/operators/OperatorProjectorBase.hpp"

namespace yrt
{
class LREM_CPU : public OSEM_CPU, public LREM
{
public:
	explicit LREM_CPU(const Scanner& pr_scanner);
	~LREM_CPU() override = default;

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
};
}  // namespace yrt
