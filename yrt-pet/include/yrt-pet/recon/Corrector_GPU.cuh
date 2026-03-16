/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/image/ImageDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/recon/Corrector.hpp"

namespace yrt
{

class OperatorProjectorDevice;

class Corrector_GPU : public Corrector
{
public:
	explicit Corrector_GPU(const Scanner& pr_scanner);

	// For reconstruction:
	// Pre-computes the attenuation correction factors (ACF)
	//  for each LOR in measurements, but also using GPU for the attenuation
	//  image forward projection if needed
	void precomputeAttenuationFactors(
	    const ProjectionData& measurements) override;
	// Pre-computes a ProjectionList of a^(i)_i for each LOR in measurements,
	//  but also using GPU for the attenuation image forward projection if
	//  needed
	void precomputeInVivoAttenuationFactors(
	    const ProjectionData& measurements) override;
};
}  // namespace yrt
