/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/OperatorProjector.hpp"
#include "recon/OSEM_CPU.hpp"

class ProjectionData;

class OSEMUpdater_CPU
{
public:
	explicit OSEMUpdater_CPU(OSEM_CPU* pp_osem);

	/*
	 * This function computes the image that will be used in the EM update
	 * (after the PSF forward has been applied and before the PSF backwards is
	 * to be applied)
	 */
	void computeEMUpdateImage(const Image& inputImage, Image& destImage) const;

	// TODO NOW: Sensitivity image generation
private:
	OSEM_CPU* mp_osem;
};
