/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageDevice.cuh"

class OSEM_GPU;

class OSEMUpdater_GPU
{
public:
	explicit OSEMUpdater_GPU(OSEM_GPU* pp_osem);

	// Iterates over all batches to compute the sensitivity image
	void computeSensitivityImage(ImageDevice& destImage) const;

	// Iterates over all batches to do the updates
	void computeEMUpdateImage(const ImageDevice& inputImage,
	                          ImageDevice& destImage) const;

private:
	OSEM_GPU* mp_osem;
};
