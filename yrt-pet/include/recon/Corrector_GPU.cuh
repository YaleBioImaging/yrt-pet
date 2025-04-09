/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/OperatorProjectorDevice.cuh"
#include "recon/Corrector.hpp"


class Corrector_GPU : public Corrector
{
public:
	explicit Corrector_GPU(const Scanner& pr_scanner);

	// Set the reference to use for the temporary device buffer
	void initializeTemporaryDeviceBuffer(const ProjectionDataDevice* master);
	void clearTemporaryDeviceBuffer();

	// For reconstruction:
	// Pre-computes the additive correction factors
	//  (randoms+scatter)/(acf*sensitivity) for each LOR in measurements,
	//  but also using GPU for the attenuation image forward projection if
	//  needed
	void precomputeAdditiveCorrectionFactors(
	    const ProjectionData& measurements) override;
	// Pre-computes a ProjectionList of a^(i)_i for each LOR in measurements,
	//  but also using GPU for the attenuation image forward projection if
	//  needed
	void precomputeInVivoAttenuationFactors(
	    const ProjectionData& measurements) override;

	void loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
	    GPULaunchConfig launchConfig);
	void loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
	    GPULaunchConfig launchConfig);

	// Getters
	const ProjectionDataDevice* getTemporaryDeviceBuffer() const;
	ProjectionDataDevice* getTemporaryDeviceBuffer();

	// Use ACF histogram to apply hardware attenuation correction in the
	//  sensitivity image
	void applyHardwareAttenuationToGivenDeviceBufferFromACFHistogram(
	    ProjectionDataDevice* destProjData, GPULaunchConfig launchConfig);

	// Use attenuation image to apply hardware attenuation correction in the
	//  sensitivity image
	void applyHardwareAttenuationToGivenDeviceBufferFromAttenuationImage(
	    ProjectionDataDevice* destProjData, OperatorProjectorDevice* projector,
	    GPULaunchConfig launchConfig);

	void initializeTemporaryDeviceImageIfNeeded(const Image* hostReference,
	                                            GPULaunchConfig launchConfig);

private:
	// Helper function
	void loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
	    const ProjectionList* factors, GPULaunchConfig launchConfig);

	std::unique_ptr<ProjectionDataDeviceOwned> mpd_temporaryCorrectionFactors;

	std::unique_ptr<ImageDeviceOwned> mpd_temporaryImage;
	const Image* mph_lastCopiedHostImage;

	// Pre-computed caches on CPU-side
	std::unique_ptr<ProjectionList> mph_additiveCorrections;
	std::unique_ptr<ProjectionList> mph_inVivoAttenuationFactors;
};
