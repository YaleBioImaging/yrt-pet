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
	    const cudaStream_t* stream = nullptr);
	void loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
	    const cudaStream_t* stream = nullptr);

	// Getters
	const ProjectionDataDevice* getTemporaryDeviceBuffer() const;
	ProjectionDataDevice* getTemporaryDeviceBuffer();

	// Use ACF histogram for sensitivity image generation
	void applyHardwareACFCorrectionFactorsToGivenDeviceBuffer(
	    ProjectionDataDevice* destProjData, const cudaStream_t* stream);
	// The projector used is in case the attenuation image needs to be projected
	void applyHardwareAttenuationImageFactorsToGivenDeviceBuffer(
	    ProjectionDataDevice* destProjData, OperatorProjectorDevice* projector,
	    const cudaStream_t* stream = nullptr);

private:
	// Helper function
	void loadPrecomputedCorrectionFactorsToTemporaryDeviceBuffer(
	    const ProjectionList* factors, const cudaStream_t* stream = nullptr);
	void initializeTemporaryDeviceImageIfNeeded(
	    const Image* hostReference, const cudaStream_t* stream = nullptr);

	// Internal management
	void initializeTemporaryDeviceImageIfNeeded(const Image* hostReference);

	std::unique_ptr<ProjectionDataDeviceOwned> mpd_temporaryCorrectionFactors;

	std::unique_ptr<ImageDeviceOwned> mpd_temporaryImage;
	const Image* mph_lastCopiedHostImage;

	// Pre-computed caches on CPU-side
	std::unique_ptr<ProjectionList> mph_additiveCorrections;
	std::unique_ptr<ProjectionList> mph_inVivoAttenuationFactors;
};
