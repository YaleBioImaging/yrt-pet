/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/OperatorProjectorDevice.cuh"
#include "recon/OSEMUpdater_GPU.cuh"
#include "recon/OSEM_GPU.cuh"

OSEMUpdater_GPU::OSEMUpdater_GPU(OSEM_GPU* pp_osem) : mp_osem(pp_osem)
{
	ASSERT(mp_osem != nullptr);
}

void OSEMUpdater_GPU::computeSensitivityImage(ImageDevice& destImage) const
{
	OperatorProjectorDevice* projector = mp_osem->getProjector();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	const ImageParams& imageParams = mp_osem->getImageParams();
	Corrector_GPU& corrector = mp_osem->getCorrector_GPU();

	const cudaStream_t* mainStream = mp_osem->getMainStream();
	const cudaStream_t* auxStream = mp_osem->getMainStream();

	ProjectionDataDeviceOwned* sensDataBuffer =
	    mp_osem->getSensitivityDataDeviceBuffer();
	const int numBatchesInCurrentSubset =
	    sensDataBuffer->getNumBatches(currentSubset);

	bool loadGlobalScalingFactor = !corrector.hasMultiplicativeCorrection();

	// TODO: do parallel batch loading here

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Batch " << batch + 1 << "/" << numBatchesInCurrentSubset
		          << "..." << std::endl;
		// Load LORs into device buffers
		sensDataBuffer->loadEventLORs(currentSubset, batch,
		                              auxStream);
		// Allocate for the projection values
		const bool hasReallocated =
		    sensDataBuffer->allocateForProjValues(auxStream);

		// Load the projection values to backproject
		// This will either load projection values from sensitivity histogram,
		//  from ACF histogram, or it will load "ones" from a uniform histogram
		sensDataBuffer->loadProjValuesFromReference(auxStream);

		// Load the projection values to the device buffer depending on the
		//  situation
		if (corrector.hasSensitivityHistogram())
		{
			// Apply global scaling factor if it's not 1.0
			if (corrector.hasGlobalScalingFactor())
			{
				sensDataBuffer->multiplyProjValues(
				    corrector.getGlobalScalingFactor(), auxStream);
			}

			// Invert sensitivity if needed
			if (corrector.mustInvertSensitivity())
			{
				sensDataBuffer->invertProjValuesDevice(auxStream);
			}
		}
		if (corrector.hasHardwareAttenuationImage())
		{
			corrector.applyHardwareAttenuationImageFactorsToGivenDeviceBuffer(
			    sensDataBuffer, projector, auxStream);
		}
		else if (corrector.doesHardwareACFComeFromHistogram())
		{
			corrector.applyHardwareACFCorrectionFactorsToGivenDeviceBuffer(
			    sensDataBuffer, auxStream);
		}

		if (!corrector.hasMultiplicativeCorrection() &&
		    (loadGlobalScalingFactor || hasReallocated))
		{
			// Need to set all bins to the global scaling factor value, but only
			//  do it the first time (unless a reallocation has occured)
			sensDataBuffer->clearProjections(
			    corrector.getGlobalScalingFactor());
			loadGlobalScalingFactor = false;
		}

		// Backproject values
		projector->applyAH(sensDataBuffer, &destImage);
	}
}

void OSEMUpdater_GPU::computeEMUpdateImage(const ImageDevice& inputImage,
                                           ImageDevice& destImage) const
{
	OperatorProjectorDevice* projector = mp_osem->getProjector();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	const ImageParams& imageParams = mp_osem->getImageParams();
	Corrector_GPU& corrector = mp_osem->getCorrector_GPU();

	const cudaStream_t* mainStream = mp_osem->getMainStream();
	const cudaStream_t* auxStream = mp_osem->getMainStream();

	ProjectionDataDeviceOwned* measurementsDevice =
	    mp_osem->getMLEMDataDeviceBuffer();
	ProjectionDataDeviceOwned* tmpBufferDevice =
	    mp_osem->getMLEMDataTmpDeviceBuffer();
	const ProjectionDataDevice* correctorTempBuffer =
	    corrector.getTemporaryDeviceBuffer();

	ASSERT(projector != nullptr);
	ASSERT(measurementsDevice != nullptr);
	ASSERT(tmpBufferDevice != nullptr);
	ASSERT(destImage.isMemoryValid());

	const int numBatchesInCurrentSubset =
	    measurementsDevice->getNumBatches(currentSubset);

	// TODO: Use parallel CUDA streams here (They are currently all
	//  synchronized)

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Batch " << batch + 1 << "/" << numBatchesInCurrentSubset
		          << "..." << std::endl;
		measurementsDevice->loadEventLORs(currentSubset, batch,
		                                  auxStream);

		measurementsDevice->allocateForProjValues(auxStream);
		measurementsDevice->loadProjValuesFromReference(auxStream);

		tmpBufferDevice->allocateForProjValues(auxStream);

		projector->applyA(&inputImage, tmpBufferDevice);

		if (corrector.hasAdditiveCorrection())
		{
			corrector.loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
			    auxStream);
			tmpBufferDevice->addProjValues(correctorTempBuffer, mainStream);
		}
		if (corrector.hasInVivoAttenuation())
		{
			corrector.loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
			    auxStream);
			tmpBufferDevice->multiplyProjValues(correctorTempBuffer,
			                                    mainStream);
		}

		tmpBufferDevice->divideMeasurementsDevice(measurementsDevice,
		                                          mainStream);

		projector->applyAH(tmpBufferDevice, &destImage);
	}
}
