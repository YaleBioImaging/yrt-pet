/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/recon/OSEMUpdater_GPU.cuh"
#include "yrt-pet/recon/OSEM_GPU.cuh"

namespace yrt
{

OSEMUpdater_GPU::OSEMUpdater_GPU(OSEM_GPU* pp_osem) : mp_osem(pp_osem)
{
	ASSERT(mp_osem != nullptr);
}

void OSEMUpdater_GPU::computeSensitivityImage(ImageDevice& destImage) const
{
	OperatorProjectorDevice* projector = mp_osem->getProjector();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	Corrector_GPU& corrector = mp_osem->getCorrector_GPU();

	const cudaStream_t* mainStream = mp_osem->getMainStream();
	const cudaStream_t* auxStream = mp_osem->getAuxStream();

	ProjectionDataDeviceOwned* sensDataBuffer =
	    mp_osem->getSensitivityDataDeviceBuffer();
	const int numBatchesInCurrentSubset =
	    sensDataBuffer->getNumBatches(currentSubset);
	const BinFilter* binFilter = projector->getBinFilter();

	bool loadGlobalScalingFactor = !corrector.hasMultiplicativeCorrection();

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Loading batch " << batch + 1 << "/"
		          << numBatchesInCurrentSubset << "..." << std::endl;

		sensDataBuffer->precomputeBatchLORs(currentSubset, batch, *binFilter);

		// Allocate for the projection values
		const bool hasReallocated =
		    sensDataBuffer->allocateForProjValues({mainStream, false});

		sensDataBuffer->loadPrecomputedLORsToDevice({mainStream, false});

		// Load the projection values to backproject
		// This will either load projection values from sensitivity histogram,
		//  from ACF histogram, or it will load "ones" from a uniform histogram
		sensDataBuffer->loadProjValuesFromReference({mainStream, false});

		// Load the projection values to the device buffer depending on the
		//  situation
		if (corrector.hasSensitivityHistogram())
		{
			// Apply global scaling factor if it's not 1.0
			if (corrector.hasGlobalScalingFactor())
			{
				sensDataBuffer->multiplyProjValues(
				    corrector.getGlobalScalingFactor(), {mainStream, false});
			}

			// Invert sensitivity if needed
			if (corrector.mustInvertSensitivity())
			{
				sensDataBuffer->invertProjValuesDevice({mainStream, false});
			}
		}
		if (corrector.hasHardwareAttenuationImage())
		{
			// TODO: it would be faster if this was done on the auxiliary stream
			//  so that a backprojection is done at the same time as a forward
			corrector
			    .applyHardwareAttenuationToGivenDeviceBufferFromAttenuationImage(
			        sensDataBuffer, projector, {mainStream, false});
		}
		else if (corrector.doesHardwareACFComeFromHistogram())
		{
			corrector
			    .applyHardwareAttenuationToGivenDeviceBufferFromACFHistogram(
			        sensDataBuffer, {mainStream, false});
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

		if (mainStream != nullptr)
		{
			cudaStreamSynchronize(*mainStream);
		}

		std::cout << "Backprojecting batch " << batch + 1 << "/"
		          << numBatchesInCurrentSubset << "..." << std::endl;

		// Backproject values
		{
			printf("\nDEBUG: In computeSensitivityImage, cudaCheckError before applyAH.\n");
			cudaCheckError();
			cudaDeviceSynchronize();
		}
		projector->applyAH(sensDataBuffer, &destImage, false);
		{
			printf("\nDEBUG: In computeSensitivityImage, cudaCheckError after applyAH.\n");
			cudaCheckError();
			cudaDeviceSynchronize();
		}
	}

	if (mainStream != nullptr)
	{
		cudaStreamSynchronize(*mainStream);
	}
}

void OSEMUpdater_GPU::computeEMUpdateImage(const ImageDevice& inputImage,
                                           ImageDevice& destImage) const
{
	{
		printf("\nDEBUG: Entering In computeEMUpdateImage\n");
		cudaCheckError();
		cudaDeviceSynchronize();
	}

	OperatorProjectorDevice* projector = mp_osem->getProjector();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	Corrector_GPU& corrector = mp_osem->getCorrector_GPU();

	const cudaStream_t* mainStream = mp_osem->getMainStream();
	const cudaStream_t* auxStream = mp_osem->getAuxStream();

	{
		printf("\nDEBUG: In computeEMUpdateImage, before ProjectionDataDeviceOwned\n");
		cudaCheckError();
		cudaDeviceSynchronize();
	}

	ProjectionDataDeviceOwned* measurementsDevice =
	    mp_osem->getMLEMDataDeviceBuffer();
	{
		printf("\nDEBUG: In computeEMUpdateImage, after measurementsDevice\n");
		cudaCheckError();
		cudaDeviceSynchronize();
	}
	ProjectionDataDeviceOwned* tmpBufferDevice =
	    mp_osem->getMLEMDataTmpDeviceBuffer();
	{
		printf("\nDEBUG: In computeEMUpdateImage, after tmpBufferDevice\n");
		cudaCheckError();
		cudaDeviceSynchronize();
	}
	const ProjectionDataDevice* correctorTempBuffer =
	    corrector.getTemporaryDeviceBuffer();
	{
		printf("\nDEBUG: In computeEMUpdateImage, after correctorTempBuffer\n");
		cudaCheckError();
		cudaDeviceSynchronize();
	}
	const BinFilter* binFilter = projector->getBinFilter();
	{
		printf("\nDEBUG: In computeEMUpdateImage, after binFilter\n");
		cudaCheckError();
		cudaDeviceSynchronize();
	}

	ASSERT(projector != nullptr);
	ASSERT(measurementsDevice != nullptr);
	ASSERT(tmpBufferDevice != nullptr);
	ASSERT(destImage.isMemoryValid());

	const int numBatchesInCurrentSubset =
	    measurementsDevice->getNumBatches(currentSubset);

	{
		printf("\nDEBUG: In computeEMUpdateImage, %d.\n", numBatchesInCurrentSubset);
		cudaCheckError();
		cudaDeviceSynchronize();
	}

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Batch " << batch + 1 << "/" << numBatchesInCurrentSubset
		          << "..." << std::endl;
		measurementsDevice->precomputeBatchLORs(currentSubset, batch,
		                                        *binFilter);

		measurementsDevice->allocateForProjValues({mainStream, false});
		measurementsDevice->loadPrecomputedLORsToDevice({mainStream, false});
		measurementsDevice->loadProjValuesFromReference({mainStream, false});

		tmpBufferDevice->allocateForProjValues({mainStream, false});

		{
			printf("\nDEBUG: In computeEMUpdateImage, before applyA\n");
			cudaCheckError();
			cudaDeviceSynchronize();
		}

		projector->applyA(&inputImage, tmpBufferDevice, false);

		{
			printf("\nDEBUG: In computeEMUpdateImage, after applyA\n");
			cudaCheckError();
			cudaDeviceSynchronize();
		}

		if (corrector.hasAdditiveCorrection(*measurementsDevice))
		{
			corrector.loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
			    {mainStream, false});

			// We need to synchronize the stream here if we want to apply both
			// the additive corrections AND the pre-correction. This is because
			// they both use the same buffer. If this is not done, and the GPU
			// forward projection takes longer than both
			// `loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(...)` and
			// `loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(...)`, the
			// code would end up treating the precorrection factors as additive
			// factors...
			const bool synchronize = corrector.hasInVivoAttenuation();

			tmpBufferDevice->addProjValues(correctorTempBuffer,
			                               {mainStream, synchronize});
		}
		if (corrector.hasInVivoAttenuation())
		{
			corrector.loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
			    {mainStream, false});
			tmpBufferDevice->multiplyProjValues(correctorTempBuffer,
			                                    {mainStream, false});
		}

		tmpBufferDevice->divideMeasurementsDevice(measurementsDevice,
		                                          {mainStream, false});

		if (mainStream != nullptr)
		{
			cudaStreamSynchronize(*mainStream);
		}

		projector->applyAH(tmpBufferDevice, &destImage, false);
	}

	if (mainStream != nullptr)
	{
		cudaStreamSynchronize(*mainStream);
	}
}
}  // namespace yrt
