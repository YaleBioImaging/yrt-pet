/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/OperatorProjectorDevice.cuh"
#include "recon/OSEMUpdater_GPU.cuh"

OSEMUpdater_GPU::OSEMUpdater_GPU(OSEM_GPU* pp_osem) : mp_osem(pp_osem)
{
	ASSERT(mp_osem != nullptr);
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

	// TODO: Add parallel CUDA streams here (They are currently all
	//  synchronized)

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		measurementsDevice->loadEventLORs(currentSubset, batch, imageParams,
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

		tmpBufferDevice->divideMeasurementsDevice(measurementsDevice, mainStream);

		projector->applyAH(tmpBufferDevice, &destImage);
	}
}
