/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#include "yrt-pet/recon/Corrector_GPU.cuh"

#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

namespace yrt
{

Corrector_GPU::Corrector_GPU(const Scanner& pr_scanner) : Corrector(pr_scanner)
{
}

void Corrector_GPU::precomputeAttenuationFactors(
    const ProjectionData& measurements)
{
	ASSERT_MSG(hasAttenuation(), "No attenuation correction needed");

	auto attenuationFactors =
	    std::make_unique<ProjectionListOwned>(&measurements);
	attenuationFactors->allocate();

	mp_attenuationFactors = std::move(attenuationFactors);

	const bin_t numBins = measurements.count();

	if (doesTotalACFComeFromHistogram())
	{
		std::cout << "Gathering ACFs to prepare for attenuation correction..."
		          << std::endl;

		float* attenuationFactorsPtr = mp_attenuationFactors->getRawPointer();
		const ProjectionData* measurementsPtr = &measurements;

		util::parallelForChunked(numBins, globals::getNumThreads(),
		                         [attenuationFactorsPtr, measurementsPtr,
		                          this](bin_t bin, size_t /*tid*/)
		                         {
			                         const histo_bin_t histoBin =
			                             measurementsPtr->getHistogramBin(bin);

			                         attenuationFactorsPtr[bin] =
			                             getTotalACFFromHistogram(histoBin);
		                         });
	}
	else if (mp_attenuationImage != nullptr)
	{
		std::cout << "Forward projecting attenuation image..." << std::endl;

		util::forwProject(measurements.getScanner(), *mp_attenuationImage,
		                  *mp_attenuationFactors, ProjectorType::SIDDON, true);

		// TODO: This part would be faster if done on GPU (inside the projection
		//  kernel)
		std::cout << "Computing attenuation coefficient factors..."
		          << std::endl;
		util::convertProjectionValuesToACF(*mp_attenuationFactors);
	}
}

void Corrector_GPU::precomputeInVivoAttenuationFactors(
    const ProjectionData& measurements)
{
	ASSERT(hasInVivoAttenuation());

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(&measurements);
	inVivoAttenuationFactors->allocate();

	mp_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);

	if (doesInVivoACFComeFromHistogram())
	{
		std::cout << "Gathering in-vivo ACFs to prepare for precorrection..."
		          << std::endl;

		const bin_t numBins = measurements.count();
		float* inVivoAttenuationFactorsPtr =
		    mp_inVivoAttenuationFactors->getRawPointer();

		util::parallelForChunked(
		    numBins, globals::getNumThreads(),
		    [numBins, &measurements, inVivoAttenuationFactorsPtr,
		     this](bin_t bin, size_t /*tid*/)
		    {
			    const histo_bin_t histoBin = measurements.getHistogramBin(bin);
			    inVivoAttenuationFactorsPtr[bin] =
			        mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
		    });
	}
	else if (mp_inVivoAttenuationImage != nullptr)
	{
		std::cout << "Forward-projecting in-vivo attenuation image to prepare "
		             "for precorrection..."
		          << std::endl;
		util::forwProject(measurements.getScanner(), *mp_inVivoAttenuationImage,
		                  *mp_inVivoAttenuationFactors, ProjectorType::SIDDON,
		                  true);

		// TODO: This part would be faster if done on GPU (inside the projection
		//  kernel)
		std::cout << "Computing attenuation coefficient factors..."
		          << std::endl;
		util::convertProjectionValuesToACF(*mp_inVivoAttenuationFactors);
	}
	else
	{
		// Not supposed to reach here
		ASSERT_MSG(false, "Unexpected error");
	}
}

}  // namespace yrt
