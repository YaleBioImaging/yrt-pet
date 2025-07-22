/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/Corrector_CPU.hpp"

#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Tools.hpp"

namespace yrt
{

Corrector_CPU::Corrector_CPU(const Scanner& pr_scanner) : Corrector(pr_scanner)
{
}

void Corrector_CPU::precomputeAdditiveCorrectionFactors(
    const ProjectionData& measurements)
{
	ASSERT_MSG(hasAdditiveCorrection(measurements),
	           "No additive corrections needed");

	const ProjectionData* measurementsPtr = &measurements;

	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(measurementsPtr);
	additiveCorrections->allocate();

	mp_additiveCorrections = std::move(additiveCorrections);
	float* additiveCorrectionsPtr = mp_additiveCorrections->getRawPointer();

	const bin_t numBins = measurements.count();
	std::cout << "Precomputing additive corrections..." << std::endl;

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurementsPtr, additiveCorrectionsPtr)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		additiveCorrectionsPtr[bin] =
		    getAdditiveCorrectionFactor(*measurementsPtr, bin);
	}
}

void Corrector_CPU::precomputeInVivoAttenuationFactors(
    const ProjectionData& measurements)
{
	ASSERT_MSG(hasInVivoAttenuation(),
	           "No in-vivo attenuation corrections needed");

	const ProjectionData* measurementsPtr = &measurements;

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(measurementsPtr);
	inVivoAttenuationFactors->allocate();

	mp_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);
	float* inVivoAttenuationFactorsPtr =
	    mp_inVivoAttenuationFactors->getRawPointer();

	const size_t numBins = measurements.count();
	std::cout << "Precomputing in-vivo attenuation corrections..." << std::endl;

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurementsPtr, inVivoAttenuationFactorsPtr)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		inVivoAttenuationFactorsPtr[bin] =
		    getInVivoAttenuationFactor(*measurementsPtr, bin);
	}
}

float Corrector_CPU::getMultiplicativeCorrectionFactor(
    const ProjectionData& measurements, bin_t binId) const
{
	if (hasMultiplicativeCorrection())
	{
		const histo_bin_t histoBin = measurements.getHistogramBin(binId);

		const float sensitivity = getSensitivity(histoBin);

		float acf;
		if (mp_hardwareAcf != nullptr)
		{
			// Hardware ACF
			acf = mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
		}
		else if (mp_hardwareAttenuationImage != nullptr)
		{
			acf = getAttenuationFactorFromAttenuationImage(
			    measurements, binId, *mp_hardwareAttenuationImage);
		}
		else
		{
			acf = 1.0f;
		}

		return acf * sensitivity;
	}
	return m_globalScalingFactor;
}

float Corrector_CPU::getAdditiveCorrectionFactor(
    const ProjectionData& measurements, bin_t binId) const
{
	const histo_bin_t histoBin = measurements.getHistogramBin(binId);

	const float randomsEstimate =
	    getRandomsEstimate(measurements, binId, histoBin);

	const float scatterEstimate = getScatterEstimate(histoBin);

	const float sensitivity = getSensitivity(histoBin);

	float acf;
	if (doesTotalACFComeFromHistogram())
	{
		acf = getTotalACFFromHistogram(histoBin);
	}
	else if (mp_attenuationImage != nullptr)
	{
		acf = getAttenuationFactorFromAttenuationImage(measurements, binId,
		                                               *mp_attenuationImage);
	}
	else
	{
		acf = 1.0f;
	}

	if (acf < StabilityEpsilon || sensitivity < StabilityEpsilon)
	{
		// To avoid numerical instability
		return 0.0f;
	}

	return (randomsEstimate + scatterEstimate) / (acf * sensitivity);
}

float Corrector_CPU::getInVivoAttenuationFactor(
    const ProjectionData& measurements, bin_t binId) const
{
	const histo_bin_t histoBin = measurements.getHistogramBin(binId);

	if (mp_inVivoAcf != nullptr)
	{
		return mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	if (mp_inVivoAttenuationImage != nullptr)
	{
		return getAttenuationFactorFromAttenuationImage(
		    measurements, binId, *mp_inVivoAttenuationImage);
	}

	return 1.0f;
}

float Corrector_CPU::getAdditiveCorrectionFactor(bin_t binId) const
{
	ASSERT(mp_additiveCorrections != nullptr &&
	       mp_additiveCorrections->isMemoryValid());
	return mp_additiveCorrections->getRawPointer()[binId];
}

float Corrector_CPU::getInVivoAttenuationFactor(bin_t binId) const
{
	ASSERT(mp_inVivoAttenuationFactors != nullptr &&
	       mp_inVivoAttenuationFactors->isMemoryValid());
	return mp_inVivoAttenuationFactors->getRawPointer()[binId];
}

const ProjectionData*
    Corrector_CPU::getCachedMeasurementsForAdditiveCorrectionFactors() const
{
	if (mp_additiveCorrections != nullptr &&
	    mp_additiveCorrections->isMemoryValid())
	{
		return mp_additiveCorrections->getReference();
	}
	return nullptr;
}

const ProjectionData*
    Corrector_CPU::getCachedMeasurementsForInVivoAttenuationFactors() const
{
	if (mp_inVivoAttenuationFactors != nullptr &&
	    mp_inVivoAttenuationFactors->isMemoryValid())
	{
		return mp_inVivoAttenuationFactors->getReference();
	}
	return nullptr;
}

float Corrector_CPU::getAttenuationFactorFromAttenuationImage(
    const ProjectionData& measurements, bin_t binId,
    const Image& attenuationImage) const
{
	const Line3D lor = measurements.getLOR(binId);

	const float tofValue =
	    measurements.hasTOF() ? measurements.getTOFValue(binId) : 0.0f;

	const float att = OperatorProjectorSiddon::singleForwardProjection(
	    &attenuationImage, lor, mp_tofHelper.get(), tofValue);

	return util::getAttenuationCoefficientFactor(att);
}

}  // namespace yrt
