/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector_CPU.hpp"

#include "operators/OperatorProjectorSiddon.hpp"
#include "utils/Assert.hpp"
#include "utils/Tools.hpp"


void Corrector_CPU::precomputeAdditiveCorrectionFactors(
    const ProjectionData* measurements)
{
	ASSERT(measurements != nullptr);

	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(measurements);
	additiveCorrections->allocate();

	mp_additiveCorrections = std::move(additiveCorrections);
	float* additiveCorrectionsPtr = mp_additiveCorrections->getRawPointer();

	const size_t numBins = measurements->count();

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurements, additiveCorrectionsPtr)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		additiveCorrectionsPtr[bin] =
		    getAdditiveCorrectionFactor(measurements, bin);
	}
}

void Corrector_CPU::precomputeInVivoAttenuationFactors(
    const ProjectionData* measurements)
{
	ASSERT(measurements != nullptr);

	auto inVivoAttenuationFactors =
	    std::make_unique<ProjectionListOwned>(measurements);
	inVivoAttenuationFactors->allocate();

	mp_inVivoAttenuationFactors = std::move(inVivoAttenuationFactors);
	float* inVivoAttenuationFactorsPtr =
	    mp_inVivoAttenuationFactors->getRawPointer();

	const size_t numBins = measurements->count();

#pragma omp parallel for default(none) \
    firstprivate(numBins, measurements, inVivoAttenuationFactorsPtr)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		inVivoAttenuationFactorsPtr[bin] =
		    getInVivoAttenuationFactor(measurements, bin);
	}
}

float Corrector_CPU::getMultiplicativeCorrectionFactor(
	const ProjectionData* measurements, bin_t binId) const
{
	ASSERT(measurements != nullptr);

	const histo_bin_t histoBin = measurements->getHistogramBin(binId);

	float sensitivity;
	if (mp_sensitivity != nullptr)
	{
		sensitivity =
			m_globalScalingFactor *
			mp_sensitivity->getProjectionValueFromHistogramBin(histoBin);
		if (m_invertSensitivity)
		{
			sensitivity = 1.0f / sensitivity;
		}
	}
	else
	{
		sensitivity = m_globalScalingFactor;
	}

	float acf;
	if (mp_hardwareAcf != nullptr)
	{
		// Hardware ACF
		acf = mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_acf != nullptr && mp_inVivoAcf == nullptr)
	{
		// All ACF is Hardware ACF (no motion)
		acf = mp_acf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_hardwareAttenuationImage != nullptr)
	{
		acf = getAttenuationFactorFromAttenuationImage(
			measurements, binId, mp_hardwareAttenuationImage);
	}
	else
	{
		acf = 1.0f;
	}

	return acf * sensitivity;
}

float Corrector_CPU::getAdditiveCorrectionFactor(const ProjectionData* measurements,
											 bin_t binId) const
{
	const histo_bin_t histoBin = measurements->getHistogramBin(binId);

	float randomsEstimate;
	if (mp_randoms != nullptr)
	{
		randomsEstimate =
			mp_randoms->getProjectionValueFromHistogramBin(histoBin);
	}
	else
	{
		randomsEstimate = measurements->getRandomsEstimate(binId);
	}

	float scatterEstimate = 0.0f;
	if (mp_scatter != nullptr)
	{
		// TODO: Support exception in case of a contiguous sinogram (future)
		scatterEstimate =
			mp_scatter->getProjectionValueFromHistogramBin(histoBin);
	}

	float sensitivity;
	if (mp_sensitivity != nullptr)
	{
		sensitivity =
			m_globalScalingFactor *
			mp_sensitivity->getProjectionValueFromHistogramBin(histoBin);
		if (m_invertSensitivity)
		{
			sensitivity = 1.0f / sensitivity;
		}
	}
	else
	{
		sensitivity = m_globalScalingFactor;
	}

	float acf;
	if (mp_acf != nullptr)
	{
		acf = mp_acf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_inVivoAcf != nullptr && mp_hardwareAcf != nullptr)
	{
		acf = mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin) *
			  mp_hardwareAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	else if (mp_attenuationImage != nullptr)
	{
		acf = getAttenuationFactorFromAttenuationImage(measurements, binId,
													   mp_attenuationImage);
	}
	else
	{
		acf = 1.0f;
	}

	return (randomsEstimate + scatterEstimate) / (acf * sensitivity);
}

float Corrector_CPU::getInVivoAttenuationFactor(const ProjectionData* measurements,
											bin_t binId) const
{
	const histo_bin_t histoBin = measurements->getHistogramBin(binId);

	if (mp_inVivoAcf != nullptr)
	{
		return mp_inVivoAcf->getProjectionValueFromHistogramBin(histoBin);
	}
	if (mp_inVivoAttenuationImage != nullptr)
	{
		return getAttenuationFactorFromAttenuationImage(
			measurements, binId, mp_inVivoAttenuationImage);
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
	const ProjectionData* measurements, bin_t binId,
	const Image* attenuationImage) const
{
	const Line3D lor = measurements->getLOR(binId);

	const float tofValue =
		measurements->hasTOF() ? measurements->getTOFValue(binId) : 0.0f;

	const float att = OperatorProjectorSiddon::singleForwardProjection(
		attenuationImage, lor, mp_tofHelper.get(), tofValue);

	return Util::getAttenuationCoefficientFactor(att);
}

