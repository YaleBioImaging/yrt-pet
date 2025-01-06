/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector.hpp"

#include "operators/OperatorProjectorSiddon.hpp"
#include "utils/Assert.hpp"

#include <utils/Tools.hpp>

Corrector::Corrector()
    : mp_randoms(nullptr),
      mp_scatter(nullptr),
      mp_acf(nullptr),
      mp_attenuationImage(nullptr),
      mp_sensitivity(nullptr),
      m_invertSensitivity(false),
      m_globalScalingFactor(1.0f),
      mp_tofHelper(nullptr)
{
}

void Corrector::setSensitivityHistogram(const Histogram* pp_sensitivity)
{
	mp_sensitivity = pp_sensitivity;
}

void Corrector::setRandomsHistogram(const Histogram* pp_randoms)
{
	mp_randoms = pp_randoms;
}

void Corrector::setScatterHistogram(const Histogram* pp_scatter)
{
	mp_scatter = pp_scatter;
}

void Corrector::setGlobalScalingFactor(float globalScalingFactor)
{
	m_globalScalingFactor = globalScalingFactor;
}

void Corrector::setAttenuationImage(const Image* pp_attenuationImage)
{
	mp_attenuationImage = pp_attenuationImage;
}

void Corrector::setACFHistogram(const Histogram* pp_acf)
{
	mp_acf = pp_acf;
}

void Corrector::setInvertSensitivity(bool invert)
{
	m_invertSensitivity = invert;
}

void Corrector::addTOF(float p_tofWidth_ps, int p_tofNumStd)
{
	mp_tofHelper =
	    std::make_unique<TimeOfFlightHelper>(p_tofWidth_ps, p_tofNumStd);
}

std::unique_ptr<ProjectionList>
    Corrector::getAdditiveCorrectionFactors(const ProjectionData* measurements,
                                            const BinIterator* binIter) const
{
	ASSERT(measurements != nullptr);
	const bool useBinIterator = binIter != nullptr;
	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(measurements);
	additiveCorrections->allocate();
	float* additiveCorrectionsPtr = additiveCorrections->getRawPointer();

	const size_t numBins =
	    useBinIterator ? binIter->size() : measurements->count();

#pragma omp parallel for default(none)                                   \
    firstprivate(numBins, measurements, additiveCorrectionsPtr, binIter, \
                     useBinIterator)
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		bin_t binId;
		if (useBinIterator)
		{
			binId = binIter->get(bin);
		}
		else
		{
			binId = bin;
		}

		additiveCorrectionsPtr[binId] =
		    getAdditiveCorrectionFactor(measurements, binId);
	}

	return additiveCorrections;
}

float Corrector::getAdditiveCorrectionFactor(const ProjectionData* measurements,
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
		scatterEstimate = mp_scatter->getProjectionValueFromHistogramBin(binId);
	}

	float sensitivity;
	if (mp_sensitivity != nullptr)
	{
		sensitivity = m_globalScalingFactor *
		              mp_sensitivity->getProjectionValueFromHistogramBin(binId);
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
		acf = mp_acf->getProjectionValueFromHistogramBin(binId);
	}
	else if (mp_attenuationImage != nullptr)
	{
		const Line3D lor = measurements->getLOR(binId);

		const float tofValue =
		    measurements->hasTOF() ? measurements->getTOFValue(binId) : 0.0f;

		const float att = OperatorProjectorSiddon::singleForwardProjection(
		    mp_attenuationImage, lor, mp_tofHelper.get(), tofValue);
		acf = Util::getAttenuationCoefficientFactor(att);
	}
	else
	{
		acf = 1.0f;
	}

	return (randomsEstimate + scatterEstimate) / (acf * sensitivity);
}
