/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/Corrector.hpp"

#include "operators/OperatorProjectorSiddon.hpp"

#include <utils/Tools.hpp>

Corrector::Corrector()
    : mp_randoms(nullptr),
      mp_scatter(nullptr),
      mp_acf(nullptr),
      mp_attenuationImage(nullptr),
      mp_sensitivity(nullptr),
      m_invertSensitivity(false),
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

std::unique_ptr<ProjectionList> Corrector::getAdditiveCorrectionFactorsHelper(
    const ProjectionData* measurements, const BinIterator* binIter,
    bool useBinIterator, bool useRandomsHistogram, bool useScatter,
    bool useAttenuationImage, bool usePrecomputedACF, bool useSensitivity,
    bool invertSensitivity, bool hasTOF) const
{
	auto additiveCorrections =
	    std::make_unique<ProjectionListOwned>(measurements);
	additiveCorrections->allocate();
	float* additiveCorrectionsPtr = additiveCorrections->getRawPointer();

	const size_t numBins =
	    useBinIterator ? measurements->count() : binIter->size();

#pragma omp parallel for default(none)                                       \
    firstprivate(numBins, measurements, additiveCorrectionsPtr, binIter,     \
                     useBinIterator, useRandomsHistogram, useScatter,        \
                     useAttenuationImage, usePrecomputedACF, useSensitivity, \
                     invertSensitivity, hasTOF)
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

		const histo_bin_t histoBin = measurements->getHistogramBin(binId);

		float randomsEstimate;
		if (useRandomsHistogram)
		{
			randomsEstimate =
			    mp_randoms->getProjectionValueFromHistogramBin(histoBin);
		}
		else
		{
			randomsEstimate = measurements->getRandomsEstimate(binId);
		}

		float scatterEstimate = 0.0f;
		if (useScatter)
		{
			// TODO: Support exception in case of a contiguous sinogram
			scatterEstimate =
			    mp_scatter->getProjectionValueFromHistogramBin(binId);
		}

		float sensitivity;
		if (useSensitivity)
		{
			sensitivity =
			    mp_sensitivity->getProjectionValueFromHistogramBin(binId);
			if (invertSensitivity)
			{
				sensitivity = 1.0f / sensitivity;
			}
		}
		else
		{
			sensitivity = 1.0f;
		}

		float acf;
		if (usePrecomputedACF)
		{
			acf = mp_acf->getProjectionValueFromHistogramBin(binId);
		}
		else if (useAttenuationImage)
		{
			Line3D lor = measurements->getLOR(binId);

			const float tofValue =
			    hasTOF ? measurements->getTOFValue(binId) : 0.0f;

			const float att = OperatorProjectorSiddon::singleForwardProjection(
			    mp_attenuationImage, lor, mp_tofHelper.get(), tofValue);
			acf = Util::getAttenuationCoefficientFactor(att);
		}
		else
		{
			acf = 1.0f;
		}

		additiveCorrectionsPtr[binId] =
		    (randomsEstimate + scatterEstimate) / (acf * sensitivity);
	}

	return additiveCorrections;
}

std::unique_ptr<ProjectionList>
    Corrector::getAdditiveCorrectionFactors(const ProjectionData* measurements,
                                            const BinIterator* binIter) const
{
	const bool useBinIterator = binIter != nullptr;
	const bool useRandomsHistogram = mp_randoms != nullptr;
	const bool useScatter = mp_scatter != nullptr;
	const bool useAttenuationImage = mp_attenuationImage != nullptr;
	const bool usePrecomputedACF = mp_acf != nullptr;
	const bool useSensitivity = mp_sensitivity != nullptr;
	const bool invertSensitivity = m_invertSensitivity;
	const bool hasTOF = measurements->hasTOF();
	return getAdditiveCorrectionFactorsHelper(
	    measurements, binIter, useBinIterator, useRandomsHistogram, useScatter,
	    useAttenuationImage, usePrecomputedACF, useSensitivity,
	    invertSensitivity, hasTOF);
}
