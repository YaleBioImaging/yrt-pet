/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/Corrector_CPU.hpp"

#include "yrt-pet/operators/ProjectorSiddon.hpp"

namespace yrt
{

Corrector_CPU::Corrector_CPU(const Scanner& pr_scanner) : Corrector(pr_scanner)
{
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
	return 1.0f;
}

}  // namespace yrt
