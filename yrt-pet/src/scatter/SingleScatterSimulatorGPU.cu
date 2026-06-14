/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/RawParameters.hpp"
#include "yrt-pet/scatter/ScatterDeviceKernels.cuh"
#include "yrt-pet/scatter/ScatterSpace.hpp"
#include "yrt-pet/scatter/SingleScatterSimulator.hpp"

#include <vector>

namespace yrt::scatter
{

void SingleScatterSimulator::runSSSDevice(ScatterSpace& outScatterSpace,
                                          bool onlyDirectPlanes,
                                          cudaStream_t* stream) const
{
	ASSERT_MSG(outScatterSpace.isMemoryValid(),
	           "Destination scatter-space array is unallocated");

	const size_t numTOFBins = outScatterSpace.getNumTOFBins();
	const size_t numPlanes = outScatterSpace.getNumPlanes();
	const size_t numAngles = outScatterSpace.getNumAngles();

	size_t numLORs;
	if (onlyDirectPlanes)
		numLORs = numTOFBins * numPlanes * numAngles * numAngles;
	else
		numLORs = outScatterSpace.getSizeTotal();

	std::vector<Line3D> lorData(numLORs);
	std::vector<float> tofValues(numLORs);
	std::vector<size_t> flatIndices(numLORs);

	size_t idx = 0;
	if (onlyDirectPlanes)
	{
		for (size_t tofBin = 0; tofBin < numTOFBins; ++tofBin)
		{
			for (size_t planeIdx = 0; planeIdx < numPlanes; ++planeIdx)
			{
				for (size_t a1 = 0; a1 < numAngles; ++a1)
				{
					for (size_t a2 = 0; a2 < numAngles; ++a2)
					{
						const ScatterSpace::ScatterSpaceIndex scsIdx{
						    tofBin, planeIdx, a1, planeIdx, a2};
						const auto [tof_ps, lor] =
						    outScatterSpace.getTOFAndLORFromIndex(scsIdx);
						lorData[idx] = lor;
						tofValues[idx] = tof_ps;
						flatIndices[idx] = outScatterSpace.getFlatIdx(scsIdx);
						++idx;
					}
				}
			}
		}
	}
	else
	{
		for (size_t i = 0; i < numLORs; ++i)
		{
			const ScatterSpace::ScatterSpaceIndex scsIdx =
			    outScatterSpace.unravelIndex(i);
			const auto [tof_ps, lor] =
			    outScatterSpace.getTOFAndLORFromIndex(scsIdx);
			lorData[i] = lor;
			tofValues[i] = tof_ps;
			flatIndices[i] = i;
		}
	}

	RawImageConst muImg = getRawImage(mr_mu);
	RawImageConst lambdaImg = getRawImage(mr_lambda);

	std::vector<float> results(numLORs);

	launchComputeSingleScatterInLOR(
	    lorData.data(), tofValues.data(), results.data(),
	    static_cast<int>(numLORs), m_xSamples.data(), m_ySamples.data(),
	    m_zSamples.data(), m_numSamples, m_energyLLD, m_sigmaEnergy,
	    m_crystalDepth, m_axialFOV, m_collimatorRadius, m_crystalMaterial,
	    m_cyl1, m_cyl2, m_endPlate1, m_endPlate2, muImg, lambdaImg, stream);

	for (size_t i = 0; i < numLORs; ++i)
	{
		outScatterSpace.setValueFlat(flatIndices[i], fmaxf(0.0f, results[i]));
	}
}

}  // namespace yrt::scatter
