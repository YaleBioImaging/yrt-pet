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

void SingleScatterSimulator::runSSSOnGPU(ScatterSpace& outScatterSpace,
                                      bool onlyDirectPlanes)
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

	std::vector<float3> lorP1(numLORs);
	std::vector<float3> lorP2(numLORs);
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
						lorP1[idx] =
						    make_float3(lor.point1.x, lor.point1.y, lor.point1.z);
						lorP2[idx] =
						    make_float3(lor.point2.x, lor.point2.y, lor.point2.z);
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
			lorP1[i] =
			    make_float3(lor.point1.x, lor.point1.y, lor.point1.z);
			lorP2[i] =
			    make_float3(lor.point2.x, lor.point2.y, lor.point2.z);
			tofValues[i] = tof_ps;
			flatIndices[i] = i;
		}
	}

	const ImageParams& mu_p = mr_mu.getParams();
	const ImageParams& lambda_p = mr_lambda.getParams();

	RawImageParams muParams;
	muParams.nx = mu_p.nx;
	muParams.ny = mu_p.ny;
	muParams.nz = mu_p.nz;
	muParams.vx = mu_p.vx;
	muParams.vy = mu_p.vy;
	muParams.vz = mu_p.vz;
	muParams.length_x = mu_p.length_x;
	muParams.length_y = mu_p.length_y;
	muParams.length_z = mu_p.length_z;
	muParams.off_x = mu_p.off_x;
	muParams.off_y = mu_p.off_y;
	muParams.off_z = mu_p.off_z;
	muParams.fovRadius = mu_p.fovRadius;

	RawImageParams lambdaParams;
	lambdaParams.nx = lambda_p.nx;
	lambdaParams.ny = lambda_p.ny;
	lambdaParams.nz = lambda_p.nz;
	lambdaParams.vx = lambda_p.vx;
	lambdaParams.vy = lambda_p.vy;
	lambdaParams.vz = lambda_p.vz;
	lambdaParams.length_x = lambda_p.length_x;
	lambdaParams.length_y = lambda_p.length_y;
	lambdaParams.length_z = lambda_p.length_z;
	lambdaParams.off_x = lambda_p.off_x;
	lambdaParams.off_y = lambda_p.off_y;
	lambdaParams.off_z = lambda_p.off_z;
	lambdaParams.fovRadius = lambda_p.fovRadius;

	const float3 imageOffset = make_float3(
	    muParams.off_x - (muParams.nx / 2.0f - 0.5f) * muParams.vx,
	    muParams.off_y - (muParams.ny / 2.0f - 0.5f) * muParams.vy,
	    muParams.off_z - (muParams.nz / 2.0f - 0.5f) * muParams.vz);

	std::vector<float> results(numLORs);

	launchComputeSingleScatterInLOR(
	    lorP1.data(), lorP2.data(), tofValues.data(), results.data(),
	    static_cast<int>(numLORs), m_xSamples.data(), m_ySamples.data(),
	    m_zSamples.data(), m_numSamples, m_energyLLD, m_sigmaEnergy,
	    m_crystalDepth, m_axialFOV, m_collimatorRadius, m_crystalMaterial,
	    m_cyl1, m_cyl2, m_endPlate1, m_endPlate2, mr_mu.getRawPointer(),
	    mr_lambda.getRawPointer(), muParams, lambdaParams, imageOffset);

	for (size_t i = 0; i < numLORs; ++i)
	{
		outScatterSpace.setValueFlat(flatIndices[i],
		                             fmaxf(0.0f, results[i]));
	}
}

}  // namespace yrt::scatter
