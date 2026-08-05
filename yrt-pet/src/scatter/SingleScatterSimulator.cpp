/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/SingleScatterSimulator.hpp"
#include "yrt-pet/scatter/SingleScatterSimulatorUtils.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_singlescattersimulator(py::module& m)
{
	auto c = py::class_<scatter::SingleScatterSimulator>(
	    m, "SingleScatterSimulator");
	c.def(py::init<const Scanner&, const Image&, const Image&, int, float,
	               float, float>(),
	      "scanner"_a, "attenuation_image"_a, "source_image"_a, "seed"_a,
	      "att_threshold_sampling"_a =
	          scatter::SingleScatterSimulator::DefaultAttThresholdSampling,
	      "num_samp_frac"_a =
	          scatter::SingleScatterSimulator::DefaultNumSampFrac,
	      "detection_threshold"_a =
	          scatter::SingleScatterSimulator::DefaultDetectionThreshold);
	c.def("runSSS", &scatter::SingleScatterSimulator::runSSS,
	      "out_scatter_space"_a, "only_direct_planes"_a = false);
#if BUILD_CUDA
	c.def(
	    "runSSSDevice",
	    [](const scatter::SingleScatterSimulator& self, ScatterSpace& outScs,
	       bool onlyDirectPlanes)
	    { self.runSSSDevice(outScs, onlyDirectPlanes); },
	    "out_scatter_space"_a, "only_direct_planes"_a = false);
#endif
	c.def("computeSingleScatterInLOR",
	      &scatter::SingleScatterSimulator::computeSingleScatterInLOR, "lor"_a,
	      "tof"_a);
	c.def("getSamplePoint", &scatter::SingleScatterSimulator::getSamplePoint,
	      "i"_a);
	c.def("getNumSamples", &scatter::SingleScatterSimulator::getNumSamples);
}
}  // namespace yrt

#endif

namespace yrt::scatter
{

SingleScatterSimulator::SingleScatterSimulator(
    const Scanner& pr_scanner, const Image& pr_mu, const Image& pr_lambda,
    int p_seed, float p_attThresholdSampling, float p_numSampFrac,
    float p_detectionThreshold)
    : m_attThresholdSampling(p_attThresholdSampling),
      m_detectionThreshold(p_detectionThreshold),
      mr_scanner(pr_scanner),
      mr_mu(pr_mu),
      mr_lambda(pr_lambda)
{
	const ImageParams& mu_params = mr_mu.getParams();
	// YP low level discriminatory energy
	m_energyLLD = mr_scanner.energyLLD;

	m_energyResolution = mr_scanner.energyResolution;
	m_scannerRadius = mr_scanner.scannerRadius;  // YP ring radius

	int seed = std::abs(p_seed);  // YP random seed
	int init = -1;
	ran1(&init);
	m_numSamples = 0;

	// Generate scatter points:
	// YP coarser cubic grid of scatter points
	ssize_t nxsamp = p_numSampFrac * mu_params.nx;
	if (nxsamp < 5)
	{
		nxsamp = 5;  // YP number of scatter points in x direction
	}
	const float nxsamp_f = static_cast<float>(nxsamp);
	ssize_t nysamp = p_numSampFrac * mu_params.ny;
	if (nysamp < 5)
	{
		nysamp = 5;
	}
	const float nysamp_f = static_cast<float>(nysamp);
	ssize_t nzsamp = p_numSampFrac * mu_params.nz;
	if (nzsamp < 5)
	{
		nzsamp = 5;
	}
	const float nzsamp_f = static_cast<float>(nzsamp);

	std::cout << "nxsamp: " << nxsamp << std::endl;
	std::cout << "nysamp: " << nysamp << std::endl;
	std::cout << "nzsamp: " << nzsamp << std::endl;
	m_xSamples.reserve(nzsamp * nysamp * nxsamp);
	m_ySamples.reserve(nzsamp * nysamp * nxsamp);
	m_zSamples.reserve(nzsamp * nysamp * nxsamp);
	// YP spacing between scatter points
	const float dxsamp = mu_params.length_x / nxsamp_f;
	const float dysamp = mu_params.length_y / nysamp_f;
	const float dzsamp = mu_params.length_z / nzsamp_f;
	Vector3D p;
	m_xSamples.clear();
	m_ySamples.clear();
	m_zSamples.clear();
	for (int k = 0; k < nzsamp; k++)
	{
		const float z = k / nzsamp_f * mu_params.length_z -
		                mu_params.length_z / 2 + mu_params.vz / 2.0 +
		                mu_params.off_z;
		for (int j = 0; j < nysamp; j++)
		{
			const float y = j / nysamp_f * mu_params.length_y -
			                mu_params.length_y / 2 + mu_params.vy / 2.0 +
			                mu_params.off_y;
			for (int i = 0; i < nxsamp; i++)
			{
				const float x = i / nxsamp_f * mu_params.length_x -
				                mu_params.length_x / 2 + mu_params.vx / 2.0 +
				                mu_params.off_x;
				const float x2 = ran1(&seed) * dxsamp + x;
				const float y2 = ran1(&seed) * dysamp + y;
				const float z2 = ran1(&seed) * dzsamp + z;
				// YP generate a random scatter point within its cell
				p.update(x2, y2, z2);

				if (mr_mu.nearestNeighbor(p) > m_attThresholdSampling &&
				    p.getNorm() < mr_scanner.scannerRadius)
				{
					// YP reject the point if the associated att. coeff is
					//  below a certain threshold
					m_numSamples++;  // nsamp: number of scatter points
					m_xSamples.push_back(x2);
					m_ySamples.push_back(y2);
					m_zSamples.push_back(z2);
				}
			}
		}
	}

	m_xSamples.shrink_to_fit();
	m_ySamples.shrink_to_fit();
	m_zSamples.shrink_to_fit();

	if (m_numSamples < 10)
	{
		std::string errorMessage =
		    "Error: Small number of scatter points in "
		    "SingleScatterSimulation::SingleScatterSimulation() : " +
		    std::to_string(m_numSamples);
		throw std::runtime_error(errorMessage);
	}
}
void SingleScatterSimulator::runSSS(ScatterSpace& outScatterSpace,
                                    bool onlyDirectPlanes) const
{
	ASSERT_MSG(outScatterSpace.isMemoryValid(),
	           "Destination scatter-space array is unallocated");

	const size_t numTOFBins = outScatterSpace.getNumTOFBins();
	const size_t numPlanes = outScatterSpace.getNumPlanes();
	const size_t numAngles = outScatterSpace.getNumAngles();

	const size_t numThreads = globals::getNumThreads();

	if (onlyDirectPlanes)
	{
		const size_t numDirectPlanesTOF = numTOFBins * numPlanes;

		util::ProgressDisplayMultiThread progressBar(numThreads,
		                                             numDirectPlanesTOF, 5);

		util::parallelForChunked(
		    numDirectPlanesTOF, numThreads,
		    [&progressBar, &outScatterSpace, numPlanes, numAngles,
		     this](size_t planeSampleIdx, size_t threadId)
		    {
			    progressBar.incrementProgress(threadId);

			    // Here, "planeSampleIdx" is a flat index encoding both the
			    //  direct plane index and the TOF bin
			    const size_t tofBin = planeSampleIdx / numPlanes;
			    const size_t planeIdx = planeSampleIdx % numPlanes;

			    for (size_t a1 = 0; a1 < numAngles; ++a1)
			    {
				    for (size_t a2 = 0; a2 < numAngles; ++a2)
				    {
					    const auto [tof_ps, lor] =
					        outScatterSpace.getTOFAndLORFromIndex(
					            {tofBin, planeIdx, a1, planeIdx, a2});

					    float scatterResult = 0.0f;

					    if (lor.isValid())
					    {
						    scatterResult =
						        computeSingleScatterInLOR(lor, tof_ps);
						    // Avoid negative values
						    scatterResult = std::max(0.0f, scatterResult);
					    }

					    outScatterSpace.setValue(tofBin, planeIdx, a1, planeIdx,
					                             a2, scatterResult);
				    }
			    }
		    });
	}
	else
	{
		const size_t numSamples = outScatterSpace.getSizeTotal();

		util::ProgressDisplayMultiThread progressBar(numThreads, numSamples, 5);

		util::parallelForChunked(
		    numSamples, numThreads,
		    [&progressBar, &outScatterSpace, this](size_t sampleId,
		                                           size_t threadId)
		    {
			    progressBar.incrementProgress(threadId);

			    const ScatterSpace::ScatterSpaceIndex scsIdx =
			        outScatterSpace.unravelIndex(sampleId);

			    const auto [tof_ps, lor] =
			        outScatterSpace.getTOFAndLORFromIndex(scsIdx);

			    float scatterResult = 0.0f;

			    if (lor.isValid())
			    {
				    scatterResult = computeSingleScatterInLOR(lor, tof_ps);
				    // Avoid negative values
				    scatterResult = std::max(0.0f, scatterResult);
			    }

			    outScatterSpace.setValue(scsIdx, scatterResult);
		    });
	}
}

float SingleScatterSimulator::computeSingleScatterInLOR(const Line3D& lor,
                                                        float tof_ps) const
{
	return scatter::computeSingleScatterInLOR(
	    lor, tof_ps, m_numSamples, m_xSamples.data(), m_ySamples.data(),
	    m_zSamples.data(), m_detectionThreshold, m_energyLLD,
	    m_energyResolution, mr_mu, mr_lambda);
}

Vector3D SingleScatterSimulator::getSamplePoint(int i) const
{
	ASSERT(i < m_numSamples);
	return Vector3D{m_xSamples[i], m_ySamples[i], m_zSamples[i]};
}

int SingleScatterSimulator::getNumSamples() const
{
	return m_numSamples;
}

float SingleScatterSimulator::ran1(int* idum)
{
	int j, k;
	static int iy = 0;
	static int iv[NTAB];
	float temp;

	if (*idum <= 0 || !iy)
	{
		if (-(*idum) < 1)
			*idum = 1;
		else
			*idum = -(*idum);
		for (j = NTAB + 7; j >= 0; j--)
		{
			k = (*idum) / IQ;
			*idum = IA * (*idum - k * IQ) - IR * k;
			if (*idum < 0)
				*idum += IM;
			if (j < NTAB)
				iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0)
		*idum += IM;
	j = iy / NDIV;
	iy = iv[j];
	iv[j] = *idum;
	if ((temp = AM * iy) > RNMX)
		return (RNMX);
	else
		return temp;
}

const Image& SingleScatterSimulator::getAttenuationImage() const
{
	return mr_mu;
}

}  // namespace yrt::scatter
