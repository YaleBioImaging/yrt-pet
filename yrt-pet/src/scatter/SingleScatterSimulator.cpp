/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/SingleScatterSimulator.hpp"
#include "yrt-pet/scatter/SingleScatterSimulatorUtils.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"
#include "yrt-pet/utils/Tools.hpp"


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
	c.def(py::init<const Scanner&, const Image&, const Image&,
	               scatter::CrystalMaterial, int, float>(),
	      "scanner"_a, "attenuation_image"_a, "source_image"_a,
	      "crystal_material"_a, "seed"_a, "num_samp_frac"_a);
	c.def("runSSS", &scatter::SingleScatterSimulator::runSSS,
	      "out_scatter_space"_a, "only_direct_planes"_a = false);
#if BUILD_CUDA
	c.def(
	    "runSSSOnGPU",
	    [](const scatter::SingleScatterSimulator& self, ScatterSpace& outScs,
	       bool onlyDirectPlanes)
	    { self.runSSSDevice(outScs, onlyDirectPlanes); },
	    "out_scatter_space"_a, "only_direct_planes"_a = false);
#endif
	c.def("setUseGPU", &scatter::SingleScatterSimulator::setUseGPU, "use"_a);
	c.def("computeSingleScatterInLOR",
	      &scatter::SingleScatterSimulator::computeSingleScatterInLOR, "lor"_a,
	      "tof"_a);
	c.def("getSamplePoint", &scatter::SingleScatterSimulator::getSamplePoint,
	      "i"_a);
	c.def("getNumSamples", &scatter::SingleScatterSimulator::getNumSamples);
	c.def("passCollimator", &scatter::SingleScatterSimulator::passCollimator,
	      "lor"_a);
}
}  // namespace yrt

#endif

namespace yrt::scatter
{

SingleScatterSimulator::SingleScatterSimulator(
    const Scanner& pr_scanner, const Image& pr_mu, const Image& pr_lambda,
    CrystalMaterial p_crystalMaterial, int seedi, float numSampFrac)
    : mr_scanner(pr_scanner),
      mr_mu(pr_mu),
      mr_lambda(pr_lambda),
      m_crystalMaterial(p_crystalMaterial)
{
	const ImageParams& mu_params = mr_mu.getParams();
	// YP low level discriminatory energy
	m_energyLLD = mr_scanner.energyLLD;

	// YP: standard deviation of scattered photons energy distribution
	m_sigmaEnergy = (mr_scanner.fwhm) / (2.0f * sqrt(2.0f * log(2.0f)));

	m_scannerRadius = mr_scanner.scannerRadius;        // YP ring radius
	m_crystalDepth = mr_scanner.crystalDepth;          // YP detector thickness
	m_axialFOV = mr_scanner.axialFOV;                  // YP Axial FOV
	m_collimatorRadius = mr_scanner.collimatorRadius;  // YP no need?

	const Vector3D c{0., 0., 0.};
	// YP: creates 2 cylinders of axial extent "afov" in millimiters xs
	m_cyl1 = Cylinder{c, m_axialFOV, m_scannerRadius};
	m_cyl2 = Cylinder{c, m_axialFOV, m_scannerRadius + m_crystalDepth};
	// YP 3 points located in the last ring of the scanner
	Vector3D p1{1.0f, 0.0f, -m_axialFOV / 2.0f},
	    p2{0.0f, 1.0f, -m_axialFOV / 2.0f}, p3{0.0f, 0.0f, -m_axialFOV / 2.0f};
	// YP defines a plane according to these 3 points
	m_endPlate1 = Plane{p1, p2, p3};

	// YP other plane located at the first ring of the scanner
	p1.z = p2.z = p3.z = m_axialFOV / 2.0;
	m_endPlate2 = Plane{p1, p2, p3};

	int seed = std::abs(seedi);  // YP random seed
	int init = -1;
	ran1(&init);
	m_numSamples = 0;

	// Generate scatter points:
	// YP coarser cubic grid of scatter points
	int nxsamp = static_cast<int>(numSampFrac * mu_params.nx);
	if (nxsamp < 5)
		nxsamp = 5;  // YP number of scatter points in x direction
	float nxsamp_f = static_cast<float>(nxsamp);
	int nysamp = static_cast<int>(numSampFrac * mu_params.ny);
	if (nysamp < 5)
		nysamp = 5;
	float nysamp_f = static_cast<float>(nysamp);
	int nzsamp = static_cast<int>(numSampFrac * mu_params.nz);
	if (nzsamp < 5)
		nzsamp = 5;
	float nzsamp_f = static_cast<float>(nzsamp);
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
				if (mr_mu.nearestNeighbor(p) > 0.005 &&
				    p.getNorm() < m_collimatorRadius)
				{
					// YP rejects the point if the associated att. coeff is
					// below
					// a certain threshold
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
		    "SingleScatterSimulation::SingleScatterSimulation(). "
		    "nsamples=" +
		    std::to_string(m_numSamples);
		throw std::runtime_error(errorMessage);
	}
}
void SingleScatterSimulator::runSSS(ScatterSpace& outScatterSpace,
                                    bool onlyDirectPlanes) const
{
	ASSERT_MSG(outScatterSpace.isMemoryValid(),
	           "Destination scatter-space array is unallocated");

	if (m_useGPU)
	{
		runSSSDevice(outScatterSpace, onlyDirectPlanes);
		return;
	}

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
	    m_zSamples.data(), m_energyLLD, m_sigmaEnergy, m_crystalDepth,
	    m_axialFOV, m_collimatorRadius, m_crystalMaterial, m_cyl1, m_cyl2,
	    m_endPlate1, m_endPlate2, mr_mu, mr_lambda);
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

float SingleScatterSimulator::getIntersectionLengthLORCrystal(
    const Line3D& lor) const
{
	return getIntersectionLengthLORCrystalRaw(lor, m_cyl1, m_cyl2);
}

bool SingleScatterSimulator::passCollimator(const Line3D& lor) const
{
	return passCollimatorRaw(lor, m_collimatorRadius, m_axialFOV, m_endPlate1,
	                         m_endPlate2);
}

const Image& SingleScatterSimulator::getAttenuationImage() const
{
	return mr_mu;
}

}  // namespace yrt::scatter
