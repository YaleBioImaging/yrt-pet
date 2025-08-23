/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/scatter/SingleScatterSimulator.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
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
	               scatter::CrystalMaterial, int>(),
	      "scanner"_a, "attenuation_image"_a, "source_image"_a,
	      "crystal_material"_a, "seed"_a);
	c.def("runSSS", &scatter::SingleScatterSimulator::runSSS, "num_z"_a,
	      "num_phi"_a, "num_r"_a, "scatter_histo"_a);
	c.def("computeSingleScatterInLOR",
	      &scatter::SingleScatterSimulator::computeSingleScatterInLOR, "lor"_a,
	      "n1"_a, "n2"_a);
	c.def("getSamplePoint", &scatter::SingleScatterSimulator::getSamplePoint,
	      "i"_a);
	c.def("getNumSamples", &scatter::SingleScatterSimulator::getNumSamples);
	c.def("passCollimator", &scatter::SingleScatterSimulator::passCollimator,
	      "lor"_a);
}
}  // namespace yrt

#endif

namespace yrt
{
namespace scatter
{

SingleScatterSimulator::SingleScatterSimulator(
    const Scanner& pr_scanner, const Image& pr_mu, const Image& pr_lambda,
    CrystalMaterial p_crystalMaterial, int seedi)
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
	int nxsamp = static_cast<int>(mu_params.nx / 1.5);
	if (nxsamp < 5)
		nxsamp = 5;  // YP number of scatter points in x direction
	int nysamp = static_cast<int>(mu_params.ny / 1.5);
	if (nysamp < 5)
		nysamp = 5;
	int nzsamp = static_cast<int>(mu_params.nz / 1.5);
	if (nzsamp < 5)
		nzsamp = 5;
	std::cout << "nxsamp: " << nxsamp << std::endl;
	std::cout << "nysamp: " << nysamp << std::endl;
	std::cout << "nzsamp: " << nzsamp << std::endl;
	m_xSamples.reserve(nzsamp * nysamp * nxsamp);
	m_ySamples.reserve(nzsamp * nysamp * nxsamp);
	m_zSamples.reserve(nzsamp * nysamp * nxsamp);
	// YP spacing between scatter points
	const float dxsamp = mu_params.length_x / (static_cast<float>(nxsamp));
	const float dysamp = mu_params.length_y / (static_cast<float>(nysamp));
	const float dzsamp = mu_params.length_z / (static_cast<float>(nzsamp));
	Vector3D p;
	m_xSamples.clear();
	m_ySamples.clear();
	m_zSamples.clear();
	for (int k = 0; k < nzsamp; k++)
	{
		const float z = k / (static_cast<float>(nzsamp)) * mu_params.length_z -
		                mu_params.length_z / 2 + mu_params.vz / 2.0 +
		                mu_params.off_z;
		for (int j = 0; j < nysamp; j++)
		{
			const float y =
			    j / (static_cast<float>(nysamp)) * mu_params.length_y -
			    mu_params.length_y / 2 + mu_params.vy / 2.0 + mu_params.off_y;
			for (int i = 0; i < nxsamp; i++)
			{
				const float x =
				    i / (static_cast<float>(nxsamp)) * mu_params.length_x -
				    mu_params.length_x / 2 + mu_params.vx / 2.0 +
				    mu_params.off_x;
				const float x2 = ran1(&seed) * dxsamp + x;
				const float y2 = ran1(&seed) * dysamp + y;
				const float z2 = ran1(&seed) * dzsamp + z;
				// YP generate a random scatter poitn within its cell
				p.update(x2, y2, z2);
				if (mr_mu.nearestNeighbor(p) > 0.005)
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
	if (m_numSamples < 10)
	{
		std::cerr << "Error: Small number of scatter points in "
		             "SingleScatterSimulation::SingleScatterSimulation(). "
		             "nsamples="
		          << m_numSamples << "\n"
		          << std::endl;
		exit(-1);
	}
}

void SingleScatterSimulator::runSSS(size_t numberZ, size_t numberPhi,
                                    size_t numberR, Histogram3D& scatterHisto)
{
	const size_t num_i_z = numberZ;
	const size_t num_i_phi = numberPhi;
	const size_t num_i_r = numberR;

	ASSERT_MSG(num_i_phi > 1,
	           "The number of phis given has to be larger than 1");
	ASSERT_MSG(num_i_r > 1, "The number of rs given has to be larger than 1");
	ASSERT_MSG(num_i_z > 0, "The number of zs given has to be larger than 0");
	ASSERT_MSG(&scatterHisto.getScanner() == &mr_scanner,
	           "The histogram's scanner is not the same as the SSS's scanner");
	ASSERT_MSG(scatterHisto.isMemoryValid(),
	           "Destination histogram is unallocated or unbound");

	constexpr size_t min_z = 0;
	constexpr size_t min_phi = 0;
	constexpr size_t min_r = 0;

	// We only fill the perpendicular bins... What does that imply for the
	// interpolation?
	const size_t num_z = mr_scanner.numRings;
	const size_t num_phi = scatterHisto.numPhi;
	const size_t num_r = scatterHisto.numR;

	// Sampling
	const float d_z = (num_z - min_z) / static_cast<float>(num_i_z - 1);
	const float d_phi = (num_phi - min_phi) / static_cast<float>(num_i_phi - 1);
	const float d_r = (num_r - min_r) / static_cast<float>(num_i_r - 1);
	m_zBinSamples.reserve(num_i_z);
	m_phiSamples.reserve(num_i_phi);
	m_rSamples.reserve(num_i_r);
	// Z-bin dimension
	for (size_t i = 0; i < num_i_z; i++)
	{
		const float z = static_cast<float>(min_z) + d_z * i;
		m_zBinSamples.push_back(std::min(static_cast<size_t>(z), num_z - 1));
	}
	// Phi dimension
	for (size_t i = 0; i < num_i_phi; i++)
	{
		const float phi = static_cast<float>(min_phi) + d_phi * i;
		m_phiSamples.push_back(std::min(static_cast<size_t>(phi), num_phi - 1));
	}
	// R dimension
	for (size_t i = 0; i < num_i_r; i++)
	{
		const float r = static_cast<float>(min_r) + d_r * i;
		m_rSamples.push_back(std::min(static_cast<size_t>(r), num_r - 1));
	}

	// Only used for printing purposes
	const int64_t progressMax = num_i_z * num_i_phi * num_i_r;
	const int numThreads = globals::getNumThreads();
	util::ProgressDisplayMultiThread progressBar{numThreads, progressMax, 5};

	util::parallelForChunked(
	    num_i_z * num_i_phi * num_i_r, globals::getNumThreads(),
	    [num_i_phi, num_i_r, &progressBar, &scatterHisto,
	     this](size_t binIdx, size_t threadNum)
	    {
		    size_t z_i = binIdx / (num_i_phi * num_i_r);
		    size_t rem_z = binIdx - z_i * num_i_phi * num_i_r;
		    size_t phi_i = rem_z / num_i_r;
		    size_t r_i = rem_z % num_i_r;
		    progressBar.progress(threadNum, 1);

		    const size_t z = m_zBinSamples[z_i];
		    const size_t phi = m_phiSamples[phi_i];
		    const size_t r = m_rSamples[r_i];

		    // Compute current LOR
		    const size_t scatterHistoBinId =
		        scatterHisto.getBinIdFromCoords(r, phi, z);

		    const auto [d1, d2] =
		        scatterHisto.getDetectorPair(scatterHistoBinId);
		    const Vector3D p1 = mr_scanner.getDetectorPos(d1);
		    const Vector3D p2 = mr_scanner.getDetectorPos(d2);
		    const Vector3D n1 = mr_scanner.getDetectorOrient(d1);
		    const Vector3D n2 = mr_scanner.getDetectorOrient(d2);

		    const Line3D lor{p1, p2};

		    const float scatterResult = computeSingleScatterInLOR(lor, n1, n2);
		    if (scatterResult <= 0.0)
			    return;  // Ignore irrelevant lines?
		    scatterHisto.setProjectionValue(scatterHistoBinId, scatterResult);
	    });

	std::cout << "Scatter simulation completed, running linear interpolation "
	             "to fill gaps..."
	          << std::endl;

	// Run interpolations to fill non-simulated bins
	const size_t num_i_z_to_take = (num_i_z == 1) ? 1 : (num_i_z - 1);
	for (size_t z_i = 0; z_i < num_i_z_to_take; z_i++)
	{
		const size_t z1 = m_zBinSamples[z_i];
		const size_t z2 =
		    (num_i_z == 1) ? m_zBinSamples[0] : m_zBinSamples[z_i + 1];
		for (size_t phi_i = 0; phi_i < num_i_phi - 1; phi_i++)
		{
			const size_t phi1 = m_phiSamples[phi_i];
			const size_t phi2 = m_phiSamples[phi_i + 1];
			for (size_t r_i = 0; r_i < num_i_r - 1; r_i++)
			{
				const size_t r1 = m_rSamples[r_i];
				const size_t r2 = m_rSamples[r_i + 1];
				util::fillBox(scatterHisto.getData(), z1, z2, phi1, phi2, r1,
				              r2);
			}
		}
	}
	std::cout << "Histogram filled in all the transaxial bins." << std::endl;

	std::cout << "Filling oblique bins..." << std::endl;
	for (coord_t z_bin_i = mr_scanner.numRings; z_bin_i < scatterHisto.numZBin;
	     ++z_bin_i)
	{
		coord_t z1, z2;
		scatterHisto.getZ1Z2(z_bin_i, z1, z2);
		const Array3DBase<float>& scatterHistoData_c = scatterHisto.getData();
		Array3DBase<float>& scatterHistoData = scatterHisto.getData();
		coord_t z_average = z1 + z2;
		if (z_average % 2 == 0)
		{
			z_average = z_average / 2;
			scatterHistoData[z_bin_i] += scatterHistoData_c[z_average];
		}
		else
		{
			const coord_t z_average_0 = z_average / 2;
			const coord_t z_average_1 = z_average_0 + 1;
			scatterHistoData[z_bin_i] += scatterHistoData_c[z_average_0];
			scatterHistoData[z_bin_i] += scatterHistoData_c[z_average_1];
			scatterHistoData[z_bin_i] *= 0.5;  // average
		}
	}
}

// YP LOR in which to compute the scatter contribution
float SingleScatterSimulator::computeSingleScatterInLOR(
    const Line3D& lor, const Vector3D& n1, const Vector3D& n2) const
{
	int i;
	float res = 0., dist1, dist2, energy, cosa, mu_scaling_factor;
	float vatt, att_s_1_511, att_s_1, att_s_2_511, att_s_2;
	float dsigcompdomega, lamb_s_1, lamb_s_2, sig_s_1, sig_s_2;
	float eps_s_1_511, eps_s_1, eps_s_2_511, eps_s_2, fac1, fac2;
	float tmp, tmp511, delta_1, delta_2, mu_det, mu_det_511;
	Line3D lor_1_s, lor_2_s;
	Vector3D ps, p1, p2, u, v;

	p1.update(lor.point1);
	p2.update(lor.point2);

	tmp511 = (m_energyLLD - 511.0) / (sqrt(2.0) * m_sigmaEnergy);
	mu_det_511 = getMuDet(511.0, m_crystalMaterial);

	for (i = 0; i < m_numSamples; i++)
	{
		// for each scatter point in the image volume

		ps.update(m_xSamples[i], m_ySamples[i], m_zSamples[i]);

		// LOR going from scatter point "ps" to detector 1
		lor_1_s.update(p1, ps);
		// LOR going from scatter point "ps" to detector 2
		lor_2_s.update(p2, ps);

		// check that the rays S-det1 and S-det2 pass the end plates
		// collimator before going further:
		if (std::abs(ps.z) > m_axialFOV / 2 &&
		    (!passCollimator(lor_1_s) || !passCollimator(lor_2_s)))
			continue;


		u.update(ps - p1);
		dist1 = u.getNorm();
		u.x /= dist1;
		u.y /= dist1;
		u.z /= dist1;
		v.update(p2 - ps);
		dist2 = v.getNorm();
		v.x /= dist2;
		v.y /= dist2;
		v.z /= dist2;

		cosa = u.scalProd(v);
		// larger angle change -> more energy loss
		energy = 511.0 / (2.0 - cosa);
		if (energy <= m_energyLLD)
		{
			continue;
		}
		tmp = (m_energyLLD - energy) / (sqrt(2.0) * m_sigmaEnergy);
		mu_scaling_factor = getMuScalingFactor(energy);

		// get scatter values:
		vatt = mr_mu.nearestNeighbor(ps);
		dsigcompdomega = getKleinNishina(cosa);

		// compute I1 and I2:
		att_s_1_511 =
		    OperatorProjectorSiddon::singleForwardProjection(&mr_mu, lor_1_s) /
		    10.0;

		att_s_1 = att_s_1_511 * mu_scaling_factor;
		lamb_s_1 = OperatorProjectorSiddon::singleForwardProjection(&mr_lambda,
		                                                            lor_1_s);
		delta_1 = getIntersectionLengthLORCrystal(lor_1_s);
		if (delta_1 > 10 * m_crystalDepth)
		{
			std::cerr << "Error computing propagation distance in detector in "
			             "SingleScatterSimulation::compute_single_scatter_in_"
			             "lor() (1).\n"
			          << std::endl;
			exit(-1);
		}

		att_s_2_511 =
		    OperatorProjectorSiddon::singleForwardProjection(&mr_mu, lor_2_s) /
		    10.0;

		att_s_2 = att_s_2_511 * mu_scaling_factor;
		lamb_s_2 = OperatorProjectorSiddon::singleForwardProjection(&mr_lambda,
		                                                            lor_2_s);
		delta_2 = getIntersectionLengthLORCrystal(lor_2_s);

		// Check that the distance between the two cylinders is not too big
		if (delta_2 > 10 * m_crystalDepth)
		{
			std::cerr << "Error computing propagation distance in detector in "
			          << "SingleScatterSimulation::compute_single_scatter_in_"
			          << "lor() (2)." << std::endl
			          << std::endl;
			exit(-1);
		}

		// geometric efficiencies (n1 and n2 must be normalized unit
		// vectors):
		sig_s_1 = std::abs(n1.scalProd(u));
		sig_s_2 = std::abs(n2.scalProd(v));

		// detection efficiencies (energy+spatial):
		eps_s_1_511 = eps_s_2_511 = util::erfc(tmp511);
		eps_s_1 = eps_s_2 = util::erfc(tmp);
		mu_det = getMuDet(energy, m_crystalMaterial);
		eps_s_1_511 *= 1 - exp(-delta_1 * mu_det_511);
		eps_s_2_511 *= 1 - exp(-delta_2 * mu_det_511);
		eps_s_1 *= 1 - exp(-delta_1 * mu_det);
		eps_s_2 *= 1 - exp(-delta_2 * mu_det);

		fac1 = lamb_s_1 * exp(-att_s_1_511 - att_s_2);
		fac1 *= eps_s_1_511 * eps_s_2;  // I^A
		fac2 = lamb_s_2 * exp(-att_s_1 - att_s_2_511);
		fac2 *= eps_s_2_511 * eps_s_1;  // I^B

		res += vatt * dsigcompdomega * (fac1 + fac2) * sig_s_1 * sig_s_2 /
		       (dist1 * dist1 * dist2 * dist2 * 4 * PI);
	}
	// divide the result by the sensitivity for trues for that LOR (don't do
	// this anymore because we use the sensitivity corrected scatter
	// sinogram in the reconstruction):
	u.update(p2 - p1);
	dist1 = u.getNorm();
	u.x /= dist1;
	u.y /= dist1;
	u.z /= dist1;
	sig_s_1 = std::abs(n1.scalProd(u));
	sig_s_2 = std::abs(n2.scalProd(u));
	eps_s_1_511 = eps_s_2_511 = util::erfc(tmp511);
	Vector3D mid{p1.x + p2.x, p1.y + p2.y, p1.z + p2.z};
	mid.x /= 2;
	mid.y /= 2;
	mid.z /= 2;
	lor_1_s.update(p1, mid);
	delta_1 = getIntersectionLengthLORCrystal(lor_1_s);
	lor_2_s.update(p2, mid);
	delta_2 = getIntersectionLengthLORCrystal(lor_2_s);
	eps_s_1_511 *= 1 - exp(-delta_1 * mu_det_511);
	eps_s_2_511 *= 1 - exp(-delta_2 * mu_det_511);
	// YN: Changed eps_s_1_511 * eps_s_1_511 to eps_s_1_511 * eps_s_2_511
	res /= eps_s_1_511 * eps_s_2_511 * sig_s_1 * sig_s_2 /
	       (dist1 * dist1 * 4 * PI);

	return res;
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

// This is the integrated KN formula up to a proportionaity constant:
float SingleScatterSimulator::getMuScalingFactor(float energy)
{
	float a = energy / 511.0;
	float res = (1 + a) / (a * a);
	res *= 2.0 * (1 + a) / (1 + 2.0 * a) - log(1 + 2.0 * a) / a;
	res += log(1 + 2 * a) / (2 * a) - (1 + 3 * a) / ((1 + 2 * a) * (1 + 2 * a));
	res /= 20.0 / 9.0 - 1.5 * log(3.0);
	return res;
}

// The first point of lor must be the detector, the second point must be the
// scatter point.
float SingleScatterSimulator::getIntersectionLengthLORCrystal(
    const Line3D& lor) const
{
	Vector3D a1, a2, inter1, inter2;
	// direction of prop.
	const Vector3D n1 = (lor.point1) - (lor.point2);

	// Compute entry point:
	m_cyl1.doesLineIntersectCylinder(lor, a1, a2);
	Vector3D n2 = a1 - (lor.point2);
	if (n2.scalProd(n1) > 0)
		inter1.update(a1);
	else
		inter1.update(a2);

	// Compute out point:
	m_cyl2.doesLineIntersectCylinder(lor, a1, a2);
	n2 = a1 - (lor.point2);
	if (n2.scalProd(n1) > 0)
		inter2.update(a1);
	else
		inter2.update(a2);

	// Return distance of prop. in detector:
	const float dist = (inter1 - inter2).getNorm();
	return dist;
}

// Return true if the line lor does not cross the end plates
// First point is detector, second point is scatter point
bool SingleScatterSimulator::passCollimator(const Line3D& lor) const
{
	if (m_collimatorRadius < 1e-7)
		return true;
	Vector3D inter;
	if (lor.point2.z < 0)
		inter = m_endPlate1.findInterLine(lor);
	else
		inter = m_endPlate2.findInterLine(lor);
	const float r = std::sqrt(inter.x * inter.x + inter.y * inter.y);
	if (r < m_collimatorRadius)
	{
		return true;
	}
	return false;
}

// This is the differential KN formula up to a proportionality constant for
// Ep=511keV.
float SingleScatterSimulator::getKleinNishina(float cosa)
{
	float res = (1 + cosa * cosa) / 2;
	res /= (2 - cosa) * (2 - cosa);
	res *= 1 + (1 - cosa) * (1 - cosa) / ((2 - cosa) * (1 + cosa * cosa));
	return res;
}

}  // namespace scatter
}  // namespace yrt
