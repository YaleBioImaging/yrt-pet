/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * Estimate detector efficiency from a measured reference histogram.
 *
 * Model: N(z,phi,r) = G x B x d x A x eps(d1) x eps(d2) x scale
 *
 * Geometry components are computed sequentially from the scanner
 * geometry (LUT):
 *   B = average G per block pair (normalized to mean 1)
 *   d = average G/B per block pair + ring (normalized to mean 1)
 *   A = average G/(B*d) per z-bin (normalized to mean 1)
 * Only detector efficiency (eps) is estimated from the reference.
 *
 * Steps:
 *   1. Compute geometric factor G per LOR (crystal positions/orientations)
 *   2. Compute block profile B from geometry (average G per block-pair)
 *   3. Compute crystal interference d from geometry (average G/B per
 *      block-pair and ring)
 *   4. Compute plane efficiency A from geometry (average G/(B*d) per z-bin)
 *   5. Estimate crystal efficiency eps(d) via iterative fan-sum
 *   6. Compute global scale factor
 *   7. Write components to output files
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../ArgumentReader.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"

using namespace yrt;

// ---------------------------------------------------------------------------
// Helper: geometric factor G for a detector pair
//   G = |n1 . d| * |n2 . d| / |d|^3
// where d = p2 - p1, n1/n2 are detector normals
// ---------------------------------------------------------------------------
static float computeGeoFactor(const Vector3D& p1, const Vector3D& p2,
                              const Vector3D& n1, const Vector3D& n2)
{
	Vector3D d = p2 - p1;
	float dist_sq = d.getNormSquared();
	if (dist_sq < 1e-12f) return 0.0f;
	float dist = std::sqrt(dist_sq);
	float cos1 = std::abs(n1.scalProd(d));
	float cos2 = std::abs(n2.scalProd(d));
	return cos1 * cos2 / (dist_sq * dist);
}

// ---------------------------------------------------------------------------
// Precompute parity-based geo lookup table (matching Python convention)
//   geo[parity][dz][rr][doi1][doi2] = G for parity=phi%2, ring-diff=dz,
//                                     r_ring=rr, DOI layers
// Uses shift=0 (Python convention) instead of per-phi shift (phi//2).
// ---------------------------------------------------------------------------
static std::vector<float> precomputeGeoParity(const Scanner& scanner,
                                              size_t maxRingDiff,
                                              size_t nR_ring,
                                              size_t nDOI)
{
	size_t nDetsRing = scanner.detsPerRing;
	int m_rCut = scanner.minAngDiff / 2;
	if (scanner.minAngDiff % 2 != 0 && nDetsRing % 4 == 0)
		m_rCut++;
	size_t nDOIPoss = nDOI * nDOI;
	size_t nTables = 2 * (maxRingDiff + 1) * nR_ring * nDOIPoss;
	std::vector<float> geo(nTables, 0.0f);
	int doi_stride = static_cast<int>(nDetsRing * scanner.numRings);

	for (size_t parity = 0; parity < 2; parity++)
	{
		for (size_t dz = 0; dz <= maxRingDiff; dz++)
		{
			for (size_t rr = 0; rr < nR_ring; rr++)
			{
				for (size_t doi1 = 0; doi1 < nDOI; doi1++)
				{
					for (size_t doi2 = 0; doi2 < nDOI; doi2++)
					{
						int d01 = 0;
						int d02 = static_cast<int>(nDetsRing / 2);
						if (parity != 0)
							d02 = static_cast<int>(
							          nDetsRing / 2 + 1);
						int dr1 = d01 + (static_cast<int>(rr)
						                - static_cast<int>(
						                      nDetsRing / 4)
						                + m_rCut);
						int dr2 = d02 - (static_cast<int>(rr)
						                - static_cast<int>(
						                      nDetsRing / 4)
						                + m_rCut);
						if (dr1 < 0)
							dr1 += static_cast<int>(nDetsRing);
						if (dr2 < 0)
							dr2 += static_cast<int>(nDetsRing);
						int d1_ring = dr1
						              % static_cast<int>(nDetsRing);
						int d2_ring = dr2
						              % static_cast<int>(nDetsRing);
						if (d1_ring > d2_ring)
							std::swap(d1_ring, d2_ring);

						det_id_t d1 = static_cast<det_id_t>(
						    d1_ring
						    + static_cast<int>(doi1) * doi_stride);
						det_id_t d2 = static_cast<det_id_t>(
						    d2_ring
						    + static_cast<int>(dz)
						      * static_cast<int>(nDetsRing)
						    + static_cast<int>(doi2) * doi_stride);

						Vector3D p1 = scanner.getDetectorPos(d1);
						Vector3D p2 = scanner.getDetectorPos(d2);
						Vector3D n1 = scanner.getDetectorOrient(d1);
						Vector3D n2 = scanner.getDetectorOrient(d2);
						float G_val = computeGeoFactor(
						    p1, p2, n1, n2);

						size_t idx = parity
						             * (maxRingDiff + 1) * nR_ring
						             * nDOIPoss
						             + dz * nR_ring * nDOIPoss
						             + rr * nDOIPoss
						             + doi1 * nDOI + doi2;
						geo[idx] = G_val;
					}
				}
			}
		}
	}
	return geo;
}

static inline float lookupGeo(const float* geo, size_t parity, size_t dz,
                              size_t rr, size_t doi1, size_t doi2,
                              size_t maxRingDiff, size_t nR_ring,
                              size_t nDOI)
{
	size_t nDOIPoss = nDOI * nDOI;
	size_t idx = parity * (maxRingDiff + 1) * nR_ring * nDOIPoss
	             + dz * nR_ring * nDOIPoss
	             + rr * nDOIPoss
	             + doi1 * nDOI + doi2;
	return geo[idx];
}

// ---------------------------------------------------------------------------
// Thread-local accumulator for epsilon estimation
// ---------------------------------------------------------------------------
struct ThreadAccumEps
{
	std::vector<double> eps_accum;
	std::vector<uint64_t> eps_cnt;
	ThreadAccumEps(size_t ndet) : eps_accum(ndet, 0.0), eps_cnt(ndet, 0) {}
};

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
	try
	{
		io::ArgumentRegistry registry{};

		std::string coreGroup   = "0. Core";
		std::string inputGroup  = "1. Input";
		std::string outputGroup = "2. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument("measured",
		                          "Measured norm Histogram3D file", true,
		                          io::TypeOfArgument::STRING, "",
		                          inputGroup, "m");
		registry.registerArgument("out", "Output directory for components",
		                          true, io::TypeOfArgument::STRING, "",
		                          outputGroup, "o");
		registry.registerArgument("block_size",
		                          "Detectors per block (default: scanner)",
		                          false, io::TypeOfArgument::INT, 0,
		                          coreGroup);
		registry.registerArgument("fan_sum_iters",
		                          "Fan-sum iterations (default: 5)",
		                          false, io::TypeOfArgument::INT, 5,
		                          coreGroup);
		registry.registerArgument("min_count",
		                          "Min counts to include LOR (default: 0)",
		                          false, io::TypeOfArgument::FLOAT, 0.0f,
		                          coreGroup);
		registry.registerArgument("is_norm",
		                          "Reference is a norm (1/sensitivity);"
		                          " invert when estimating eps",
		                          false, io::TypeOfArgument::BOOL, false,
		                          coreGroup);

		io::ArgumentReader config{
		    registry,
		    "Estimate detector efficiency from a measured norm\n"
		    "histogram.\n"
		    "Model: N = G x B x d x A x eps(d1) x eps(d2) x scale\n"
		    "Geometry (G, B, d, A) computed from scanner LUT.\n"
		    "Only detector efficiency (eps) estimated from reference.\n"
		    "Saves eps and scale to output directory."};

		if (!config.loadFromCommandLine(argc, argv)) return 0;
		if (!config.validate()) return -1;

		auto scanner_fname   = config.getValue<std::string>("scanner");
		auto measured_fname  = config.getValue<std::string>("measured");
		auto out_dir         = config.getValue<std::string>("out");
		auto numThreads      = config.getValue<int>("num_threads");
		int fan_sum_iters    = config.getValue<int>("fan_sum_iters");
		float min_count      = config.getValue<float>("min_count");
		int block_size_opt   = config.getValue<int>("block_size");
		bool is_norm         = config.getValue<bool>("is_norm");

		globals::setNumThreads(numThreads);
		numThreads = globals::getNumThreads();

		// -----------------------------------------------------------------
		// Load scanner and histogram
		// -----------------------------------------------------------------
		std::cout << "Loading scanner: " << scanner_fname << std::endl;
		auto scanner = std::make_unique<Scanner>(scanner_fname);

		std::cout << "Loading measured norm: " << measured_fname << std::endl;
		auto histo = std::make_unique<Histogram3DOwned>(
		    *scanner, measured_fname);

		size_t num_dets  = scanner->getExpectedNumDets();
		size_t total_bins = histo->count();
		size_t nRings    = scanner->numRings;
		size_t nDetsRing = scanner->detsPerRing;
		size_t nDOI      = scanner->numDOI;
		size_t maxRingDiff = scanner->maxRingDiff;
		size_t nZ        = histo->numZBin;
		size_t nPhi      = histo->numPhi;
		size_t nR        = histo->numR;
		size_t bs        = (block_size_opt > 0)
		                       ? static_cast<size_t>(block_size_opt)
		                       : scanner->detsPerBlock;
		size_t nblocks   = bs * bs;
		float min_count_f = min_count;

		// Validate block size
		if (bs == 0)
		{
			std::cerr << "ERROR: block_size cannot be 0. "
			          << "Set via --block_size or in scanner config."
			          << std::endl;
			return -1;
		}

		std::cout << "  Rings: " << nRings
		          << "  Dets/ring: " << nDetsRing
		          << "  DOI: " << nDOI
		          << "  nZ: " << nZ
		          << "  nPhi: " << nPhi
		          << "  nR: " << nR
		          << "  Total bins: " << total_bins
		          << "  Num dets: " << num_dets
		          << "  Block size: " << bs
		          << std::endl;
		std::cout << "  Fan-sum iters: " << fan_sum_iters
		          << "  min_count: " << min_count_f
		          << std::endl;

		// Access histogram data as a flat array for fast iteration
		const Array3DBase<float>& data = histo->getData();
		const float* data_ptr = data.getRawPointer();

		// -----------------------------------------------------------------
		// Step 1: Precompute geometric factor G (Python-compatible parity)
		// -----------------------------------------------------------------
		size_t nDOIPoss  = nDOI * nDOI;
		size_t nR_ring   = nR / nDOIPoss;
		std::cout << "\n[1/6] Precomputing geometric factor G (parity-based,"
		          << " Python-compatible)..." << std::endl;
		auto geo_parity = precomputeGeoParity(
		    *scanner, maxRingDiff, nR_ring, nDOI);
		const float* geo_ptr = geo_parity.data();
		std::cout << "  Geo table size: "
		          << 2 << "x" << (maxRingDiff + 1) << "x" << nR_ring
		          << "x" << nDOI << "x" << nDOI
		          << "  range=[" << *std::min_element(geo_parity.begin(),
		                                             geo_parity.end())
		          << "," << *std::max_element(geo_parity.begin(),
		                                      geo_parity.end())
		          << "]" << std::endl;

		// -----------------------------------------------------------------
		// Step 2: Compute block profile B from geometry
		//   B(b1, b2) = average G over all LORs with same block-pair,
		//               normalized to mean 1
		// -----------------------------------------------------------------
		std::cout << "\n[2/6] Computing block profile B from geometry..."
		          << std::endl;

		std::vector<double> B_accum(nblocks, 0.0);
		std::vector<uint64_t> B_count(nblocks, 0);

		for (size_t binId = 0; binId < total_bins; ++binId)
		{
			det_pair_t dp = histo->getDetectorPair(binId);
			if (dp.d1 >= num_dets || dp.d2 >= num_dets) continue;
			if (dp.d1 == dp.d2) continue;

			coord_t r, phi, z_bin;
			histo->getCoordsFromBinId(binId, r, phi, z_bin);
			coord_t z1, z2;
			histo->getZ1Z2(z_bin, z1, z2);
			coord_t dz_g = (z1 > z2) ? (z1 - z2) : (z2 - z1);

			coord_t r_ring_g = r / nDOIPoss;
			size_t doi_case_g = r % nDOIPoss;
			size_t doi1_g = doi_case_g % nDOI;
			size_t doi2_g = doi_case_g / nDOI;
			float G = lookupGeo(geo_ptr, phi % 2, dz_g, r_ring_g,
			                    doi1_g, doi2_g,
			                    maxRingDiff, nR_ring, nDOI);
			if (G <= 0.0f) continue;

			size_t b1 = dp.d1 % bs;
			size_t b2 = dp.d2 % bs;
			size_t bidx = b1 * bs + b2;

			B_accum[bidx] += G;
			B_count[bidx]++;
		}

		std::vector<float> B(nblocks, 1.0f);
		double B_sum_valid = 0.0;
		size_t B_count_valid = 0;
		for (size_t i = 0; i < nblocks; ++i)
		{
			if (B_count[i] > 0)
			{
				B[i] = static_cast<float>(
				    B_accum[i] / static_cast<double>(B_count[i]));
				B_sum_valid += B[i];
				B_count_valid++;
			}
		}
		float B_mean = (B_count_valid > 0)
		                   ? static_cast<float>(
		                         B_sum_valid / B_count_valid)
		                   : 1.0f;
		for (size_t i = 0; i < nblocks; ++i)
		{
			if (B_count[i] > 0)
				B[i] /= B_mean;
		}

		float B_min = *std::min_element(B.begin(), B.end());
		float B_max = *std::max_element(B.begin(), B.end());
		std::cout << "  B shape=(" << bs << "," << bs << ")"
		          << "  range=[" << B_min << "," << B_max << "]"
		          << std::endl;

		// -----------------------------------------------------------------
		// Step 3: Compute crystal interference d from geometry (sequential)
		//   d(b1, b2, r) = average G/B over all LORs with same block-pair
		//                   and ring, normalized to mean 1
		// -----------------------------------------------------------------
		std::cout << "\n[3/6] Computing crystal interference d from geometry..."
		          << std::endl;

		size_t n_bins_d = nR * nblocks;
		std::vector<double> D_accum(n_bins_d, 0.0);
		std::vector<uint64_t> D_count(n_bins_d, 0);

		for (size_t binId = 0; binId < total_bins; ++binId)
		{
			det_pair_t dp = histo->getDetectorPair(binId);
			if (dp.d1 >= num_dets || dp.d2 >= num_dets) continue;
			if (dp.d1 == dp.d2) continue;

			coord_t r, phi, z_bin;
			histo->getCoordsFromBinId(binId, r, phi, z_bin);
			coord_t z1, z2;
			histo->getZ1Z2(z_bin, z1, z2);
			coord_t dz_g = (z1 > z2) ? (z1 - z2) : (z2 - z1);

			coord_t r_ring_g = r / nDOIPoss;
			size_t doi_case_g = r % nDOIPoss;
			size_t doi1_g = doi_case_g % nDOI;
			size_t doi2_g = doi_case_g / nDOI;
			float G = lookupGeo(geo_ptr, phi % 2, dz_g, r_ring_g,
			                    doi1_g, doi2_g,
			                    maxRingDiff, nR_ring, nDOI);
			if (G <= 0.0f) continue;

			size_t b1 = dp.d1 % bs;
			size_t b2 = dp.d2 % bs;
			size_t bidx = b1 * bs + b2;
			if (B[bidx] <= 0.0f) continue;

			// Sequential: weight by G/B
			double G_over_B = static_cast<double>(G)
			                  / static_cast<double>(B[bidx]);
			size_t ridx = r * nblocks + bidx;

			D_accum[ridx] += G_over_B;
			D_count[ridx]++;
		}

		std::vector<float> d_flat(n_bins_d, 1.0f);
		double D_sum_valid = 0.0;
		size_t D_count_valid = 0;
		for (size_t i = 0; i < n_bins_d; ++i)
		{
			if (D_count[i] > 0)
			{
				d_flat[i] = static_cast<float>(
				    D_accum[i] / static_cast<double>(D_count[i]));
				D_sum_valid += d_flat[i];
				D_count_valid++;
			}
		}
		float D_mean = (D_count_valid > 0)
		                   ? static_cast<float>(
		                         D_sum_valid / D_count_valid)
		                   : 1.0f;
		for (size_t i = 0; i < n_bins_d; ++i)
		{
			if (D_count[i] > 0)
				d_flat[i] /= D_mean;
		}

		float d_min = *std::min_element(d_flat.begin(), d_flat.end());
		float d_max = *std::max_element(d_flat.begin(), d_flat.end());
		std::cout << "  d shape=(" << nR << "," << bs << "," << bs << ")"
		          << "  range=[" << d_min << "," << d_max << "]"
		          << std::endl;

		// -----------------------------------------------------------------
		// Step 4: Compute plane efficiency A from geometry (sequential)
		//   A(z_bin) = average G/(B*d) over all LORs in same z-bin,
		//              normalized to mean 1
		// -----------------------------------------------------------------
		std::cout << "\n[4/6] Computing plane efficiency A from geometry..."
		          << std::endl;

		std::vector<double> A_accum(nZ, 0.0);
		std::vector<uint64_t> A_count(nZ, 0);

		for (size_t binId = 0; binId < total_bins; ++binId)
		{
			det_pair_t dp = histo->getDetectorPair(binId);
			if (dp.d1 >= num_dets || dp.d2 >= num_dets) continue;
			if (dp.d1 == dp.d2) continue;

			coord_t r, phi, z_bin;
			histo->getCoordsFromBinId(binId, r, phi, z_bin);
			coord_t z1_a, z2_a;
			histo->getZ1Z2(z_bin, z1_a, z2_a);
			coord_t dz_g = (z1_a > z2_a) ? (z1_a - z2_a)
			                             : (z2_a - z1_a);

			coord_t r_ring_g = r / nDOIPoss;
			size_t doi_case_g = r % nDOIPoss;
			size_t doi1_g = doi_case_g % nDOI;
			size_t doi2_g = doi_case_g / nDOI;
			float G = lookupGeo(geo_ptr, phi % 2, dz_g, r_ring_g,
			                    doi1_g, doi2_g,
			                    maxRingDiff, nR_ring, nDOI);
			if (G <= 0.0f) continue;

			size_t b1 = dp.d1 % bs;
			size_t b2 = dp.d2 % bs;
			size_t bidx = b1 * bs + b2;
			if (B[bidx] <= 0.0f) continue;

			float d_val = d_flat[r * nblocks + bidx];
			if (d_val <= 0.0f) continue;

			// Sequential: weight by G/(B*d)
			double G_over_Bd = static_cast<double>(G)
			                   / (static_cast<double>(B[bidx])
			                      * static_cast<double>(d_val));

			A_accum[z_bin] += G_over_Bd;
			A_count[z_bin]++;
		}

		std::vector<float> A(nZ, 1.0f);
		double A_sum_valid = 0.0;
		size_t A_count_valid = 0;
		for (size_t i = 0; i < nZ; ++i)
		{
			if (A_count[i] > 0)
			{
				A[i] = static_cast<float>(
				    A_accum[i] / static_cast<double>(A_count[i]));
				A_sum_valid += A[i];
				A_count_valid++;
			}
		}
		float A_mean = (A_count_valid > 0)
		                   ? static_cast<float>(
		                         A_sum_valid / A_count_valid)
		                   : 1.0f;
		for (size_t i = 0; i < nZ; ++i)
		{
			if (A_count[i] > 0)
				A[i] /= A_mean;
		}

		float A_min = *std::min_element(A.begin(), A.end());
		float A_max = *std::max_element(A.begin(), A.end());
		std::cout << "  A shape=(" << nZ << ")"
		          << "  range=[" << A_min << "," << A_max << "]"
		          << std::endl;

		// -----------------------------------------------------------------
		// Step 5: Estimate epsilon via fan-sum iteration
		// -----------------------------------------------------------------
		std::cout << "\n[5/6] Estimating epsilon via fan-sum ("
		          << fan_sum_iters << " iterations)..."
		          << (is_norm ? " [norm mode]" : "") << std::endl;

		std::vector<double> log_eps(num_dets, 0.0);

		for (int iter = 0; iter < fan_sum_iters; ++iter)
		{
			std::vector<ThreadAccumEps> e_accums;
			e_accums.reserve(numThreads);
			for (int t = 0; t < numThreads; ++t)
				e_accums.emplace_back(num_dets);

			util::ProgressDisplayMultiThread progressE(
			    numThreads, static_cast<int64_t>(total_bins), 5);

			util::parallelForChunked(
			    total_bins, numThreads,
			    [&](size_t binId, size_t threadId)
			    {
				    progressE.incrementProgress(threadId, 1);

				    float m_val = data_ptr[binId];
				    if (m_val <= min_count_f) return;

				    det_pair_t dp = histo->getDetectorPair(binId);
				    if (dp.d1 >= num_dets || dp.d2 >= num_dets) return;
				    if (dp.d1 == dp.d2) return;

				    coord_t r, phi, z_bin;
				    histo->getCoordsFromBinId(binId, r, phi, z_bin);

				    // Compute G from precomputed table
				    coord_t r_ring_g = r / nDOIPoss;
				    coord_t z1_e, z2_e;
				    histo->getZ1Z2(z_bin, z1_e, z2_e);
				    coord_t dz_g = (z1_e > z2_e) ? (z1_e - z2_e)
				                                 : (z2_e - z1_e);
				    size_t doi_case_g = r % nDOIPoss;
				    size_t doi1_g = doi_case_g % nDOI;
				    size_t doi2_g = doi_case_g / nDOI;
				    float G = lookupGeo(geo_ptr, phi % 2, dz_g, r_ring_g,
				                        doi1_g, doi2_g,
				                        maxRingDiff, nR_ring, nDOI);
				    if (G <= 0.0f) return;

				    size_t b1 = dp.d1 % bs;
				    size_t b2 = dp.d2 % bs;
				    size_t bidx = b1 * bs + b2;
				    float B_val = B[bidx];
				    if (B_val <= 0.0f) return;

				    float d_val = d_flat[static_cast<size_t>(r) * nblocks
				                         + b1 * bs + b2];
				    if (d_val <= 0.0f) return;

				    float A_val = A[z_bin];
				    if (A_val <= 0.0f) return;

				    double gbdA = static_cast<double>(G) * B_val * d_val
				                  * A_val;
				    double log_m = std::log(
				        static_cast<double>(m_val));
				    double log_res = (is_norm ? -log_m : log_m)
				                     - std::log(gbdA);

				    auto& acc = e_accums[threadId];
				    acc.eps_accum[dp.d1] += log_res - log_eps[dp.d2];
				    acc.eps_accum[dp.d2] += log_res - log_eps[dp.d1];
				    acc.eps_cnt[dp.d1]++;
				    acc.eps_cnt[dp.d2]++;
			    });

			// Merge
			std::vector<double> e_accum(num_dets, 0.0);
			std::vector<uint64_t> e_cnt(num_dets, 0);
			for (auto& acc : e_accums)
			{
				for (size_t i = 0; i < num_dets; ++i)
				{
					e_accum[i] += acc.eps_accum[i];
					e_cnt[i]   += acc.eps_cnt[i];
				}
			}

			// Update log_eps
			double log_eps_sum = 0.0;
			uint64_t e_valid = 0;
			for (size_t i = 0; i < num_dets; ++i)
			{
				if (e_cnt[i] > 0)
				{
					log_eps[i] = e_accum[i]
					             / static_cast<double>(e_cnt[i]);
					log_eps_sum += log_eps[i];
					e_valid++;
				}
			}
			double log_eps_mean = (e_valid > 0)
			                          ? log_eps_sum
			                            / static_cast<double>(e_valid)
			                          : 0.0;
			for (auto& le : log_eps) le -= log_eps_mean;

			// Stats
			double eps_mean = 0.0;
			double eps_var = 0.0;
			uint64_t eps_valid = 0;
			for (size_t i = 0; i < num_dets; ++i)
			{
				double e = std::exp(log_eps[i]);
				eps_mean += e;
				eps_valid++;
			}
			eps_mean /= static_cast<double>(eps_valid);
			for (size_t i = 0; i < num_dets; ++i)
			{
				double e = std::exp(log_eps[i]);
				eps_var += (e - eps_mean) * (e - eps_mean);
			}
			eps_var /= static_cast<double>(eps_valid);

			uint64_t e_avg_cnt = 0;
			for (auto& c : e_cnt) e_avg_cnt += c;
			e_avg_cnt /= num_dets;

			std::cout << "  iter " << (iter + 1) << "/" << fan_sum_iters
			          << ": eps std=" << std::sqrt(eps_var)
			          << "  cnt_avg=" << e_avg_cnt
			          << std::endl;
		}

		// Final epsilon
		std::vector<float> eps(num_dets);
		double eps_final_mean = 0.0;
		for (size_t i = 0; i < num_dets; ++i)
		{
			eps[i] = static_cast<float>(std::exp(log_eps[i]));
			eps_final_mean += eps[i];
		}
		eps_final_mean /= static_cast<double>(num_dets);
		for (auto& e : eps) e /= static_cast<float>(eps_final_mean);

		double eps_std = 0.0;
		for (auto& e : eps) eps_std += (e - 1.0) * (e - 1.0);
		eps_std = std::sqrt(eps_std / num_dets);
		std::cout << "  Final eps: mean=1.0  std=" << eps_std << std::endl;

		// -----------------------------------------------------------------
		// Step 6: Compute global scale from first direct plane
		// -----------------------------------------------------------------
		std::cout << "\n[6/6] Computing scale..." << std::endl;

		double sum_ref = 0.0;
		double sum_pred = 0.0;

		for (size_t binId = 0; binId < total_bins; ++binId)
		{
			float m_val = data_ptr[binId];
			if (m_val <= 0.0f) continue;

			det_pair_t dp = histo->getDetectorPair(binId);
			if (dp.d1 >= num_dets || dp.d2 >= num_dets) continue;
			if (dp.d1 == dp.d2) continue;

			coord_t r, phi, z_bin;
			histo->getCoordsFromBinId(binId, r, phi, z_bin);
			coord_t z1_s, z2_s;
			histo->getZ1Z2(z_bin, z1_s, z2_s);
			if (z_bin != 0) continue;  // only first z-bin

			// Compute G from precomputed table
			coord_t r_ring_g = r / nDOIPoss;
			coord_t dz_g = (z1_s > z2_s) ? (z1_s - z2_s) : (z2_s - z1_s);
			size_t doi_case_g = r % nDOIPoss;
			size_t doi1_g = doi_case_g % nDOI;
			size_t doi2_g = doi_case_g / nDOI;
			float G = lookupGeo(geo_ptr, phi % 2, dz_g, r_ring_g,
			                    doi1_g, doi2_g,
			                    maxRingDiff, nR_ring, nDOI);
			if (G <= 0.0f) continue;

			size_t b1 = dp.d1 % bs;
			size_t b2 = dp.d2 % bs;
			size_t bidx = b1 * bs + b2;
			if (B[bidx] <= 0.0f) continue;

			float d_val = d_flat[static_cast<size_t>(r) * nblocks + bidx];
			if (d_val <= 0.0f) continue;

			double pred = static_cast<double>(G) * B[bidx] * d_val
			              * A[z_bin] * eps[dp.d1] * eps[dp.d2];
			if (pred <= 0.0) continue;

			if (is_norm)
				sum_ref += 1.0 / m_val;
			else
				sum_ref += m_val;
			sum_pred += pred;
		}

		float scale = (sum_pred > 0.0)
		                  ? static_cast<float>(sum_ref / sum_pred)
		                  : 1.0f;
		std::cout << "  Scale = " << scale << std::endl;

		// -----------------------------------------------------------------
		// Save components
		// -----------------------------------------------------------------
		std::cout << "\nSaving components to " << out_dir << "/" << std::endl;

		// Create output directory
		std::filesystem::create_directories(out_dir);

		// B: (bs, bs) as RWD 2D
		{
			Array2DOwned<float> arr;
			arr.allocate(bs, bs);
			std::copy(B.begin(), B.end(), arr.getRawPointer());
			arr.writeToFile(out_dir + "/block_profile.rwd");
		}
		std::cout << "  block_profile.rwd  (" << bs << "," << bs << ")"
		          << std::endl;

		// d: (nR, bs, bs) as RWD 3D
		{
			Array3DOwned<float> arr;
			arr.allocate(nR, bs, bs);
			std::copy(d_flat.begin(), d_flat.end(), arr.getRawPointer());
			arr.writeToFile(out_dir + "/d_pattern.rwd");
		}
		std::cout << "  d_pattern.rwd      (" << nR << "," << bs << "," << bs
		          << ")" << std::endl;

		// geo_lut: (maxRingDiff+1, nR_ring, 2, nDOI, nDOI) as RWD 5D
		{
			Array5DOwned<float> arr;
			arr.allocate(maxRingDiff + 1, nR_ring, 2, nDOI, nDOI);
			size_t nDOIPoss = nDOI * nDOI;
			size_t dz_stride = nR_ring * nDOIPoss;
			size_t p_stride = (maxRingDiff + 1) * dz_stride;
			for (size_t dz = 0; dz <= maxRingDiff; dz++)
			{
				for (size_t rr = 0; rr < nR_ring; rr++)
				{
					for (size_t parity = 0; parity < 2;
					     parity++)
					{
						for (size_t doi1 = 0; doi1 < nDOI;
						     doi1++)
						{
							for (size_t doi2 = 0;
							     doi2 < nDOI; doi2++)
							{
								size_t src_idx =
								    parity * p_stride
								    + dz * dz_stride
								    + rr * nDOIPoss
								    + doi1 * nDOI + doi2;
								arr.set(
								    {dz, rr, parity,
								     doi1, doi2},
								    geo_parity
								        [src_idx]);
							}
						}
					}
				}
			}
			arr.writeToFile(out_dir + "/geo_lut.rwd");
		}
		std::cout << "  geo_lut.rwd       ("
		          << (maxRingDiff + 1) << "," << nR_ring << ",2,"
		          << nDOI << "," << nDOI << ")" << std::endl;

		// A: (nZ) as RWD 1D
		{
			Array1DOwned<float> arr;
			arr.allocate(nZ);
			std::copy(A.begin(), A.end(), arr.getRawPointer());
			arr.writeToFile(out_dir + "/plane_eff_A.rwd");
		}
		std::cout << "  plane_eff_A.rwd    (" << nZ << ")" << std::endl;

		// eps: (num_dets) as RWD 1D
		{
			Array1DOwned<float> arr;
			arr.allocate(num_dets);
			std::copy(eps.begin(), eps.end(), arr.getRawPointer());
			arr.writeToFile(out_dir + "/eps.rwd");
		}
		std::cout << "  eps.rwd            (" << num_dets << ")" << std::endl;

		// Detector positions LUT from scanner geometry
		{
			Array2DOwned<float> detLUT;
			scanner->createLUT(detLUT);
			detLUT.writeToFile(out_dir + "/det_positions.rwd");
		}
		std::cout << "  det_positions.rwd  (" << num_dets << ",6)" << std::endl;

		// Scale: plain text
		{
			std::ofstream f(out_dir + "/scale.txt");
			if (f) f << scale << std::endl;
			else std::cerr << "ERROR: Failed to write scale" << std::endl;
		}
		std::cout << "  scale.txt          = " << scale << std::endl;

		std::cout << "\nDone." << std::endl;

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}
