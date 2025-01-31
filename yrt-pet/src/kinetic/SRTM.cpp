/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "kinetic/SRTM.hpp"

#include "omp.h"

#include <limits>
#include <memory>

#if BUILD_PYBIND11

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>

pybind11::array_t<double> fit_srtm_basis(
    pybind11::array_t<double> tac_all, pybind11::array_t<double> kin_p,
    pybind11::array_t<double> A_all, pybind11::array_t<double> B_all,
    pybind11::array_t<double> Rinv_Qt_all, pybind11::array_t<double> W,
    pybind11::array_t<double> Lambda, double alpha,
    pybind11::array_t<double> kappa_list, int num_threads)
{
	pybind11::buffer_info buf_tac_all = tac_all.request();
	pybind11::buffer_info buf_kin_p = kin_p.request();
	pybind11::buffer_info buf_A_all = A_all.request();
	pybind11::buffer_info buf_B_all = B_all.request();
	pybind11::buffer_info buf_Rinv_Qt_all = Rinv_Qt_all.request();
	pybind11::buffer_info buf_W = W.request();
	pybind11::buffer_info buf_Lambda = Lambda.request();
	pybind11::buffer_info buf_kappa_list = kappa_list.request();

	size_t num_pix = buf_tac_all.shape[1];
	int num_frames = buf_tac_all.shape[0];
	int num_kappa = buf_kappa_list.size;
	const int num_k = 3;

	if (buf_kin_p.ndim == 2 &&
	    (buf_kin_p.shape[0] != num_k ||
	     buf_kin_p.shape[1] != static_cast<long>(num_pix)))
	{
		throw std::runtime_error("kin_p matrix should have shape [P, X]");
	}
	if (buf_A_all.shape[0] != num_kappa || buf_A_all.shape[1] != num_frames ||
	    buf_A_all.shape[2] != 2)
	{
		std::stringstream err;
		err << "A matrix should have shape [K, T, 2] " << "([" << num_kappa
		    << ", " << num_frames << ", 2])";
		throw std::runtime_error(err.str());
	}
	if (buf_B_all.shape[0] != num_kappa || buf_B_all.shape[1] != num_k ||
	    buf_B_all.shape[2] != 2)
	{
		std::stringstream err;
		err << "B matrix should have shape [K, P, 2] " << "([" << num_kappa
		    << ", " << num_k << ", 2])";
		throw std::runtime_error(err.str());
	}
	if (buf_Rinv_Qt_all.shape[0] != num_kappa ||
	    buf_Rinv_Qt_all.shape[1] != 2 || buf_Rinv_Qt_all.shape[2] != 2)
	{
		std::stringstream err;
		err << "Rinv_Qt matrix should have shape [K, 2, 2] " << "(["
		    << num_kappa << ", 2, 2])";
		throw std::runtime_error(err.str());
	}
	if (buf_W.size != num_frames)
	{
		throw std::runtime_error("W matrix should have shape [T]");
	}
	if (buf_Lambda.ndim == 1 && buf_Lambda.size != num_k)
	{
		throw std::runtime_error("Lambda matrix should have shape [3]");
	}

	/* No pointer is passed, so NumPy will allocate the buffer */
	auto kin_out = pybind11::array_t<double>(
	    std::vector<ptrdiff_t>{3, static_cast<long>(num_pix)});
	pybind11::buffer_info buf_kin_out = kin_out.request();
	double* ptr_kin_out = static_cast<double*>(buf_kin_out.ptr);

	double* ptr_tac_all = static_cast<double*>(buf_tac_all.ptr);
	double* ptr_A_all = static_cast<double*>(buf_A_all.ptr);
	double* ptr_B_all = static_cast<double*>(buf_B_all.ptr);
	double* ptr_Rinv_Qt_all = static_cast<double*>(buf_Rinv_Qt_all.ptr);
	double* ptr_W = static_cast<double*>(buf_W.ptr);
	double* ptr_kappa_list = static_cast<double*>(buf_kappa_list.ptr);
	double* ptr_kin_p = nullptr;
	double* ptr_Lambda = nullptr;
	if (buf_kin_p.ndim == 2)
	{
		ptr_kin_p = static_cast<double*>(buf_kin_p.ptr);
		ptr_Lambda = static_cast<double*>(buf_Lambda.ptr);
	}

	solveSRTMBasis(ptr_tac_all, ptr_kin_out, ptr_kin_p, ptr_A_all, ptr_B_all,
	               ptr_Rinv_Qt_all, ptr_W, ptr_Lambda, alpha, ptr_kappa_list,
	               num_kappa, num_pix, num_frames, num_threads);

	return kin_out;
}

pybind11::array_t<double> fit_srtm_basis_joint(
    pybind11::array_t<double> tac_all, pybind11::array_t<double> kin_p,
    pybind11::array_t<double> A_all, pybind11::array_t<double> B_all,
    pybind11::array_t<double> Rinv_Qt_all, pybind11::array_t<double> W,
    pybind11::array_t<double> Lambda, double alpha,
    pybind11::array_t<double> kappa_list, int num_threads)
{
	pybind11::buffer_info buf_tac_all = tac_all.request();
	pybind11::buffer_info buf_kin_p = kin_p.request();
	pybind11::buffer_info buf_A_all = A_all.request();
	pybind11::buffer_info buf_B_all = B_all.request();
	pybind11::buffer_info buf_Rinv_Qt_all = Rinv_Qt_all.request();
	pybind11::buffer_info buf_W = W.request();
	pybind11::buffer_info buf_Lambda = Lambda.request();
	pybind11::buffer_info buf_kappa_list = kappa_list.request();

	size_t num_pix = buf_tac_all.shape[2];
	int num_frames = buf_tac_all.shape[1];
	int num_kappa = buf_kappa_list.shape[1];
	const int num_k = 6;

	if (buf_kin_p.ndim == 2 &&
	    (buf_kin_p.shape[0] != num_k ||
	     buf_kin_p.shape[1] != static_cast<long>(num_pix)))
	{
		throw std::runtime_error("kin_p matrix should have shape [P, X]");
	}
	if (buf_A_all.shape[0] != num_kappa ||
	    buf_A_all.shape[1] != 2 * num_frames || buf_A_all.shape[2] != 4)
	{
		std::stringstream err;
		err << "A matrix should have shape [K, 2T, 4] " << "([" << num_kappa
		    << ", " << 2 * num_frames << ", 4])";
		throw std::runtime_error(err.str());
	}
	if (buf_B_all.shape[0] != num_kappa || buf_B_all.shape[1] != num_k ||
	    buf_B_all.shape[2] != 4)
	{
		std::stringstream err;
		err << "B matrix should have shape [K, P, 4] " << "([" << num_kappa
		    << ", " << num_k << ", 4])";
		throw std::runtime_error(err.str());
	}
	if (buf_Rinv_Qt_all.shape[0] != num_kappa ||
	    buf_Rinv_Qt_all.shape[1] != 4 || buf_Rinv_Qt_all.shape[2] != 4)
	{
		std::stringstream err;
		err << "Rinv_Qt matrix should have shape [K, 4, 4] " << "(["
		    << num_kappa << ", 4, 4])";
		throw std::runtime_error(err.str());
	}
	if (buf_W.size != 2 * num_frames)
	{
		throw std::runtime_error("W matrix should have shape [2T]");
	}
	if (buf_Lambda.ndim == 1 && buf_Lambda.size != num_k)
	{
		throw std::runtime_error("Lambda matrix should have shape [6]");
	}

	/* No pointer is passed, so NumPy will allocate the buffer */
	auto kin_out = pybind11::array_t<double>(
	    std::vector<ptrdiff_t>{num_k, static_cast<long>(num_pix)});
	pybind11::buffer_info buf_kin_out = kin_out.request();
	double* ptr_kin_out = static_cast<double*>(buf_kin_out.ptr);

	double* ptr_tac_all = static_cast<double*>(buf_tac_all.ptr);
	double* ptr_A_all = static_cast<double*>(buf_A_all.ptr);
	double* ptr_B_all = static_cast<double*>(buf_B_all.ptr);
	double* ptr_Rinv_Qt_all = static_cast<double*>(buf_Rinv_Qt_all.ptr);
	double* ptr_W = static_cast<double*>(buf_W.ptr);
	double* ptr_kappa_list = static_cast<double*>(buf_kappa_list.ptr);
	double* ptr_kin_p = nullptr;
	double* ptr_Lambda = nullptr;
	if (buf_kin_p.ndim == 2)
	{
		ptr_kin_p = static_cast<double*>(buf_kin_p.ptr);
		ptr_Lambda = static_cast<double*>(buf_Lambda.ptr);
	}

	solveSRTMBasisJoint(ptr_tac_all, ptr_kin_out, ptr_kin_p, ptr_A_all,
	                    ptr_B_all, ptr_Rinv_Qt_all, ptr_W, ptr_Lambda, alpha,
	                    ptr_kappa_list, num_kappa, num_pix, num_frames,
	                    num_threads);

	return kin_out;
}

void py_setup_srtm(pybind11::module& m)
{
	m.def("fit_srtm_basis", &fit_srtm_basis, "Fit SRTM model using bases");
	m.def("fit_srtm_basis_joint", &fit_srtm_basis_joint,
	      "Fit joint SRTM model using bases");
}

#endif  // if BUILD_PYBIND11

template <typename T>
void solveSRTMBasis(const T* tac_all, T* kin_out, const T* kin_p,
                    const T* A_all, const T* B_all, const T* Rinv_Qt_all,
                    const T* W, const T* Lambda, const T alpha,
                    const T* kappa_list, const int num_kappa,
                    const size_t num_pix, const int num_frames,
                    const int num_threads)
{
	auto tac_buff = std::make_unique<T[]>(num_frames * num_threads);

#pragma omp parallel for num_threads(num_threads)
	for (size_t pi = 0; pi < num_pix; pi++)
	{
		int tid = omp_get_thread_num();
		// Fill buffer
		T* tac_buff_t = tac_buff.get() + tid * num_frames;
		for (int ti = 0; ti < num_frames; ti++)
		{
			tac_buff_t[ti] = tac_all[ti * num_pix + pi];
		}
		T cost_min = std::numeric_limits<T>::max();
		T opt_bp = -1.f;
		T opt_k2 = -1.f;
		T opt_r1 = -1.f;
		for (int ki = 0; ki < num_kappa; ki++)
		{
			// Get precomputed matrices for current basis
			const T* A = &A_all[ki * num_frames * 2];
			const T* B = &B_all[ki * 3 * 2];
			const T* Rinv_Qt = &Rinv_Qt_all[ki * 2 * 2];
			const T kappa = kappa_list[ki];

			// Right-hand side
			T y0 = 0.f;
			T y1 = 0.f;
			for (int ti = 0; ti < num_frames; ti++)
			{
				y0 += A[ti * 2] * W[ti] * tac_buff_t[ti];
				y1 += A[ti * 2 + 1] * W[ti] * tac_buff_t[ti];
			}
			if (alpha > 0.f && kin_p != nullptr)
			{
				y0 +=
				    alpha * (B[0] * Lambda[0] * (kin_p[0 * num_pix + pi] + 1) +
				             B[2] * Lambda[1] * kin_p[1 * num_pix + pi] +
				             B[4] * Lambda[2] * kin_p[2 * num_pix + pi]);
				y1 +=
				    alpha * (B[1] * Lambda[0] * (kin_p[0 * num_pix + pi] + 1) +
				             B[3] * Lambda[1] * kin_p[1 * num_pix + pi] +
				             B[5] * Lambda[2] * kin_p[2 * num_pix + pi]);
			}

			// Calculate inverse
			T theta_0 = Rinv_Qt[0] * y0 + Rinv_Qt[1] * y1;
			T theta_1 = Rinv_Qt[2] * y0 + Rinv_Qt[3] * y1;

			// Compute cost
			T cost = 0.f;
			for (int ti = 0; ti < num_frames; ti++)
			{
				T res = tac_buff_t[ti] -
				        (A[ti * 2] * theta_0 + A[ti * 2 + 1] * theta_1);
				cost += W[ti] * res * res;
			}
			if (alpha > 0.f && kin_p != nullptr)
			{
				T k_diff_0 = kin_p[0 * num_pix + pi] -
				             ((theta_0 * kappa + theta_1) / kappa - 1);
				T k_diff_1 =
				    kin_p[1 * num_pix + pi] - (theta_0 * kappa + theta_1);
				T k_diff_2 = kin_p[2 * num_pix + pi] - theta_0;
				cost += alpha * (Lambda[0] * k_diff_0 * k_diff_0 +
				                 Lambda[1] * k_diff_1 * k_diff_1 +
				                 Lambda[2] * k_diff_2 * k_diff_2);
			}
			cost *= 0.5f;

			// Track minimum cost
			if (cost < cost_min)
			{
				cost_min = cost;
				opt_bp = (theta_0 * kappa + theta_1) / kappa - 1;
				opt_k2 = theta_0 * kappa + theta_1;
				opt_r1 = theta_0;
			}
		}

		// Store output
		kin_out[0 * num_pix + pi] = opt_bp;
		kin_out[1 * num_pix + pi] = opt_k2;
		kin_out[2 * num_pix + pi] = opt_r1;
	}
}

template void solveSRTMBasis(const float* tac_all, float* kin_out,
                             const float* kin_p, const float* A_all,
                             const float* B_all, const float* Rinv_Qt_all,
                             const float* W, const float* Lambda,
                             const float alpha, const float* kappa_list,
                             const int num_kappa, const size_t num_pix,
                             const int num_frames, const int num_threads);
template void solveSRTMBasis(const double* tac_all, double* kin_out,
                             const double* kin_p, const double* A_all,
                             const double* B_all, const double* Rinv_Qt_all,
                             const double* W, const double* Lambda,
                             const double alpha, const double* kappa_list,
                             const int num_kappa, const size_t num_pix,
                             const int num_frames, const int num_threads);

template <typename T>
void solveSRTMBasisJoint(const T* tac_all, T* kin_out, const T* kin_p,
                         const T* A_all, const T* B_all, const T* Rinv_Qt_all,
                         const T* W, const T* Lambda, const T alpha,
                         const T* kappa_list, const int num_kappa,
                         const size_t num_pix, const int num_frames,
                         const int num_threads)
{
	auto tac_buff = std::make_unique<T[]>(2 * num_frames * num_threads);

#pragma omp parallel for num_threads(num_threads)
	for (size_t pi = 0; pi < num_pix; pi++)
	{
		int tid = omp_get_thread_num();
		// Fill buffer
		T* tac_buff_t0 = tac_buff.get() + tid * 2 * num_frames;
		T* tac_buff_t1 = tac_buff_t0 + num_frames;
		for (int ti = 0; ti < num_frames; ti++)
		{
			tac_buff_t0[ti] = tac_all[ti * num_pix + pi];
			tac_buff_t1[ti] = tac_all[num_pix * num_frames + ti * num_pix + pi];
		}
		T cost_min = std::numeric_limits<T>::max();
		T opt_bp_b = -1.f;
		T opt_k2_b = -1.f;
		T opt_r1_b = -1.f;
		T opt_bp_d = -1.f;
		T opt_k2_d = -1.f;
		T opt_r1_d = -1.f;
		T opt_ro = -1.f;
		const T* W0 = W;
		const T* W1 = W + num_frames;
		for (int ki = 0; ki < num_kappa; ki++)
		{
			// Get precomputed matrices for current basis
			const T* A0 = &A_all[ki * 2 * num_frames * 4];
			const T* A1 = A0 + num_frames * 4;
			const T* B = &B_all[ki * 6 * 4];
			const T* Rinv_Qt = &Rinv_Qt_all[ki * 4 * 4];
			const T kappa_0 = kappa_list[ki];
			const T kappa_1 = kappa_list[ki + num_kappa];

			// Right-hand side
			T y0 = 0.f;
			T y1 = 0.f;
			T y2 = 0.f;
			T y3 = 0.f;
			for (int ti = 0; ti < num_frames; ti++)
			{
				y0 += A0[ti * 4] * W0[ti] * tac_buff_t0[ti];
				y1 += A0[ti * 4 + 1] * W0[ti] * tac_buff_t0[ti];
				y2 += A1[ti * 4 + 2] * W1[ti] * tac_buff_t1[ti];
				y3 += A1[ti * 4 + 3] * W1[ti] * tac_buff_t1[ti];
			}
			if (alpha > 0.f && kin_p != nullptr)
			{
				y0 +=
				    alpha * (B[0] * Lambda[0] * (kin_p[0 * num_pix + pi] + 1) +
				             B[4] * Lambda[1] * kin_p[1 * num_pix + pi] +
				             B[8] * Lambda[2] * kin_p[2 * num_pix + pi] +
				             B[12] * Lambda[3] * (kin_p[3 * num_pix + pi] + 1) +
				             B[16] * Lambda[4] * kin_p[4 * num_pix + pi] +
				             B[20] * Lambda[5] * kin_p[5 * num_pix + pi]);
				y1 +=
				    alpha * (B[1] * Lambda[0] * (kin_p[0 * num_pix + pi] + 1) +
				             B[5] * Lambda[1] * kin_p[1 * num_pix + pi] +
				             B[9] * Lambda[2] * kin_p[2 * num_pix + pi] +
				             B[13] * Lambda[3] * (kin_p[3 * num_pix + pi] + 1) +
				             B[17] * Lambda[4] * kin_p[4 * num_pix + pi] +
				             B[21] * Lambda[5] * kin_p[5 * num_pix + pi]);
				y2 +=
				    alpha * (B[2] * Lambda[0] * (kin_p[0 * num_pix + pi] + 1) +
				             B[6] * Lambda[1] * kin_p[1 * num_pix + pi] +
				             B[10] * Lambda[2] * kin_p[2 * num_pix + pi] +
				             B[14] * Lambda[3] * (kin_p[3 * num_pix + pi] + 1) +
				             B[18] * Lambda[4] * kin_p[4 * num_pix + pi] +
				             B[22] * Lambda[5] * kin_p[5 * num_pix + pi]);
				y3 +=
				    alpha * (B[3] * Lambda[0] * (kin_p[0 * num_pix + pi] + 1) +
				             B[7] * Lambda[1] * kin_p[1 * num_pix + pi] +
				             B[11] * Lambda[2] * kin_p[2 * num_pix + pi] +
				             B[15] * Lambda[3] * (kin_p[3 * num_pix + pi] + 1) +
				             B[19] * Lambda[4] * kin_p[4 * num_pix + pi] +
				             B[23] * Lambda[5] * kin_p[5 * num_pix + pi]);
			}

			// Calculate inverse
			T theta_0 = Rinv_Qt[0] * y0 + Rinv_Qt[1] * y1 + Rinv_Qt[2] * y2 +
			            Rinv_Qt[3] * y3;
			T theta_1 = Rinv_Qt[4] * y0 + Rinv_Qt[5] * y1 + Rinv_Qt[6] * y2 +
			            Rinv_Qt[7] * y3;
			T theta_2 = Rinv_Qt[8] * y0 + Rinv_Qt[9] * y1 + Rinv_Qt[10] * y2 +
			            Rinv_Qt[11] * y3;
			T theta_3 = Rinv_Qt[12] * y0 + Rinv_Qt[13] * y1 + Rinv_Qt[14] * y2 +
			            Rinv_Qt[15] * y3;

			// Compute cost
			T cost = 0.f;
			for (int ti = 0; ti < num_frames; ti++)
			{
				T res0 = tac_buff_t0[ti] -
				         (A0[ti * 4] * theta_0 + A0[ti * 4 + 1] * theta_1);
				T res1 = tac_buff_t1[ti] -
				         (A1[ti * 4 + 2] * theta_2 + A1[ti * 4 + 3] * theta_3);
				cost += W0[ti] * res0 * res0 + W1[ti] * res1 * res1;
			}
			if (alpha > 0.f && kin_p != nullptr)
			{
				T k_diff_0 = kin_p[0 * num_pix + pi] - (theta_0 / kappa_0 - 1);
				T k_diff_1 = kin_p[1 * num_pix + pi] - theta_0;
				T k_diff_2 = kin_p[2 * num_pix + pi] - theta_1;
				T k_diff_3 = kin_p[3 * num_pix + pi] - (theta_2 / kappa_1 - 1);
				T k_diff_4 = kin_p[4 * num_pix + pi] - theta_2;
				T k_diff_5 = kin_p[5 * num_pix + pi] - theta_3;
				cost += alpha * (Lambda[0] * k_diff_0 * k_diff_0 +
				                 Lambda[1] * k_diff_1 * k_diff_1 +
				                 Lambda[2] * k_diff_2 * k_diff_2 +
				                 Lambda[3] * k_diff_3 * k_diff_3 +
				                 Lambda[4] * k_diff_4 * k_diff_4 +
				                 Lambda[5] * k_diff_5 * k_diff_5);
			}
			cost *= 0.5f;

			// Track minimum cost
			if (cost < cost_min)
			{
				cost_min = cost;
				opt_bp_b = theta_0 / kappa_0 - 1;
				opt_bp_d = theta_2 / kappa_1 - 1;
				opt_k2_b = theta_0;
				opt_k2_d = theta_2;
				opt_r1_b = theta_1;
				opt_r1_d = theta_3;
				T opt_bp_b_div = opt_bp_b;
				if (std::abs(opt_bp_b) < 1e-8)
				{
					opt_bp_b_div = ((opt_bp_b >= 0) - (opt_bp_b < 0)) * 1e-8;
				}
				opt_ro = 1 - opt_bp_d / opt_bp_b_div;
			}
		}

		// Store output
		kin_out[0 * num_pix + pi] = opt_bp_b;
		kin_out[1 * num_pix + pi] = opt_k2_b;
		kin_out[2 * num_pix + pi] = opt_r1_b;
		kin_out[3 * num_pix + pi] = opt_ro;
		kin_out[4 * num_pix + pi] = opt_k2_d;
		kin_out[5 * num_pix + pi] = opt_r1_d;
	}
}

template void solveSRTMBasisJoint(const float* tac_all, float* kin_out,
                                  const float* kin_p, const float* A_all,
                                  const float* B_all, const float* Rinv_Qt_all,
                                  const float* W, const float* Lambda,
                                  const float alpha, const float* kappa_list,
                                  const int num_kappa, const size_t num_pix,
                                  const int num_frames, const int num_threads);
template void solveSRTMBasisJoint(const double* tac_all, double* kin_out,
                                  const double* kin_p, const double* A_all,
                                  const double* B_all,
                                  const double* Rinv_Qt_all, const double* W,
                                  const double* Lambda, const double alpha,
                                  const double* kappa_list, const int num_kappa,
                                  const size_t num_pix, const int num_frames,
                                  const int num_threads);
