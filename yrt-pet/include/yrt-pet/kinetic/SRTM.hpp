/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cmath>
#include <cstddef>

namespace yrt
{

template <typename T>
void solveSRTMBasis(const T* tac_all, T* kin_out, const T* kin_p,
                    const T* A_all, const T* B_all, const T* Rinv_Qt_all,
                    const T* W, const T* Lambda, const T alpha,
                    const T* kappa_list, const int num_kappa,
                    const size_t num_pix, const int num_frames,
                    const int num_threads);

template <typename T>
void solveSRTMBasisJoint(const T* tac_all, T* kin_out, const T* kin_p,
                         const T* A_all, const T* B_all, const T* Rinv_Qt_all,
                         const T* W, const T* Lambda, const T alpha,
                         const T* kappa_list, const int num_kappa,
                         const size_t num_pix, const int num_frames,
                         const int num_threads);
}  // namespace yrt
