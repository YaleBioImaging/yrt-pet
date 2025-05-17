/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

// constants:
#define PIHALF 1.57079632679489661923
#define PI 3.14159265358979
#define TWOPI 6.28318530717959
#define SPEED_OF_LIGHT_MM_PS 0.299792458
#define DOUBLE_PRECISION 10e-25
#define LARGE_VALUE 10e25
#define SIZE_STRING_BUFFER 1024
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1e-20
#define ITMAX 10000
#define EPS 1.0e-8
#define EPS_FLT 1.0e-8f
#define SIGMA_TO_FWHM 2.354820045031
#define SMALL 1.0e-6
#define SMALL_FLT 1.0e-6f
#define MAX_ARRAY 1000
#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1 + (IM - 1) / NTAB)
#define RNMX (1.0 - EPS)

#define NS_TO_S 1e-9
#define NS_TO_S_FLT 1e-9f

// macros:
#define GET_MIN(a, b, c) ((((a > b) ? b : a) > c) ? c : ((a > b) ? b : a))
#define GET_SGN(a) ((a > 0) ? 1 : -1)
#define GET_SQ(a) ((a) * (a))
#define APPROX_EQ(a, b) (std::abs((a) - (b)) < 1e-6)
#define APPROX_EQ_THRESH(a, b, thresh) (std::abs((a) - (b)) < (thresh))
#define SIGN(a, b) ((b) > 0.0 ? std::abs(a) : -std::abs(a))
#define SHFT(a, b, c, d) \
	(a) = (b);           \
	(b) = (c);           \
	(c) = (d);
