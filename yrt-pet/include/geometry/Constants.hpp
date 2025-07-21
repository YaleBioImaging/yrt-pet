
/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

// constants:
constexpr double PIHALF = 1.57079632679489661923;
constexpr double PI = 3.14159265358979;
constexpr double TWOPI = 6.28318530717959;
constexpr float PIHALF_FLT = 1.57079632679489661923f;
constexpr float PI_FLT = 3.14159265358979f;
constexpr float TWOPI_FLT = 6.28318530717959f;
constexpr double SPEED_OF_LIGHT_MM_PS = 0.299792458;
constexpr double DOUBLE_PRECISION = 10e-25;
constexpr double LARGE_VALUE = 10e25;
constexpr double SIZE_STRING_BUFFER = 1024;
constexpr double GOLD = 1.618034;
constexpr double GLIMIT = 100.0;
constexpr double TINY = 1e-20;
constexpr double ITMAX = 10000;
constexpr double EPS = 1.0e-8;
constexpr float EPS_FLT = 1.0e-8f;
constexpr double SIGMA_TO_FWHM = 2.354820045031;
constexpr double SMALL = 1.0e-6;
constexpr float SMALL_FLT = 1.0e-6f;
constexpr int IA = 16807;
constexpr int IM = 2147483647;
constexpr double AM = (1.0 / IM);
constexpr int IQ = 127773;
constexpr int IR = 2836;
constexpr int NTAB = 32;
constexpr int NDIV = (1 + (IM - 1) / NTAB);
constexpr double RNMX = (1.0 - EPS);
constexpr double NS_TO_S = 1e-9;
constexpr float NS_TO_S_FLT = 1e-9f;

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