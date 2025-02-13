/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ inline float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline float4& operator+=(float4& a, const float4& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

__device__ inline float4 operator-(const float4& a, const float4& b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ inline float4& operator-=(float4& a, const float4& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

__device__ inline float4 operator*(const float4& a, const float f)
{
	return make_float4(a.x * f, a.y * f, a.z * f, a.w * f);
}

__device__ inline float4& operator*=(float4& a, const float4& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

__device__ inline float4 operator/(const float4& a, const float f)
{
	return make_float4(a.x / f, a.y / f, a.z / f, a.w / f);
}

__device__ inline float4& operator/=(float4& a, const float4& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

__device__ inline float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3& operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__device__ inline float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3& operator-=(float3& a, const float3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

__device__ inline float3 operator*(const float3& a, const float f)
{
	return make_float3(a.x * f, a.y * f, a.z * f);
}

__device__ inline float3& operator*=(float3& a, const float3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

__device__ inline float3 operator/(const float3& a, const float f)
{
	return make_float3(a.x / f, a.y / f, a.z / f);
}

__device__ inline float3& operator/=(float3& a, const float3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

// Base case: single argument
template <typename T>
__device__ inline T max(T a) {
	return a;
}

template <typename T, typename... Args>
__device__ inline T max(T first, Args... args) {
	T remaining_max = max(args...);
	return (first > remaining_max) ? first : remaining_max;
}

// Base case: single argument
template <typename T>
__device__ inline T min(T a) {
	return a;
}

template <typename T, typename... Args>
__device__ inline T min(T first, Args... args) {
	T remaining_max = min(args...);
	return (first < remaining_max) ? first : remaining_max;
}

__global__ void applyAttenuationFactors_kernel(const float* pd_attImgProjData,
											   const float* pd_imgProjData,
											   float* pd_destProjData,
											   float unitFactor,
											   size_t maxNumberOfEvents);
