/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cuda_runtime.h>

__device__ inline float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, const float f)
{
	return make_float3(a.x * f, a.y * f, a.z * f);
}

__device__ inline float4 operator*(const float4& a, const float f)
{
	return make_float4(a.x * f, a.y * f, a.z * f, a.w * f);
}

__device__ inline float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline float4 operator-(const float4& a, const float4& b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ inline float3 operator/(const float3& a, const float f)
{
	return make_float3(a.x / f, a.y / f, a.z / f);
}

__global__ void applyAttenuationFactors_kernel(const float* pd_attImgProjData,
											   const float* pd_imgProjData,
											   float* pd_destProjData,
											   float unitFactor,
											   size_t maxNumberOfEvents);
