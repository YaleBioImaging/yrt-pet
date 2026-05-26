#include <metal_stdlib>

using namespace metal;

kernel void smoke_add_one(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          uint id [[thread_position_in_grid]])
{
	output[id] = input[id] + 1.0f;
}

kernel void projection_clear(device float* values [[buffer(0)]],
                             constant float& value [[buffer(1)]],
                             uint id [[thread_position_in_grid]])
{
	values[id] = value;
}

kernel void projection_add(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint id [[thread_position_in_grid]])
{
	output[id] += input[id];
}

kernel void projection_multiply_scalar(device float* output [[buffer(0)]],
                                       constant float& scalar [[buffer(1)]],
                                       uint id [[thread_position_in_grid]])
{
	output[id] *= scalar;
}

kernel void
projection_multiply_elementwise(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                uint id [[thread_position_in_grid]])
{
	output[id] *= input[id];
}

kernel void
projection_divide_measurements(device const float* measurements [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               uint id [[thread_position_in_grid]])
{
	if (output[id] != 0.0f)
	{
		output[id] = measurements[id] / output[id];
	}
}

kernel void projection_invert(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              uint id [[thread_position_in_grid]])
{
	const float value = input[id];
	output[id] = value != 0.0f ? 1.0f / value : 0.0f;
}

kernel void projection_to_acf(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant float& unitFactor [[buffer(2)]],
                              uint id [[thread_position_in_grid]])
{
	output[id] = exp(-input[id] * unitFactor);
}

struct ProjectionLineEndpoints
{
	float p1x;
	float p1y;
	float p1z;
	float p2x;
	float p2y;
	float p2z;
};

struct ProjectionImageBounds
{
	float lengthX;
	float lengthY;
	float lengthZ;
	float fovRadius;
};

struct ProjectionAlphaRange
{
	float alphaMin;
	float alphaMax;
	uint valid;
};

inline void projection_get_alpha(float r0, float r1, float p1, float p2,
                                 float inv_p12, thread float& amin,
                                 thread float& amax)
{
	amin = 0.0f;
	amax = 1.0f;
	if (p1 != p2)
	{
		const float a0 = (r0 - p1) * inv_p12;
		const float a1 = (r1 - p1) * inv_p12;
		if (a0 < a1)
		{
			amin = a0;
			amax = a1;
		}
		else
		{
			amin = a1;
			amax = a0;
		}
	}
	else if (p1 < r0 || p1 > r1)
	{
		amax = 0.0f;
		amin = 1.0f;
	}
}

kernel void projection_siddon_entry_range(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device ProjectionAlphaRange* ranges [[buffer(1)]],
    constant ProjectionImageBounds& bounds [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
	const ProjectionLineEndpoints line = lines[id];
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;

	float fovMin = 0.0f;
	float fovMax = 1.0f;
	const float a = dx * dx + dy * dy;
	const float b = 2.0f * (dx * line.p1x + dy * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                bounds.fovRadius * bounds.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			ProjectionAlphaRange invalidRange;
			invalidRange.alphaMin = 1.0f;
			invalidRange.alphaMax = 0.0f;
			invalidRange.valid = 0u;
			ranges[id] = invalidRange;
			return;
		}
		const float sqrtDelta = sqrt(delta);
		fovMin = (-b - sqrtDelta) / (2.0f * a);
		fovMax = (-b + sqrtDelta) / (2.0f * a);
	}

	const float invX = dx == 0.0f ? 0.0f : 1.0f / dx;
	const float invY = dy == 0.0f ? 0.0f : 1.0f / dy;
	const float invZ = dz == 0.0f ? 0.0f : 1.0f / dz;
	float axMin;
	float axMax;
	float ayMin;
	float ayMax;
	float azMin;
	float azMax;
	projection_get_alpha(-0.5f * bounds.lengthX, 0.5f * bounds.lengthX,
	    line.p1x, line.p2x, invX, axMin, axMax);
	projection_get_alpha(-0.5f * bounds.lengthY, 0.5f * bounds.lengthY,
	    line.p1y, line.p2y, invY, ayMin, ayMax);
	projection_get_alpha(-0.5f * bounds.lengthZ, 0.5f * bounds.lengthZ,
	    line.p1z, line.p2z, invZ, azMin, azMax);

	const float alphaMin =
	    max(max(max(max(0.0f, fovMin), axMin), ayMin), azMin);
	const float alphaMax =
	    min(min(min(min(1.0f, fovMax), axMax), ayMax), azMax);
	ProjectionAlphaRange outputRange;
	outputRange.alphaMin = alphaMin;
	outputRange.alphaMax = alphaMax;
	outputRange.valid = alphaMin < alphaMax ? 1u : 0u;
	ranges[id] = outputRange;
}

struct SiddonForwardImageParams
{
	uint nx;
	uint ny;
	uint nz;
	uint nt;
	uint frame;
	float lengthX;
	float lengthY;
	float lengthZ;
	float voxelX;
	float voxelY;
	float voxelZ;
	float fovRadius;
};

inline uint siddon_image_offset(uint vx, uint vy, uint vz,
                                SiddonForwardImageParams params)
{
	return vx + params.nx * (vy + params.ny * vz);
}

inline void atomic_add_float(device atomic_uint* valueBits, float value)
{
	uint oldBits = atomic_load_explicit(valueBits, memory_order_relaxed);
	while (true)
	{
		const float oldValue = as_type<float>(oldBits);
		const uint newBits = as_type<uint>(oldValue + value);
		uint expectedBits = oldBits;
		if (atomic_compare_exchange_weak_explicit(valueBits, &expectedBits,
		        newBits, memory_order_relaxed, memory_order_relaxed))
		{
			return;
		}
		oldBits = expectedBits;
	}
}

inline float siddon_forward_single_ray_value(
    device const float* image, ProjectionLineEndpoints line,
    SiddonForwardImageParams params)
{
	const float px = line.p2x - line.p1x;
	const float py = line.p2y - line.p1y;
	const float pz = line.p2z - line.p1z;
	const float a = px * px + py * py;
	const float b = 2.0f * (px * line.p1x + py * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                params.fovRadius * params.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	float t0 = 0.0f;
	float t1 = 1.0f;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			return 0.0f;
		}
		const float sqrtDelta = sqrt(delta);
		t0 = (-b - sqrtDelta) / (2.0f * a);
		t1 = (-b + sqrtDelta) / (2.0f * a);
	}

	const float dNorm = sqrt(px * px + py * py + pz * pz);
	const bool flatX = line.p1x == line.p2x;
	const bool flatY = line.p1y == line.p2y;
	const bool flatZ = line.p1z == line.p2z;
	const float invX = flatX ? 0.0f : 1.0f / px;
	const float invY = flatY ? 0.0f : 1.0f / py;
	const float invZ = flatZ ? 0.0f : 1.0f / pz;
	const int dirX = invX >= 0.0f ? 1 : -1;
	const int dirY = invY >= 0.0f ? 1 : -1;
	const int dirZ = invZ >= 0.0f ? 1 : -1;

	const float x0 = -0.5f * params.lengthX;
	const float x1 = 0.5f * params.lengthX;
	const float y0 = -0.5f * params.lengthY;
	const float y1 = 0.5f * params.lengthY;
	const float z0 = -0.5f * params.lengthZ;
	const float z1 = 0.5f * params.lengthZ;
	float axMin;
	float axMax;
	float ayMin;
	float ayMax;
	float azMin;
	float azMax;
	projection_get_alpha(x0, x1, line.p1x, line.p2x, invX, axMin, axMax);
	projection_get_alpha(y0, y1, line.p1y, line.p2y, invY, ayMin, ayMax);
	projection_get_alpha(z0, z1, line.p1z, line.p2z, invZ, azMin, azMax);

	const float alphaMin =
	    max(max(max(max(0.0f, t0), axMin), ayMin), azMin);
	const float alphaMax =
	    min(min(min(min(1.0f, t1), axMax), ayMax), azMax);
	float alphaCur = alphaMin;
	if (alphaCur >= alphaMax)
	{
		return 0.0f;
	}

	float xCur = invX > 0.0f ? x0 : x1;
	float yCur = invY > 0.0f ? y0 : y1;
	float zCur = invZ > 0.0f ? z0 : z1;
	if ((invX >= 0.0f && line.p1x > x1) ||
	    (invX < 0.0f && line.p1x < x0) ||
	    (invY >= 0.0f && line.p1y > y1) ||
	    (invY < 0.0f && line.p1y < y0) ||
	    (invZ >= 0.0f && line.p1z > z1) ||
	    (invZ < 0.0f && line.p1z < z0))
	{
		return 0.0f;
	}

	const float maxFloat = 3.402823466e+38f;
	float axNext = flatX ? maxFloat : axMin;
	if (!flatX)
	{
		const int kx = int(ceil(float(dirX) *
		                       (alphaCur * px - xCur + line.p1x) /
		                       params.voxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) /
		                       params.voxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) /
		                       params.voxelZ));
		zCur += float(kz * dirZ) * params.voxelZ;
		azNext = (zCur - line.p1z) * invZ;
	}

	bool first = true;
	int vx = -1;
	int vy = -1;
	int vz = -1;
	int dirPrevious = -1;
	float axNextPrevious = axNext;
	float ayNextPrevious = ayNext;
	float azNextPrevious = azNext;
	float projection = 0.0f;
	const uint spatialCount = params.nx * params.ny * params.nz;
	const uint frameBase = params.frame * spatialCount;

	while (alphaCur < alphaMax)
	{
		int dirNext = 0;
		float alphaNext = -1.0f;
		if (axNextPrevious <= ayNextPrevious &&
		    axNextPrevious <= azNextPrevious)
		{
			alphaNext = axNext;
			xCur += float(dirX) * params.voxelX;
			axNext = (xCur - line.p1x) * invX;
			dirNext |= 1;
		}
		if (ayNextPrevious <= axNextPrevious &&
		    ayNextPrevious <= azNextPrevious)
		{
			alphaNext = ayNext;
			yCur += float(dirY) * params.voxelY;
			ayNext = (yCur - line.p1y) * invY;
			dirNext |= 2;
		}
		if (azNextPrevious <= axNextPrevious &&
		    azNextPrevious <= ayNextPrevious)
		{
			alphaNext = azNext;
			zCur += float(dirZ) * params.voxelZ;
			azNext = (zCur - line.p1z) * invZ;
			dirNext |= 4;
		}
		if (alphaNext > alphaMax)
		{
			alphaNext = alphaMax;
		}
		if (alphaCur >= alphaNext)
		{
			axNextPrevious = axNext;
			ayNextPrevious = ayNext;
			azNextPrevious = azNext;
			continue;
		}

		bool done = false;
		const float alphaMid = 0.5f * (alphaCur + alphaNext);
		if (first)
		{
			vx = int((line.p1x + alphaMid * px + 0.5f * params.lengthX) /
			         params.voxelX);
			vy = int((line.p1y + alphaMid * py + 0.5f * params.lengthY) /
			         params.voxelY);
			vz = int((line.p1z + alphaMid * pz + 0.5f * params.lengthZ) /
			         params.voxelZ);
			first = false;
			if (vx < 0 || vx >= int(params.nx) || vy < 0 ||
			    vy >= int(params.ny) || vz < 0 || vz >= int(params.nz))
			{
				done = true;
			}
		}
		else
		{
			if ((dirPrevious & 1) != 0)
			{
				vx += dirX;
				if (vx < 0 || vx >= int(params.nx))
				{
					done = true;
				}
			}
			if ((dirPrevious & 2) != 0)
			{
				vy += dirY;
				if (vy < 0 || vy >= int(params.ny))
				{
					done = true;
				}
			}
			if ((dirPrevious & 4) != 0)
			{
				vz += dirZ;
				if (vz < 0 || vz >= int(params.nz))
				{
					done = true;
				}
			}
		}
		if (done)
		{
			break;
		}

		dirPrevious = dirNext;
		const float weight = (alphaNext - alphaCur) * dNorm;
		const uint imageOffset =
		    frameBase +
		    siddon_image_offset(uint(vx), uint(vy), uint(vz), params);
		projection += weight * image[imageOffset];
		alphaCur = alphaNext;
		axNextPrevious = axNext;
		ayNextPrevious = ayNext;
		azNextPrevious = azNext;
	}

	return projection;
}

inline void siddon_backproject_single_ray_atomic_value(
    device atomic_uint* image, ProjectionLineEndpoints line,
    float projectionValue, SiddonForwardImageParams params)
{
	if (projectionValue == 0.0f)
	{
		return;
	}

	const float px = line.p2x - line.p1x;
	const float py = line.p2y - line.p1y;
	const float pz = line.p2z - line.p1z;
	const float a = px * px + py * py;
	const float b = 2.0f * (px * line.p1x + py * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                params.fovRadius * params.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	float t0 = 0.0f;
	float t1 = 1.0f;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			return;
		}
		const float sqrtDelta = sqrt(delta);
		t0 = (-b - sqrtDelta) / (2.0f * a);
		t1 = (-b + sqrtDelta) / (2.0f * a);
	}

	const float dNorm = sqrt(px * px + py * py + pz * pz);
	const bool flatX = line.p1x == line.p2x;
	const bool flatY = line.p1y == line.p2y;
	const bool flatZ = line.p1z == line.p2z;
	const float invX = flatX ? 0.0f : 1.0f / px;
	const float invY = flatY ? 0.0f : 1.0f / py;
	const float invZ = flatZ ? 0.0f : 1.0f / pz;
	const int dirX = invX >= 0.0f ? 1 : -1;
	const int dirY = invY >= 0.0f ? 1 : -1;
	const int dirZ = invZ >= 0.0f ? 1 : -1;

	const float x0 = -0.5f * params.lengthX;
	const float x1 = 0.5f * params.lengthX;
	const float y0 = -0.5f * params.lengthY;
	const float y1 = 0.5f * params.lengthY;
	const float z0 = -0.5f * params.lengthZ;
	const float z1 = 0.5f * params.lengthZ;
	float axMin;
	float axMax;
	float ayMin;
	float ayMax;
	float azMin;
	float azMax;
	projection_get_alpha(x0, x1, line.p1x, line.p2x, invX, axMin, axMax);
	projection_get_alpha(y0, y1, line.p1y, line.p2y, invY, ayMin, ayMax);
	projection_get_alpha(z0, z1, line.p1z, line.p2z, invZ, azMin, azMax);

	const float alphaMin =
	    max(max(max(max(0.0f, t0), axMin), ayMin), azMin);
	const float alphaMax =
	    min(min(min(min(1.0f, t1), axMax), ayMax), azMax);
	float alphaCur = alphaMin;
	if (alphaCur >= alphaMax)
	{
		return;
	}

	float xCur = invX > 0.0f ? x0 : x1;
	float yCur = invY > 0.0f ? y0 : y1;
	float zCur = invZ > 0.0f ? z0 : z1;
	if ((invX >= 0.0f && line.p1x > x1) ||
	    (invX < 0.0f && line.p1x < x0) ||
	    (invY >= 0.0f && line.p1y > y1) ||
	    (invY < 0.0f && line.p1y < y0) ||
	    (invZ >= 0.0f && line.p1z > z1) ||
	    (invZ < 0.0f && line.p1z < z0))
	{
		return;
	}

	const float maxFloat = 3.402823466e+38f;
	float axNext = flatX ? maxFloat : axMin;
	if (!flatX)
	{
		const int kx = int(ceil(float(dirX) *
		                       (alphaCur * px - xCur + line.p1x) /
		                       params.voxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) /
		                       params.voxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) /
		                       params.voxelZ));
		zCur += float(kz * dirZ) * params.voxelZ;
		azNext = (zCur - line.p1z) * invZ;
	}

	bool first = true;
	int vx = -1;
	int vy = -1;
	int vz = -1;
	int dirPrevious = -1;
	float axNextPrevious = axNext;
	float ayNextPrevious = ayNext;
	float azNextPrevious = azNext;
	const uint spatialCount = params.nx * params.ny * params.nz;
	const uint frameBase = params.frame * spatialCount;

	while (alphaCur < alphaMax)
	{
		int dirNext = 0;
		float alphaNext = -1.0f;
		if (axNextPrevious <= ayNextPrevious &&
		    axNextPrevious <= azNextPrevious)
		{
			alphaNext = axNext;
			xCur += float(dirX) * params.voxelX;
			axNext = (xCur - line.p1x) * invX;
			dirNext |= 1;
		}
		if (ayNextPrevious <= axNextPrevious &&
		    ayNextPrevious <= azNextPrevious)
		{
			alphaNext = ayNext;
			yCur += float(dirY) * params.voxelY;
			ayNext = (yCur - line.p1y) * invY;
			dirNext |= 2;
		}
		if (azNextPrevious <= axNextPrevious &&
		    azNextPrevious <= ayNextPrevious)
		{
			alphaNext = azNext;
			zCur += float(dirZ) * params.voxelZ;
			azNext = (zCur - line.p1z) * invZ;
			dirNext |= 4;
		}
		if (alphaNext > alphaMax)
		{
			alphaNext = alphaMax;
		}
		if (alphaCur >= alphaNext)
		{
			axNextPrevious = axNext;
			ayNextPrevious = ayNext;
			azNextPrevious = azNext;
			continue;
		}

		bool done = false;
		const float alphaMid = 0.5f * (alphaCur + alphaNext);
		if (first)
		{
			vx = int((line.p1x + alphaMid * px + 0.5f * params.lengthX) /
			         params.voxelX);
			vy = int((line.p1y + alphaMid * py + 0.5f * params.lengthY) /
			         params.voxelY);
			vz = int((line.p1z + alphaMid * pz + 0.5f * params.lengthZ) /
			         params.voxelZ);
			first = false;
			if (vx < 0 || vx >= int(params.nx) || vy < 0 ||
			    vy >= int(params.ny) || vz < 0 || vz >= int(params.nz))
			{
				done = true;
			}
		}
		else
		{
			if ((dirPrevious & 1) != 0)
			{
				vx += dirX;
				if (vx < 0 || vx >= int(params.nx))
				{
					done = true;
				}
			}
			if ((dirPrevious & 2) != 0)
			{
				vy += dirY;
				if (vy < 0 || vy >= int(params.ny))
				{
					done = true;
				}
			}
			if ((dirPrevious & 4) != 0)
			{
				vz += dirZ;
				if (vz < 0 || vz >= int(params.nz))
				{
					done = true;
				}
			}
		}
		if (done)
		{
			break;
		}

		dirPrevious = dirNext;
		const float update = projectionValue * (alphaNext - alphaCur) * dNorm;
		const uint imageOffset =
		    frameBase +
		    siddon_image_offset(uint(vx), uint(vy), uint(vz), params);
		atomic_add_float(&image[imageOffset], update);
		alphaCur = alphaNext;
		axNextPrevious = axNext;
		ayNextPrevious = ayNext;
		azNextPrevious = azNext;
	}
}

inline float siddon_single_ray_weight_for_voxel(
    ProjectionLineEndpoints line, uint targetVx, uint targetVy, uint targetVz,
    SiddonForwardImageParams params)
{
	const float px = line.p2x - line.p1x;
	const float py = line.p2y - line.p1y;
	const float pz = line.p2z - line.p1z;
	const float a = px * px + py * py;
	const float b = 2.0f * (px * line.p1x + py * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                params.fovRadius * params.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	float t0 = 0.0f;
	float t1 = 1.0f;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			return 0.0f;
		}
		const float sqrtDelta = sqrt(delta);
		t0 = (-b - sqrtDelta) / (2.0f * a);
		t1 = (-b + sqrtDelta) / (2.0f * a);
	}

	const float dNorm = sqrt(px * px + py * py + pz * pz);
	const bool flatX = line.p1x == line.p2x;
	const bool flatY = line.p1y == line.p2y;
	const bool flatZ = line.p1z == line.p2z;
	const float invX = flatX ? 0.0f : 1.0f / px;
	const float invY = flatY ? 0.0f : 1.0f / py;
	const float invZ = flatZ ? 0.0f : 1.0f / pz;
	const int dirX = invX >= 0.0f ? 1 : -1;
	const int dirY = invY >= 0.0f ? 1 : -1;
	const int dirZ = invZ >= 0.0f ? 1 : -1;

	const float x0 = -0.5f * params.lengthX;
	const float x1 = 0.5f * params.lengthX;
	const float y0 = -0.5f * params.lengthY;
	const float y1 = 0.5f * params.lengthY;
	const float z0 = -0.5f * params.lengthZ;
	const float z1 = 0.5f * params.lengthZ;
	float axMin;
	float axMax;
	float ayMin;
	float ayMax;
	float azMin;
	float azMax;
	projection_get_alpha(x0, x1, line.p1x, line.p2x, invX, axMin, axMax);
	projection_get_alpha(y0, y1, line.p1y, line.p2y, invY, ayMin, ayMax);
	projection_get_alpha(z0, z1, line.p1z, line.p2z, invZ, azMin, azMax);

	const float alphaMin =
	    max(max(max(max(0.0f, t0), axMin), ayMin), azMin);
	const float alphaMax =
	    min(min(min(min(1.0f, t1), axMax), ayMax), azMax);
	float alphaCur = alphaMin;
	if (alphaCur >= alphaMax)
	{
		return 0.0f;
	}

	float xCur = invX > 0.0f ? x0 : x1;
	float yCur = invY > 0.0f ? y0 : y1;
	float zCur = invZ > 0.0f ? z0 : z1;
	if ((invX >= 0.0f && line.p1x > x1) ||
	    (invX < 0.0f && line.p1x < x0) ||
	    (invY >= 0.0f && line.p1y > y1) ||
	    (invY < 0.0f && line.p1y < y0) ||
	    (invZ >= 0.0f && line.p1z > z1) ||
	    (invZ < 0.0f && line.p1z < z0))
	{
		return 0.0f;
	}

	const float maxFloat = 3.402823466e+38f;
	float axNext = flatX ? maxFloat : axMin;
	if (!flatX)
	{
		const int kx = int(ceil(float(dirX) *
		                       (alphaCur * px - xCur + line.p1x) /
		                       params.voxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) /
		                       params.voxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) /
		                       params.voxelZ));
		zCur += float(kz * dirZ) * params.voxelZ;
		azNext = (zCur - line.p1z) * invZ;
	}

	bool first = true;
	int vx = -1;
	int vy = -1;
	int vz = -1;
	int dirPrevious = -1;
	float axNextPrevious = axNext;
	float ayNextPrevious = ayNext;
	float azNextPrevious = azNext;
	float totalWeight = 0.0f;

	while (alphaCur < alphaMax)
	{
		int dirNext = 0;
		float alphaNext = -1.0f;
		if (axNextPrevious <= ayNextPrevious &&
		    axNextPrevious <= azNextPrevious)
		{
			alphaNext = axNext;
			xCur += float(dirX) * params.voxelX;
			axNext = (xCur - line.p1x) * invX;
			dirNext |= 1;
		}
		if (ayNextPrevious <= axNextPrevious &&
		    ayNextPrevious <= azNextPrevious)
		{
			alphaNext = ayNext;
			yCur += float(dirY) * params.voxelY;
			ayNext = (yCur - line.p1y) * invY;
			dirNext |= 2;
		}
		if (azNextPrevious <= axNextPrevious &&
		    azNextPrevious <= ayNextPrevious)
		{
			alphaNext = azNext;
			zCur += float(dirZ) * params.voxelZ;
			azNext = (zCur - line.p1z) * invZ;
			dirNext |= 4;
		}
		if (alphaNext > alphaMax)
		{
			alphaNext = alphaMax;
		}
		if (alphaCur >= alphaNext)
		{
			axNextPrevious = axNext;
			ayNextPrevious = ayNext;
			azNextPrevious = azNext;
			continue;
		}

		bool done = false;
		const float alphaMid = 0.5f * (alphaCur + alphaNext);
		if (first)
		{
			vx = int((line.p1x + alphaMid * px + 0.5f * params.lengthX) /
			         params.voxelX);
			vy = int((line.p1y + alphaMid * py + 0.5f * params.lengthY) /
			         params.voxelY);
			vz = int((line.p1z + alphaMid * pz + 0.5f * params.lengthZ) /
			         params.voxelZ);
			first = false;
			if (vx < 0 || vx >= int(params.nx) || vy < 0 ||
			    vy >= int(params.ny) || vz < 0 || vz >= int(params.nz))
			{
				done = true;
			}
		}
		else
		{
			if ((dirPrevious & 1) != 0)
			{
				vx += dirX;
				if (vx < 0 || vx >= int(params.nx))
				{
					done = true;
				}
			}
			if ((dirPrevious & 2) != 0)
			{
				vy += dirY;
				if (vy < 0 || vy >= int(params.ny))
				{
					done = true;
				}
			}
			if ((dirPrevious & 4) != 0)
			{
				vz += dirZ;
				if (vz < 0 || vz >= int(params.nz))
				{
					done = true;
				}
			}
		}
		if (done)
		{
			break;
		}

		dirPrevious = dirNext;
		if (vx == int(targetVx) && vy == int(targetVy) &&
		    vz == int(targetVz))
		{
			totalWeight += (alphaNext - alphaCur) * dNorm;
		}
		alphaCur = alphaNext;
		axNextPrevious = axNext;
		ayNextPrevious = ayNext;
		azNextPrevious = azNext;
	}

	return totalWeight;
}

kernel void siddon_forward_single_ray(
    device const float* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	projectionValues[id] =
	    siddon_forward_single_ray_value(image, lines[id], params);
}

kernel void siddon_backproject_single_ray(
    device atomic_uint* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device const float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	siddon_backproject_single_ray_atomic_value(
	    image, lines[id], projectionValues[id], params);
}

struct ImageShape
{
	uint nx;
	uint ny;
	uint nz;
	uint nt;
};

struct ImageScalarKernelParams
{
	ImageShape shape;
	float value;
};

struct ImageThresholdKernelParams
{
	ImageShape shape;
	float threshold;
	float valLeScale;
	float valLeOffset;
	float valGtScale;
	float valGtOffset;
};

struct ImageEMKernelParams
{
	ImageShape shape;
	float threshold;
};

struct ImageConvolutionKernelParams
{
	ImageShape shape;
	uint kernelSize;
};

inline uint image_spatial_count(ImageShape shape)
{
	return shape.nx * shape.ny * shape.nz;
}

inline uint image_idx3(uint x, uint y, uint z, ImageShape shape)
{
	return x + shape.nx * (y + shape.ny * z);
}

inline uint image_circular(uint size, int value)
{
	if (value < 0)
	{
		return uint(value + int(size));
	}
	if (value >= int(size))
	{
		return uint(value - int(size));
	}
	return uint(value);
}

kernel void image_fill(device float* image [[buffer(0)]],
                       constant ImageScalarKernelParams& params [[buffer(1)]],
                       uint id [[thread_position_in_grid]])
{
	image[id] = params.value;
}

kernel void
image_multiply_scalar(device float* image [[buffer(0)]],
                      constant ImageScalarKernelParams& params [[buffer(1)]],
                      uint id [[thread_position_in_grid]])
{
	image[id] *= params.value;
}

kernel void image_add_3d_to_3d(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               uint id [[thread_position_in_grid]])
{
	output[id] += input[id];
}

kernel void image_add_3d_to_4d(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant ImageShape& shape [[buffer(2)]],
                               uint id [[thread_position_in_grid]])
{
	const uint spatialCount = image_spatial_count(shape);
	output[id] += input[id % spatialCount];
}

kernel void
image_apply_threshold(device float* image [[buffer(0)]],
                      device const float* mask [[buffer(1)]],
                      constant ImageThresholdKernelParams& params [[buffer(2)]],
                      uint id [[thread_position_in_grid]])
{
	if (mask[id] <= params.threshold)
	{
		image[id] = image[id] * params.valLeScale + params.valLeOffset;
	}
	else
	{
		image[id] = image[id] * params.valGtScale + params.valGtOffset;
	}
}

kernel void
image_apply_threshold_broadcast(
    device float* image [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    constant ImageThresholdKernelParams& params [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
	const uint spatialCount = image_spatial_count(params.shape);
	const uint spatialId = id % spatialCount;
	if (mask[spatialId] <= params.threshold)
	{
		image[id] = image[id] * params.valLeScale + params.valLeOffset;
	}
	else
	{
		image[id] = image[id] * params.valGtScale + params.valGtOffset;
	}
}

kernel void image_update_em_static(
    device const float* update [[buffer(0)]],
    device float* image [[buffer(1)]],
    device const float* sensitivity [[buffer(2)]],
    constant ImageEMKernelParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	const float sens = sensitivity[id];
	if (sens > params.threshold)
	{
		image[id] *= update[id] / sens;
	}
}

kernel void image_update_em_dynamic(
    device const float* update [[buffer(0)]],
    device float* image [[buffer(1)]],
    device const float* sensitivity [[buffer(2)]],
    constant ImageEMKernelParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	const uint spatialCount = image_spatial_count(params.shape);
	const uint spatialId = id % spatialCount;
	const float sens = sensitivity[spatialId];
	if (sens > params.threshold)
	{
		image[id] *= update[id] / sens;
	}
}

kernel void image_update_em_dynamic_scaled(
    device const float* update [[buffer(0)]],
    device float* image [[buffer(1)]],
    device const float* sensitivity [[buffer(2)]],
    device const float* sensitivityScaling [[buffer(3)]],
    constant ImageEMKernelParams& params [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
	const uint spatialCount = image_spatial_count(params.shape);
	const uint frame = id / spatialCount;
	const uint spatialId = id % spatialCount;
	const float invScaling = 1.0f / sensitivityScaling[frame];
	const float threshold = params.threshold * invScaling;
	const float sens = sensitivity[spatialId];
	if (sens > threshold)
	{
		image[id] *= (update[id] * invScaling) / sens;
	}
}

kernel void image_convolve3d_separable_x(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    constant ImageConvolutionKernelParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	const ImageShape shape = params.shape;
	const uint spatialCount = image_spatial_count(shape);
	const uint frameBase = (id / spatialCount) * spatialCount;
	const uint spatialId = id % spatialCount;
	const uint x = spatialId % shape.nx;
	const uint y = (spatialId / shape.nx) % shape.ny;
	const uint z = spatialId / (shape.nx * shape.ny);
	const int halfKernelSize = int(params.kernelSize / 2);

	float sum = 0.0f;
	for (int kk = -halfKernelSize; kk <= halfKernelSize; ++kk)
	{
		const uint wrappedX = image_circular(shape.nx, int(x) - kk);
		const uint imageIndex = frameBase + image_idx3(wrappedX, y, z, shape);
		sum += coefficients[uint(kk + halfKernelSize)] * input[imageIndex];
	}
	output[id] = sum;
}

kernel void image_convolve3d_separable_y(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    constant ImageConvolutionKernelParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	const ImageShape shape = params.shape;
	const uint spatialCount = image_spatial_count(shape);
	const uint frameBase = (id / spatialCount) * spatialCount;
	const uint spatialId = id % spatialCount;
	const uint x = spatialId % shape.nx;
	const uint y = (spatialId / shape.nx) % shape.ny;
	const uint z = spatialId / (shape.nx * shape.ny);
	const int halfKernelSize = int(params.kernelSize / 2);

	float sum = 0.0f;
	for (int kk = -halfKernelSize; kk <= halfKernelSize; ++kk)
	{
		const uint wrappedY = image_circular(shape.ny, int(y) - kk);
		const uint imageIndex = frameBase + image_idx3(x, wrappedY, z, shape);
		sum += coefficients[uint(kk + halfKernelSize)] * input[imageIndex];
	}
	output[id] = sum;
}

kernel void image_convolve3d_separable_z(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    constant ImageConvolutionKernelParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	const ImageShape shape = params.shape;
	const uint spatialCount = image_spatial_count(shape);
	const uint frameBase = (id / spatialCount) * spatialCount;
	const uint spatialId = id % spatialCount;
	const uint x = spatialId % shape.nx;
	const uint y = (spatialId / shape.nx) % shape.ny;
	const uint z = spatialId / (shape.nx * shape.ny);
	const int halfKernelSize = int(params.kernelSize / 2);

	float sum = 0.0f;
	for (int kk = -halfKernelSize; kk <= halfKernelSize; ++kk)
	{
		const uint wrappedZ = image_circular(shape.nz, int(z) - kk);
		const uint imageIndex = frameBase + image_idx3(x, y, wrappedZ, shape);
		sum += coefficients[uint(kk + halfKernelSize)] * input[imageIndex];
	}
	output[id] = sum;
}
