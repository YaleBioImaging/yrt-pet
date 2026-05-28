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

struct ProjectionOsemRatioParams
{
	float globalScaleFactor;
	float denomThreshold;
	uint hasSensitivity;
	uint hasAttenuation;
	uint hasRandoms;
	uint hasScatter;
	uint hasInVivoAttenuation;
};

kernel void
projection_osem_ratio(device float* estimatesAndOutput [[buffer(0)]],
                      device const float* measurements [[buffer(1)]],
                      device const float* sensitivity [[buffer(2)]],
                      device const float* attenuation [[buffer(3)]],
                      device const float* randoms [[buffer(4)]],
                      device const float* scatter [[buffer(5)]],
                      device const float* inVivoAttenuation [[buffer(6)]],
                      constant ProjectionOsemRatioParams& params [[buffer(7)]],
                      uint id [[thread_position_in_grid]])
{
	float update = estimatesAndOutput[id];
	if (params.hasSensitivity != 0u)
	{
		update *= sensitivity[id];
	}
	if (params.hasAttenuation != 0u)
	{
		update *= attenuation[id];
	}
	update *= params.globalScaleFactor;

	if (params.hasRandoms != 0u)
	{
		update += randoms[id];
	}
	if (params.hasScatter != 0u)
	{
		update += scatter[id];
	}
	if (params.hasInVivoAttenuation != 0u)
	{
		update *= inVivoAttenuation[id];
	}

	if (abs(update) > params.denomThreshold)
	{
		update = measurements[id] / update;
		if (params.hasSensitivity != 0u)
		{
			update *= sensitivity[id];
		}
		if (params.hasAttenuation != 0u)
		{
			update *= attenuation[id];
		}
		update *= params.globalScaleFactor;
	}
	else
	{
		update = 0.0f;
	}
	estimatesAndOutput[id] = update;
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
	float invVoxelX;
	float invVoxelY;
	float invVoxelZ;
	float halfLengthX;
	float halfLengthY;
	float halfLengthZ;
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

inline void atomic_add_float(device atomic_float* value, float update)
{
	atomic_fetch_add_explicit(value, update, memory_order_relaxed);
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

	const float x0 = -params.halfLengthX;
	const float x1 = params.halfLengthX;
	const float y0 = -params.halfLengthY;
	const float y1 = params.halfLengthY;
	const float z0 = -params.halfLengthZ;
	const float z1 = params.halfLengthZ;
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
		                       (alphaCur * px - xCur + line.p1x) *
		                       params.invVoxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) *
		                       params.invVoxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) *
		                       params.invVoxelZ));
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
			vx = int((line.p1x + alphaMid * px + params.halfLengthX) *
			         params.invVoxelX);
			vy = int((line.p1y + alphaMid * py + params.halfLengthY) *
			         params.invVoxelY);
			vz = int((line.p1z + alphaMid * pz + params.halfLengthZ) *
			         params.invVoxelZ);
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

template <typename AtomicImagePointer>
inline void siddon_backproject_single_ray_atomic_value(
    AtomicImagePointer image, ProjectionLineEndpoints line,
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

	const float x0 = -params.halfLengthX;
	const float x1 = params.halfLengthX;
	const float y0 = -params.halfLengthY;
	const float y1 = params.halfLengthY;
	const float z0 = -params.halfLengthZ;
	const float z1 = params.halfLengthZ;
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
		                       (alphaCur * px - xCur + line.p1x) *
		                       params.invVoxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) *
		                       params.invVoxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) *
		                       params.invVoxelZ));
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
			vx = int((line.p1x + alphaMid * px + params.halfLengthX) *
			         params.invVoxelX);
			vy = int((line.p1y + alphaMid * py + params.halfLengthY) *
			         params.invVoxelY);
			vz = int((line.p1z + alphaMid * pz + params.halfLengthZ) *
			         params.invVoxelZ);
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

inline uint siddon_backproject_single_ray_update_count_value(
    ProjectionLineEndpoints line, float projectionValue,
    SiddonForwardImageParams params)
{
	if (projectionValue == 0.0f)
	{
		return 0;
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
			return 0;
		}
		const float sqrtDelta = sqrt(delta);
		t0 = (-b - sqrtDelta) / (2.0f * a);
		t1 = (-b + sqrtDelta) / (2.0f * a);
	}

	const bool flatX = line.p1x == line.p2x;
	const bool flatY = line.p1y == line.p2y;
	const bool flatZ = line.p1z == line.p2z;
	const float invX = flatX ? 0.0f : 1.0f / px;
	const float invY = flatY ? 0.0f : 1.0f / py;
	const float invZ = flatZ ? 0.0f : 1.0f / pz;
	const int dirX = invX >= 0.0f ? 1 : -1;
	const int dirY = invY >= 0.0f ? 1 : -1;
	const int dirZ = invZ >= 0.0f ? 1 : -1;

	const float x0 = -params.halfLengthX;
	const float x1 = params.halfLengthX;
	const float y0 = -params.halfLengthY;
	const float y1 = params.halfLengthY;
	const float z0 = -params.halfLengthZ;
	const float z1 = params.halfLengthZ;
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
		return 0;
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
		return 0;
	}

	const float maxFloat = 3.402823466e+38f;
	float axNext = flatX ? maxFloat : axMin;
	if (!flatX)
	{
		const int kx = int(ceil(float(dirX) *
		                       (alphaCur * px - xCur + line.p1x) *
		                       params.invVoxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) *
		                       params.invVoxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) *
		                       params.invVoxelZ));
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
	uint updateCount = 0;

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
			vx = int((line.p1x + alphaMid * px + params.halfLengthX) *
			         params.invVoxelX);
			vy = int((line.p1y + alphaMid * py + params.halfLengthY) *
			         params.invVoxelY);
			vz = int((line.p1z + alphaMid * pz + params.halfLengthZ) *
			         params.invVoxelZ);
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
		updateCount += 1;
		alphaCur = alphaNext;
		axNextPrevious = axNext;
		ayNextPrevious = ayNext;
		azNextPrevious = azNext;
	}

	return updateCount;
}

inline void siddon_backproject_single_ray_voxel_hit_count_value(
    device atomic_uint* hitCounts, ProjectionLineEndpoints line,
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

	const bool flatX = line.p1x == line.p2x;
	const bool flatY = line.p1y == line.p2y;
	const bool flatZ = line.p1z == line.p2z;
	const float invX = flatX ? 0.0f : 1.0f / px;
	const float invY = flatY ? 0.0f : 1.0f / py;
	const float invZ = flatZ ? 0.0f : 1.0f / pz;
	const int dirX = invX >= 0.0f ? 1 : -1;
	const int dirY = invY >= 0.0f ? 1 : -1;
	const int dirZ = invZ >= 0.0f ? 1 : -1;

	const float x0 = -params.halfLengthX;
	const float x1 = params.halfLengthX;
	const float y0 = -params.halfLengthY;
	const float y1 = params.halfLengthY;
	const float z0 = -params.halfLengthZ;
	const float z1 = params.halfLengthZ;
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
		                       (alphaCur * px - xCur + line.p1x) *
		                       params.invVoxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) *
		                       params.invVoxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) *
		                       params.invVoxelZ));
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
			vx = int((line.p1x + alphaMid * px + params.halfLengthX) *
			         params.invVoxelX);
			vy = int((line.p1y + alphaMid * py + params.halfLengthY) *
			         params.invVoxelY);
			vz = int((line.p1z + alphaMid * pz + params.halfLengthZ) *
			         params.invVoxelZ);
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
		const uint imageOffset =
		    frameBase +
		    siddon_image_offset(uint(vx), uint(vy), uint(vz), params);
		atomic_fetch_add_explicit(&hitCounts[imageOffset], 1u,
		                          memory_order_relaxed);
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

	const float x0 = -params.halfLengthX;
	const float x1 = params.halfLengthX;
	const float y0 = -params.halfLengthY;
	const float y1 = params.halfLengthY;
	const float z0 = -params.halfLengthZ;
	const float z1 = params.halfLengthZ;
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
		                       (alphaCur * px - xCur + line.p1x) *
		                       params.invVoxelX));
		xCur += float(kx * dirX) * params.voxelX;
		axNext = (xCur - line.p1x) * invX;
	}
	float ayNext = flatY ? maxFloat : ayMin;
	if (!flatY)
	{
		const int ky = int(ceil(float(dirY) *
		                       (alphaCur * py - yCur + line.p1y) *
		                       params.invVoxelY));
		yCur += float(ky * dirY) * params.voxelY;
		ayNext = (yCur - line.p1y) * invY;
	}
	float azNext = flatZ ? maxFloat : azMin;
	if (!flatZ)
	{
		const int kz = int(ceil(float(dirZ) *
		                       (alphaCur * pz - zCur + line.p1z) *
		                       params.invVoxelZ));
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
			vx = int((line.p1x + alphaMid * px + params.halfLengthX) *
			         params.invVoxelX);
			vy = int((line.p1y + alphaMid * py + params.halfLengthY) *
			         params.invVoxelY);
			vz = int((line.p1z + alphaMid * pz + params.halfLengthZ) *
			         params.invVoxelZ);
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

inline bool joseph_alpha_range(ProjectionLineEndpoints line,
                               SiddonForwardImageParams params,
                               thread float& alphaMin,
                               thread float& alphaMax)
{
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	if (dx == 0.0f && dy == 0.0f && dz == 0.0f)
	{
		alphaMin = 1.0f;
		alphaMax = 0.0f;
		return false;
	}

	float fovMin = 0.0f;
	float fovMax = 1.0f;
	const float a = dx * dx + dy * dy;
	const float b = 2.0f * (dx * line.p1x + dy * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                params.fovRadius * params.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			alphaMin = 1.0f;
			alphaMax = 0.0f;
			return false;
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
	projection_get_alpha(-params.halfLengthX, params.halfLengthX,
	    line.p1x, line.p2x, invX, axMin, axMax);
	projection_get_alpha(-params.halfLengthY, params.halfLengthY,
	    line.p1y, line.p2y, invY, ayMin, ayMax);
	projection_get_alpha(-params.halfLengthZ, params.halfLengthZ,
	    line.p1z, line.p2z, invZ, azMin, azMax);

	alphaMin = max(max(max(max(0.0f, fovMin), axMin), ayMin), azMin);
	alphaMax = min(min(min(min(1.0f, fovMax), axMax), ayMax), azMax);
	return alphaMin < alphaMax;
}

inline uint joseph_major_axis(ProjectionLineEndpoints line,
                              SiddonForwardImageParams params)
{
	const float sx = abs(line.p2x - line.p1x) * params.invVoxelX;
	const float sy = abs(line.p2y - line.p1y) * params.invVoxelY;
	const float sz = abs(line.p2z - line.p1z) * params.invVoxelZ;
	if (sx >= sy && sx >= sz)
	{
		return 0u;
	}
	return sy >= sz ? 1u : 2u;
}

inline float joseph_axis_coord(ProjectionLineEndpoints line, uint axis,
                               float alpha)
{
	if (axis == 0u)
	{
		return line.p1x + alpha * (line.p2x - line.p1x);
	}
	if (axis == 1u)
	{
		return line.p1y + alpha * (line.p2y - line.p1y);
	}
	return line.p1z + alpha * (line.p2z - line.p1z);
}

inline float joseph_axis_delta(ProjectionLineEndpoints line, uint axis)
{
	if (axis == 0u)
	{
		return line.p2x - line.p1x;
	}
	if (axis == 1u)
	{
		return line.p2y - line.p1y;
	}
	return line.p2z - line.p1z;
}

inline float joseph_axis_start(ProjectionLineEndpoints line, uint axis)
{
	if (axis == 0u)
	{
		return line.p1x;
	}
	if (axis == 1u)
	{
		return line.p1y;
	}
	return line.p1z;
}

inline float joseph_axis_half_length(SiddonForwardImageParams params,
                                     uint axis)
{
	if (axis == 0u)
	{
		return params.halfLengthX;
	}
	if (axis == 1u)
	{
		return params.halfLengthY;
	}
	return params.halfLengthZ;
}

inline float joseph_axis_voxel(SiddonForwardImageParams params, uint axis)
{
	if (axis == 0u)
	{
		return params.voxelX;
	}
	if (axis == 1u)
	{
		return params.voxelY;
	}
	return params.voxelZ;
}

inline float joseph_axis_inv_voxel(SiddonForwardImageParams params,
                                   uint axis)
{
	if (axis == 0u)
	{
		return params.invVoxelX;
	}
	if (axis == 1u)
	{
		return params.invVoxelY;
	}
	return params.invVoxelZ;
}

inline uint joseph_axis_size(SiddonForwardImageParams params, uint axis)
{
	if (axis == 0u)
	{
		return params.nx;
	}
	if (axis == 1u)
	{
		return params.ny;
	}
	return params.nz;
}

inline float joseph_grid_coord(float coord, float halfLength,
                               float invVoxel)
{
	return (coord + halfLength) * invVoxel - 0.5f;
}

inline bool joseph_sample_bounds(ProjectionLineEndpoints line,
                                 SiddonForwardImageParams params, uint axis,
                                 float alphaMin, float alphaMax,
                                 thread int& first,
                                 thread int& last)
{
	const float grid0 = joseph_grid_coord(
	    joseph_axis_coord(line, axis, alphaMin),
	    joseph_axis_half_length(params, axis),
	    joseph_axis_inv_voxel(params, axis));
	const float grid1 = joseph_grid_coord(
	    joseph_axis_coord(line, axis, alphaMax),
	    joseph_axis_half_length(params, axis),
	    joseph_axis_inv_voxel(params, axis));
	first = int(ceil(min(grid0, grid1)));
	last = int(floor(max(grid0, grid1)));
	if (first < 0)
	{
		first = 0;
	}
	const int maxIndex = int(joseph_axis_size(params, axis)) - 1;
	if (last > maxIndex)
	{
		last = maxIndex;
	}
	return first <= last;
}

struct JosephAxisCache
{
	float halfLength;
	float voxel;
	float start;
	float invDelta;
	float halfAlphaStep;
	float rayLength;
};

inline JosephAxisCache joseph_make_axis_cache(
    ProjectionLineEndpoints line, SiddonForwardImageParams params, uint axis)
{
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	const float dAxis = joseph_axis_delta(line, axis);

	JosephAxisCache cache;
	cache.halfLength = joseph_axis_half_length(params, axis);
	cache.voxel = joseph_axis_voxel(params, axis);
	cache.start = joseph_axis_start(line, axis);
	cache.invDelta = dAxis == 0.0f ? 0.0f : 1.0f / dAxis;
	cache.halfAlphaStep =
	    dAxis == 0.0f ? 0.0f : 0.5f * cache.voxel / abs(dAxis);
	cache.rayLength = sqrt(dx * dx + dy * dy + dz * dz);
	return cache;
}

inline float joseph_cached_sample_alpha(JosephAxisCache cache,
                                        int majorIndex)
{
	const float centerCoord =
	    -cache.halfLength + (float(majorIndex) + 0.5f) * cache.voxel;
	return (centerCoord - cache.start) * cache.invDelta;
}

inline float joseph_cached_sample_weight(JosephAxisCache cache,
                                         float centerAlpha, float alphaMin,
                                         float alphaMax)
{
	const float segmentStart =
	    max(alphaMin, centerAlpha - cache.halfAlphaStep);
	const float segmentEnd =
	    min(alphaMax, centerAlpha + cache.halfAlphaStep);
	if (segmentStart >= segmentEnd)
	{
		return 0.0f;
	}
	return cache.rayLength * (segmentEnd - segmentStart);
}

inline float joseph_cached_sample_weight_stride(JosephAxisCache cache,
                                                float centerAlpha,
                                                float alphaMin,
                                                float alphaMax,
                                                uint sampleStride)
{
	const float strideScale = float(sampleStride <= 1u ? 1u : sampleStride);
	const float halfAlphaStep = cache.halfAlphaStep * strideScale;
	const float segmentStart = max(alphaMin, centerAlpha - halfAlphaStep);
	const float segmentEnd = min(alphaMax, centerAlpha + halfAlphaStep);
	if (segmentStart >= segmentEnd)
	{
		return 0.0f;
	}
	return cache.rayLength * (segmentEnd - segmentStart);
}

inline float joseph_sample_weight(ProjectionLineEndpoints line,
                                  SiddonForwardImageParams params, uint axis,
                                  int majorIndex, float alphaMin,
                                  float alphaMax)
{
	const float dAxis = joseph_axis_delta(line, axis);
	if (dAxis == 0.0f)
	{
		return 0.0f;
	}
	const float voxel = joseph_axis_voxel(params, axis);
	const float centerCoord = -joseph_axis_half_length(params, axis) +
	                          (float(majorIndex) + 0.5f) * voxel;
	const float centerAlpha =
	    (centerCoord - joseph_axis_start(line, axis)) / dAxis;
	const float halfAlphaStep = 0.5f * voxel / abs(dAxis);
	const float segmentStart = max(alphaMin, centerAlpha - halfAlphaStep);
	const float segmentEnd = min(alphaMax, centerAlpha + halfAlphaStep);
	if (segmentStart >= segmentEnd)
	{
		return 0.0f;
	}
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	return sqrt(dx * dx + dy * dy + dz * dz) *
	       (segmentEnd - segmentStart);
}

inline float joseph_sample_alpha(ProjectionLineEndpoints line,
                                 SiddonForwardImageParams params, uint axis,
                                 int majorIndex)
{
	const float dAxis = joseph_axis_delta(line, axis);
	if (dAxis == 0.0f)
	{
		return 0.0f;
	}
	const float centerCoord = -joseph_axis_half_length(params, axis) +
	                          (float(majorIndex) + 0.5f) *
	                              joseph_axis_voxel(params, axis);
	return (centerCoord - joseph_axis_start(line, axis)) / dAxis;
}

inline float joseph_image_value(device const float* image, int vx, int vy,
                                int vz, SiddonForwardImageParams params)
{
	if (vx < 0 || vy < 0 || vz < 0 || vx >= int(params.nx) ||
	    vy >= int(params.ny) || vz >= int(params.nz))
	{
		return 0.0f;
	}
	const uint spatialCount = params.nx * params.ny * params.nz;
	const uint frameBase = params.frame * spatialCount;
	return image[frameBase +
	             siddon_image_offset(uint(vx), uint(vy), uint(vz), params)];
}

inline float joseph_bilinear_forward(device const float* image, uint axis,
                                     int majorIndex, float alpha,
                                     ProjectionLineEndpoints line,
                                     SiddonForwardImageParams params)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0u)
	{
		const float gy = joseph_grid_coord(y, params.halfLengthY,
		                                   params.invVoxelY);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int y0 = int(floor(gy));
		const int z0 = int(floor(gz));
		const float fy = gy - float(y0);
		const float fz = gz - float(z0);
		return (1.0f - fy) * (1.0f - fz) *
		           joseph_image_value(image, majorIndex, y0, z0, params) +
		       fy * (1.0f - fz) *
		           joseph_image_value(image, majorIndex, y0 + 1, z0, params) +
		       (1.0f - fy) * fz *
		           joseph_image_value(image, majorIndex, y0, z0 + 1, params) +
		       fy * fz *
		           joseph_image_value(image, majorIndex, y0 + 1, z0 + 1,
		                              params);
	}
	if (axis == 1u)
	{
		const float gx = joseph_grid_coord(x, params.halfLengthX,
		                                   params.invVoxelX);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int x0 = int(floor(gx));
		const int z0 = int(floor(gz));
		const float fx = gx - float(x0);
		const float fz = gz - float(z0);
		return (1.0f - fx) * (1.0f - fz) *
		           joseph_image_value(image, x0, majorIndex, z0, params) +
		       fx * (1.0f - fz) *
		           joseph_image_value(image, x0 + 1, majorIndex, z0, params) +
		       (1.0f - fx) * fz *
		           joseph_image_value(image, x0, majorIndex, z0 + 1, params) +
		       fx * fz *
		           joseph_image_value(image, x0 + 1, majorIndex, z0 + 1,
		                              params);
	}
	const float gx = joseph_grid_coord(x, params.halfLengthX,
	                                   params.invVoxelX);
	const float gy = joseph_grid_coord(y, params.halfLengthY,
	                                   params.invVoxelY);
	const int x0 = int(floor(gx));
	const int y0 = int(floor(gy));
	const float fx = gx - float(x0);
	const float fy = gy - float(y0);
	return (1.0f - fx) * (1.0f - fy) *
	           joseph_image_value(image, x0, y0, majorIndex, params) +
	       fx * (1.0f - fy) *
	           joseph_image_value(image, x0 + 1, y0, majorIndex, params) +
	       (1.0f - fx) * fy *
	           joseph_image_value(image, x0, y0 + 1, majorIndex, params) +
	       fx * fy *
	           joseph_image_value(image, x0 + 1, y0 + 1, majorIndex, params);
}

inline float joseph_texture_coord(float gridCoord, uint size)
{
	return (gridCoord + 0.5f) / float(size);
}

inline float joseph_texture_forward(
    texture3d<float, access::sample> imageTexture, sampler imageSampler,
    uint axis, int majorIndex, float alpha, ProjectionLineEndpoints line,
    SiddonForwardImageParams params)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);

	float gx = 0.0f;
	float gy = 0.0f;
	float gz = 0.0f;
	if (axis == 0u)
	{
		gx = float(majorIndex);
		gy = joseph_grid_coord(y, params.halfLengthY, params.invVoxelY);
		gz = joseph_grid_coord(z, params.halfLengthZ, params.invVoxelZ);
	}
	else if (axis == 1u)
	{
		gx = joseph_grid_coord(x, params.halfLengthX, params.invVoxelX);
		gy = float(majorIndex);
		gz = joseph_grid_coord(z, params.halfLengthZ, params.invVoxelZ);
	}
	else
	{
		gx = joseph_grid_coord(x, params.halfLengthX, params.invVoxelX);
		gy = joseph_grid_coord(y, params.halfLengthY, params.invVoxelY);
		gz = float(majorIndex);
	}

	return imageTexture.sample(imageSampler,
	    float3(joseph_texture_coord(gx, params.nx),
	           joseph_texture_coord(gy, params.ny),
	           joseph_texture_coord(gz, params.nz))).r;
}

template <typename AtomicImagePointer>
inline void joseph_add_voxel(AtomicImagePointer image, int vx, int vy, int vz,
                             float update,
                             SiddonForwardImageParams params)
{
	if (update == 0.0f || vx < 0 || vy < 0 || vz < 0 ||
	    vx >= int(params.nx) || vy >= int(params.ny) ||
	    vz >= int(params.nz))
	{
		return;
	}
	const uint spatialCount = params.nx * params.ny * params.nz;
	const uint frameBase = params.frame * spatialCount;
	const uint imageOffset =
	    frameBase + siddon_image_offset(uint(vx), uint(vy), uint(vz),
	                                    params);
	atomic_add_float(&image[imageOffset], update);
}

template <typename AtomicImagePointer>
inline void joseph_bilinear_backproject(AtomicImagePointer image, uint axis,
                                        int majorIndex, float alpha,
                                        float update,
                                        ProjectionLineEndpoints line,
                                        SiddonForwardImageParams params)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0u)
	{
		const float gy = joseph_grid_coord(y, params.halfLengthY,
		                                   params.invVoxelY);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int y0 = int(floor(gy));
		const int z0 = int(floor(gz));
		const float fy = gy - float(y0);
		const float fz = gz - float(z0);
		joseph_add_voxel(image, majorIndex, y0, z0,
		    update * (1.0f - fy) * (1.0f - fz), params);
		joseph_add_voxel(image, majorIndex, y0 + 1, z0,
		    update * fy * (1.0f - fz), params);
		joseph_add_voxel(image, majorIndex, y0, z0 + 1,
		    update * (1.0f - fy) * fz, params);
		joseph_add_voxel(image, majorIndex, y0 + 1, z0 + 1,
		    update * fy * fz, params);
		return;
	}
	if (axis == 1u)
	{
		const float gx = joseph_grid_coord(x, params.halfLengthX,
		                                   params.invVoxelX);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int x0 = int(floor(gx));
		const int z0 = int(floor(gz));
		const float fx = gx - float(x0);
		const float fz = gz - float(z0);
		joseph_add_voxel(image, x0, majorIndex, z0,
		    update * (1.0f - fx) * (1.0f - fz), params);
		joseph_add_voxel(image, x0 + 1, majorIndex, z0,
		    update * fx * (1.0f - fz), params);
		joseph_add_voxel(image, x0, majorIndex, z0 + 1,
		    update * (1.0f - fx) * fz, params);
		joseph_add_voxel(image, x0 + 1, majorIndex, z0 + 1,
		    update * fx * fz, params);
		return;
	}
	const float gx = joseph_grid_coord(x, params.halfLengthX,
	                                   params.invVoxelX);
	const float gy = joseph_grid_coord(y, params.halfLengthY,
	                                   params.invVoxelY);
	const int x0 = int(floor(gx));
	const int y0 = int(floor(gy));
	const float fx = gx - float(x0);
	const float fy = gy - float(y0);
	joseph_add_voxel(image, x0, y0, majorIndex,
	    update * (1.0f - fx) * (1.0f - fy), params);
	joseph_add_voxel(image, x0 + 1, y0, majorIndex,
	    update * fx * (1.0f - fy), params);
	joseph_add_voxel(image, x0, y0 + 1, majorIndex,
	    update * (1.0f - fx) * fy, params);
	joseph_add_voxel(image, x0 + 1, y0 + 1, majorIndex,
	    update * fx * fy, params);
}

inline bool joseph_has_voxel_update(int vx, int vy, int vz, float update,
                                    SiddonForwardImageParams params)
{
	return update != 0.0f && vx >= 0 && vy >= 0 && vz >= 0 &&
	       vx < int(params.nx) && vy < int(params.ny) &&
	       vz < int(params.nz);
}

inline uint joseph_count_voxel_update(int vx, int vy, int vz, float update,
                                      SiddonForwardImageParams params)
{
	return joseph_has_voxel_update(vx, vy, vz, update, params) ? 1u : 0u;
}

inline void joseph_count_voxel_hit(device atomic_uint* hitCounts, int vx,
                                   int vy, int vz, float update,
                                   SiddonForwardImageParams params)
{
	if (!joseph_has_voxel_update(vx, vy, vz, update, params))
	{
		return;
	}
	const uint spatialCount = params.nx * params.ny * params.nz;
	const uint frameBase = params.frame * spatialCount;
	const uint imageOffset =
	    frameBase + siddon_image_offset(uint(vx), uint(vy), uint(vz),
	                                    params);
	atomic_fetch_add_explicit(&hitCounts[imageOffset], 1u,
	                          memory_order_relaxed);
}

inline uint joseph_bilinear_update_count(uint axis, int majorIndex,
                                         float alpha, float update,
                                         ProjectionLineEndpoints line,
                                         SiddonForwardImageParams params)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0u)
	{
		const float gy = joseph_grid_coord(y, params.halfLengthY,
		                                   params.invVoxelY);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int y0 = int(floor(gy));
		const int z0 = int(floor(gz));
		const float fy = gy - float(y0);
		const float fz = gz - float(z0);
		return joseph_count_voxel_update(
		           majorIndex, y0, z0,
		           update * (1.0f - fy) * (1.0f - fz), params) +
		       joseph_count_voxel_update(
		           majorIndex, y0 + 1, z0, update * fy * (1.0f - fz),
		           params) +
		       joseph_count_voxel_update(
		           majorIndex, y0, z0 + 1, update * (1.0f - fy) * fz,
		           params) +
		       joseph_count_voxel_update(
		           majorIndex, y0 + 1, z0 + 1, update * fy * fz, params);
	}
	if (axis == 1u)
	{
		const float gx = joseph_grid_coord(x, params.halfLengthX,
		                                   params.invVoxelX);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int x0 = int(floor(gx));
		const int z0 = int(floor(gz));
		const float fx = gx - float(x0);
		const float fz = gz - float(z0);
		return joseph_count_voxel_update(
		           x0, majorIndex, z0,
		           update * (1.0f - fx) * (1.0f - fz), params) +
		       joseph_count_voxel_update(
		           x0 + 1, majorIndex, z0,
		           update * fx * (1.0f - fz), params) +
		       joseph_count_voxel_update(
		           x0, majorIndex, z0 + 1, update * (1.0f - fx) * fz,
		           params) +
		       joseph_count_voxel_update(
		           x0 + 1, majorIndex, z0 + 1, update * fx * fz, params);
	}
	const float gx = joseph_grid_coord(x, params.halfLengthX,
	                                   params.invVoxelX);
	const float gy = joseph_grid_coord(y, params.halfLengthY,
	                                   params.invVoxelY);
	const int x0 = int(floor(gx));
	const int y0 = int(floor(gy));
	const float fx = gx - float(x0);
	const float fy = gy - float(y0);
	return joseph_count_voxel_update(
	           x0, y0, majorIndex, update * (1.0f - fx) * (1.0f - fy),
	           params) +
	       joseph_count_voxel_update(
	           x0 + 1, y0, majorIndex, update * fx * (1.0f - fy),
	           params) +
	       joseph_count_voxel_update(
	           x0, y0 + 1, majorIndex, update * (1.0f - fx) * fy,
	           params) +
	       joseph_count_voxel_update(
	           x0 + 1, y0 + 1, majorIndex, update * fx * fy, params);
}

inline void joseph_bilinear_voxel_hit_count(
    device atomic_uint* hitCounts, uint axis, int majorIndex, float alpha,
    float update, ProjectionLineEndpoints line,
    SiddonForwardImageParams params)
{
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0u)
	{
		const float gy = joseph_grid_coord(y, params.halfLengthY,
		                                   params.invVoxelY);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int y0 = int(floor(gy));
		const int z0 = int(floor(gz));
		const float fy = gy - float(y0);
		const float fz = gz - float(z0);
		joseph_count_voxel_hit(hitCounts, majorIndex, y0, z0,
		    update * (1.0f - fy) * (1.0f - fz), params);
		joseph_count_voxel_hit(hitCounts, majorIndex, y0 + 1, z0,
		    update * fy * (1.0f - fz), params);
		joseph_count_voxel_hit(hitCounts, majorIndex, y0, z0 + 1,
		    update * (1.0f - fy) * fz, params);
		joseph_count_voxel_hit(hitCounts, majorIndex, y0 + 1, z0 + 1,
		    update * fy * fz, params);
		return;
	}
	if (axis == 1u)
	{
		const float gx = joseph_grid_coord(x, params.halfLengthX,
		                                   params.invVoxelX);
		const float gz = joseph_grid_coord(z, params.halfLengthZ,
		                                   params.invVoxelZ);
		const int x0 = int(floor(gx));
		const int z0 = int(floor(gz));
		const float fx = gx - float(x0);
		const float fz = gz - float(z0);
		joseph_count_voxel_hit(hitCounts, x0, majorIndex, z0,
		    update * (1.0f - fx) * (1.0f - fz), params);
		joseph_count_voxel_hit(hitCounts, x0 + 1, majorIndex, z0,
		    update * fx * (1.0f - fz), params);
		joseph_count_voxel_hit(hitCounts, x0, majorIndex, z0 + 1,
		    update * (1.0f - fx) * fz, params);
		joseph_count_voxel_hit(hitCounts, x0 + 1, majorIndex, z0 + 1,
		    update * fx * fz, params);
		return;
	}
	const float gx = joseph_grid_coord(x, params.halfLengthX,
	                                   params.invVoxelX);
	const float gy = joseph_grid_coord(y, params.halfLengthY,
	                                   params.invVoxelY);
	const int x0 = int(floor(gx));
	const int y0 = int(floor(gy));
	const float fx = gx - float(x0);
	const float fy = gy - float(y0);
	joseph_count_voxel_hit(hitCounts, x0, y0, majorIndex,
	    update * (1.0f - fx) * (1.0f - fy), params);
	joseph_count_voxel_hit(hitCounts, x0 + 1, y0, majorIndex,
	    update * fx * (1.0f - fy), params);
	joseph_count_voxel_hit(hitCounts, x0, y0 + 1, majorIndex,
	    update * (1.0f - fx) * fy, params);
	joseph_count_voxel_hit(hitCounts, x0 + 1, y0 + 1, majorIndex,
	    update * fx * fy, params);
}

inline float joseph_forward_single_ray_value(
    device const float* image, ProjectionLineEndpoints line,
    SiddonForwardImageParams params)
{
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return 0.0f;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0.0f;
	}

	float projection = 0.0f;
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight(axisCache, alpha, alphaMin,
		                                alphaMax);
		if (weight == 0.0f)
		{
			continue;
		}
		projection += weight * joseph_bilinear_forward(
		                            image, axis, majorIndex, alpha, line,
		                            params);
	}
	return projection;
}

inline float joseph_forward_single_ray_texture_value(
    texture3d<float, access::sample> imageTexture, sampler imageSampler,
    ProjectionLineEndpoints line, SiddonForwardImageParams params)
{
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return 0.0f;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0.0f;
	}

	float projection = 0.0f;
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight(axisCache, alpha, alphaMin,
		                                alphaMax);
		if (weight == 0.0f)
		{
			continue;
		}
		projection += weight *
		              joseph_texture_forward(
		                  imageTexture, imageSampler, axis, majorIndex,
		                  alpha, line, params);
	}
	return projection;
}

template <typename AtomicImagePointer>
inline void joseph_backproject_single_ray_atomic_value(
    AtomicImagePointer image, ProjectionLineEndpoints line,
    float projectionValue, SiddonForwardImageParams params)
{
	if (projectionValue == 0.0f)
	{
		return;
	}
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return;
	}

	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight(axisCache, alpha, alphaMin,
		                                alphaMax);
		if (weight == 0.0f)
		{
			continue;
		}
		joseph_bilinear_backproject(
		    image, axis, majorIndex, alpha, projectionValue * weight, line,
		    params);
	}
}

inline uint joseph_backproject_single_ray_update_count_value(
    ProjectionLineEndpoints line, float projectionValue,
    SiddonForwardImageParams params)
{
	if (projectionValue == 0.0f)
	{
		return 0u;
	}
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return 0u;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0u;
	}

	uint updateCount = 0u;
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight(axisCache, alpha, alphaMin,
		                                alphaMax);
		if (weight == 0.0f)
		{
			continue;
		}
		updateCount += joseph_bilinear_update_count(
		    axis, majorIndex, alpha, projectionValue * weight, line, params);
	}
	return updateCount;
}

inline void joseph_backproject_single_ray_voxel_hit_count_value(
    device atomic_uint* hitCounts, ProjectionLineEndpoints line,
    float projectionValue, SiddonForwardImageParams params)
{
	if (projectionValue == 0.0f)
	{
		return;
	}
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return;
	}

	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight(axisCache, alpha, alphaMin,
		                                alphaMax);
		if (weight == 0.0f)
		{
			continue;
		}
		joseph_bilinear_voxel_hit_count(hitCounts, axis, majorIndex, alpha,
		    projectionValue * weight, line, params);
	}
}

inline float joseph_forward_single_ray_sample_stride_value(
    device const float* image, ProjectionLineEndpoints line,
    SiddonForwardImageParams params, uint sampleStride)
{
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return 0.0f;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0.0f;
	}

	float projection = 0.0f;
	const int step = sampleStride <= 1u ? 1 : int(sampleStride);
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; majorIndex += step)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight_stride(
		        axisCache, alpha, alphaMin, alphaMax, sampleStride);
		if (weight == 0.0f)
		{
			continue;
		}
		projection += weight * joseph_bilinear_forward(
		                            image, axis, majorIndex, alpha, line,
		                            params);
	}
	return projection;
}

inline float joseph_forward_single_ray_texture_sample_stride_value(
    texture3d<float, access::sample> imageTexture, sampler imageSampler,
    ProjectionLineEndpoints line, SiddonForwardImageParams params,
    uint sampleStride)
{
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return 0.0f;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0.0f;
	}

	float projection = 0.0f;
	const int step = sampleStride <= 1u ? 1 : int(sampleStride);
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; majorIndex += step)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight_stride(
		        axisCache, alpha, alphaMin, alphaMax, sampleStride);
		if (weight == 0.0f)
		{
			continue;
		}
		projection += weight *
		              joseph_texture_forward(
		                  imageTexture, imageSampler, axis, majorIndex,
		                  alpha, line, params);
	}
	return projection;
}

template <typename AtomicImagePointer>
inline void joseph_backproject_single_ray_atomic_sample_stride_value(
    AtomicImagePointer image, ProjectionLineEndpoints line,
    float projectionValue, SiddonForwardImageParams params, uint sampleStride)
{
	if (projectionValue == 0.0f)
	{
		return;
	}
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return;
	}

	const int step = sampleStride <= 1u ? 1 : int(sampleStride);
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; majorIndex += step)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight_stride(
		        axisCache, alpha, alphaMin, alphaMax, sampleStride);
		if (weight == 0.0f)
		{
			continue;
		}
		joseph_bilinear_backproject(
		    image, axis, majorIndex, alpha, projectionValue * weight, line,
		    params);
	}
}

inline uint joseph_backproject_single_ray_update_count_sample_stride_value(
    ProjectionLineEndpoints line, float projectionValue,
    SiddonForwardImageParams params, uint sampleStride)
{
	if (projectionValue == 0.0f)
	{
		return 0u;
	}
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return 0u;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0u;
	}

	uint updateCount = 0u;
	const int step = sampleStride <= 1u ? 1 : int(sampleStride);
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; majorIndex += step)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight_stride(
		        axisCache, alpha, alphaMin, alphaMax, sampleStride);
		if (weight == 0.0f)
		{
			continue;
		}
		updateCount += joseph_bilinear_update_count(
		    axis, majorIndex, alpha, projectionValue * weight, line, params);
	}
	return updateCount;
}

inline void joseph_backproject_single_ray_voxel_hit_count_sample_stride_value(
    device atomic_uint* hitCounts, ProjectionLineEndpoints line,
    float projectionValue, SiddonForwardImageParams params, uint sampleStride)
{
	if (projectionValue == 0.0f)
	{
		return;
	}
	float alphaMin;
	float alphaMax;
	if (!joseph_alpha_range(line, params, alphaMin, alphaMax))
	{
		return;
	}
	const uint axis = joseph_major_axis(line, params);
	int first;
	int last;
	if (!joseph_sample_bounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return;
	}

	const int step = sampleStride <= 1u ? 1 : int(sampleStride);
	const JosephAxisCache axisCache =
	    joseph_make_axis_cache(line, params, axis);
	for (int majorIndex = first; majorIndex <= last; majorIndex += step)
	{
		const float alpha =
		    joseph_cached_sample_alpha(axisCache, majorIndex);
		const float weight =
		    joseph_cached_sample_weight_stride(
		        axisCache, alpha, alphaMin, alphaMax, sampleStride);
		if (weight == 0.0f)
		{
			continue;
		}
		joseph_bilinear_voxel_hit_count(hitCounts, axis, majorIndex, alpha,
		    projectionValue * weight, line, params);
	}
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

kernel void siddon_backproject_single_ray_native_atomic_float(
    device atomic_float* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device const float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	siddon_backproject_single_ray_atomic_value(
	    image, lines[id], projectionValues[id], params);
}

kernel void joseph_forward_single_ray(
    device const float* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	projectionValues[id] =
	    joseph_forward_single_ray_value(image, lines[id], params);
}

kernel void joseph_forward_single_ray_sample_stride(
    device const float* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    constant uint& sampleStride [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
	projectionValues[id] = joseph_forward_single_ray_sample_stride_value(
	    image, lines[id], params, sampleStride);
}

kernel void joseph_forward_single_ray_texture(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device float* projectionValues [[buffer(1)]],
    constant SiddonForwardImageParams& params [[buffer(2)]],
    texture3d<float, access::sample> imageTexture [[texture(0)]],
    sampler imageSampler [[sampler(0)]],
    uint id [[thread_position_in_grid]])
{
	projectionValues[id] = joseph_forward_single_ray_texture_value(
	    imageTexture, imageSampler, lines[id], params);
}

kernel void joseph_forward_single_ray_texture_sample_stride(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device float* projectionValues [[buffer(1)]],
    constant SiddonForwardImageParams& params [[buffer(2)]],
    constant uint& sampleStride [[buffer(3)]],
    texture3d<float, access::sample> imageTexture [[texture(0)]],
    sampler imageSampler [[sampler(0)]],
    uint id [[thread_position_in_grid]])
{
	projectionValues[id] =
	    joseph_forward_single_ray_texture_sample_stride_value(
	        imageTexture, imageSampler, lines[id], params, sampleStride);
}

kernel void joseph_backproject_single_ray(
    device atomic_uint* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device const float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	joseph_backproject_single_ray_atomic_value(
	    image, lines[id], projectionValues[id], params);
}

kernel void joseph_backproject_single_ray_sample_stride(
    device atomic_uint* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device const float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    constant uint& sampleStride [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
	joseph_backproject_single_ray_atomic_sample_stride_value(
	    image, lines[id], projectionValues[id], params, sampleStride);
}

kernel void joseph_backproject_single_ray_native_atomic_float(
    device atomic_float* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device const float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	joseph_backproject_single_ray_atomic_value(
	    image, lines[id], projectionValues[id], params);
}

kernel void joseph_backproject_single_ray_sample_stride_native_atomic_float(
    device atomic_float* image [[buffer(0)]],
    device const ProjectionLineEndpoints* lines [[buffer(1)]],
    device const float* projectionValues [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    constant uint& sampleStride [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
	joseph_backproject_single_ray_atomic_sample_stride_value(
	    image, lines[id], projectionValues[id], params, sampleStride);
}

kernel void joseph_backproject_single_ray_update_count(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device const float* projectionValues [[buffer(1)]],
    device uint* updateCounts [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	updateCounts[id] = joseph_backproject_single_ray_update_count_value(
	    lines[id], projectionValues[id], params);
}

kernel void joseph_backproject_single_ray_update_count_sample_stride(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device const float* projectionValues [[buffer(1)]],
    device uint* updateCounts [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    constant uint& sampleStride [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
	updateCounts[id] =
	    joseph_backproject_single_ray_update_count_sample_stride_value(
	        lines[id], projectionValues[id], params, sampleStride);
}

kernel void joseph_backproject_single_ray_voxel_hit_count(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device const float* projectionValues [[buffer(1)]],
    device atomic_uint* hitCounts [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	joseph_backproject_single_ray_voxel_hit_count_value(
	    hitCounts, lines[id], projectionValues[id], params);
}

kernel void joseph_backproject_single_ray_voxel_hit_count_sample_stride(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device const float* projectionValues [[buffer(1)]],
    device atomic_uint* hitCounts [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    constant uint& sampleStride [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
	joseph_backproject_single_ray_voxel_hit_count_sample_stride_value(
	    hitCounts, lines[id], projectionValues[id], params, sampleStride);
}

kernel void siddon_backproject_single_ray_update_count(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device const float* projectionValues [[buffer(1)]],
    device uint* updateCounts [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	updateCounts[id] = siddon_backproject_single_ray_update_count_value(
	    lines[id], projectionValues[id], params);
}

kernel void siddon_backproject_single_ray_voxel_hit_count(
    device const ProjectionLineEndpoints* lines [[buffer(0)]],
    device const float* projectionValues [[buffer(1)]],
    device atomic_uint* hitCounts [[buffer(2)]],
    constant SiddonForwardImageParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
	siddon_backproject_single_ray_voxel_hit_count_value(
	    hitCounts, lines[id], projectionValues[id], params);
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
