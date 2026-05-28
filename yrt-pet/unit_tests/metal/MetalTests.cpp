/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalSmoke.hpp"
#include "yrt-pet/backends/metal/ExperimentalBackend.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ImageMetal.hpp"
#include "yrt-pet/backends/metal/ImageOps.hpp"
#include "yrt-pet/backends/metal/JosephProjectorMetal.hpp"
#include "yrt-pet/backends/metal/JosephProjectorOps.hpp"
#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"
#include "yrt-pet/backends/metal/OperatorPsfMetal.hpp"
#include "yrt-pet/backends/metal/PsfFileOps.hpp"
#include "yrt-pet/backends/metal/PsfOps.hpp"
#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorKernels.hpp"
#include "yrt-pet/backends/metal/ProjectionGeometryOps.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorMetal.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorOps.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorMetal.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageParams.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"
#include "yrt-pet/utils/Array.hpp"
#include "yrt-pet/utils/Tools.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

struct TestCase
{
	std::string name;
	bool (*run)();
};

class ScopedEnvVar
{
public:
	ScopedEnvVar(std::string name, const char* value) : m_name(std::move(name))
	{
		const char* previous = std::getenv(m_name.c_str());
		if (previous != nullptr)
		{
			m_hadPreviousValue = true;
			m_previousValue = previous;
		}
		setenv(m_name.c_str(), value, 1);
	}

	~ScopedEnvVar()
	{
		if (m_hadPreviousValue)
		{
			setenv(m_name.c_str(), m_previousValue.c_str(), 1);
		}
		else
		{
			unsetenv(m_name.c_str());
		}
	}

private:
	std::string m_name;
	std::string m_previousValue;
	bool m_hadPreviousValue = false;
};

bool almostEqual(float actual, float expected)
{
	const float scale = std::max(1.0f, std::fabs(expected));
	return std::fabs(actual - expected) <= 1.0e-5f * scale;
}

bool imagesMatch(const yrt::Image& actual, const yrt::Image& expected)
{
	const yrt::ImageParams& params = actual.getParams();
	const std::size_t count = static_cast<std::size_t>(params.nx) *
	                          static_cast<std::size_t>(params.ny) *
	                          static_cast<std::size_t>(params.nz) *
	                          static_cast<std::size_t>(params.nt);
	const float* actualPtr = actual.getRawPointer();
	const float* expectedPtr = expected.getRawPointer();
	for (std::size_t i = 0; i < count; ++i)
	{
		if (!almostEqual(actualPtr[i], expectedPtr[i]))
		{
			return false;
		}
	}
	return true;
}

bool valuesMatch(const std::vector<float>& actual,
                 const std::vector<float>& expected)
{
	if (actual.size() != expected.size())
	{
		return false;
	}
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		if (!almostEqual(actual[i], expected[i]))
		{
			return false;
		}
	}
	return true;
}

bool dotAlmostEqual(double actual, double expected)
{
	const double scale = std::max({1.0, std::fabs(actual), std::fabs(expected)});
	return std::fabs(actual - expected) <= 1.0e-4 * scale;
}

double dotValues(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
	if (lhs.size() != rhs.size())
	{
		return std::numeric_limits<double>::quiet_NaN();
	}

	double sum = 0.0;
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		sum += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
	}
	return sum;
}

double imageDotFrame(const yrt::Image& lhs, const yrt::Image& rhs,
                     std::uint32_t frame)
{
	const yrt::ImageParams& params = lhs.getParams();
	const std::size_t spatialCount = static_cast<std::size_t>(params.nx) *
	                                 static_cast<std::size_t>(params.ny) *
	                                 static_cast<std::size_t>(params.nz);
	const std::size_t offset = spatialCount * frame;
	const float* lhsPtr = lhs.getRawPointer();
	const float* rhsPtr = rhs.getRawPointer();

	double sum = 0.0;
	for (std::size_t i = 0; i < spatialCount; ++i)
	{
		sum += static_cast<double>(lhsPtr[offset + i]) *
		       static_cast<double>(rhsPtr[offset + i]);
	}
	return sum;
}

void getAlpha(float r0, float r1, float p1, float p2, float invP12,
              float& alphaMin, float& alphaMax)
{
	alphaMin = 0.0f;
	alphaMax = 1.0f;
	if (p1 != p2)
	{
		const float a0 = (r0 - p1) * invP12;
		const float a1 = (r1 - p1) * invP12;
		if (a0 < a1)
		{
			alphaMin = a0;
			alphaMax = a1;
		}
		else
		{
			alphaMin = a1;
			alphaMax = a0;
		}
	}
	else if (p1 < r0 || p1 > r1)
	{
		alphaMax = 0.0f;
		alphaMin = 1.0f;
	}
}

yrt::backend::metal::ProjectionAlphaRange cpuSiddonEntryRange(
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    const yrt::backend::metal::ProjectionImageBounds& bounds)
{
	float fovMin = 0.0f;
	float fovMax = 1.0f;
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	const float a = dx * dx + dy * dy;
	const float b = 2.0f * (dx * line.p1x + dy * line.p1y);
	const float c = line.p1x * line.p1x + line.p1y * line.p1y -
	                bounds.fovRadius * bounds.fovRadius;
	const float delta = b * b - 4.0f * a * c;
	if (a != 0.0f)
	{
		if (delta <= 0.0f)
		{
			return {1.0f, 0.0f, 0u};
		}
		const float sqrtDelta = std::sqrt(delta);
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
	getAlpha(-0.5f * bounds.lengthX, 0.5f * bounds.lengthX, line.p1x,
	         line.p2x, invX, axMin, axMax);
	getAlpha(-0.5f * bounds.lengthY, 0.5f * bounds.lengthY, line.p1y,
	         line.p2y, invY, ayMin, ayMax);
	getAlpha(-0.5f * bounds.lengthZ, 0.5f * bounds.lengthZ, line.p1z,
	         line.p2z, invZ, azMin, azMax);

	const float alphaMin =
	    std::max({0.0f, fovMin, axMin, ayMin, azMin});
	const float alphaMax =
	    std::min({1.0f, fovMax, axMax, ayMax, azMax});
	return {alphaMin, alphaMax, alphaMin < alphaMax ? 1u : 0u};
}

bool rangesMatch(
    const std::vector<yrt::backend::metal::ProjectionAlphaRange>& actual,
    const std::vector<yrt::backend::metal::ProjectionAlphaRange>& expected)
{
	if (actual.size() != expected.size())
	{
		return false;
	}
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		if (actual[i].valid != expected[i].valid ||
		    !almostEqual(actual[i].alphaMin, expected[i].alphaMin) ||
		    !almostEqual(actual[i].alphaMax, expected[i].alphaMax))
		{
			return false;
		}
	}
	return true;
}

bool lineEndpointVectorsMatch(
    const std::vector<yrt::backend::metal::ProjectionLineEndpoints>& actual,
    const std::vector<yrt::backend::metal::ProjectionLineEndpoints>& expected)
{
	if (actual.size() != expected.size())
	{
		return false;
	}
	for (std::size_t i = 0; i < actual.size(); ++i)
	{
		if (!almostEqual(actual[i].p1x, expected[i].p1x) ||
		    !almostEqual(actual[i].p1y, expected[i].p1y) ||
		    !almostEqual(actual[i].p1z, expected[i].p1z) ||
		    !almostEqual(actual[i].p2x, expected[i].p2x) ||
		    !almostEqual(actual[i].p2y, expected[i].p2y) ||
		    !almostEqual(actual[i].p2z, expected[i].p2z))
		{
			return false;
		}
	}
	return true;
}

yrt::Line3D makeLine(
    const yrt::backend::metal::ProjectionLineEndpoints& endpoints)
{
	yrt::Line3D line;
	line.point1.x = endpoints.p1x;
	line.point1.y = endpoints.p1y;
	line.point1.z = endpoints.p1z;
	line.point2.x = endpoints.p2x;
	line.point2.y = endpoints.p2y;
	line.point2.z = endpoints.p2z;
	return line;
}

std::size_t imageVoxelCount(const yrt::ImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

void copyValuesToImage(yrt::Image& image, const std::vector<float>& values)
{
	std::copy(values.begin(), values.end(), image.getRawPointer());
}

bool josephReferenceAlphaRange(
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    const yrt::ImageParams& params, float& alphaMin, float& alphaMax)
{
	const yrt::backend::metal::ProjectionImageBounds bounds{
	    params.length_x, params.length_y, params.length_z, params.fovRadius};
	const auto range = cpuSiddonEntryRange(line, bounds);
	alphaMin = range.alphaMin;
	alphaMax = range.alphaMax;
	return range.valid != 0u;
}

float josephGridCoord(float coord, float halfLength, float invVoxel)
{
	return (coord + halfLength) * invVoxel - 0.5f;
}

float josephAxisCoord(
    const yrt::backend::metal::ProjectionLineEndpoints& line, int axis,
    float alpha)
{
	if (axis == 0)
	{
		return line.p1x + alpha * (line.p2x - line.p1x);
	}
	if (axis == 1)
	{
		return line.p1y + alpha * (line.p2y - line.p1y);
	}
	return line.p1z + alpha * (line.p2z - line.p1z);
}

float josephAxisDelta(
    const yrt::backend::metal::ProjectionLineEndpoints& line, int axis)
{
	if (axis == 0)
	{
		return line.p2x - line.p1x;
	}
	if (axis == 1)
	{
		return line.p2y - line.p1y;
	}
	return line.p2z - line.p1z;
}

float josephAxisStart(
    const yrt::backend::metal::ProjectionLineEndpoints& line, int axis)
{
	if (axis == 0)
	{
		return line.p1x;
	}
	if (axis == 1)
	{
		return line.p1y;
	}
	return line.p1z;
}

int josephMajorAxis(
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    const yrt::ImageParams& params)
{
	const float sx = std::fabs(line.p2x - line.p1x) / params.vx;
	const float sy = std::fabs(line.p2y - line.p1y) / params.vy;
	const float sz = std::fabs(line.p2z - line.p1z) / params.vz;
	if (sx >= sy && sx >= sz)
	{
		return 0;
	}
	return sy >= sz ? 1 : 2;
}

float josephHalfLength(const yrt::ImageParams& params, int axis)
{
	if (axis == 0)
	{
		return 0.5f * params.length_x;
	}
	if (axis == 1)
	{
		return 0.5f * params.length_y;
	}
	return 0.5f * params.length_z;
}

float josephVoxelSize(const yrt::ImageParams& params, int axis)
{
	if (axis == 0)
	{
		return params.vx;
	}
	if (axis == 1)
	{
		return params.vy;
	}
	return params.vz;
}

float josephInvVoxelSize(const yrt::ImageParams& params, int axis)
{
	return 1.0f / josephVoxelSize(params, axis);
}

int josephAxisSize(const yrt::ImageParams& params, int axis)
{
	if (axis == 0)
	{
		return static_cast<int>(params.nx);
	}
	if (axis == 1)
	{
		return static_cast<int>(params.ny);
	}
	return static_cast<int>(params.nz);
}

bool josephSampleBounds(
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    const yrt::ImageParams& params, int axis, float alphaMin, float alphaMax,
    int& first, int& last)
{
	const float grid0 = josephGridCoord(
	    josephAxisCoord(line, axis, alphaMin), josephHalfLength(params, axis),
	    josephInvVoxelSize(params, axis));
	const float grid1 = josephGridCoord(
	    josephAxisCoord(line, axis, alphaMax), josephHalfLength(params, axis),
	    josephInvVoxelSize(params, axis));
	first = static_cast<int>(std::ceil(std::min(grid0, grid1)));
	last = static_cast<int>(std::floor(std::max(grid0, grid1)));
	first = std::max(first, 0);
	last = std::min(last, josephAxisSize(params, axis) - 1);
	return first <= last;
}

float josephSampleAlpha(
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    const yrt::ImageParams& params, int axis, int majorIndex)
{
	const float dAxis = josephAxisDelta(line, axis);
	if (dAxis == 0.0f)
	{
		return 0.0f;
	}
	const float centerCoord = -josephHalfLength(params, axis) +
	                          (static_cast<float>(majorIndex) + 0.5f) *
	                              josephVoxelSize(params, axis);
	return (centerCoord - josephAxisStart(line, axis)) / dAxis;
}

float josephSampleWeight(
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    const yrt::ImageParams& params, int axis, int majorIndex, float alphaMin,
    float alphaMax)
{
	const float dAxis = josephAxisDelta(line, axis);
	if (dAxis == 0.0f)
	{
		return 0.0f;
	}
	const float centerAlpha =
	    josephSampleAlpha(line, params, axis, majorIndex);
	const float halfAlphaStep =
	    0.5f * josephVoxelSize(params, axis) / std::fabs(dAxis);
	const float segmentStart = std::max(alphaMin, centerAlpha - halfAlphaStep);
	const float segmentEnd = std::min(alphaMax, centerAlpha + halfAlphaStep);
	if (segmentStart >= segmentEnd)
	{
		return 0.0f;
	}
	const float dx = line.p2x - line.p1x;
	const float dy = line.p2y - line.p1y;
	const float dz = line.p2z - line.p1z;
	return std::sqrt(dx * dx + dy * dy + dz * dz) *
	       (segmentEnd - segmentStart);
}

std::size_t imageOffset(const yrt::ImageParams& params, int x, int y, int z,
                        std::uint32_t frame)
{
	return static_cast<std::size_t>(frame) *
	           static_cast<std::size_t>(params.nx) *
	           static_cast<std::size_t>(params.ny) *
	           static_cast<std::size_t>(params.nz) +
	       static_cast<std::size_t>(x) +
	       static_cast<std::size_t>(params.nx) *
	           (static_cast<std::size_t>(y) +
	            static_cast<std::size_t>(params.ny) *
	                static_cast<std::size_t>(z));
}

float imageValue(const yrt::Image& image, int x, int y, int z,
                 std::uint32_t frame)
{
	const yrt::ImageParams& params = image.getParams();
	if (x < 0 || y < 0 || z < 0 || x >= params.nx || y >= params.ny ||
	    z >= params.nz)
	{
		return 0.0f;
	}
	return image.getRawPointer()[imageOffset(params, x, y, z, frame)];
}

float josephBilinearForward(const yrt::Image& image, int axis, int majorIndex,
                            float alpha,
                            const yrt::backend::metal::ProjectionLineEndpoints&
                                line,
                            std::uint32_t frame)
{
	const yrt::ImageParams& params = image.getParams();
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0)
	{
		const float gy = josephGridCoord(
		    y, 0.5f * params.length_y, 1.0f / params.vy);
		const float gz = josephGridCoord(
		    z, 0.5f * params.length_z, 1.0f / params.vz);
		const int y0 = static_cast<int>(std::floor(gy));
		const int z0 = static_cast<int>(std::floor(gz));
		const float fy = gy - static_cast<float>(y0);
		const float fz = gz - static_cast<float>(z0);
		return (1.0f - fy) * (1.0f - fz) *
		           imageValue(image, majorIndex, y0, z0, frame) +
		       fy * (1.0f - fz) *
		           imageValue(image, majorIndex, y0 + 1, z0, frame) +
		       (1.0f - fy) * fz *
		           imageValue(image, majorIndex, y0, z0 + 1, frame) +
		       fy * fz *
		           imageValue(image, majorIndex, y0 + 1, z0 + 1, frame);
	}
	if (axis == 1)
	{
		const float gx = josephGridCoord(
		    x, 0.5f * params.length_x, 1.0f / params.vx);
		const float gz = josephGridCoord(
		    z, 0.5f * params.length_z, 1.0f / params.vz);
		const int x0 = static_cast<int>(std::floor(gx));
		const int z0 = static_cast<int>(std::floor(gz));
		const float fx = gx - static_cast<float>(x0);
		const float fz = gz - static_cast<float>(z0);
		return (1.0f - fx) * (1.0f - fz) *
		           imageValue(image, x0, majorIndex, z0, frame) +
		       fx * (1.0f - fz) *
		           imageValue(image, x0 + 1, majorIndex, z0, frame) +
		       (1.0f - fx) * fz *
		           imageValue(image, x0, majorIndex, z0 + 1, frame) +
		       fx * fz *
		           imageValue(image, x0 + 1, majorIndex, z0 + 1, frame);
	}
	const float gx = josephGridCoord(
	    x, 0.5f * params.length_x, 1.0f / params.vx);
	const float gy = josephGridCoord(
	    y, 0.5f * params.length_y, 1.0f / params.vy);
	const int x0 = static_cast<int>(std::floor(gx));
	const int y0 = static_cast<int>(std::floor(gy));
	const float fx = gx - static_cast<float>(x0);
	const float fy = gy - static_cast<float>(y0);
	return (1.0f - fx) * (1.0f - fy) *
	           imageValue(image, x0, y0, majorIndex, frame) +
	       fx * (1.0f - fy) *
	           imageValue(image, x0 + 1, y0, majorIndex, frame) +
	       (1.0f - fx) * fy *
	           imageValue(image, x0, y0 + 1, majorIndex, frame) +
	       fx * fy *
	           imageValue(image, x0 + 1, y0 + 1, majorIndex, frame);
}

float josephForwardReference(
    const yrt::Image& image,
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    std::uint32_t frame)
{
	const yrt::ImageParams& params = image.getParams();
	float alphaMin = 0.0f;
	float alphaMax = 0.0f;
	if (!josephReferenceAlphaRange(line, params, alphaMin, alphaMax))
	{
		return 0.0f;
	}
	const int axis = josephMajorAxis(line, params);
	int first = 0;
	int last = -1;
	if (!josephSampleBounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return 0.0f;
	}
	float projection = 0.0f;
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float weight =
		    josephSampleWeight(line, params, axis, majorIndex, alphaMin,
		                       alphaMax);
		const float alpha =
		    josephSampleAlpha(line, params, axis, majorIndex);
		projection += weight *
		              josephBilinearForward(image, axis, majorIndex, alpha,
		                                    line, frame);
	}
	return projection;
}

void addImageValue(yrt::Image& image, int x, int y, int z, std::uint32_t frame,
                   float value)
{
	yrt::ImageParams params = image.getParams();
	if (value == 0.0f || x < 0 || y < 0 || z < 0 || x >= params.nx ||
	    y >= params.ny || z >= params.nz)
	{
		return;
	}
	image.getRawPointer()[imageOffset(params, x, y, z, frame)] += value;
}

void josephBilinearBackProject(yrt::Image& image, int axis, int majorIndex,
                               float alpha, float update,
                               const yrt::backend::metal::
                                   ProjectionLineEndpoints& line,
                               std::uint32_t frame)
{
	const yrt::ImageParams& params = image.getParams();
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	if (axis == 0)
	{
		const float gy = josephGridCoord(
		    y, 0.5f * params.length_y, 1.0f / params.vy);
		const float gz = josephGridCoord(
		    z, 0.5f * params.length_z, 1.0f / params.vz);
		const int y0 = static_cast<int>(std::floor(gy));
		const int z0 = static_cast<int>(std::floor(gz));
		const float fy = gy - static_cast<float>(y0);
		const float fz = gz - static_cast<float>(z0);
		addImageValue(image, majorIndex, y0, z0, frame,
		              update * (1.0f - fy) * (1.0f - fz));
		addImageValue(image, majorIndex, y0 + 1, z0, frame,
		              update * fy * (1.0f - fz));
		addImageValue(image, majorIndex, y0, z0 + 1, frame,
		              update * (1.0f - fy) * fz);
		addImageValue(image, majorIndex, y0 + 1, z0 + 1, frame,
		              update * fy * fz);
		return;
	}
	if (axis == 1)
	{
		const float gx = josephGridCoord(
		    x, 0.5f * params.length_x, 1.0f / params.vx);
		const float gz = josephGridCoord(
		    z, 0.5f * params.length_z, 1.0f / params.vz);
		const int x0 = static_cast<int>(std::floor(gx));
		const int z0 = static_cast<int>(std::floor(gz));
		const float fx = gx - static_cast<float>(x0);
		const float fz = gz - static_cast<float>(z0);
		addImageValue(image, x0, majorIndex, z0, frame,
		              update * (1.0f - fx) * (1.0f - fz));
		addImageValue(image, x0 + 1, majorIndex, z0, frame,
		              update * fx * (1.0f - fz));
		addImageValue(image, x0, majorIndex, z0 + 1, frame,
		              update * (1.0f - fx) * fz);
		addImageValue(image, x0 + 1, majorIndex, z0 + 1, frame,
		              update * fx * fz);
		return;
	}
	const float gx = josephGridCoord(
	    x, 0.5f * params.length_x, 1.0f / params.vx);
	const float gy = josephGridCoord(
	    y, 0.5f * params.length_y, 1.0f / params.vy);
	const int x0 = static_cast<int>(std::floor(gx));
	const int y0 = static_cast<int>(std::floor(gy));
	const float fx = gx - static_cast<float>(x0);
	const float fy = gy - static_cast<float>(y0);
	addImageValue(image, x0, y0, majorIndex, frame,
	              update * (1.0f - fx) * (1.0f - fy));
	addImageValue(image, x0 + 1, y0, majorIndex, frame,
	              update * fx * (1.0f - fy));
	addImageValue(image, x0, y0 + 1, majorIndex, frame,
	              update * (1.0f - fx) * fy);
	addImageValue(image, x0 + 1, y0 + 1, majorIndex, frame,
	              update * fx * fy);
}

void josephBackProjectReference(
    yrt::Image& image,
    const yrt::backend::metal::ProjectionLineEndpoints& line,
    float projectionValue, std::uint32_t frame)
{
	const yrt::ImageParams& params = image.getParams();
	float alphaMin = 0.0f;
	float alphaMax = 0.0f;
	if (projectionValue == 0.0f ||
	    !josephReferenceAlphaRange(line, params, alphaMin, alphaMax))
	{
		return;
	}
	const int axis = josephMajorAxis(line, params);
	int first = 0;
	int last = -1;
	if (!josephSampleBounds(line, params, axis, alphaMin, alphaMax, first,
	        last))
	{
		return;
	}
	for (int majorIndex = first; majorIndex <= last; ++majorIndex)
	{
		const float weight =
		    josephSampleWeight(line, params, axis, majorIndex, alphaMin,
		                       alphaMax);
		const float alpha =
		    josephSampleAlpha(line, params, axis, majorIndex);
		josephBilinearBackProject(image, axis, majorIndex, alpha,
		                          projectionValue * weight, line, frame);
	}
}

yrt::Line3D makeBridgeLine(float p1x, float p1y, float p1z, float p2x,
                           float p2y, float p2z)
{
	yrt::Line3D line;
	line.point1.x = p1x;
	line.point1.y = p1y;
	line.point1.z = p1z;
	line.point2.x = p2x;
	line.point2.y = p2y;
	line.point2.z = p2z;
	return line;
}

yrt::Scanner makeBridgeScanner()
{
	return yrt::Scanner("metal_bridge_test", 25.0f, 5.0f, 3.0f, 10.0f,
	                    300.0f, 256, 5, 1, 4, 2, 8);
}

class MetalBridgeProjectionData final : public yrt::ProjectionData
{
public:
	MetalBridgeProjectionData(const yrt::Scanner& scanner,
	                          std::vector<yrt::Line3D> lines,
	                          std::vector<float> values,
	                          std::vector<yrt::frame_t> dynamicFrames = {},
	                          std::vector<yrt::det_pair_t> detectorPairs = {})
	    : ProjectionData(scanner),
	      m_lines{std::move(lines)},
	      m_values{std::move(values)},
	      m_dynamicFrames{std::move(dynamicFrames)},
	      m_detectorPairs{std::move(detectorPairs)}
	{
	}

	yrt::size_t count() const override
	{
		return m_lines.size();
	}

	float getProjectionValue(yrt::bin_t id) const override
	{
		return m_values[static_cast<std::size_t>(id)];
	}

	void setProjectionValue(yrt::bin_t id, float val) override
	{
		m_values[static_cast<std::size_t>(id)] = val;
	}

	yrt::det_id_t getDetector1(yrt::bin_t id) const override
	{
		if (!m_detectorPairs.empty())
		{
			return m_detectorPairs[static_cast<std::size_t>(id)].d1;
		}
		return 0;
	}

	yrt::det_id_t getDetector2(yrt::bin_t id) const override
	{
		if (!m_detectorPairs.empty())
		{
			return m_detectorPairs[static_cast<std::size_t>(id)].d2;
		}
		return 1;
	}

	std::unique_ptr<yrt::BinIterator>
	    getBinIter(int /*numSubsets*/, int /*idxSubset*/) const override
	{
		return std::make_unique<yrt::BinIteratorRange>(
		    static_cast<yrt::bin_t>(m_lines.size()));
	}

	bool hasDynamicFraming() const override
	{
		return !m_dynamicFrames.empty();
	}

	yrt::frame_t getDynamicFrame(yrt::bin_t id) const override
	{
		return hasDynamicFraming() ? m_dynamicFrames[static_cast<std::size_t>(id)]
		                           : 0;
	}

	yrt::size_t getNumDynamicFrames() const override
	{
		if (!hasDynamicFraming())
		{
			return 1;
		}

		yrt::frame_t maxFrame = 0;
		for (const yrt::frame_t frame : m_dynamicFrames)
		{
			if (frame >= 0)
			{
				maxFrame = std::max(maxFrame, frame);
			}
		}
		return static_cast<std::size_t>(maxFrame) + 1;
	}

	bool hasArbitraryLORs() const override
	{
		return true;
	}

	yrt::Line3D getArbitraryLOR(yrt::bin_t id) const override
	{
		return m_lines[static_cast<std::size_t>(id)];
	}

	std::set<yrt::ProjectionPropertyType>
	    getProjectionPropertyTypes() const override
	{
		auto projectionProperties = ProjectionData::getProjectionPropertyTypes();
		if (hasDynamicFraming())
		{
			projectionProperties.insert(
			    yrt::ProjectionPropertyType::DYNAMIC_FRAME);
		}
		return projectionProperties;
	}

	const std::vector<float>& values() const
	{
		return m_values;
	}

private:
	std::vector<yrt::Line3D> m_lines;
	std::vector<float> m_values;
	std::vector<yrt::frame_t> m_dynamicFrames;
	std::vector<yrt::det_pair_t> m_detectorPairs;
};

struct ScopedPathCleanup
{
	std::filesystem::path path;
	~ScopedPathCleanup()
	{
		std::error_code ec;
		std::filesystem::remove_all(path, ec);
	}
};

bool writeTextFile(const std::filesystem::path& path, const std::string& text)
{
	std::ofstream file(path);
	if (!file)
	{
		return false;
	}
	file << text;
	return true;
}

std::string uniformPsfCsv()
{
	return "0.25,-0.5,1.25\n"
	       "-0.1,0.8,0.3\n"
	       "0.6,-0.2,0.6\n"
	       "3,3,3\n";
}

std::string projectionPsfCsv()
{
	return "1.0,1.0,3,0.25,0.5,0.25\n";
}

std::string projectorLorCsv()
{
	return "-2.0,0.0,0.0,2.0,0.0,0.0,1.0,0\n"
	       "0.0,-2.0,0.0,0.0,2.0,0.0,-0.25,1\n"
	       "0.0,0.0,-2.0,0.0,0.0,2.0,2.0,-1\n"
	       "-2.0,-2.0,0.0,2.0,2.0,0.0,0.5,1\n"
	       "2.0,2.0,0.0,3.0,2.0,0.0,4.0,0\n";
}

struct ProjectorLorCsvData
{
	std::vector<yrt::Line3D> lines;
	std::vector<float> values;
	std::vector<yrt::frame_t> frames;
};

ProjectorLorCsvData readProjectorLorCsvForTest(
    const std::filesystem::path& path)
{
	yrt::Array2DOwned<float> csv;
	yrt::util::readCSV<float>(path.string(), csv);
	if (csv.getSize(1) != 8)
	{
		throw std::runtime_error("Projector LOR test CSV must have 8 columns");
	}

	ProjectorLorCsvData data;
	data.lines.reserve(csv.getSize(0));
	data.values.reserve(csv.getSize(0));
	data.frames.reserve(csv.getSize(0));
	for (std::size_t row = 0; row < csv.getSize(0); ++row)
	{
		data.lines.push_back(makeBridgeLine(csv[row][0], csv[row][1],
		                                    csv[row][2], csv[row][3],
		                                    csv[row][4], csv[row][5]));
		data.values.push_back(csv[row][6]);
		data.frames.push_back(static_cast<yrt::frame_t>(csv[row][7]));
	}
	return data;
}

bool expectThrows(const std::function<void()>& action)
{
	try
	{
		action();
	}
	catch (const std::exception&)
	{
		return true;
	}
	return false;
}

bool runBackendContextTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid() || !context.errorMessage().empty())
	{
		return false;
	}

	const std::size_t count = 8;
	yrt::backend::metal::Buffer privateBuffer =
	    yrt::backend::metal::Buffer::allocatePrivate(
	        context.device(), sizeof(float) * count);
	std::vector<float> actual(count, 0.0f);
	std::vector<float> expected(count, 3.5f);
	return privateBuffer.isValid() && !privateBuffer.isHostVisible() &&
	       yrt::backend::metal::launchProjectionClear(context.device(),
	           context.library(), context.commandQueue(), privateBuffer,
	           expected.front(), count) &&
	       !privateBuffer.copyToHost(actual.data(), sizeof(float) * count) &&
	       privateBuffer.copyToHost(context.commandQueue(), actual.data(),
	           sizeof(float) * count) &&
	       valuesMatch(actual, expected);
}

bool runProjectionVectorOpsHostApiGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const std::vector<float> lhs = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, -8.0f, 16.0f,
	    3.25f, -0.125f, 7.5f, 11.0f, -13.0f, 19.0f, 23.0f, -29.0f};
	const std::vector<float> rhs = {
	    0.0f, -4.0f, 2.0f, 0.25f, -0.5f, 5.0f, -10.0f, 8.0f, 1.5f,
	    -3.0f, 6.0f, -7.5f, 0.0f, 13.0f, -19.0f, 0.75f, 29.0f};

	std::vector<float> actual = lhs;
	std::vector<float> expected(lhs.size());
	const float clearValue = -7.25f;
	if (!yrt::backend::metal::clear(context, actual, clearValue))
	{
		return false;
	}
	std::fill(expected.begin(), expected.end(), clearValue);
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	actual = rhs;
	if (!yrt::backend::metal::add(context, lhs, actual))
	{
		return false;
	}
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		expected[i] = rhs[i] + lhs[i];
	}
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	actual = rhs;
	const float scalar = -1.75f;
	if (!yrt::backend::metal::multiplyByScalar(context, actual, scalar))
	{
		return false;
	}
	for (std::size_t i = 0; i < rhs.size(); ++i)
	{
		expected[i] = rhs[i] * scalar;
	}
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	actual = rhs;
	if (!yrt::backend::metal::multiplyElementwise(context, lhs, actual))
	{
		return false;
	}
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		expected[i] = rhs[i] * lhs[i];
	}
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	actual = rhs;
	if (!yrt::backend::metal::divideMeasurements(context, lhs, actual))
	{
		return false;
	}
	expected = rhs;
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if (expected[i] != 0.0f)
		{
			expected[i] = lhs[i] / expected[i];
		}
	}
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	actual.assign(lhs.size(), -999.0f);
	if (!yrt::backend::metal::invert(context, lhs, actual))
	{
		return false;
	}
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		expected[i] = lhs[i] != 0.0f ? 1.0f / lhs[i] : 0.0f;
	}
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	actual.assign(lhs.size(), 0.0f);
	const float unitFactor = 0.1f;
	if (!yrt::backend::metal::convertToACF(context, lhs, actual, unitFactor))
	{
		return false;
	}
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		expected[i] = std::exp(-lhs[i] * unitFactor);
	}
	if (!valuesMatch(actual, expected))
	{
		return false;
	}

	return true;
}

bool runProjectionVectorMetalGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const std::vector<float> input = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, -8.0f, 16.0f};
	const std::vector<float> output = {
	    0.0f, -4.0f, 2.0f, 0.25f, -0.5f, 5.0f, -10.0f, 8.0f, 1.5f};

	yrt::backend::metal::ProjectionVectorMetal vector(context, output);
	if (!vector.isValid() || vector.size() != output.size() || vector.empty())
	{
		return false;
	}

	std::vector<float> expected(output.size(), 3.5f);
	if (!vector.clear(3.5f) || !valuesMatch(vector.values(), expected))
	{
		return false;
	}

	vector.values() = output;
	if (!vector.add(input))
	{
		return false;
	}
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		expected[i] = output[i] + input[i];
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	if (!vector.multiplyByScalar(-0.5f))
	{
		return false;
	}
	for (float& value : expected)
	{
		value *= -0.5f;
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	yrt::backend::metal::ProjectionVectorMetal multiplier(context, input);
	if (!vector.multiplyElementwise(multiplier))
	{
		return false;
	}
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		expected[i] *= input[i];
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	vector.values() = output;
	if (!vector.divideMeasurements(input))
	{
		return false;
	}
	expected = output;
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		if (expected[i] != 0.0f)
		{
			expected[i] = input[i] / expected[i];
		}
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	vector.values() = input;
	if (!vector.invert())
	{
		return false;
	}
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		expected[i] = input[i] != 0.0f ? 1.0f / input[i] : 0.0f;
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	vector.values() = input;
	const float unitFactor = 0.25f;
	if (!vector.convertToACF(unitFactor))
	{
		return false;
	}
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		expected[i] = std::exp(-input[i] * unitFactor);
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	const std::vector<float> shortInput = {1.0f, 2.0f};
	const std::vector<float> beforeMismatch = vector.values();
	if (vector.add(shortInput) || !valuesMatch(vector.values(), beforeMismatch))
	{
		return false;
	}

	return true;
}

bool runProjectionGeometryGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::backend::metal::ProjectionImageBounds bounds{
	    6.0f, 4.0f, 8.0f, 2.0f};
	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-4.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f},
	    {0.0f, 0.0f, -5.0f, 0.0f, 0.0f, 5.0f},
	    {0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f},
	    {0.0f, 2.0f, 0.0f, 0.0f, -2.0f, 0.0f},
	    {3.0f, 3.0f, 0.0f, 5.0f, 3.0f, 0.0f},
	    {0.0f, 3.0f, 0.0f, 0.0f, 3.0f, 2.0f}};

	std::vector<yrt::backend::metal::ProjectionAlphaRange> expected;
	expected.reserve(lines.size());
	for (const auto& line : lines)
	{
		expected.push_back(cpuSiddonEntryRange(line, bounds));
	}

	std::vector<yrt::backend::metal::ProjectionAlphaRange> actual;
	if (!yrt::backend::metal::computeSiddonEntryRanges(
	        context, lines, bounds, actual) ||
	    !rangesMatch(actual, expected))
	{
		return false;
	}

	const yrt::ImageParams params(6, 4, 8, 6.0f, 4.0f, 8.0f);
	const auto imageBounds =
	    yrt::backend::metal::makeProjectionImageBounds(params);
	if (!almostEqual(imageBounds.lengthX, params.length_x) ||
	    !almostEqual(imageBounds.lengthY, params.length_y) ||
	    !almostEqual(imageBounds.lengthZ, params.length_z) ||
	    !almostEqual(imageBounds.fovRadius, params.fovRadius))
	{
		return false;
	}

	const yrt::backend::metal::ProjectionImageBounds invalidBounds{
	    0.0f, 4.0f, 8.0f, 2.0f};
	return !yrt::backend::metal::computeSiddonEntryRanges(
	    context, lines, invalidBounds, actual);
}

bool runProjectionBatchMetalGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::backend::metal::ProjectionImageBounds bounds{
	    6.0f, 4.0f, 8.0f, 2.0f};
	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-4.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f},
	    {0.0f, 0.0f, -5.0f, 0.0f, 0.0f, 5.0f},
	    {0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f},
	    {0.0f, 2.0f, 0.0f, 0.0f, -2.0f, 0.0f}};
	const std::vector<float> initialValues = {1.0f, -2.0f, 3.5f, 0.25f};
	yrt::backend::metal::ProjectionBatchMetal batch(context, lines,
	                                                initialValues);
	if (!batch.isValid() || batch.errorMessage() != "" ||
	    batch.size() != lines.size() || batch.empty() ||
	    batch.shape().eventCount != lines.size() ||
	    batch.shape().hasDetectorOrientations || batch.shape().hasTof ||
	    batch.shape().hasDynamicFrames ||
	    !lineEndpointVectorsMatch(batch.lines(), lines) ||
	    !batch.lorBuffer().isValid() ||
	    !batch.projectionValuesBuffer().isValid())
	{
		return false;
	}

	std::vector<float> valuesFromMetal;
	if (!batch.copyProjectionValuesToHost(valuesFromMetal) ||
	    !valuesMatch(valuesFromMetal, initialValues))
	{
		return false;
	}

	const std::vector<float> updatedValues = {-8.0f, 0.0f, 2.25f, 7.5f};
	if (!batch.setProjectionValues(updatedValues) ||
	    !batch.copyProjectionValuesToHost(valuesFromMetal) ||
	    !valuesMatch(valuesFromMetal, updatedValues))
	{
		return false;
	}

	const std::vector<float> mismatchedValues = {1.0f, 2.0f};
	if (batch.setProjectionValues(mismatchedValues) ||
	    !batch.copyProjectionValuesToHost(valuesFromMetal) ||
	    !valuesMatch(valuesFromMetal, updatedValues))
	{
		return false;
	}

	std::vector<yrt::backend::metal::ProjectionAlphaRange> expectedRanges;
	expectedRanges.reserve(lines.size());
	for (const auto& line : lines)
	{
		expectedRanges.push_back(cpuSiddonEntryRange(line, bounds));
	}

	std::vector<yrt::backend::metal::ProjectionAlphaRange> actualRanges;
	if (!batch.computeSiddonEntryRanges(bounds, actualRanges) ||
	    !rangesMatch(actualRanges, expectedRanges))
	{
		return false;
	}

	yrt::backend::metal::ProjectionBatchMetal emptyBatch(
	    context,
	    std::vector<yrt::backend::metal::ProjectionLineEndpoints>{});
	yrt::backend::metal::ProjectionBatchMetal mismatchedBatch(
	    context, lines, {1.0f, 2.0f});
	return !emptyBatch.isValid() && !emptyBatch.errorMessage().empty() &&
	       !mismatchedBatch.isValid() &&
	       !mismatchedBatch.errorMessage().empty();
}

bool runSiddonSingleRayForwardGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	yrt::ImageParams params(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(params); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.2f +
		    static_cast<float>(i / 7) * 0.15f;
	}

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
	    {0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f},
	    {0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f},
	    {-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f},
	    {2.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f},
	    {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f}};
	yrt::backend::metal::ProjectionBatchMetal batch(
	    context, lines, std::vector<float>(lines.size(), -99.0f));
	if (!batch.isValid())
	{
		return false;
	}

	for (std::uint32_t frame = 0; frame < 2; ++frame)
	{
		const std::vector<float> sentinel(lines.size(), -99.0f);
		if (!batch.setProjectionValues(sentinel) ||
		    !yrt::backend::metal::forwardProjectSiddonSingleRay(
		        context, image, batch, frame))
		{
			return false;
		}

		std::vector<float> actual;
		if (!batch.copyProjectionValuesToHost(actual))
		{
			return false;
		}

		std::vector<float> expected;
		expected.reserve(lines.size());
		for (const auto& line : lines)
		{
			expected.push_back(yrt::ProjectorSiddon::singleForwardProjection(
			    &image, makeLine(line), static_cast<yrt::frame_t>(frame)));
		}
		if (!valuesMatch(actual, expected))
		{
			return false;
		}
	}

	const std::vector<float> beforeInvalidFrame = {1.0f, 2.0f, 3.0f, 4.0f,
	                                              5.0f, 6.0f};
	std::vector<float> afterInvalidFrame;
	return batch.setProjectionValues(beforeInvalidFrame) &&
	       !yrt::backend::metal::forwardProjectSiddonSingleRay(
	           context, image, batch, 2) &&
	       batch.copyProjectionValuesToHost(afterInvalidFrame) &&
	       valuesMatch(afterInvalidFrame, beforeInvalidFrame);
}

bool runSiddonSingleRayAdjointGoldenTestImpl()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	yrt::ImageParams params(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	const std::size_t count = imageVoxelCount(params);
	std::vector<float> seed(count);
	for (std::size_t i = 0; i < count; ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.1f;
	}

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
	    {-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
	    {0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f},
	    {0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f},
	    {3.0f, 3.0f, 0.0f, 5.0f, 3.0f, 0.0f},
	    {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f}};
	const std::vector<float> projectionValues = {1.0f, -0.25f, 2.0f, 0.0f,
	                                             4.0f, 0.5f};
	yrt::backend::metal::ProjectionBatchMetal batch(context, lines,
	                                                projectionValues);
	if (!batch.isValid())
	{
		return false;
	}

	for (std::uint32_t frame = 0; frame < 2; ++frame)
	{
		yrt::ImageOwned cpuImage(params);
		yrt::ImageOwned metalImage(params);
		cpuImage.allocate();
		metalImage.allocate();
		copyValuesToImage(cpuImage, seed);
		copyValuesToImage(metalImage, seed);

		for (std::size_t i = 0; i < lines.size(); ++i)
		{
			yrt::ProjectorSiddon::singleBackProjection(&cpuImage,
			    makeLine(lines[i]), projectionValues[i],
			    static_cast<yrt::frame_t>(frame));
		}

		if (!yrt::backend::metal::backProjectSiddonSingleRay(
		        context, batch, metalImage, frame) ||
		    !imagesMatch(metalImage, cpuImage))
		{
			return false;
		}
	}

	yrt::ImageOwned diagCpuImage(params);
	yrt::ImageOwned diagMetalImage(params);
	diagCpuImage.allocate();
	diagMetalImage.allocate();
	copyValuesToImage(diagCpuImage, seed);
	copyValuesToImage(diagMetalImage, seed);
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		yrt::ProjectorSiddon::singleBackProjection(&diagCpuImage,
		    makeLine(lines[i]), projectionValues[i], 0);
	}
	yrt::backend::metal::SiddonProjectorKernelProfile profile;
	profile.diagnoseAdjointUpdateCounts = true;
	profile.diagnoseAdjointVoxelHits = true;
	if (!yrt::backend::metal::backProjectSiddonSingleRay(
	        context, batch, diagMetalImage, 0, &profile) ||
	    !imagesMatch(diagMetalImage, diagCpuImage) ||
	    profile.adjointUpdateCountSeconds <= 0.0 ||
	    profile.adjointVoxelHitCountSeconds <= 0.0 ||
	    profile.adjointVoxelUpdates == 0 ||
	    profile.adjointRaysWithUpdates == 0 ||
	    profile.adjointMaxUpdatesPerRay == 0 ||
	    profile.adjointVoxelHitMaps == 0 ||
	    profile.adjointBatchHitVoxels == 0 ||
	    profile.adjointVoxelHitTotalUpdates != profile.adjointVoxelUpdates ||
	    profile.adjointMaxVoxelHits == 0 ||
	    profile.adjointMaxBatchP95VoxelHits == 0 ||
	    profile.adjointMaxBatchP99VoxelHits == 0)
	{
		return false;
	}

	yrt::ImageOwned invalidFrameImage(params);
	yrt::ImageOwned expectedInvalidFrame(params);
	invalidFrameImage.allocate();
	expectedInvalidFrame.allocate();
	copyValuesToImage(invalidFrameImage, seed);
	copyValuesToImage(expectedInvalidFrame, seed);
	return !yrt::backend::metal::backProjectSiddonSingleRay(
	           context, batch, invalidFrameImage, 2) &&
	       imagesMatch(invalidFrameImage, expectedInvalidFrame);
}

bool runSiddonSingleRayAdjointGoldenTest()
{
	return runSiddonSingleRayAdjointGoldenTestImpl();
}

bool runSiddonSingleRayNativeAtomicFloatAdjointGoldenTest()
{
	const ScopedEnvVar nativeFloatAtomics(
	    "YRTPET_METAL_USE_NATIVE_FLOAT_ATOMICS", "1");
	return runSiddonSingleRayAdjointGoldenTestImpl();
}

bool runSiddonProjectorMetalAdapterGoldenTest()
{
	const yrt::backend::metal::Context context;
	yrt::backend::metal::SiddonProjectorMetal projector(context);
	if (!projector.isValid() || !projector.context().isValid())
	{
		return false;
	}

	yrt::ImageParams params(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(params); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.125f +
		    static_cast<float>(i / 5) * 0.03125f;
	}

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
	    {0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f},
	    {0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f},
	    {-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f},
	    {2.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f},
	    {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f}};
	const std::vector<float> projectionWeights = {1.25f, -0.75f, 0.5f,
	                                              2.0f, 3.0f, -1.5f};
	const std::uint32_t frame = 1;

	auto forwardBatch =
	    projector.makeBatch(lines, std::vector<float>(lines.size(), 0.0f));
	if (!forwardBatch.isValid() ||
	    !projector.forwardProjectSingleRay(image, forwardBatch, frame))
	{
		return false;
	}

	std::vector<float> actualForward;
	if (!forwardBatch.copyProjectionValuesToHost(actualForward))
	{
		return false;
	}

	std::vector<float> expectedForward;
	expectedForward.reserve(lines.size());
	for (const auto& line : lines)
	{
		expectedForward.push_back(yrt::ProjectorSiddon::singleForwardProjection(
		    &image, makeLine(line), static_cast<yrt::frame_t>(frame)));
	}
	if (!valuesMatch(actualForward, expectedForward))
	{
		return false;
	}

	auto adjointBatch = projector.makeBatch(lines, projectionWeights);
	if (!adjointBatch.isValid())
	{
		return false;
	}

	yrt::ImageOwned metalAdjoint(params);
	yrt::ImageOwned cpuAdjoint(params);
	metalAdjoint.allocate();
	cpuAdjoint.allocate();
	metalAdjoint.fill(0.0f);
	cpuAdjoint.fill(0.0f);
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		yrt::ProjectorSiddon::singleBackProjection(&cpuAdjoint,
		    makeLine(lines[i]), projectionWeights[i],
		    static_cast<yrt::frame_t>(frame));
	}

	if (!projector.backProjectSingleRay(adjointBatch, metalAdjoint, frame) ||
	    !imagesMatch(metalAdjoint, cpuAdjoint))
	{
		return false;
	}

	const double projectionDot = dotValues(actualForward, projectionWeights);
	const double imageDot = imageDotFrame(image, metalAdjoint, frame);
	return dotAlmostEqual(projectionDot, imageDot);
}

bool runJosephProjectorMetalAdapterGoldenTest()
{
	const yrt::backend::metal::Context context;
	yrt::backend::metal::JosephProjectorMetal projector(context);
	if (!projector.isValid() || !projector.context().isValid())
	{
		return false;
	}

	yrt::ImageParams params(5, 4, 4, 5.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(params); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.0625f +
		    static_cast<float>(i / 7) * 0.015625f;
	}

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-3.0f, -1.25f, -1.0f, 3.0f, 1.25f, 1.0f},
	    {-2.0f, 0.25f, -1.5f, 2.0f, 0.25f, 1.5f},
	    {0.5f, -3.0f, -1.0f, -0.5f, 3.0f, 1.0f},
	    {-2.5f, -2.0f, 0.0f, 2.5f, 2.0f, 0.0f},
	    {4.0f, 4.0f, 0.0f, 5.0f, 4.0f, 0.0f}};
	const std::vector<float> projectionWeights = {1.25f, -0.5f, 0.75f,
	                                              2.0f, 3.0f};
	const std::uint32_t frame = 1;

	auto forwardBatch =
	    projector.makeBatch(lines, std::vector<float>(lines.size(), 0.0f));
	if (!forwardBatch.isValid() ||
	    !projector.forwardProjectSingleRay(image, forwardBatch, frame))
	{
		return false;
	}

	std::vector<float> actualForward;
	if (!forwardBatch.copyProjectionValuesToHost(actualForward))
	{
		return false;
	}
	std::vector<float> expectedForward;
	expectedForward.reserve(lines.size());
	for (const auto& line : lines)
	{
		expectedForward.push_back(josephForwardReference(image, line, frame));
	}
	if (!valuesMatch(actualForward, expectedForward))
	{
		return false;
	}

	yrt::ImageOwned metalAdjoint(params);
	yrt::ImageOwned expectedAdjoint(params);
	metalAdjoint.allocate();
	expectedAdjoint.allocate();
	metalAdjoint.fill(0.0f);
	expectedAdjoint.fill(0.0f);
	auto adjointBatch = projector.makeBatch(lines, projectionWeights);
	if (!adjointBatch.isValid() ||
	    !projector.backProjectSingleRay(adjointBatch, metalAdjoint, frame))
	{
		return false;
	}
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		josephBackProjectReference(expectedAdjoint, lines[i],
		                           projectionWeights[i], frame);
	}
	if (!imagesMatch(metalAdjoint, expectedAdjoint))
	{
		return false;
	}

	return dotAlmostEqual(dotValues(expectedForward, projectionWeights),
	    imageDotFrame(image, metalAdjoint, frame));
}

bool runJosephProjectorMetalTextureForwardGoldenTest()
{
	const yrt::backend::metal::Context context;
	yrt::backend::metal::JosephProjectorMetal projector(context);
	if (!projector.isValid())
	{
		return false;
	}

	yrt::ImageParams params(5, 4, 4, 5.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(params); ++i)
	{
		image.getRawPointer()[i] =
		    0.125f + static_cast<float>((static_cast<int>(i) % 19) - 9) *
		                 0.03125f +
		    static_cast<float>(i / 11) * 0.0078125f;
	}

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-3.0f, -1.25f, -1.0f, 3.0f, 1.25f, 1.0f},
	    {-2.0f, 0.25f, -1.5f, 2.0f, 0.25f, 1.5f},
	    {0.5f, -3.0f, -1.0f, -0.5f, 3.0f, 1.0f},
	    {-2.5f, -2.0f, 0.0f, 2.5f, 2.0f, 0.0f},
	    {4.0f, 4.0f, 0.0f, 5.0f, 4.0f, 0.0f}};
	const std::uint32_t frame = 1;

	auto textureBatch =
	    projector.makeBatch(lines, std::vector<float>(lines.size(), 0.0f));
	auto bufferBatch =
	    projector.makeBatch(lines, std::vector<float>(lines.size(), 0.0f));
	if (!textureBatch.isValid() || !bufferBatch.isValid())
	{
		return false;
	}

	yrt::backend::metal::SiddonForwardImageParams imageParams{};
	yrt::backend::metal::Texture3D texture;
	yrt::backend::metal::Sampler sampler;
	if (!yrt::backend::metal::makeSiddonForwardImageParams(
	        image, frame, imageParams) ||
	    !yrt::backend::metal::uploadJosephImageFrameTexture(
	        context, image, frame, texture, sampler, nullptr) ||
	    !yrt::backend::metal::forwardProjectJosephSingleRayTexture(
	        context, texture, sampler, textureBatch, imageParams, nullptr) ||
	    !projector.forwardProjectSingleRay(image, bufferBatch, frame))
	{
		return false;
	}

	std::vector<float> textureForward;
	std::vector<float> bufferForward;
	if (!textureBatch.copyProjectionValuesToHost(textureForward) ||
	    !bufferBatch.copyProjectionValuesToHost(bufferForward))
	{
		return false;
	}

	std::vector<float> expectedForward;
	expectedForward.reserve(lines.size());
	for (const auto& line : lines)
	{
		expectedForward.push_back(josephForwardReference(image, line, frame));
	}

	return valuesMatch(textureForward, expectedForward) &&
	       valuesMatch(textureForward, bufferForward);
}

bool runSiddonEmptyOrMissGoldenTest()
{
	const yrt::backend::metal::Context context;
	yrt::backend::metal::SiddonProjectorMetal projector(context);
	if (!projector.isValid())
	{
		return false;
	}

	yrt::ImageParams params(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	image.allocate();
	const std::size_t count = imageVoxelCount(params);
	std::vector<float> seed(count);
	for (std::size_t i = 0; i < count; ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.25f;
	}
	copyValuesToImage(image, seed);

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> missLines = {
	    {0.0f, 3.0f, 0.0f, 1.0f, 3.0f, 0.0f},
	    {-1.0f, 0.0f, 3.0f, 1.0f, 0.0f, 3.0f},
	    {2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f}};
	auto forwardBatch =
	    projector.makeBatch(missLines, std::vector<float>(missLines.size(), -7.0f));
	if (!forwardBatch.isValid() ||
	    !projector.forwardProjectSingleRay(image, forwardBatch, 0))
	{
		return false;
	}

	std::vector<float> actualForward;
	if (!forwardBatch.copyProjectionValuesToHost(actualForward) ||
	    !valuesMatch(actualForward, std::vector<float>(missLines.size(), 0.0f)))
	{
		return false;
	}

	yrt::ImageOwned adjointImage(params);
	yrt::ImageOwned expectedImage(params);
	adjointImage.allocate();
	expectedImage.allocate();
	copyValuesToImage(adjointImage, seed);
	copyValuesToImage(expectedImage, seed);

	auto adjointBatch = projector.makeBatch(missLines, {2.0f, -3.0f, 5.0f});
	return adjointBatch.isValid() &&
	       projector.backProjectSingleRay(adjointBatch, adjointImage, 1) &&
	       imagesMatch(adjointImage, expectedImage);
}

bool runSiddonProjectorMetalFailureModeGoldenTest()
{
	const yrt::backend::metal::Context context;
	yrt::backend::metal::SiddonProjectorMetal projector(context);
	if (!projector.isValid())
	{
		return false;
	}

	yrt::ImageParams params(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	yrt::ImageOwned expectedImage(params);
	image.allocate();
	expectedImage.allocate();
	const std::size_t count = imageVoxelCount(params);
	std::vector<float> seed(count);
	for (std::size_t i = 0; i < count; ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) * 0.5f;
	}
	copyValuesToImage(image, seed);
	copyValuesToImage(expectedImage, seed);

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
	    {0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f}};
	auto batch = projector.makeBatch(lines, {7.0f, -4.0f});
	if (!batch.isValid())
	{
		return false;
	}

	std::vector<float> valuesAfterInvalidFrame;
	if (projector.forwardProjectSingleRay(image, batch, 2) ||
	    !batch.copyProjectionValuesToHost(valuesAfterInvalidFrame) ||
	    !valuesMatch(valuesAfterInvalidFrame, {7.0f, -4.0f}) ||
	    projector.backProjectSingleRay(batch, image, 2) ||
	    !imagesMatch(image, expectedImage))
	{
		return false;
	}

	auto emptyBatch =
	    projector.makeBatch(std::vector<yrt::backend::metal::ProjectionLineEndpoints>{});
	auto mismatchedBatch = projector.makeBatch(lines, {1.0f});
	if (emptyBatch.isValid() || emptyBatch.errorMessage().empty() ||
	    mismatchedBatch.isValid() || mismatchedBatch.errorMessage().empty() ||
	    projector.forwardProjectSingleRay(image, emptyBatch, 0) ||
	    projector.forwardProjectSingleRay(image, mismatchedBatch, 0) ||
	    projector.backProjectSingleRay(emptyBatch, image, 0) ||
	    projector.backProjectSingleRay(mismatchedBatch, image, 0))
	{
		return false;
	}
	return imagesMatch(image, expectedImage);
}

bool runSiddonProjectorMetalFrameIsolationGoldenTest()
{
	const yrt::backend::metal::Context context;
	yrt::backend::metal::SiddonProjectorMetal projector(context);
	if (!projector.isValid())
	{
		return false;
	}

	yrt::ImageParams params(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned image(params);
	image.allocate();
	const std::size_t spatialCount = static_cast<std::size_t>(params.nx) *
	                                 static_cast<std::size_t>(params.ny) *
	                                 static_cast<std::size_t>(params.nz);
	for (std::size_t i = 0; i < spatialCount; ++i)
	{
		image.getRawPointer()[i] = static_cast<float>(i + 1) * 0.1f;
		image.getRawPointer()[spatialCount + i] =
		    -1.0f + static_cast<float>(i) * 0.07f;
	}

	const std::vector<yrt::backend::metal::ProjectionLineEndpoints> lines = {
	    {-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
	    {0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f},
	    {-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f}};

	std::vector<float> frame0Values;
	std::vector<float> frame1Values;
	auto frame0Batch =
	    projector.makeBatch(lines, std::vector<float>(lines.size(), 0.0f));
	auto frame1Batch =
	    projector.makeBatch(lines, std::vector<float>(lines.size(), 0.0f));
	if (!frame0Batch.isValid() || !frame1Batch.isValid() ||
	    !projector.forwardProjectSingleRay(image, frame0Batch, 0) ||
	    !projector.forwardProjectSingleRay(image, frame1Batch, 1) ||
	    !frame0Batch.copyProjectionValuesToHost(frame0Values) ||
	    !frame1Batch.copyProjectionValuesToHost(frame1Values))
	{
		return false;
	}

	std::vector<float> expectedFrame0;
	std::vector<float> expectedFrame1;
	for (const auto& line : lines)
	{
		expectedFrame0.push_back(yrt::ProjectorSiddon::singleForwardProjection(
		    &image, makeLine(line), 0));
		expectedFrame1.push_back(yrt::ProjectorSiddon::singleForwardProjection(
		    &image, makeLine(line), 1));
	}
	if (!valuesMatch(frame0Values, expectedFrame0) ||
	    !valuesMatch(frame1Values, expectedFrame1))
	{
		return false;
	}

	const std::vector<float> projectionValues = {1.0f, -2.0f, 0.5f};
	auto adjointBatch = projector.makeBatch(lines, projectionValues);
	yrt::ImageOwned metalAdjoint(params);
	yrt::ImageOwned cpuAdjoint(params);
	metalAdjoint.allocate();
	cpuAdjoint.allocate();
	copyValuesToImage(metalAdjoint,
	                  std::vector<float>(image.getRawPointer(),
	                                     image.getRawPointer() +
	                                         imageVoxelCount(params)));
	cpuAdjoint.copyFromImage(&metalAdjoint);
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		yrt::ProjectorSiddon::singleBackProjection(&cpuAdjoint,
		    makeLine(lines[i]), projectionValues[i], 1);
	}
	return adjointBatch.isValid() &&
	       projector.backProjectSingleRay(adjointBatch, metalAdjoint, 1) &&
	       imagesMatch(metalAdjoint, cpuAdjoint);
}

bool runOperatorProjectorMetalBridgeForwardGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.15f +
		    static_cast<float>(i / 6) * 0.05f;
	}

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f),
	    makeBridgeLine(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f),
	    makeBridgeLine(2.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<yrt::frame_t> frames = {0, 1, -1, 0, 1, 1};
	const std::vector<float> initialValues(lines.size(), -99.0f);

	MetalBridgeProjectionData cpuData(scanner, lines, initialValues, frames);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyA(&image, &cpuData);

	MetalBridgeProjectionData metalData(scanner, lines, initialValues, frames);
	auto metalBinIterator = metalData.getBinIter(1, 0);
	yrt::OperatorProjector metalProjector(projectorParams,
	                                      metalBinIterator.get());
	const yrt::backend::metal::OperatorProjectorMetalBridge bridge(context);
	return bridge.canRunSiddon(metalProjector).supported &&
	       bridge.applyA(metalProjector, image, metalData, *metalBinIterator,
	                     *metalProjector.getBinLoader()) &&
	       valuesMatch(metalData.values(), cpuData.values());
}

bool runOperatorProjectorMetalBridgeAdjointGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned cpuImage(imageParams);
	yrt::ImageOwned metalImage(imageParams);
	cpuImage.allocate();
	metalImage.allocate();
	std::vector<float> seed(imageVoxelCount(imageParams));
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.05f;
	}
	copyValuesToImage(cpuImage, seed);
	copyValuesToImage(metalImage, seed);

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f),
	    makeBridgeLine(3.0f, 3.0f, 0.0f, 5.0f, 3.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<float> projectionValues = {1.0f, -0.25f, 2.0f, 0.0f,
	                                             4.0f, 0.5f};
	const std::vector<yrt::frame_t> frames = {0, 1, -1, 0, 1, 1};

	MetalBridgeProjectionData cpuData(scanner, lines, projectionValues, frames);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyAH(&cpuData, &cpuImage);

	MetalBridgeProjectionData metalData(scanner, lines, projectionValues,
	                                    frames);
	auto metalBinIterator = metalData.getBinIter(1, 0);
	yrt::OperatorProjector metalProjector(projectorParams,
	                                      metalBinIterator.get());
	const yrt::backend::metal::OperatorProjectorMetalBridge bridge(context);
	return bridge.canRunSiddon(metalProjector).supported &&
	       bridge.applyAH(metalProjector, metalData, metalImage,
	                      *metalBinIterator,
	                      *metalProjector.getBinLoader()) &&
	       imagesMatch(metalImage, cpuImage);
}

bool runOperatorProjectorMetalBridgePartialCacheGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.075f +
		    static_cast<float>(i / 6) * 0.035f;
	}

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f),
	    makeBridgeLine(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f),
	    makeBridgeLine(2.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<yrt::frame_t> frames = {0, 1, -1, 0, 1, 1};
	const std::vector<float> projectionValues = {1.0f, -0.25f, 2.0f, 0.0f,
	                                             4.0f, 0.5f};

	MetalBridgeProjectionData cpuForwardData(scanner, lines,
	                                         std::vector<float>(lines.size(),
	                                                            -13.0f),
	                                         frames);
	auto cpuForwardBinIterator = cpuForwardData.getBinIter(1, 0);
	yrt::OperatorProjector cpuForwardProjector(
	    projectorParams, cpuForwardBinIterator.get());
	cpuForwardProjector.applyA(&image, &cpuForwardData);

	MetalBridgeProjectionData metalForwardData(scanner, lines,
	                                           std::vector<float>(lines.size(),
	                                                              -13.0f),
	                                           frames);
	auto metalForwardBinIterator = metalForwardData.getBinIter(1, 0);
	yrt::OperatorProjector metalForwardProjector(
	    projectorParams, metalForwardBinIterator.get());
	yrt::backend::metal::OperatorProjectorMetalProfile forwardProfile;
	yrt::backend::metal::OperatorProjectorMetalCache forwardCache;
	forwardCache.setMaxBytes(150);
	forwardCache.setMaxBatchEvents(2);
	const yrt::backend::metal::OperatorProjectorMetalBridge forwardBridge(
	    context, &forwardProfile, &forwardCache);
	if (!forwardBridge.applyA(metalForwardProjector, image, metalForwardData,
	        *metalForwardBinIterator,
	        *metalForwardProjector.getBinLoader()) ||
	    !valuesMatch(metalForwardData.values(), cpuForwardData.values()) ||
	    forwardProfile.cacheBuilds == 0 ||
	    forwardProfile.cacheSkipsOverBudget == 0 ||
	    forwardProfile.uncachedBatches == 0 || forwardCache.usedBytes() == 0)
	{
		return false;
	}

	if (!forwardBridge.applyA(metalForwardProjector, image, metalForwardData,
	        *metalForwardBinIterator,
	        *metalForwardProjector.getBinLoader()) ||
	    !valuesMatch(metalForwardData.values(), cpuForwardData.values()) ||
	    forwardProfile.cacheHits == 0)
	{
		return false;
	}

	yrt::ImageOwned cpuAdjointImage(imageParams);
	yrt::ImageOwned metalAdjointImage(imageParams);
	cpuAdjointImage.allocate();
	metalAdjointImage.allocate();
	std::vector<float> seed(imageVoxelCount(imageParams), 0.0f);
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) * 0.025f;
	}
	copyValuesToImage(cpuAdjointImage, seed);
	copyValuesToImage(metalAdjointImage, seed);

	MetalBridgeProjectionData cpuAdjointData(scanner, lines, projectionValues,
	                                        frames);
	auto cpuAdjointBinIterator = cpuAdjointData.getBinIter(1, 0);
	yrt::OperatorProjector cpuAdjointProjector(
	    projectorParams, cpuAdjointBinIterator.get());
	cpuAdjointProjector.applyAH(&cpuAdjointData, &cpuAdjointImage);

	MetalBridgeProjectionData metalAdjointData(scanner, lines,
	                                          projectionValues, frames);
	auto metalAdjointBinIterator = metalAdjointData.getBinIter(1, 0);
	yrt::OperatorProjector metalAdjointProjector(
	    projectorParams, metalAdjointBinIterator.get());
	yrt::backend::metal::OperatorProjectorMetalProfile adjointProfile;
	yrt::backend::metal::OperatorProjectorMetalCache adjointCache;
	adjointCache.setMaxBytes(150);
	adjointCache.setMaxBatchEvents(2);
	const yrt::backend::metal::OperatorProjectorMetalBridge adjointBridge(
	    context, &adjointProfile, &adjointCache);
	return adjointBridge.applyAH(metalAdjointProjector, metalAdjointData,
	           metalAdjointImage, *metalAdjointBinIterator,
	           *metalAdjointProjector.getBinLoader()) &&
	       imagesMatch(metalAdjointImage, cpuAdjointImage) &&
	       adjointProfile.cacheBuilds > 0 &&
	       adjointProfile.cacheSkipsOverBudget > 0 &&
	       adjointProfile.uncachedBatches > 0 && adjointCache.usedBytes() > 0;
}

bool runOperatorProjectorMetalBridgeUnsupportedGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	const yrt::backend::metal::OperatorProjectorMetalBridge bridge(context);

	yrt::ProjectorParams supportedParams(scanner);
	supportedParams.projectorType = yrt::ProjectorType::SIDDON;
	supportedParams.numRays = 1;
	yrt::OperatorProjector supportedProjector(supportedParams);
	if (!bridge.canRunSiddon(supportedProjector).supported)
	{
		return false;
	}

	yrt::ProjectorParams ddParams(scanner);
	ddParams.projectorType = yrt::ProjectorType::DD;
	yrt::OperatorProjector ddProjector(ddParams);
	const auto ddSupport = bridge.canRunSiddon(ddProjector);
	if (ddSupport.supported || ddSupport.reason.empty())
	{
		return false;
	}

	yrt::ProjectorParams multiRayParams(scanner);
	multiRayParams.projectorType = yrt::ProjectorType::SIDDON;
	multiRayParams.numRays = 2;
	yrt::OperatorProjector multiRayProjector(multiRayParams);
	const auto multiRaySupport = bridge.canRunSiddon(multiRayProjector);
	if (multiRaySupport.supported || multiRaySupport.reason.empty())
	{
		return false;
	}

	yrt::ProjectorParams tofParams(scanner);
	tofParams.projectorType = yrt::ProjectorType::SIDDON;
	tofParams.addTOF(300.0f, 3);
	yrt::OperatorProjector tofProjector(tofParams);
	const auto tofSupport = bridge.canRunSiddon(tofProjector);
	if (tofSupport.supported || tofSupport.reason.empty())
	{
		return false;
	}

	yrt::Array2DOwned<float> basis;
	basis.allocate(2, 2);
	basis[0][0] = 1.0f;
	basis[0][1] = 0.25f;
	basis[1][0] = -0.5f;
	basis[1][1] = 0.75f;

	yrt::ProjectorParams lrParams(scanner);
	lrParams.projectorType = yrt::ProjectorType::SIDDON;
	lrParams.updaterType = yrt::UpdaterType::LR;
	lrParams.bindHBasis(basis.getRawPointer(), 2, 2);
	yrt::OperatorProjector lrProjector(lrParams);
	const auto lrSupport = bridge.canRunSiddon(lrProjector);
	if (lrSupport.supported || lrSupport.reason.empty())
	{
		return false;
	}

	yrt::ProjectorParams lrDualParams(scanner);
	lrDualParams.projectorType = yrt::ProjectorType::SIDDON;
	lrDualParams.updaterType = yrt::UpdaterType::LRDUALUPDATE;
	lrDualParams.bindHBasis(basis.getRawPointer(), 2, 2);
	yrt::OperatorProjector lrDualProjector(lrDualParams);
	const auto lrDualSupport = bridge.canRunSiddon(lrDualProjector);
	return !lrDualSupport.supported && !lrDualSupport.reason.empty();
}

bool runOperatorProjectorMetalDispatchDefaultGoldenTest()
{
	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.2f +
		    static_cast<float>(i / 5) * 0.04f;
	}

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<yrt::frame_t> frames = {0, 1, -1, 1};
	const std::vector<float> initialValues(lines.size(), -11.0f);

	MetalBridgeProjectionData cpuData(scanner, lines, initialValues, frames);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyA(&image, &cpuData);

	MetalBridgeProjectionData defaultData(scanner, lines, initialValues, frames);
	auto defaultBinIterator = defaultData.getBinIter(1, 0);
	yrt::OperatorProjector defaultProjector(projectorParams,
	                                        defaultBinIterator.get());
	if (defaultProjector.isExperimentalMetalProjectorEnabled())
	{
		return false;
	}
	defaultProjector.applyA(&image, &defaultData);
	return valuesMatch(defaultData.values(), cpuData.values());
}

bool runOperatorProjectorMetalDispatchEnabledForwardGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.125f +
		    static_cast<float>(i / 4) * 0.03f;
	}

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f),
	    makeBridgeLine(2.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<yrt::frame_t> frames = {0, 1, 0, -1, 1};
	const std::vector<float> initialValues(lines.size(), -19.0f);

	MetalBridgeProjectionData cpuData(scanner, lines, initialValues, frames);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyA(&image, &cpuData);

	MetalBridgeProjectionData metalData(scanner, lines, initialValues, frames);
	auto metalBinIterator = metalData.getBinIter(1, 0);
	yrt::OperatorProjector metalProjector(projectorParams,
	                                      metalBinIterator.get());
	metalProjector.setExperimentalMetalProjectorEnabled(true);
	if (!metalProjector.isExperimentalMetalProjectorEnabled())
	{
		return false;
	}
	metalProjector.applyA(&image, &metalData);
	return valuesMatch(metalData.values(), cpuData.values());
}

bool runOperatorProjectorMetalDispatchEnabledAdjointGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned cpuImage(imageParams);
	yrt::ImageOwned metalImage(imageParams);
	cpuImage.allocate();
	metalImage.allocate();
	std::vector<float> seed(imageVoxelCount(imageParams));
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) * 0.05f;
	}
	copyValuesToImage(cpuImage, seed);
	copyValuesToImage(metalImage, seed);

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(3.0f, 3.0f, 0.0f, 5.0f, 3.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<float> projectionValues = {1.0f, -0.25f, 2.0f, 4.0f,
	                                             0.5f};
	const std::vector<yrt::frame_t> frames = {0, 1, -1, 1, 1};

	MetalBridgeProjectionData cpuData(scanner, lines, projectionValues, frames);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyAH(&cpuData, &cpuImage);

	MetalBridgeProjectionData metalData(scanner, lines, projectionValues,
	                                    frames);
	auto metalBinIterator = metalData.getBinIter(1, 0);
	yrt::OperatorProjector metalProjector(projectorParams,
	                                      metalBinIterator.get());
	metalProjector.setExperimentalMetalProjectorEnabled(true);
	metalProjector.applyAH(&metalData, &metalImage);
	return imagesMatch(metalImage, cpuImage);
}

bool runOperatorProjectorMetalDispatchJosephGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;

	yrt::ImageParams imageParams(4, 4, 4, 4.0f, 4.0f, 4.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		image.getRawPointer()[i] =
		    0.25f + static_cast<float>((static_cast<int>(i) % 13) - 6) *
		                0.03125f;
	}

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.5f, -0.4f, -0.2f, 2.3f, 0.8f, 0.4f),
	    makeBridgeLine(-0.3f, -2.4f, 0.5f, 0.4f, 2.2f, -0.7f),
	    makeBridgeLine(-0.8f, 0.1f, -2.4f, 0.9f, -0.2f, 2.1f),
	    makeBridgeLine(3.5f, 3.5f, 3.5f, 4.5f, 3.5f, 3.5f)};
	const std::vector<yrt::frame_t> frames = {0, 1, 0, -1};
	const std::vector<float> initialValues(lines.size(), -17.0f);

	auto toEndpoints = [](const yrt::Line3D& line)
	{
		return yrt::backend::metal::ProjectionLineEndpoints{
		    line.point1.x, line.point1.y, line.point1.z,
		    line.point2.x, line.point2.y, line.point2.z};
	};

	std::vector<float> expectedForward = initialValues;
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		if (frames[i] >= 0)
		{
			expectedForward[i] =
			    josephForwardReference(image, toEndpoints(lines[i]),
			                           static_cast<std::uint32_t>(frames[i]));
		}
	}

	MetalBridgeProjectionData forwardData(scanner, lines, initialValues,
	                                      frames);
	auto forwardBinIterator = forwardData.getBinIter(1, 0);
	yrt::OperatorProjector forwardProjector(projectorParams,
	                                        forwardBinIterator.get());
	forwardProjector.setExperimentalMetalProjectorEnabled(true);
	forwardProjector.setExperimentalMetalProjectorKernel("joseph");
	if (forwardProjector.getExperimentalMetalProjectorKernel() != "joseph")
	{
		return false;
	}
	forwardProjector.applyA(&image, &forwardData);
	if (!valuesMatch(forwardData.values(), expectedForward))
	{
		return false;
	}

	MetalBridgeProjectionData textureForwardData(scanner, lines,
	                                             initialValues, frames);
	auto textureForwardBinIterator = textureForwardData.getBinIter(1, 0);
	yrt::OperatorProjector textureForwardProjector(
	    projectorParams, textureForwardBinIterator.get());
	textureForwardProjector.setExperimentalMetalProjectorEnabled(true);
	textureForwardProjector.setExperimentalMetalProjectorKernel(
	    "joseph_texture_forward");
	if (textureForwardProjector.getExperimentalMetalProjectorKernel() !=
	    "joseph_texture_forward")
	{
		return false;
	}
	textureForwardProjector.applyA(&image, &textureForwardData);
	if (!valuesMatch(textureForwardData.values(), expectedForward))
	{
		return false;
	}

	yrt::ImageOwned adjointImage(imageParams);
	yrt::ImageOwned expectedAdjoint(imageParams);
	adjointImage.allocate();
	expectedAdjoint.allocate();
	std::vector<float> seed(imageVoxelCount(imageParams), 0.0f);
	copyValuesToImage(adjointImage, seed);
	copyValuesToImage(expectedAdjoint, seed);

	const std::vector<float> projectionValues = {1.0f, -0.5f, 0.75f, 2.0f};
	for (std::size_t i = 0; i < lines.size(); ++i)
	{
		if (frames[i] >= 0)
		{
			josephBackProjectReference(
			    expectedAdjoint, toEndpoints(lines[i]), projectionValues[i],
			    static_cast<std::uint32_t>(frames[i]));
		}
	}

	MetalBridgeProjectionData adjointData(scanner, lines, projectionValues,
	                                      frames);
	auto adjointBinIterator = adjointData.getBinIter(1, 0);
	yrt::OperatorProjector adjointProjector(projectorParams,
	                                        adjointBinIterator.get());
	adjointProjector.setExperimentalMetalProjectorEnabled(true);
	adjointProjector.setExperimentalMetalProjectorKernel("joseph");
	adjointProjector.applyAH(&adjointData, &adjointImage);
	if (!imagesMatch(adjointImage, expectedAdjoint))
	{
		return false;
	}

	yrt::ImageOwned textureAdjointImage(imageParams);
	textureAdjointImage.allocate();
	copyValuesToImage(textureAdjointImage, seed);
	MetalBridgeProjectionData textureAdjointData(scanner, lines,
	                                             projectionValues, frames);
	auto textureAdjointBinIterator = textureAdjointData.getBinIter(1, 0);
	yrt::OperatorProjector textureAdjointProjector(
	    projectorParams, textureAdjointBinIterator.get());
	textureAdjointProjector.setExperimentalMetalProjectorEnabled(true);
	textureAdjointProjector.setExperimentalMetalProjectorKernel(
	    "joseph_texture_forward");
	textureAdjointProjector.applyAH(&textureAdjointData,
	                                &textureAdjointImage);
	return imagesMatch(textureAdjointImage, expectedAdjoint);
}

bool bridgeReportsUnsupported(const yrt::ProjectorParams& projectorParams)
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	yrt::OperatorProjector projector(projectorParams);
	const yrt::backend::metal::OperatorProjectorMetalBridge bridge(context);
	const auto support = bridge.canRunSiddon(projector);
	return !support.supported && !support.reason.empty();
}

std::vector<yrt::Line3D> makeBridgeFallbackLines()
{
	return {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
}

std::vector<yrt::det_pair_t>
    makeBridgeFallbackDetectorPairs(const yrt::Scanner& scanner)
{
	const yrt::det_id_t opposite =
	    static_cast<yrt::det_id_t>(scanner.detsPerRing / 2);
	return {{0, opposite},
	        {1, static_cast<yrt::det_id_t>(opposite + 1)},
	        {2, static_cast<yrt::det_id_t>(opposite + 2)},
	        {3, static_cast<yrt::det_id_t>(opposite + 3)}};
}

std::vector<yrt::Line3D> makeBridgeFallbackDetectorLines(
    const yrt::Scanner& scanner,
    const std::vector<yrt::det_pair_t>& detectorPairs)
{
	std::vector<yrt::Line3D> lines;
	lines.reserve(detectorPairs.size());
	for (const yrt::det_pair_t& pair : detectorPairs)
	{
		const yrt::Vector3D p1 = scanner.getDetectorPos(pair.d1);
		const yrt::Vector3D p2 = scanner.getDetectorPos(pair.d2);
		lines.push_back(makeBridgeLine(p1.x, p1.y, p1.z, p2.x, p2.y,
		                               p2.z));
	}
	return lines;
}

void fillBridgeDispatchImage(yrt::Image& image)
{
	const std::size_t count = imageVoxelCount(image.getParams());
	for (std::size_t i = 0; i < count; ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.1f +
		    static_cast<float>(i / 5) * 0.025f;
	}
}

bool runOperatorProjectorMetalDispatchForwardFallback(
    const yrt::ProjectorParams& projectorParams, const yrt::Image& image,
    const std::vector<yrt::Line3D>& lines,
    const std::vector<yrt::frame_t>& frames = {},
    const std::vector<yrt::det_pair_t>& detectorPairs = {})
{
	if (!bridgeReportsUnsupported(projectorParams))
	{
		return false;
	}

	const std::vector<float> initialValues(lines.size(), -31.0f);
	MetalBridgeProjectionData cpuData(projectorParams.scanner, lines,
	                                  initialValues, frames, detectorPairs);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyA(&image, &cpuData);

	MetalBridgeProjectionData fallbackData(projectorParams.scanner, lines,
	                                       initialValues, frames,
	                                       detectorPairs);
	auto fallbackBinIterator = fallbackData.getBinIter(1, 0);
	yrt::OperatorProjector fallbackProjector(projectorParams,
	                                         fallbackBinIterator.get());
	fallbackProjector.setExperimentalMetalProjectorEnabled(true);
	fallbackProjector.applyA(&image, &fallbackData);
	return valuesMatch(fallbackData.values(), cpuData.values());
}

bool runOperatorProjectorMetalDispatchAdjointFallback(
    const yrt::ProjectorParams& projectorParams,
    const yrt::ImageParams& imageParams,
    const std::vector<float>& seedValues,
    const std::vector<yrt::Line3D>& lines,
    const std::vector<yrt::frame_t>& frames = {},
    const std::vector<yrt::det_pair_t>& detectorPairs = {})
{
	if (!bridgeReportsUnsupported(projectorParams))
	{
		return false;
	}

	yrt::ImageOwned cpuImage(imageParams);
	yrt::ImageOwned fallbackImage(imageParams);
	cpuImage.allocate();
	fallbackImage.allocate();
	copyValuesToImage(cpuImage, seedValues);
	copyValuesToImage(fallbackImage, seedValues);

	std::vector<float> projectionValues(lines.size());
	for (std::size_t i = 0; i < projectionValues.size(); ++i)
	{
		projectionValues[i] = 0.75f + static_cast<float>(i) * 0.5f;
	}
	MetalBridgeProjectionData cpuData(projectorParams.scanner, lines,
	                                  projectionValues, frames,
	                                  detectorPairs);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyAH(&cpuData, &cpuImage);

	MetalBridgeProjectionData fallbackData(projectorParams.scanner, lines,
	                                       projectionValues, frames,
	                                       detectorPairs);
	auto fallbackBinIterator = fallbackData.getBinIter(1, 0);
	yrt::OperatorProjector fallbackProjector(projectorParams,
	                                         fallbackBinIterator.get());
	fallbackProjector.setExperimentalMetalProjectorEnabled(true);
	fallbackProjector.applyAH(&fallbackData, &fallbackImage);
	return imagesMatch(fallbackImage, cpuImage);
}

bool runOperatorProjectorMetalDispatchUnsupportedFallbackGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;
	projectorParams.addTOF(300.0f, 3);

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		image.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.25f;
	}

	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<float> initialValues(lines.size(), -23.0f);

	MetalBridgeProjectionData cpuData(scanner, lines, initialValues);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(projectorParams,
	                                    cpuBinIterator.get());
	cpuProjector.applyA(&image, &cpuData);

	MetalBridgeProjectionData fallbackData(scanner, lines, initialValues);
	auto fallbackBinIterator = fallbackData.getBinIter(1, 0);
	yrt::OperatorProjector fallbackProjector(projectorParams,
	                                         fallbackBinIterator.get());
	fallbackProjector.setExperimentalMetalProjectorEnabled(true);
	fallbackProjector.applyA(&image, &fallbackData);
	return valuesMatch(fallbackData.values(), cpuData.values());
}

bool runOperatorProjectorMetalDispatchDDFallbackGoldenTest()
{
	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::DD;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	fillBridgeDispatchImage(image);

	const std::vector<yrt::det_pair_t> detectorPairs =
	    makeBridgeFallbackDetectorPairs(scanner);
	const std::vector<yrt::Line3D> lines =
	    makeBridgeFallbackDetectorLines(scanner, detectorPairs);
	if (!runOperatorProjectorMetalDispatchForwardFallback(projectorParams,
	        image, lines, {}, detectorPairs))
	{
		return false;
	}

	std::vector<float> seed(imageVoxelCount(imageParams));
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.04f;
	}
	const std::vector<yrt::Line3D> adjointLines = {lines.front()};
	const std::vector<yrt::det_pair_t> adjointPairs = {detectorPairs.front()};
	if (!runOperatorProjectorMetalDispatchAdjointFallback(
	        projectorParams, imageParams, seed, adjointLines, {}, adjointPairs))
	{
		return false;
	}
	return true;
}

bool runOperatorProjectorMetalDispatchMultiRayFallbackGoldenTest()
{
	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 2;

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	fillBridgeDispatchImage(image);

	const std::vector<yrt::det_pair_t> detectorPairs =
	    makeBridgeFallbackDetectorPairs(scanner);
	const std::vector<yrt::Line3D> lines =
	    makeBridgeFallbackDetectorLines(scanner, detectorPairs);
	if (!runOperatorProjectorMetalDispatchForwardFallback(projectorParams,
	        image, lines, {}, detectorPairs))
	{
		return false;
	}

	std::vector<float> seed(imageVoxelCount(imageParams));
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.03f;
	}
	const std::vector<yrt::Line3D> adjointLines = {lines.front()};
	const std::vector<yrt::det_pair_t> adjointPairs = {detectorPairs.front()};
	return runOperatorProjectorMetalDispatchAdjointFallback(
	    projectorParams, imageParams, seed, adjointLines, {}, adjointPairs);
}

bool runOperatorProjectorMetalDispatchProjectionPsfFallbackGoldenTest()
{
	const std::filesystem::path psfPath =
	    std::filesystem::temp_directory_path() /
	    "yrtpet_metal_projection_psf_fallback.csv";
	ScopedPathCleanup cleanup{psfPath};
	if (!writeTextFile(psfPath, projectionPsfCsv()))
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::DD;
	projectorParams.projPsf_fname = psfPath.string();

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	fillBridgeDispatchImage(image);

	const std::vector<yrt::det_pair_t> detectorPairs =
	    makeBridgeFallbackDetectorPairs(scanner);
	const std::vector<yrt::Line3D> lines =
	    makeBridgeFallbackDetectorLines(scanner, detectorPairs);
	if (!runOperatorProjectorMetalDispatchForwardFallback(projectorParams,
	        image, lines, {}, detectorPairs))
	{
		return false;
	}

	std::vector<float> seed(imageVoxelCount(imageParams));
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) * 0.05f;
	}
	const std::vector<yrt::Line3D> adjointLines = {lines.front()};
	const std::vector<yrt::det_pair_t> adjointPairs = {detectorPairs.front()};
	return runOperatorProjectorMetalDispatchAdjointFallback(
	    projectorParams, imageParams, seed, adjointLines, {}, adjointPairs);
}

bool runOperatorProjectorMetalDispatchLRFallbackGoldenTest()
{
	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::Array2DOwned<float> basis;
	basis.allocate(2, 2);
	basis[0][0] = 1.0f;
	basis[0][1] = 0.25f;
	basis[1][0] = -0.5f;
	basis[1][1] = 0.75f;

	yrt::ProjectorParams lrParams(scanner);
	lrParams.projectorType = yrt::ProjectorType::SIDDON;
	lrParams.numRays = 1;
	lrParams.updaterType = yrt::UpdaterType::LR;
	lrParams.bindHBasis(basis.getRawPointer(), 2, 2);

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned image(imageParams);
	image.allocate();
	fillBridgeDispatchImage(image);

	const std::vector<yrt::Line3D> lines = makeBridgeFallbackLines();
	const std::vector<yrt::frame_t> frames = {0, 1, 0, 1};
	if (!runOperatorProjectorMetalDispatchForwardFallback(lrParams, image,
	        lines, frames))
	{
		return false;
	}

	std::vector<float> seed(imageVoxelCount(imageParams));
	for (std::size_t i = 0; i < seed.size(); ++i)
	{
		seed[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.02f;
	}
	if (!runOperatorProjectorMetalDispatchAdjointFallback(
	        lrParams, imageParams, seed, {lines.front()}, {frames.front()}))
	{
		return false;
	}

	yrt::ProjectorParams lrDualParams(scanner);
	lrDualParams.projectorType = yrt::ProjectorType::SIDDON;
	lrDualParams.numRays = 1;
	lrDualParams.updaterType = yrt::UpdaterType::LRDUALUPDATE;
	lrDualParams.bindHBasis(basis.getRawPointer(), 2, 2);
	return runOperatorProjectorMetalDispatchForwardFallback(lrDualParams,
	    image, lines, frames);
}

void fillOsemImage(yrt::Image& image, float base, float step)
{
	const std::size_t count = imageVoxelCount(image.getParams());
	for (std::size_t i = 0; i < count; ++i)
	{
		image.getRawPointer()[i] =
		    base + step * static_cast<float>((static_cast<int>(i) % 17) + 1);
	}
}

std::unique_ptr<yrt::Image> runOsemCpuReconForMetalTest(
    yrt::ProjectionData& dataInput, yrt::Image& initialEstimate,
    yrt::Image& sensitivityImage, yrt::ProjectorType projectorType,
    bool enableMetalProjector, yrt::OSEM_CPU& osem)
{
	osem.num_MLEM_iterations = 1;
	osem.num_OSEM_subsets = 1;
	osem.setProjector(projectorType);
	osem.setNumRays(1);
	osem.setDataInput(&dataInput);
	osem.setImageParams(initialEstimate.getParams());
	osem.setInitialEstimate(&initialEstimate);
	osem.setSensitivityImage(&sensitivityImage);
	osem.setExperimentalMetalProjectorEnabled(enableMetalProjector);
	return osem.reconstruct("");
}

bool runOsemCpuExperimentalMetalProjectorGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	const std::vector<yrt::Line3D> lines = {
	    makeBridgeLine(-2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f),
	    makeBridgeLine(0.0f, -2.0f, 0.0f, 0.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 2.0f),
	    makeBridgeLine(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, 0.0f),
	    makeBridgeLine(2.0f, 2.0f, 0.0f, 3.0f, 2.0f, 0.0f),
	    makeBridgeLine(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f)};
	const std::vector<float> measurements = {1.0f, 0.75f, 1.25f,
	                                          0.0f, 2.0f, 0.5f};
	const std::vector<yrt::frame_t> frames = {0, 1, -1, 0, 1, 1};

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageParams sensitivityParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned cpuInitial(imageParams);
	yrt::ImageOwned metalInitial(imageParams);
	yrt::ImageOwned cpuSensitivity(sensitivityParams);
	yrt::ImageOwned metalSensitivity(sensitivityParams);
	cpuInitial.allocate();
	metalInitial.allocate();
	cpuSensitivity.allocate();
	metalSensitivity.allocate();
	fillOsemImage(cpuInitial, 0.45f, 0.035f);
	metalInitial.copyFromImage(&cpuInitial);
	fillOsemImage(cpuSensitivity, 1.0f, 0.01f);
	metalSensitivity.copyFromImage(&cpuSensitivity);

	MetalBridgeProjectionData cpuData(scanner, lines, measurements, frames);
	MetalBridgeProjectionData metalData(scanner, lines, measurements, frames);

	yrt::OSEM_CPU cpuOsem(scanner);
	yrt::OSEM_CPU metalOsem(scanner);
	auto cpuOutput = runOsemCpuReconForMetalTest(
	    cpuData, cpuInitial, cpuSensitivity, yrt::ProjectorType::SIDDON, false,
	    cpuOsem);
	auto metalOutput = runOsemCpuReconForMetalTest(
	    metalData, metalInitial, metalSensitivity, yrt::ProjectorType::SIDDON,
	    true, metalOsem);

	return !cpuOsem.didLastExperimentalMetalProjectorRun() &&
	       metalOsem.isExperimentalMetalProjectorEnabled() &&
	       metalOsem.didLastExperimentalMetalProjectorRun() &&
	       imagesMatch(*metalOutput, *cpuOutput);
}

bool runOsemCpuExperimentalMetalProjectorDDFallbackGoldenTest()
{
	const yrt::Scanner scanner = makeBridgeScanner();
	const std::vector<yrt::det_pair_t> detectorPairs =
	    makeBridgeFallbackDetectorPairs(scanner);
	const std::vector<yrt::Line3D> lines =
	    makeBridgeFallbackDetectorLines(scanner, detectorPairs);
	const std::vector<float> measurements = {1.0f, 0.25f, 1.5f, 0.5f};

	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned cpuInitial(imageParams);
	yrt::ImageOwned metalInitial(imageParams);
	yrt::ImageOwned cpuSensitivity(imageParams);
	yrt::ImageOwned metalSensitivity(imageParams);
	cpuInitial.allocate();
	metalInitial.allocate();
	cpuSensitivity.allocate();
	metalSensitivity.allocate();
	fillOsemImage(cpuInitial, 0.35f, 0.025f);
	metalInitial.copyFromImage(&cpuInitial);
	fillOsemImage(cpuSensitivity, 1.0f, 0.02f);
	metalSensitivity.copyFromImage(&cpuSensitivity);

	MetalBridgeProjectionData cpuData(scanner, lines, measurements, {},
	                                  detectorPairs);
	MetalBridgeProjectionData metalData(scanner, lines, measurements, {},
	                                    detectorPairs);

	yrt::OSEM_CPU cpuOsem(scanner);
	yrt::OSEM_CPU metalOsem(scanner);
	auto cpuOutput = runOsemCpuReconForMetalTest(
	    cpuData, cpuInitial, cpuSensitivity, yrt::ProjectorType::DD, false,
	    cpuOsem);
	auto metalOutput = runOsemCpuReconForMetalTest(
	    metalData, metalInitial, metalSensitivity, yrt::ProjectorType::DD, true,
	    metalOsem);

	return !metalOsem.didLastExperimentalMetalProjectorRun() &&
	       imagesMatch(*metalOutput, *cpuOutput);
}

bool runOperatorProjectorMetalFileInputGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	const std::filesystem::path fixtureDir =
	    std::filesystem::temp_directory_path() /
	    "yrtpet_metal_projector_real_input_golden";
	ScopedPathCleanup cleanup{fixtureDir};
	std::error_code ec;
	std::filesystem::remove_all(fixtureDir, ec);
	if (!std::filesystem::create_directories(fixtureDir))
	{
		return false;
	}

	const std::filesystem::path imagePath = fixtureDir / "input.nii";
	const std::filesystem::path lorPath = fixtureDir / "lors.csv";
	yrt::ImageParams imageParams(3, 3, 3, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                             0.0f, 2);
	yrt::ImageOwned seed(imageParams);
	seed.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(imageParams); ++i)
	{
		seed.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.075f +
		    static_cast<float>(i / 6) * 0.025f;
	}
	seed.writeToFile(imagePath.string());
	if (!writeTextFile(lorPath, projectorLorCsv()))
	{
		return false;
	}

	ProjectorLorCsvData lorData;
	try
	{
		lorData = readProjectorLorCsvForTest(lorPath);
	}
	catch (...)
	{
		return false;
	}

	const yrt::Scanner scanner = makeBridgeScanner();
	yrt::ProjectorParams projectorParams(scanner);
	projectorParams.projectorType = yrt::ProjectorType::SIDDON;
	projectorParams.numRays = 1;
	yrt::ImageOwned inputFromFile(imagePath.string());

	MetalBridgeProjectionData supportData(scanner, lorData.lines,
	                                      lorData.values, lorData.frames);
	auto supportBinIterator = supportData.getBinIter(1, 0);
	yrt::OperatorProjector supportProjector(projectorParams,
	                                        supportBinIterator.get());
	const yrt::backend::metal::OperatorProjectorMetalBridge bridge(context);
	if (!bridge.canRunSiddon(supportProjector).supported)
	{
		return false;
	}

	MetalBridgeProjectionData cpuForwardData(scanner, lorData.lines,
	                                         lorData.values, lorData.frames);
	auto cpuForwardBinIterator = cpuForwardData.getBinIter(1, 0);
	yrt::OperatorProjector cpuForwardProjector(projectorParams,
	                                           cpuForwardBinIterator.get());
	cpuForwardProjector.applyA(&inputFromFile, &cpuForwardData);

	MetalBridgeProjectionData bridgeForwardData(scanner, lorData.lines,
	                                            lorData.values, lorData.frames);
	auto bridgeForwardBinIterator = bridgeForwardData.getBinIter(1, 0);
	yrt::OperatorProjector bridgeForwardProjector(
	    projectorParams, bridgeForwardBinIterator.get());
	if (!bridge.applyA(bridgeForwardProjector, inputFromFile, bridgeForwardData,
	        *bridgeForwardBinIterator,
	        *bridgeForwardProjector.getBinLoader()) ||
	    !valuesMatch(bridgeForwardData.values(), cpuForwardData.values()))
	{
		return false;
	}

	MetalBridgeProjectionData optInForwardData(scanner, lorData.lines,
	                                           lorData.values, lorData.frames);
	auto optInForwardBinIterator = optInForwardData.getBinIter(1, 0);
	yrt::OperatorProjector optInForwardProjector(
	    projectorParams, optInForwardBinIterator.get());
	optInForwardProjector.setExperimentalMetalProjectorEnabled(true);
	optInForwardProjector.applyA(&inputFromFile, &optInForwardData);
	if (!valuesMatch(optInForwardData.values(), cpuForwardData.values()))
	{
		return false;
	}

	yrt::ImageOwned cpuAdjoint(imageParams);
	yrt::ImageOwned bridgeAdjoint(imageParams);
	yrt::ImageOwned optInAdjoint(imageParams);
	cpuAdjoint.allocate();
	bridgeAdjoint.allocate();
	optInAdjoint.allocate();
	cpuAdjoint.copyFromImage(&inputFromFile);
	bridgeAdjoint.copyFromImage(&inputFromFile);
	optInAdjoint.copyFromImage(&inputFromFile);

	MetalBridgeProjectionData cpuAdjointData(scanner, lorData.lines,
	                                         lorData.values, lorData.frames);
	auto cpuAdjointBinIterator = cpuAdjointData.getBinIter(1, 0);
	yrt::OperatorProjector cpuAdjointProjector(projectorParams,
	                                           cpuAdjointBinIterator.get());
	cpuAdjointProjector.applyAH(&cpuAdjointData, &cpuAdjoint);

	MetalBridgeProjectionData bridgeAdjointData(scanner, lorData.lines,
	                                            lorData.values, lorData.frames);
	auto bridgeAdjointBinIterator = bridgeAdjointData.getBinIter(1, 0);
	yrt::OperatorProjector bridgeAdjointProjector(
	    projectorParams, bridgeAdjointBinIterator.get());
	if (!bridge.applyAH(bridgeAdjointProjector, bridgeAdjointData,
	        bridgeAdjoint, *bridgeAdjointBinIterator,
	        *bridgeAdjointProjector.getBinLoader()) ||
	    !imagesMatch(bridgeAdjoint, cpuAdjoint))
	{
		return false;
	}

	MetalBridgeProjectionData optInAdjointData(scanner, lorData.lines,
	                                           lorData.values, lorData.frames);
	auto optInAdjointBinIterator = optInAdjointData.getBinIter(1, 0);
	yrt::OperatorProjector optInAdjointProjector(
	    projectorParams, optInAdjointBinIterator.get());
	optInAdjointProjector.setExperimentalMetalProjectorEnabled(true);
	optInAdjointProjector.applyAH(&optInAdjointData, &optInAdjoint);
	return imagesMatch(optInAdjoint, cpuAdjoint);
}

bool runImageOpsHostApiGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	yrt::ImageParams params3D(3, 2, 2, 3.0f, 2.0f, 2.0f);
	yrt::ImageParams params4D(3, 2, 2, 3.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f,
	                          2);
	const std::size_t spatialCount = imageVoxelCount(params3D);
	const std::size_t voxelCount4D = imageVoxelCount(params4D);

	const std::vector<float> image3D = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f,
	    4.0f, -8.0f, 16.0f, 3.25f, -0.125f, 7.5f};
	const std::vector<float> addend3D = {
	    1.0f, 0.0f, -1.0f, 2.0f, -2.0f, 3.0f,
	    -3.0f, 4.0f, -4.0f, 5.0f, -5.0f, 6.0f};
	const std::vector<float> mask3D = {
	    -1.0f, 0.5f, 0.5001f, 2.0f, -0.25f, 0.0f,
	    0.49f, 0.51f, 1.0f, -2.0f, 0.5f, 3.0f};
	const std::vector<float> sensitivity3D = {
	    0.0f, 0.05f, 0.1f, 0.1001f, 1.0f, 2.0f,
	    0.2f, 4.0f, 0.099f, 8.0f, 16.0f, 32.0f};
	const std::vector<float> update3D = {
	    1.0f, 2.0f, 4.0f, 0.5f, 8.0f, 2.0f,
	    3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f};

	std::vector<float> image4D(voxelCount4D);
	std::vector<float> update4D(voxelCount4D);
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		image4D[i] = static_cast<float>(i) * 0.5f - 3.0f;
		update4D[i] = 0.75f + static_cast<float>(i) * 0.25f;
	}

	const float fillValue = -4.25f;
	yrt::ImageOwned cpuFill(params4D);
	yrt::ImageOwned metalFill(params4D);
	cpuFill.allocate();
	metalFill.allocate();
	copyValuesToImage(cpuFill, image4D);
	copyValuesToImage(metalFill, image4D);
	cpuFill.fill(fillValue);
	if (!yrt::backend::metal::fill(context, metalFill, fillValue) ||
	    !imagesMatch(metalFill, cpuFill))
	{
		return false;
	}

	const float scalar = -1.5f;
	yrt::ImageOwned cpuScalar(params4D);
	yrt::ImageOwned metalScalar(params4D);
	cpuScalar.allocate();
	metalScalar.allocate();
	copyValuesToImage(cpuScalar, image4D);
	copyValuesToImage(metalScalar, image4D);
	cpuScalar.multWithScalar(scalar);
	if (!yrt::backend::metal::multiplyByScalar(context, metalScalar, scalar) ||
	    !imagesMatch(metalScalar, cpuScalar))
	{
		return false;
	}

	yrt::ImageOwned addInput3D(params3D);
	yrt::ImageOwned cpuAdd3D(params3D);
	yrt::ImageOwned metalAdd3D(params3D);
	addInput3D.allocate();
	cpuAdd3D.allocate();
	metalAdd3D.allocate();
	copyValuesToImage(addInput3D, addend3D);
	copyValuesToImage(cpuAdd3D, image3D);
	copyValuesToImage(metalAdd3D, image3D);
	addInput3D.addFirstImageToSecond(&cpuAdd3D);
	if (!yrt::backend::metal::add3DTo3D(context, addInput3D, metalAdd3D) ||
	    !imagesMatch(metalAdd3D, cpuAdd3D))
	{
		return false;
	}

	yrt::ImageOwned metalAdd4D(params4D);
	yrt::ImageOwned expectedAdd4D(params4D);
	metalAdd4D.allocate();
	expectedAdd4D.allocate();
	copyValuesToImage(metalAdd4D, image4D);
	copyValuesToImage(expectedAdd4D, image4D);
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		expectedAdd4D.getRawPointer()[i] += addend3D[i % spatialCount];
	}
	if (!yrt::backend::metal::add3DTo4D(context, addInput3D, metalAdd4D) ||
	    !imagesMatch(metalAdd4D, expectedAdd4D))
	{
		return false;
	}

	const float threshold = 0.5f;
	const float valLeScale = -0.5f;
	const float valLeOffset = 2.0f;
	const float valGtScale = 1.5f;
	const float valGtOffset = -3.0f;
	yrt::ImageOwned mask(params3D);
	yrt::ImageOwned cpuThreshold(params3D);
	yrt::ImageOwned metalThreshold(params3D);
	mask.allocate();
	cpuThreshold.allocate();
	metalThreshold.allocate();
	copyValuesToImage(mask, mask3D);
	copyValuesToImage(cpuThreshold, image3D);
	copyValuesToImage(metalThreshold, image3D);
	cpuThreshold.applyThreshold(&mask, threshold, valLeScale, valLeOffset,
	                            valGtScale, valGtOffset);
	if (!yrt::backend::metal::applyThreshold(context, metalThreshold, mask,
	        threshold, valLeScale, valLeOffset, valGtScale, valGtOffset) ||
	    !imagesMatch(metalThreshold, cpuThreshold))
	{
		return false;
	}

	yrt::ImageOwned cpuBroadcast(params4D);
	yrt::ImageOwned metalBroadcast(params4D);
	cpuBroadcast.allocate();
	metalBroadcast.allocate();
	copyValuesToImage(cpuBroadcast, image4D);
	copyValuesToImage(metalBroadcast, image4D);
	cpuBroadcast.applyThresholdBroadcast(&mask, threshold, valLeScale,
	                                     valLeOffset, valGtScale, valGtOffset);
	if (!yrt::backend::metal::applyThresholdBroadcast(context, metalBroadcast,
	        mask, threshold, valLeScale, valLeOffset, valGtScale, valGtOffset) ||
	    !imagesMatch(metalBroadcast, cpuBroadcast))
	{
		return false;
	}

	const float emThreshold = 0.1f;
	yrt::ImageOwned updateStatic(params3D);
	yrt::ImageOwned sensitivity(params3D);
	yrt::ImageOwned cpuStatic(params3D);
	yrt::ImageOwned metalStatic(params3D);
	updateStatic.allocate();
	sensitivity.allocate();
	cpuStatic.allocate();
	metalStatic.allocate();
	copyValuesToImage(updateStatic, update3D);
	copyValuesToImage(sensitivity, sensitivity3D);
	copyValuesToImage(cpuStatic, image3D);
	copyValuesToImage(metalStatic, image3D);
	cpuStatic.updateEMThresholdStatic(&updateStatic, &sensitivity, emThreshold);
	if (!yrt::backend::metal::updateEMStatic(context, metalStatic, updateStatic,
	        sensitivity, emThreshold) ||
	    !imagesMatch(metalStatic, cpuStatic))
	{
		return false;
	}

	yrt::ImageOwned updateDynamic(params4D);
	yrt::ImageOwned cpuDynamic(params4D);
	yrt::ImageOwned metalDynamic(params4D);
	updateDynamic.allocate();
	cpuDynamic.allocate();
	metalDynamic.allocate();
	copyValuesToImage(updateDynamic, update4D);
	copyValuesToImage(cpuDynamic, image4D);
	copyValuesToImage(metalDynamic, image4D);
	cpuDynamic.updateEMThresholdDynamic(&updateDynamic, &sensitivity,
	                                    emThreshold);
	if (!yrt::backend::metal::updateEMDynamic(context, metalDynamic,
	        updateDynamic, sensitivity, emThreshold) ||
	    !imagesMatch(metalDynamic, cpuDynamic))
	{
		return false;
	}

	return true;
}

bool runImageMetalGoldenTest()
{
	const yrt::backend::metal::Context context;
	if (!context.isValid())
	{
		return false;
	}

	yrt::ImageParams params3D(3, 2, 2, 3.0f, 2.0f, 2.0f);
	yrt::ImageParams params4D(3, 2, 2, 3.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f,
	                          2);
	const std::size_t spatialCount = imageVoxelCount(params3D);
	const std::size_t voxelCount4D = imageVoxelCount(params4D);

	const std::vector<float> image3D = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f,
	    4.0f, -8.0f, 16.0f, 3.25f, -0.125f, 7.5f};
	const std::vector<float> addend3D = {
	    1.0f, 0.0f, -1.0f, 2.0f, -2.0f, 3.0f,
	    -3.0f, 4.0f, -4.0f, 5.0f, -5.0f, 6.0f};
	const std::vector<float> mask3D = {
	    -1.0f, 0.5f, 0.5001f, 2.0f, -0.25f, 0.0f,
	    0.49f, 0.51f, 1.0f, -2.0f, 0.5f, 3.0f};
	const std::vector<float> sensitivity3D = {
	    0.0f, 0.05f, 0.1f, 0.1001f, 1.0f, 2.0f,
	    0.2f, 4.0f, 0.099f, 8.0f, 16.0f, 32.0f};
	const std::vector<float> update3D = {
	    1.0f, 2.0f, 4.0f, 0.5f, 8.0f, 2.0f,
	    3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f};

	std::vector<float> image4D(voxelCount4D);
	std::vector<float> update4D(voxelCount4D);
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		image4D[i] = static_cast<float>(i) * 0.5f - 3.0f;
		update4D[i] = 0.75f + static_cast<float>(i) * 0.25f;
	}

	yrt::ImageOwned seed4D(params4D);
	yrt::ImageOwned seed3D(params3D);
	yrt::ImageOwned addInput3D(params3D);
	yrt::ImageOwned mask(params3D);
	yrt::ImageOwned updateStatic(params3D);
	yrt::ImageOwned sensitivity(params3D);
	yrt::ImageOwned updateDynamic(params4D);
	seed4D.allocate();
	seed3D.allocate();
	addInput3D.allocate();
	mask.allocate();
	updateStatic.allocate();
	sensitivity.allocate();
	updateDynamic.allocate();
	copyValuesToImage(seed4D, image4D);
	copyValuesToImage(seed3D, image3D);
	copyValuesToImage(addInput3D, addend3D);
	copyValuesToImage(mask, mask3D);
	copyValuesToImage(updateStatic, update3D);
	copyValuesToImage(sensitivity, sensitivity3D);
	copyValuesToImage(updateDynamic, update4D);

	yrt::ImageOwned cpuFill(params4D);
	cpuFill.allocate();
	cpuFill.copyFromImage(&seed4D);
	yrt::backend::metal::ImageMetal metalFill(context, seed4D);
	const float fillValue = -4.25f;
	cpuFill.fill(fillValue);
	if (!metalFill.isValid() || !metalFill.fill(fillValue) ||
	    !imagesMatch(metalFill.image(), cpuFill))
	{
		return false;
	}

	yrt::ImageOwned cpuScalar(params4D);
	cpuScalar.allocate();
	cpuScalar.copyFromImage(&seed4D);
	yrt::backend::metal::ImageMetal metalScalar(context, seed4D);
	const float scalar = -1.5f;
	cpuScalar.multWithScalar(scalar);
	if (!metalScalar.multiplyByScalar(scalar) ||
	    !imagesMatch(metalScalar.image(), cpuScalar))
	{
		return false;
	}

	yrt::ImageOwned cpuAdd3D(params3D);
	cpuAdd3D.allocate();
	cpuAdd3D.copyFromImage(&seed3D);
	yrt::backend::metal::ImageMetal metalAdd3D(context, seed3D);
	addInput3D.addFirstImageToSecond(&cpuAdd3D);
	if (!metalAdd3D.add3DTo3D(addInput3D) ||
	    !imagesMatch(metalAdd3D.image(), cpuAdd3D))
	{
		return false;
	}

	yrt::ImageOwned expectedAdd4D(params4D);
	expectedAdd4D.allocate();
	expectedAdd4D.copyFromImage(&seed4D);
	yrt::backend::metal::ImageMetal metalAdd4D(context, seed4D);
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		expectedAdd4D.getRawPointer()[i] += addend3D[i % spatialCount];
	}
	if (!metalAdd4D.add3DTo4D(addInput3D) ||
	    !imagesMatch(metalAdd4D.image(), expectedAdd4D))
	{
		return false;
	}

	const float threshold = 0.5f;
	const float valLeScale = -0.5f;
	const float valLeOffset = 2.0f;
	const float valGtScale = 1.5f;
	const float valGtOffset = -3.0f;
	yrt::ImageOwned cpuThreshold(params3D);
	cpuThreshold.allocate();
	cpuThreshold.copyFromImage(&seed3D);
	yrt::backend::metal::ImageMetal metalThreshold(context, seed3D);
	yrt::backend::metal::ImageMetal metalMask(context, mask);
	cpuThreshold.applyThreshold(&mask, threshold, valLeScale, valLeOffset,
	                            valGtScale, valGtOffset);
	if (!metalThreshold.applyThreshold(metalMask, threshold, valLeScale,
	        valLeOffset, valGtScale, valGtOffset) ||
	    !imagesMatch(metalThreshold.image(), cpuThreshold))
	{
		return false;
	}

	yrt::ImageOwned cpuBroadcast(params4D);
	cpuBroadcast.allocate();
	cpuBroadcast.copyFromImage(&seed4D);
	yrt::backend::metal::ImageMetal metalBroadcast(context, seed4D);
	cpuBroadcast.applyThresholdBroadcast(&mask, threshold, valLeScale,
	                                     valLeOffset, valGtScale, valGtOffset);
	if (!metalBroadcast.applyThresholdBroadcast(metalMask, threshold,
	        valLeScale, valLeOffset, valGtScale, valGtOffset) ||
	    !imagesMatch(metalBroadcast.image(), cpuBroadcast))
	{
		return false;
	}

	const float emThreshold = 0.1f;
	yrt::ImageOwned cpuStatic(params3D);
	cpuStatic.allocate();
	cpuStatic.copyFromImage(&seed3D);
	yrt::backend::metal::ImageMetal metalStatic(context, seed3D);
	yrt::backend::metal::ImageMetal metalUpdateStatic(context, updateStatic);
	yrt::backend::metal::ImageMetal metalSensitivity(context, sensitivity);
	cpuStatic.updateEMThresholdStatic(&updateStatic, &sensitivity,
	                                  emThreshold);
	if (!metalStatic.updateEMStatic(metalUpdateStatic, metalSensitivity,
	        emThreshold) ||
	    !imagesMatch(metalStatic.image(), cpuStatic))
	{
		return false;
	}

	yrt::ImageOwned cpuDynamic(params4D);
	cpuDynamic.allocate();
	cpuDynamic.copyFromImage(&seed4D);
	yrt::backend::metal::ImageMetal metalDynamic(context, seed4D);
	yrt::backend::metal::ImageMetal metalUpdateDynamic(context, updateDynamic);
	cpuDynamic.updateEMThresholdDynamic(&updateDynamic, &sensitivity,
	                                    emThreshold);
	if (!metalDynamic.updateEMDynamic(metalUpdateDynamic, metalSensitivity,
	        emThreshold) ||
	    !imagesMatch(metalDynamic.image(), cpuDynamic))
	{
		return false;
	}

	yrt::backend::metal::ImageMetal allocated(context, params3D);
	if (!allocated.fill(1.25f))
	{
		return false;
	}
	yrt::ImageOwned expectedAllocated(params3D);
	expectedAllocated.allocate();
	expectedAllocated.fill(1.25f);
	return imagesMatch(allocated.image(), expectedAllocated);
}

bool runPsfOpsHostApiGoldenTest()
{
	yrt::ImageParams params(4, 3, 3, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned input(params);
	yrt::ImageOwned cpuOutput(params);
	yrt::ImageOwned metalOutput(params);
	input.allocate();
	cpuOutput.allocate();
	metalOutput.allocate();

	const std::size_t count = static_cast<std::size_t>(params.nx) *
	                          static_cast<std::size_t>(params.ny) *
	                          static_cast<std::size_t>(params.nz) *
	                          static_cast<std::size_t>(params.nt);
	for (std::size_t i = 0; i < count; ++i)
	{
		input.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.25f +
		    static_cast<float>(i / 5) * 0.125f;
	}

	const std::vector<float> kernelX = {0.25f, -0.5f, 1.25f};
	const std::vector<float> kernelY = {-0.1f, 0.8f, 0.3f};
	const std::vector<float> kernelZ = {0.6f, -0.2f, 0.6f};

	yrt::OperatorPsf psf(kernelX, kernelY, kernelZ);
	psf.applyA(&input, &cpuOutput);

	const yrt::backend::metal::Context context;
	if (!yrt::backend::metal::convolve3DSeparableHost(
	        context, input, metalOutput, kernelX, kernelY, kernelZ))
	{
		return false;
	}

	return imagesMatch(metalOutput, cpuOutput);
}

bool runOperatorPsfMetalGoldenTest()
{
	yrt::ImageParams params(4, 3, 3, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned input(params);
	yrt::ImageOwned cpuOutput(params);
	yrt::ImageOwned metalOutput(params);
	input.allocate();
	cpuOutput.allocate();
	metalOutput.allocate();

	const std::size_t count = imageVoxelCount(params);
	for (std::size_t i = 0; i < count; ++i)
	{
		input.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.2f +
		    static_cast<float>(i / 9) * 0.15f;
	}

	const std::vector<float> kernelX = {0.25f, -0.5f, 1.25f};
	const std::vector<float> kernelY = {-0.1f, 0.8f, 0.3f};
	const std::vector<float> kernelZ = {0.6f, -0.2f, 0.6f};

	const yrt::backend::metal::Context context;
	const yrt::backend::metal::OperatorPsfMetal metalPsf(
	    context, kernelX, kernelY, kernelZ);
	if (!metalPsf.isValid())
	{
		return false;
	}

	yrt::OperatorPsf cpuPsf(kernelX, kernelY, kernelZ);
	cpuPsf.applyA(&input, &cpuOutput);
	if (!metalPsf.applyA(input, metalOutput) ||
	    !imagesMatch(metalOutput, cpuOutput))
	{
		return false;
	}

	cpuPsf.applyAH(&input, &cpuOutput);
	if (!metalPsf.applyAH(input, metalOutput) ||
	    !imagesMatch(metalOutput, cpuOutput))
	{
		return false;
	}

	return true;
}

bool runExperimentalBackendGoldenTest()
{
	const yrt::backend::metal::ExperimentalBackend backend;
	if (!backend.isAvailable() || !backend.isValid() ||
	    !backend.errorMessage().empty())
	{
		return false;
	}

	const std::vector<float> input = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, -8.0f, 16.0f};
	const std::vector<float> output = {
	    3.0f, -4.0f, 2.0f, 0.25f, -0.5f, 5.0f, -10.0f, 8.0f, 1.5f};
	std::vector<float> expected(output.size());

	auto vector = backend.makeProjectionVector(output);
	if (!vector.isValid() || !vector.add(input))
	{
		return false;
	}
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		expected[i] = output[i] + input[i];
	}
	if (!valuesMatch(vector.values(), expected))
	{
		return false;
	}

	yrt::ImageParams params(4, 3, 3, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                        0.0f, 2);
	yrt::ImageOwned seed(params);
	yrt::ImageOwned cpuImage(params);
	seed.allocate();
	cpuImage.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(params); ++i)
	{
		seed.getRawPointer()[i] = static_cast<float>(i) * 0.25f - 1.0f;
	}
	cpuImage.copyFromImage(&seed);

	auto image = backend.makeImage(seed);
	const float scalar = -1.25f;
	cpuImage.multWithScalar(scalar);
	if (!image.isValid() || !image.multiplyByScalar(scalar) ||
	    !imagesMatch(image.image(), cpuImage))
	{
		return false;
	}

	yrt::ImageOwned psfOutputCpu(params);
	yrt::ImageOwned psfOutputMetal(params);
	yrt::ImageOwned psfAdjointCpu(params);
	yrt::ImageOwned psfAdjointMetal(params);
	psfOutputCpu.allocate();
	psfOutputMetal.allocate();
	psfAdjointCpu.allocate();
	psfAdjointMetal.allocate();
	const std::vector<float> kernelX = {0.25f, -0.5f, 1.25f};
	const std::vector<float> kernelY = {-0.1f, 0.8f, 0.3f};
	const std::vector<float> kernelZ = {0.6f, -0.2f, 0.6f};

	yrt::OperatorPsf cpuPsf(kernelX, kernelY, kernelZ);
	cpuPsf.applyA(&seed, &psfOutputCpu);
	cpuPsf.applyAH(&seed, &psfAdjointCpu);

	if (!backend.applyOperatorPsfForward(seed, psfOutputMetal, kernelX,
	        kernelY, kernelZ) ||
	    !imagesMatch(psfOutputMetal, psfOutputCpu))
	{
		return false;
	}

	if (!backend.applyOperatorPsfAdjoint(seed, psfAdjointMetal, kernelX,
	        kernelY, kernelZ) ||
	    !imagesMatch(psfAdjointMetal, psfAdjointCpu))
	{
		return false;
	}

	const std::filesystem::path psfCsvPath =
	    std::filesystem::temp_directory_path() /
	    "yrtpet_metal_experimental_backend_psf.csv";
	ScopedPathCleanup cleanup{psfCsvPath};
	if (!writeTextFile(psfCsvPath, uniformPsfCsv()))
	{
		return false;
	}

	yrt::OperatorPsf cpuPsfFromFile(psfCsvPath.string());
	cpuPsfFromFile.applyA(&seed, &psfOutputCpu);
	cpuPsfFromFile.applyAH(&seed, &psfAdjointCpu);

	yrt::ImageOwned psfFileOutputMetal(params);
	yrt::ImageOwned psfFileAdjointMetal(params);
	psfFileOutputMetal.allocate();
	psfFileAdjointMetal.allocate();
	const bool filePsfMatches =
	    backend.applyOperatorPsfForward(seed, psfFileOutputMetal,
	        psfCsvPath.string()) &&
	    imagesMatch(psfFileOutputMetal, psfOutputCpu) &&
	    backend.applyOperatorPsfAdjoint(seed, psfFileAdjointMetal,
	        psfCsvPath.string()) &&
	    imagesMatch(psfFileAdjointMetal, psfAdjointCpu);
	return filePsfMatches;
}

bool runPsfFileOpsRealInputGoldenTest()
{
	const yrt::backend::metal::ExperimentalBackend backend;
	if (!backend.isAvailable() || !backend.isValid())
	{
		return false;
	}

	const std::filesystem::path fixtureDir =
	    std::filesystem::temp_directory_path() /
	    "yrtpet_metal_file_psf_real_input_golden";
	ScopedPathCleanup cleanup{fixtureDir};
	std::error_code ec;
	std::filesystem::remove_all(fixtureDir, ec);
	if (!std::filesystem::create_directories(fixtureDir))
	{
		return false;
	}

	const std::filesystem::path imagePath = fixtureDir / "input.nii";
	const std::filesystem::path psfPath = fixtureDir / "image_psf.csv";
	yrt::ImageParams params(4, 3, 3, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f,
	                        0.0f, 2);
	yrt::ImageOwned seed(params);
	seed.allocate();
	for (std::size_t i = 0; i < imageVoxelCount(params); ++i)
	{
		seed.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 19) - 9) * 0.125f +
		    static_cast<float>(i / 7) * 0.05f;
	}
	seed.writeToFile(imagePath.string());
	if (!writeTextFile(psfPath, uniformPsfCsv()))
	{
		return false;
	}

	yrt::ImageOwned inputFromFile(imagePath.string());
	yrt::OperatorPsf cpuPsf(psfPath.string());
	yrt::ImageOwned cpuForward(params);
	yrt::ImageOwned metalForward(params);
	yrt::ImageOwned cpuAdjoint(params);
	yrt::ImageOwned metalAdjoint(params);
	cpuForward.allocate();
	metalForward.allocate();
	cpuAdjoint.allocate();
	metalAdjoint.allocate();

	cpuPsf.applyA(&inputFromFile, &cpuForward);
	cpuPsf.applyAH(&inputFromFile, &cpuAdjoint);

	return yrt::backend::metal::applyPsfForward(inputFromFile, metalForward,
	           psfPath.string()) &&
	       imagesMatch(metalForward, cpuForward) &&
	       yrt::backend::metal::applyPsfAdjoint(inputFromFile, metalAdjoint,
	           psfPath.string()) &&
	       imagesMatch(metalAdjoint, cpuAdjoint);
}

bool runPsfFileOpsErrorPathTest()
{
	const yrt::backend::metal::ExperimentalBackend backend;
	if (!backend.isAvailable() || !backend.isValid())
	{
		return false;
	}

	const std::filesystem::path fixtureDir =
	    std::filesystem::temp_directory_path() /
	    "yrtpet_metal_file_psf_error_paths";
	ScopedPathCleanup cleanup{fixtureDir};
	std::error_code ec;
	std::filesystem::remove_all(fixtureDir, ec);
	if (!std::filesystem::create_directories(fixtureDir))
	{
		return false;
	}

	const std::filesystem::path goodPsfPath = fixtureDir / "good_psf.csv";
	const std::filesystem::path shortPsfPath = fixtureDir / "short_psf.csv";
	const std::filesystem::path evenPsfPath = fixtureDir / "even_psf.csv";
	if (!writeTextFile(goodPsfPath, uniformPsfCsv()) ||
	    !writeTextFile(shortPsfPath, "0.25,0.5,0.25\n0.2,0.6,0.2\n") ||
	    !writeTextFile(evenPsfPath,
	        "0.25,0.5,0.25,0.0\n"
	        "0.2,0.6,0.2,0.0\n"
	        "0.1,0.8,0.1,0.0\n"
	        "4,3,3,0\n"))
	{
		return false;
	}

	yrt::ImageParams inputParams(4, 3, 3, 4.0f, 3.0f, 3.0f);
	yrt::ImageParams outputParams(3, 3, 3, 3.0f, 3.0f, 3.0f);
	yrt::ImageOwned input(inputParams);
	yrt::ImageOwned mismatchedOutput(outputParams);
	input.allocate();
	mismatchedOutput.allocate();
	input.fill(1.0f);
	mismatchedOutput.fill(0.0f);

	const std::filesystem::path missingPsfPath =
	    fixtureDir / "missing_psf.csv";
	if (!expectThrows(
	        [&input, &mismatchedOutput, &missingPsfPath]()
	        { static_cast<void>(yrt::backend::metal::applyPsfForward(
	              input, mismatchedOutput, missingPsfPath.string())); }) ||
	    !expectThrows(
	        [&input, &mismatchedOutput, &shortPsfPath]()
	        { static_cast<void>(yrt::backend::metal::applyPsfForward(
	              input, mismatchedOutput, shortPsfPath.string())); }) ||
	    !expectThrows(
	        [&input, &mismatchedOutput, &evenPsfPath]()
	        { static_cast<void>(yrt::backend::metal::applyPsfForward(
	              input, mismatchedOutput, evenPsfPath.string())); }))
	{
		return false;
	}

	if (yrt::backend::metal::applyPsfForward(input, mismatchedOutput,
	        goodPsfPath.string()) ||
	    yrt::backend::metal::applyPsfAdjoint(input, mismatchedOutput,
	        goodPsfPath.string()))
	{
		return false;
	}

	return true;
}

}  // namespace

int main()
{
	if (!yrt::backend::metal::isAvailable())
	{
		std::cout << "Metal tests: SKIP (Metal device unavailable)\n";
		return 77;
	}

	const std::vector<TestCase> testCases = {
	    {"device_metallib_buffer_smoke", yrt::backend::metal::runSmokeKernel},
	    {"backend_context_valid", runBackendContextTest},
	    {"projection_vector_golden",
	        yrt::backend::metal::runProjectionVectorGoldenTests},
	    {"projection_vector_ops_host_api_golden",
	     runProjectionVectorOpsHostApiGoldenTest},
	    {"projection_vector_metal_golden", runProjectionVectorMetalGoldenTest},
	    {"projection_geometry_siddon_entry_range_golden",
	     runProjectionGeometryGoldenTest},
	    {"projection_batch_metal_buffer_golden",
	     runProjectionBatchMetalGoldenTest},
	    {"siddon_single_ray_forward_golden",
	     runSiddonSingleRayForwardGoldenTest},
	    {"siddon_single_ray_adjoint_golden",
	     runSiddonSingleRayAdjointGoldenTest},
	    {"siddon_single_ray_native_atomic_float_adjoint_golden",
	     runSiddonSingleRayNativeAtomicFloatAdjointGoldenTest},
	    {"siddon_projector_metal_adjointness_golden",
	     runSiddonProjectorMetalAdapterGoldenTest},
	    {"joseph_projector_metal_adjointness_golden",
	     runJosephProjectorMetalAdapterGoldenTest},
	    {"joseph_projector_metal_texture_forward_golden",
	     runJosephProjectorMetalTextureForwardGoldenTest},
	    {"siddon_empty_or_miss_golden", runSiddonEmptyOrMissGoldenTest},
	    {"siddon_projector_metal_failure_modes_golden",
	     runSiddonProjectorMetalFailureModeGoldenTest},
	    {"siddon_projector_metal_frame_isolation_golden",
	     runSiddonProjectorMetalFrameIsolationGoldenTest},
	    {"operator_projector_metal_bridge_forward_golden",
	     runOperatorProjectorMetalBridgeForwardGoldenTest},
	    {"operator_projector_metal_bridge_adjoint_golden",
	     runOperatorProjectorMetalBridgeAdjointGoldenTest},
	    {"operator_projector_metal_bridge_partial_cache_golden",
	     runOperatorProjectorMetalBridgePartialCacheGoldenTest},
	    {"operator_projector_metal_bridge_unsupported_golden",
	     runOperatorProjectorMetalBridgeUnsupportedGoldenTest},
	    {"operator_projector_metal_dispatch_default_golden",
	     runOperatorProjectorMetalDispatchDefaultGoldenTest},
	    {"operator_projector_metal_dispatch_forward_golden",
	     runOperatorProjectorMetalDispatchEnabledForwardGoldenTest},
	    {"operator_projector_metal_dispatch_adjoint_golden",
	     runOperatorProjectorMetalDispatchEnabledAdjointGoldenTest},
	    {"operator_projector_metal_dispatch_joseph_golden",
	     runOperatorProjectorMetalDispatchJosephGoldenTest},
	    {"operator_projector_metal_dispatch_fallback_golden",
	     runOperatorProjectorMetalDispatchUnsupportedFallbackGoldenTest},
	    {"operator_projector_metal_dispatch_dd_fallback_golden",
	     runOperatorProjectorMetalDispatchDDFallbackGoldenTest},
	    {"operator_projector_metal_dispatch_multi_ray_fallback_golden",
	     runOperatorProjectorMetalDispatchMultiRayFallbackGoldenTest},
	    {"operator_projector_metal_dispatch_projection_psf_fallback_golden",
	     runOperatorProjectorMetalDispatchProjectionPsfFallbackGoldenTest},
	    {"operator_projector_metal_dispatch_lr_fallback_golden",
	     runOperatorProjectorMetalDispatchLRFallbackGoldenTest},
	    {"osem_cpu_experimental_metal_projector_golden",
	     runOsemCpuExperimentalMetalProjectorGoldenTest},
	    {"osem_cpu_experimental_metal_projector_dd_fallback_golden",
	     runOsemCpuExperimentalMetalProjectorDDFallbackGoldenTest},
	    {"operator_projector_metal_file_input_golden",
	     runOperatorProjectorMetalFileInputGoldenTest},
	    {"image_scalar_ops_golden",
	     yrt::backend::metal::runImageScalarOpsGoldenTests},
	    {"image_ops_host_api_golden", runImageOpsHostApiGoldenTest},
	    {"image_metal_golden", runImageMetalGoldenTest},
	    {"psf_convolution_golden",
	        yrt::backend::metal::runPsfConvolutionGoldenTests},
	    {"psf_ops_host_api_golden", runPsfOpsHostApiGoldenTest},
	    {"operator_psf_metal_golden", runOperatorPsfMetalGoldenTest},
	    {"experimental_backend_golden", runExperimentalBackendGoldenTest},
	    {"psf_file_ops_real_input_golden", runPsfFileOpsRealInputGoldenTest},
	    {"psf_file_ops_error_paths", runPsfFileOpsErrorPathTest},
	};

	for (const TestCase& testCase : testCases)
	{
		std::cout << "[ RUN      ] " << testCase.name << '\n';
		if (!testCase.run())
		{
			std::cerr << "[  FAILED  ] " << testCase.name << '\n';
			return 1;
		}
		std::cout << "[       OK ] " << testCase.name << '\n';
	}

	std::cout << "Metal tests: PASS\n";
	return 0;
}
