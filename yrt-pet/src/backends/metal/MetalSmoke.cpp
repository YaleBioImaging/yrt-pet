/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalSmoke.hpp"

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"
#include "yrt-pet/backends/metal/JosephProjectorKernels.hpp"
#include "yrt-pet/backends/metal/MetalBackend.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/MotionOps.hpp"
#include "yrt-pet/backends/metal/ProjectionGeometryKernels.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorKernels.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorKernels.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageParams.hpp"
#include "yrt-pet/datastruct/projection/LORMotion.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/utils/ReconstructionUtils.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

#ifndef YRTPET_METAL_SMOKE_METALLIB
#define YRTPET_METAL_SMOKE_METALLIB ""
#endif

namespace yrt::backend::metal
{
namespace
{

bool almostEqual(float actual, float expected)
{
	const float scale = std::max(1.0f, std::fabs(expected));
	return std::fabs(actual - expected) <= 1.0e-5f * scale;
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

bool readBuffer(const Buffer& buffer, std::vector<float>& output)
{
	return buffer.copyToHost(output.data(), sizeof(float) * output.size());
}

std::vector<float> runOperatorPsfReference(const std::vector<float>& input,
    const ImageShape& shape, const std::vector<float>& kernelX,
    const std::vector<float>& kernelY, const std::vector<float>& kernelZ)
{
	yrt::ImageParams params(static_cast<int>(shape.nx),
	    static_cast<int>(shape.ny), static_cast<int>(shape.nz),
	    static_cast<float>(shape.nx), static_cast<float>(shape.ny),
	    static_cast<float>(shape.nz), 0.0f, 0.0f, 0.0f,
	    static_cast<yrt::frame_t>(shape.nt));
	yrt::ImageOwned inputImage(params);
	inputImage.allocate();
	std::copy(input.begin(), input.end(), inputImage.getRawPointer());

	yrt::ImageOwned outputImage(params);
	outputImage.allocate();

	yrt::OperatorPsf psf(kernelX, kernelY, kernelZ);
	psf.applyA(&inputImage, &outputImage);

	const float* outputPtr = outputImage.getRawPointer();
	return std::vector<float>(outputPtr, outputPtr + input.size());
}

bool runMetalPsfConvolution(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const std::vector<float>& input,
    const ImageShape& shape, const std::vector<float>& kernelX,
    const std::vector<float>& kernelY, const std::vector<float>& kernelZ,
    std::vector<float>& output)
{
	const auto byteCount = sizeof(float) * input.size();
	Buffer inputBuffer = Buffer::copyFromHost(device, input.data(), byteCount);
	Buffer tempA = Buffer::allocate(device, byteCount);
	Buffer tempB = Buffer::allocate(device, byteCount);
	Buffer kernelXBuffer =
	    Buffer::copyFromHost(device, kernelX.data(), sizeof(float) * kernelX.size());
	Buffer kernelYBuffer =
	    Buffer::copyFromHost(device, kernelY.data(), sizeof(float) * kernelY.size());
	Buffer kernelZBuffer =
	    Buffer::copyFromHost(device, kernelZ.data(), sizeof(float) * kernelZ.size());
	if (!inputBuffer.isValid() || !tempA.isValid() || !tempB.isValid() ||
	    !kernelXBuffer.isValid() || !kernelYBuffer.isValid() ||
	    !kernelZBuffer.isValid())
	{
		return false;
	}

	if (!launchImageConvolve3DSeparableX(device, library, commandQueue,
	        inputBuffer, tempA, kernelXBuffer,
	        static_cast<std::uint32_t>(kernelX.size()), shape))
	{
		return false;
	}
	if (!launchImageConvolve3DSeparableY(device, library, commandQueue, tempA,
	        tempB, kernelYBuffer, static_cast<std::uint32_t>(kernelY.size()),
	        shape))
	{
		return false;
	}
	if (!launchImageConvolve3DSeparableZ(device, library, commandQueue, tempB,
	        tempA, kernelZBuffer, static_cast<std::uint32_t>(kernelZ.size()),
	        shape))
	{
		return false;
	}

	return tempA.copyToHost(output.data(), byteCount);
}

}  // namespace

bool isAvailable()
{
	return isDeviceAvailable();
}

bool runSmokeKernel()
{
	const Device device = Device::createSystemDefault();
	if (!device.isValid())
	{
		return false;
	}

	const Library library =
	    Library::loadFromFile(device, YRTPET_METAL_SMOKE_METALLIB);
	const CommandQueue commandQueue = CommandQueue::create(device);
	if (!library.isValid() || !commandQueue.isValid())
	{
		return false;
	}

	constexpr std::size_t ValueCount = 16;
	std::array<float, ValueCount> input{};
	for (std::size_t i = 0; i < ValueCount; ++i)
	{
		input[i] = static_cast<float>(i);
	}

	Buffer inputBuffer =
	    Buffer::copyFromHost(device, input.data(), sizeof(float) * input.size());
	Buffer outputBuffer = Buffer::allocate(device, sizeof(float) * input.size());
	if (!inputBuffer.isValid() || !outputBuffer.isValid())
	{
		return false;
	}

	if (!launchSmokeAddOne(device, library, commandQueue, inputBuffer, outputBuffer,
	        input.size()))
	{
		return false;
	}

	std::array<float, ValueCount> output{};
	if (!outputBuffer.copyToHost(output.data(), sizeof(float) * output.size()))
	{
		return false;
	}

	for (std::size_t i = 0; i < ValueCount; ++i)
	{
		const float expected = input[i] + 1.0f;
		if (std::fabs(output[i] - expected) > 1e-6f)
		{
			return false;
		}
	}

	return true;
}

bool runJosephAdjointAccumulationSmoke()
{
	const Device device = Device::createSystemDefault();
	if (!device.isValid())
	{
		return false;
	}

	const Library library =
	    Library::loadFromFile(device, YRTPET_METAL_SMOKE_METALLIB);
	const CommandQueue commandQueue = CommandQueue::create(device);
	if (!library.isValid() || !commandQueue.isValid())
	{
		return false;
	}

	const SiddonForwardImageParams params{
	    4,
	    4,
	    4,
	    1,
	    0,
	    4.0f,
	    4.0f,
	    4.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    2.0f,
	    2.0f,
	    2.0f,
	    10.0f};
	const std::vector<ProjectionLineEndpoints> lines = {
	    {-1.5f, -0.25f, 0.25f, 1.5f, -0.25f, 0.25f},
	    {-1.5f, 0.25f, -0.25f, 1.5f, 0.25f, -0.25f}};
	const std::vector<float> values = {1.0f, 0.75f};
	const std::size_t voxelCount = static_cast<std::size_t>(params.nx) *
	                               params.ny * params.nz * params.nt;
	const std::vector<float> zeros(voxelCount, 0.0f);

	auto runKernel = [&](const char* functionName,
	                     std::vector<float>& output) -> bool
	{
		Buffer imageBuffer = Buffer::copyFromHost(
		    device, zeros.data(), sizeof(float) * zeros.size());
		Buffer lineBuffer = Buffer::copyFromHost(
		    device, lines.data(), sizeof(ProjectionLineEndpoints) * lines.size());
		Buffer valueBuffer = Buffer::copyFromHost(
		    device, values.data(), sizeof(float) * values.size());
		if (!imageBuffer.isValid() || !lineBuffer.isValid() ||
		    !valueBuffer.isValid())
		{
			return false;
		}
		if (!launchKernel1D(device, library, commandQueue, functionName,
		        {{&imageBuffer, 0}, {&lineBuffer, 1}, {&valueBuffer, 2}},
		        {{&params, sizeof(params), 3}}, lines.size()))
		{
			return false;
		}
		return imageBuffer.copyToHost(
		    output.data(), sizeof(float) * output.size());
	};

	std::vector<float> reference(voxelCount);
	std::vector<float> threadgroupOutput(voxelCount);
	std::vector<float> axisSwitchOutput(voxelCount);
	std::vector<float> incrementalOutput(voxelCount);
	if (!runKernel("joseph_backproject_single_ray_native_atomic_float",
	        reference) ||
	    !runKernel(
	        "joseph_backproject_single_ray_threadgroup_sample_native_atomic_float",
	        threadgroupOutput) ||
	    !runKernel(
	        "joseph_backproject_single_ray_axis_switch_native_atomic_float",
	        axisSwitchOutput) ||
	    !runKernel(
	        "joseph_backproject_single_ray_incremental_native_atomic_float",
	        incrementalOutput))
	{
		return false;
	}
	return valuesMatch(threadgroupOutput, reference) &&
	       valuesMatch(axisSwitchOutput, reference) &&
	       valuesMatch(incrementalOutput, reference);
}

bool runJosephAxisSpecializedSmoke()
{
	const Device device = Device::createSystemDefault();
	if (!device.isValid())
	{
		return false;
	}

	const Library library =
	    Library::loadFromFile(device, YRTPET_METAL_SMOKE_METALLIB);
	const CommandQueue commandQueue = CommandQueue::create(device);
	if (!library.isValid() || !commandQueue.isValid())
	{
		return false;
	}

	const SiddonForwardImageParams params{
	    4,
	    4,
	    4,
	    1,
	    0,
	    4.0f,
	    4.0f,
	    4.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    1.0f,
	    2.0f,
	    2.0f,
	    2.0f,
	    10.0f};
	const std::vector<ProjectionLineEndpoints> lines = {
	    {-1.5f, -0.25f, 0.25f, 1.5f, -0.25f, 0.25f},
	    {-0.25f, -1.5f, 0.25f, -0.25f, 1.5f, 0.25f},
	    {-0.25f, 0.25f, -1.5f, -0.25f, 0.25f, 1.5f}};
	const std::size_t voxelCount = static_cast<std::size_t>(params.nx) *
	                               params.ny * params.nz * params.nt;
	std::vector<float> image(voxelCount);
	for (std::size_t i = 0; i < image.size(); ++i)
	{
		image[i] = static_cast<float>((i % 17) + 1) * 0.125f;
	}
	const std::vector<float> projectionValue = {1.25f};
	const std::vector<float> zeros(voxelCount, 0.0f);

	for (std::uint32_t axis = 0; axis < lines.size(); ++axis)
	{
		const ProjectionLineEndpoints line = lines[axis];
		Buffer imageBuffer =
		    Buffer::copyFromHost(device, image.data(),
		        sizeof(float) * image.size());
		Buffer axisImageBuffer =
		    Buffer::copyFromHost(device, image.data(),
		        sizeof(float) * image.size());
		Buffer lineBuffer =
		    Buffer::copyFromHost(device, &line,
		        sizeof(ProjectionLineEndpoints));
		Buffer genericProjection =
		    Buffer::allocate(device, sizeof(float));
		Buffer axisProjection =
		    Buffer::allocate(device, sizeof(float));
		if (!imageBuffer.isValid() || !axisImageBuffer.isValid() ||
		    !lineBuffer.isValid() || !genericProjection.isValid() ||
		    !axisProjection.isValid())
		{
			return false;
		}
		if (!launchJosephForwardSingleRay(device, library, commandQueue,
		        imageBuffer, lineBuffer, genericProjection, params, 1) ||
		    !launchJosephForwardSingleRayAxis(device, library, commandQueue,
		        axisImageBuffer, lineBuffer, axisProjection, params, 1, axis))
		{
			return false;
		}
		std::vector<float> genericForward(1);
		std::vector<float> axisForward(1);
		if (!genericProjection.copyToHost(genericForward.data(),
		        sizeof(float)) ||
		    !axisProjection.copyToHost(axisForward.data(), sizeof(float)) ||
		    !valuesMatch(axisForward, genericForward))
		{
			return false;
		}

		Buffer genericBackImage =
		    Buffer::copyFromHost(device, zeros.data(),
		        sizeof(float) * zeros.size());
		Buffer axisBackImage =
		    Buffer::copyFromHost(device, zeros.data(),
		        sizeof(float) * zeros.size());
		Buffer valueBuffer = Buffer::copyFromHost(
		    device, projectionValue.data(), sizeof(float));
		if (!genericBackImage.isValid() || !axisBackImage.isValid() ||
		    !valueBuffer.isValid())
		{
			return false;
		}
		if (!launchJosephBackProjectSingleRay(device, library, commandQueue,
		        genericBackImage, lineBuffer, valueBuffer, params, 1) ||
		    !launchJosephBackProjectSingleRayAxis(device, library, commandQueue,
		        axisBackImage, lineBuffer, valueBuffer, params, 1, axis))
		{
			return false;
		}
		std::vector<float> genericBack(voxelCount);
		std::vector<float> axisBack(voxelCount);
		if (!genericBackImage.copyToHost(
		        genericBack.data(), sizeof(float) * genericBack.size()) ||
		    !axisBackImage.copyToHost(
		        axisBack.data(), sizeof(float) * axisBack.size()) ||
		    !valuesMatch(axisBack, genericBack))
		{
			return false;
		}
	}
	return true;
}

bool runProjectionVectorGoldenTests()
{
	const Device device = Device::createSystemDefault();
	if (!device.isValid())
	{
		return false;
	}

	const Library library =
	    Library::loadFromFile(device, YRTPET_METAL_SMOKE_METALLIB);
	const CommandQueue commandQueue = CommandQueue::create(device);
	if (!library.isValid() || !commandQueue.isValid())
	{
		return false;
	}

	const std::vector<float> lhs = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, -8.0f, 16.0f,
	    3.25f, -0.125f, 7.5f, 11.0f, -13.0f, 19.0f, 23.0f, -29.0f};
	const std::vector<float> rhs = {
	    0.0f, -4.0f, 2.0f, 0.25f, -0.5f, 5.0f, -10.0f, 8.0f, 1.5f,
	    -3.0f, 6.0f, -7.5f, 0.0f, 13.0f, -19.0f, 0.75f, 29.0f};
	const std::size_t count = lhs.size();

	std::vector<float> actual(count);
	std::vector<float> expected(count);

	Buffer clearBuffer =
	    Buffer::copyFromHost(device, lhs.data(), sizeof(float) * count);
	const float clearValue = -7.25f;
	if (!clearBuffer.isValid() ||
	    !launchProjectionClear(device, library, commandQueue, clearBuffer,
	        clearValue, count))
	{
		return false;
	}
	std::fill(expected.begin(), expected.end(), clearValue);
	if (!readBuffer(clearBuffer, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	Buffer addInput =
	    Buffer::copyFromHost(device, lhs.data(), sizeof(float) * count);
	Buffer addOutput =
	    Buffer::copyFromHost(device, rhs.data(), sizeof(float) * count);
	if (!addInput.isValid() || !addOutput.isValid() ||
	    !launchProjectionAdd(device, library, commandQueue, addInput, addOutput,
	        count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		expected[i] = rhs[i] + lhs[i];
	}
	if (!readBuffer(addOutput, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	Buffer scalarOutput =
	    Buffer::copyFromHost(device, rhs.data(), sizeof(float) * count);
	const float scalar = -1.75f;
	if (!scalarOutput.isValid() ||
	    !launchProjectionMultiplyScalar(device, library, commandQueue,
	        scalarOutput, scalar, count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		expected[i] = rhs[i] * scalar;
	}
	if (!readBuffer(scalarOutput, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	Buffer multiplyInput =
	    Buffer::copyFromHost(device, lhs.data(), sizeof(float) * count);
	Buffer multiplyOutput =
	    Buffer::copyFromHost(device, rhs.data(), sizeof(float) * count);
	if (!multiplyInput.isValid() || !multiplyOutput.isValid() ||
	    !launchProjectionMultiplyElementwise(device, library, commandQueue,
	        multiplyInput, multiplyOutput, count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		expected[i] = rhs[i] * lhs[i];
	}
	if (!readBuffer(multiplyOutput, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	Buffer divideMeasurements =
	    Buffer::copyFromHost(device, lhs.data(), sizeof(float) * count);
	Buffer divideOutput =
	    Buffer::copyFromHost(device, rhs.data(), sizeof(float) * count);
	if (!divideMeasurements.isValid() || !divideOutput.isValid() ||
	    !launchProjectionDivideMeasurements(device, library, commandQueue,
	        divideMeasurements, divideOutput, count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		expected[i] = rhs[i] != 0.0f ? lhs[i] / rhs[i] : rhs[i];
	}
	if (!readBuffer(divideOutput, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	Buffer invertInput =
	    Buffer::copyFromHost(device, lhs.data(), sizeof(float) * count);
	Buffer invertOutput = Buffer::allocate(device, sizeof(float) * count);
	if (!invertInput.isValid() || !invertOutput.isValid() ||
	    !launchProjectionInvert(device, library, commandQueue, invertInput,
	        invertOutput, count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		expected[i] = lhs[i] != 0.0f ? 1.0f / lhs[i] : 0.0f;
	}
	if (!readBuffer(invertOutput, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	Buffer acfInput =
	    Buffer::copyFromHost(device, lhs.data(), sizeof(float) * count);
	Buffer acfOutput = Buffer::allocate(device, sizeof(float) * count);
	const float unitFactor = 0.1f;
	if (!acfInput.isValid() || !acfOutput.isValid() ||
	    !launchProjectionConvertToACF(device, library, commandQueue, acfInput,
	        acfOutput, unitFactor, count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		expected[i] = std::exp(-lhs[i] * unitFactor);
	}
	if (!readBuffer(acfOutput, actual) || !valuesMatch(actual, expected))
	{
		return false;
	}

	std::vector<float> estimates = {
	    1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 0.5f, -2.0f, 3.0f,
	    6.0f, 9.0f, 12.0f, 15.0f, -4.0f, 5.0f, 7.0f, 11.0f, 13.0f};
	const std::vector<float> measurements = {
	    2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 1.0f, 3.0f, 6.0f,
	    9.0f, 12.0f, 15.0f, 18.0f, 21.0f, 24.0f, 27.0f, 30.0f, 33.0f};
	const std::vector<float> multiplicative = {
	    1.0f, 0.5f, 2.0f, 0.25f, 1.5f, 3.0f, -1.0f, 0.75f,
	    1.25f, 2.5f, 0.125f, 4.0f, 1.0f, -0.5f, 0.8f, 1.2f, 2.0f};
	const std::vector<float> additive = {
	    0.0f, 1.0f, -2.0f, 0.5f, 4.0f, -1.0f, 0.0f, 2.0f,
	    -3.0f, 1.5f, 0.25f, -4.0f, 4.0f, 2.5f, -5.0f, 6.0f, -10.0f};
	const std::vector<float> inVivo = {
	    1.0f, 2.0f, 0.5f, 1.5f, 0.75f, 1.25f, 2.0f, 0.25f,
	    1.1f, 0.9f, 1.3f, 0.7f, 1.4f, 0.6f, 1.8f, 0.8f, 1.0f};
	estimates[12] = -4.0f;
	Buffer compactEstimate =
	    Buffer::copyFromHost(device, estimates.data(), sizeof(float) * count);
	Buffer compactMeasurements = Buffer::copyFromHost(
	    device, measurements.data(), sizeof(float) * count);
	Buffer compactMultiplicative = Buffer::copyFromHost(
	    device, multiplicative.data(), sizeof(float) * count);
	Buffer compactAdditive =
	    Buffer::copyFromHost(device, additive.data(), sizeof(float) * count);
	Buffer compactInVivo =
	    Buffer::copyFromHost(device, inVivo.data(), sizeof(float) * count);
	const ProjectionCompactOsemRatioParams compactParams{0.05f, 1u};
	if (!compactEstimate.isValid() || !compactMeasurements.isValid() ||
	    !compactMultiplicative.isValid() || !compactAdditive.isValid() ||
	    !compactInVivo.isValid() ||
	    !launchProjectionCompactOsemRatio(device, library, commandQueue,
	        compactEstimate, compactMeasurements, compactMultiplicative,
	        compactAdditive, compactInVivo, compactParams, count))
	{
		return false;
	}
	for (std::size_t i = 0; i < count; ++i)
	{
		float estimate = estimates[i] * multiplicative[i] + additive[i];
		estimate *= inVivo[i];
		expected[i] = std::fabs(estimate) > compactParams.denomThreshold ?
		                  measurements[i] / estimate * multiplicative[i] :
		                  0.0f;
	}
	if (!readBuffer(compactEstimate, actual) ||
	    !valuesMatch(actual, expected))
	{
		return false;
	}

	return true;
}

bool runImageScalarOpsGoldenTests()
{
	const Device device = Device::createSystemDefault();
	if (!device.isValid())
	{
		return false;
	}

	const Library library =
	    Library::loadFromFile(device, YRTPET_METAL_SMOKE_METALLIB);
	const CommandQueue commandQueue = CommandQueue::create(device);
	if (!library.isValid() || !commandQueue.isValid())
	{
		return false;
	}

	const ImageShape shape3D{3, 2, 2, 1};
	const ImageShape shape4D{3, 2, 2, 2};
	const std::size_t spatialCount = shape3D.spatialVoxelCount();
	const std::size_t voxelCount4D = shape4D.voxelCount();

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

	std::vector<float> image4D(voxelCount4D);
	std::vector<float> update4D(voxelCount4D);
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		image4D[i] = static_cast<float>(i) * 0.5f - 3.0f;
		update4D[i] = 0.75f + static_cast<float>(i) * 0.25f;
	}

	std::vector<float> actual3D(spatialCount);
	std::vector<float> expected3D(spatialCount);
	std::vector<float> actual4D(voxelCount4D);
	std::vector<float> expected4D(voxelCount4D);

	Buffer fillBuffer =
	    Buffer::copyFromHost(device, image4D.data(), sizeof(float) * image4D.size());
	const float fillValue = -4.25f;
	if (!fillBuffer.isValid() ||
	    !launchImageFill(device, library, commandQueue, fillBuffer, shape4D,
	        fillValue))
	{
		return false;
	}
	std::fill(expected4D.begin(), expected4D.end(), fillValue);
	std::vector<float> fillActual(voxelCount4D);
	if (!readBuffer(fillBuffer, fillActual) || !valuesMatch(fillActual, expected4D))
	{
		return false;
	}

	Buffer scalarBuffer =
	    Buffer::copyFromHost(device, image4D.data(), sizeof(float) * image4D.size());
	const float scalar = -1.5f;
	if (!scalarBuffer.isValid() ||
	    !launchImageMultiplyScalar(device, library, commandQueue, scalarBuffer,
	        shape4D, scalar))
	{
		return false;
	}
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		expected4D[i] = image4D[i] * scalar;
	}
	if (!readBuffer(scalarBuffer, actual4D) ||
	    !valuesMatch(actual4D, expected4D))
	{
		return false;
	}

	Buffer add3DInput =
	    Buffer::copyFromHost(device, addend3D.data(), sizeof(float) * spatialCount);
	Buffer add3DOutput =
	    Buffer::copyFromHost(device, image3D.data(), sizeof(float) * spatialCount);
	if (!add3DInput.isValid() || !add3DOutput.isValid() ||
	    !launchImageAdd3DTo3D(device, library, commandQueue, add3DInput,
	        add3DOutput, shape3D))
	{
		return false;
	}
	for (std::size_t i = 0; i < spatialCount; ++i)
	{
		expected3D[i] = image3D[i] + addend3D[i];
	}
	if (!readBuffer(add3DOutput, actual3D) ||
	    !valuesMatch(actual3D, expected3D))
	{
		return false;
	}

	Buffer add4DInput =
	    Buffer::copyFromHost(device, addend3D.data(), sizeof(float) * spatialCount);
	Buffer add4DOutput =
	    Buffer::copyFromHost(device, image4D.data(), sizeof(float) * image4D.size());
	if (!add4DInput.isValid() || !add4DOutput.isValid() ||
	    !launchImageAdd3DTo4D(device, library, commandQueue, add4DInput,
	        add4DOutput, shape4D))
	{
		return false;
	}
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		expected4D[i] = image4D[i] + addend3D[i % spatialCount];
	}
	if (!readBuffer(add4DOutput, actual4D) ||
	    !valuesMatch(actual4D, expected4D))
	{
		return false;
	}

	const ImageThresholdParams thresholdParams{
	    0.5f, -0.5f, 2.0f, 1.5f, -3.0f};
	Buffer thresholdImage =
	    Buffer::copyFromHost(device, image3D.data(), sizeof(float) * spatialCount);
	Buffer thresholdMask =
	    Buffer::copyFromHost(device, mask3D.data(), sizeof(float) * spatialCount);
	if (!thresholdImage.isValid() || !thresholdMask.isValid() ||
	    !launchImageApplyThreshold(device, library, commandQueue, thresholdImage,
	        thresholdMask, shape3D, thresholdParams))
	{
		return false;
	}
	for (std::size_t i = 0; i < spatialCount; ++i)
	{
		expected3D[i] = mask3D[i] <= thresholdParams.threshold
		                  ? image3D[i] * thresholdParams.valLeScale +
		                        thresholdParams.valLeOffset
		                  : image3D[i] * thresholdParams.valGtScale +
		                        thresholdParams.valGtOffset;
	}
	if (!readBuffer(thresholdImage, actual3D) ||
	    !valuesMatch(actual3D, expected3D))
	{
		return false;
	}

	Buffer thresholdBroadcastImage =
	    Buffer::copyFromHost(device, image4D.data(), sizeof(float) * image4D.size());
	Buffer thresholdBroadcastMask =
	    Buffer::copyFromHost(device, mask3D.data(), sizeof(float) * spatialCount);
	if (!thresholdBroadcastImage.isValid() || !thresholdBroadcastMask.isValid() ||
	    !launchImageApplyThresholdBroadcast(device, library, commandQueue,
	        thresholdBroadcastImage, thresholdBroadcastMask, shape4D,
	        thresholdParams))
	{
		return false;
	}
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		const std::size_t j = i % spatialCount;
		expected4D[i] = mask3D[j] <= thresholdParams.threshold
		                  ? image4D[i] * thresholdParams.valLeScale +
		                        thresholdParams.valLeOffset
		                  : image4D[i] * thresholdParams.valGtScale +
		                        thresholdParams.valGtOffset;
	}
	if (!readBuffer(thresholdBroadcastImage, actual4D) ||
	    !valuesMatch(actual4D, expected4D))
	{
		return false;
	}

	const std::vector<float> update3D = {
	    1.0f, 2.0f, 4.0f, 0.5f, 8.0f, 2.0f,
	    3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f};
	const float emThreshold = 0.1f;
	Buffer staticUpdate =
	    Buffer::copyFromHost(device, update3D.data(), sizeof(float) * spatialCount);
	Buffer staticImage =
	    Buffer::copyFromHost(device, image3D.data(), sizeof(float) * spatialCount);
	Buffer staticSens = Buffer::copyFromHost(device, sensitivity3D.data(),
	    sizeof(float) * spatialCount);
	if (!staticUpdate.isValid() || !staticImage.isValid() ||
	    !staticSens.isValid() ||
	    !launchImageUpdateEMStatic(device, library, commandQueue, staticUpdate,
	        staticImage, staticSens, shape3D, emThreshold))
	{
		return false;
	}
	expected3D = image3D;
	for (std::size_t i = 0; i < spatialCount; ++i)
	{
		if (sensitivity3D[i] > emThreshold)
		{
			expected3D[i] *= update3D[i] / sensitivity3D[i];
		}
	}
	if (!readBuffer(staticImage, actual3D) ||
	    !valuesMatch(actual3D, expected3D))
	{
		return false;
	}

	Buffer dynamicUpdate =
	    Buffer::copyFromHost(device, update4D.data(), sizeof(float) * update4D.size());
	Buffer dynamicImage =
	    Buffer::copyFromHost(device, image4D.data(), sizeof(float) * image4D.size());
	Buffer dynamicSens = Buffer::copyFromHost(device, sensitivity3D.data(),
	    sizeof(float) * spatialCount);
	if (!dynamicUpdate.isValid() || !dynamicImage.isValid() ||
	    !dynamicSens.isValid() ||
	    !launchImageUpdateEMDynamic(device, library, commandQueue, dynamicUpdate,
	        dynamicImage, dynamicSens, shape4D, emThreshold))
	{
		return false;
	}
	expected4D = image4D;
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		const std::size_t j = i % spatialCount;
		if (sensitivity3D[j] > emThreshold)
		{
			expected4D[i] *= update4D[i] / sensitivity3D[j];
		}
	}
	if (!readBuffer(dynamicImage, actual4D) ||
	    !valuesMatch(actual4D, expected4D))
	{
		return false;
	}

	const std::vector<float> sensitivityScaling = {2.0f, 0.5f};
	Buffer scaledUpdate =
	    Buffer::copyFromHost(device, update4D.data(), sizeof(float) * update4D.size());
	Buffer scaledImage =
	    Buffer::copyFromHost(device, image4D.data(), sizeof(float) * image4D.size());
	Buffer scaledSens = Buffer::copyFromHost(device, sensitivity3D.data(),
	    sizeof(float) * spatialCount);
	Buffer scaledFactors = Buffer::copyFromHost(device, sensitivityScaling.data(),
	    sizeof(float) * sensitivityScaling.size());
	if (!scaledUpdate.isValid() || !scaledImage.isValid() ||
	    !scaledSens.isValid() || !scaledFactors.isValid() ||
	    !launchImageUpdateEMDynamicScaled(device, library, commandQueue,
	        scaledUpdate, scaledImage, scaledSens, scaledFactors, shape4D,
	        emThreshold))
	{
		return false;
	}
	expected4D = image4D;
	for (std::size_t i = 0; i < voxelCount4D; ++i)
	{
		const std::size_t frame = i / spatialCount;
		const std::size_t j = i % spatialCount;
		const float invScaling = 1.0f / sensitivityScaling[frame];
		const float threshold = emThreshold * invScaling;
		if (sensitivity3D[j] > threshold)
		{
			expected4D[i] *= (update4D[i] * invScaling) / sensitivity3D[j];
		}
	}
	if (!readBuffer(scaledImage, actual4D) ||
	    !valuesMatch(actual4D, expected4D))
	{
		return false;
	}

	return true;
}

bool runPsfConvolutionGoldenTests()
{
	const Device device = Device::createSystemDefault();
	if (!device.isValid())
	{
		return false;
	}

	const Library library =
	    Library::loadFromFile(device, YRTPET_METAL_SMOKE_METALLIB);
	const CommandQueue commandQueue = CommandQueue::create(device);
	if (!library.isValid() || !commandQueue.isValid())
	{
		return false;
	}

	const ImageShape shape{4, 3, 3, 1};
	std::vector<float> input(shape.voxelCount());
	for (std::size_t i = 0; i < input.size(); ++i)
	{
		input[i] = static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.5f +
		           static_cast<float>(i / 7) * 0.25f;
	}

	const std::vector<float> identity = {0.0f, 1.0f, 0.0f};
	const std::vector<std::vector<float>> kernelXs = {
	    {0.25f, -0.5f, 1.25f}, identity, identity,
	    {0.2f, 0.3f, 0.5f}};
	const std::vector<std::vector<float>> kernelYs = {
	    identity, {-0.75f, 0.5f, 1.25f}, identity,
	    {-0.1f, 0.8f, 0.3f}};
	const std::vector<std::vector<float>> kernelZs = {
	    identity, identity, {1.5f, -0.25f, -0.25f},
	    {0.6f, -0.2f, 0.6f}};

	for (std::size_t testId = 0; testId < kernelXs.size(); ++testId)
	{
		std::vector<float> metalOutput(input.size());
		if (!runMetalPsfConvolution(device, library, commandQueue, input, shape,
		        kernelXs[testId], kernelYs[testId], kernelZs[testId],
		        metalOutput))
		{
			return false;
		}

		const std::vector<float> referenceOutput = runOperatorPsfReference(
		    input, shape, kernelXs[testId], kernelYs[testId], kernelZs[testId]);
		if (!valuesMatch(metalOutput, referenceOutput))
		{
			return false;
		}
	}

	return true;
}

bool runImageMotionGoldenTests()
{
	Context context;
	if (!context.isValid())
	{
		return false;
	}

	yrt::ImageParams params(4, 3, 2, 4.0f, 3.0f, 2.0f);
	const std::size_t voxelCount = static_cast<std::size_t>(params.nx) *
	                               static_cast<std::size_t>(params.ny) *
	                               static_cast<std::size_t>(params.nz);
	yrt::ImageOwned inputImage(params);
	inputImage.allocate();
	for (std::size_t i = 0; i < voxelCount; ++i)
	{
		inputImage.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 13) - 4) * 0.25f +
		    static_cast<float>(i / 5) * 0.1f;
	}

	yrt::LORMotion motion(3);
	motion.setStartingTimestamp(0, 0);
	motion.setStartingTimestamp(1, 10);
	motion.setStartingTimestamp(2, 20);
	motion.setTransform(0,
	    yrt::transform_t{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
	        0.0f, 0.0f, 1.0f, 0.0f});
	motion.setTransform(1,
	    yrt::transform_t{1.0f, 0.0f, 0.0f, 0.35f, 0.0f, 1.0f, 0.0f,
	        -0.25f, 0.0f, 0.0f, 1.0f, 0.2f});
	motion.setTransform(2,
	    yrt::transform_t{0.9848077f, -0.1736482f, 0.0f, -0.15f, 0.1736482f,
	        0.9848077f, 0.0f, 0.1f, 0.0f, 0.0f, 1.0f, -0.3f});

	auto cpuReference =
	    yrt::util::timeAverageMoveImage<false>(motion, &inputImage,
	        static_cast<yrt::timestamp_t>(0),
	        static_cast<yrt::timestamp_t>(30));
	if (!cpuReference)
	{
		return false;
	}

	yrt::ImageOwned metalOutput(params);
	metalOutput.allocate();
	metalOutput.fill(0.0f);
	if (!timeAverageMoveImage(context, motion, inputImage, metalOutput,
	        static_cast<yrt::timestamp_t>(0),
	        static_cast<yrt::timestamp_t>(30)))
	{
		return false;
	}

	const float* cpu = cpuReference->getRawPointer();
	const float* metal = metalOutput.getRawPointer();
	for (std::size_t i = 0; i < voxelCount; ++i)
	{
		const float scale = std::max(1.0f, std::fabs(cpu[i]));
		if (std::fabs(metal[i] - cpu[i]) > 5.0e-5f * scale)
		{
			return false;
		}
	}

	return true;
}

}  // namespace yrt::backend::metal
