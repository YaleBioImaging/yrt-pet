/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ExperimentalBackend.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/image/ImageParams.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

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

std::size_t imageVoxelCount(const yrt::ImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

bool runProjectionVectorSample(
    const yrt::backend::metal::ExperimentalBackend& backend)
{
	const std::vector<float> input = {
	    -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, -8.0f, 16.0f};
	const std::vector<float> output = {
	    3.0f, -4.0f, 2.0f, 0.25f, -0.5f, 5.0f, -10.0f, 8.0f, 1.5f};

	auto values = backend.makeProjectionVector(output);
	if (!values.isValid() || !values.add(input))
	{
		return false;
	}

	std::vector<float> expected(output.size());
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		expected[i] = output[i] + input[i];
	}
	return valuesMatch(values.values(), expected);
}

bool runImageSample(const yrt::backend::metal::ExperimentalBackend& backend)
{
	yrt::ImageParams params(4, 3, 3, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f,
	                        2);
	yrt::ImageOwned input(params);
	yrt::ImageOwned expected(params);
	input.allocate();
	expected.allocate();

	const std::size_t count = imageVoxelCount(params);
	for (std::size_t i = 0; i < count; ++i)
	{
		input.getRawPointer()[i] =
		    static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.2f;
	}
	expected.copyFromImage(&input);

	const float scalar = -1.25f;
	auto image = backend.makeImage(input);
	expected.multWithScalar(scalar);
	return image.isValid() && image.multiplyByScalar(scalar) &&
	       imagesMatch(image.image(), expected);
}

bool runPsfSample(const yrt::backend::metal::ExperimentalBackend& backend)
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

	yrt::OperatorPsf cpuPsf(kernelX, kernelY, kernelZ);
	cpuPsf.applyA(&input, &cpuOutput);

	return backend.applyOperatorPsfForward(input, metalOutput, kernelX,
	           kernelY, kernelZ) &&
	       imagesMatch(metalOutput, cpuOutput);
}

}  // namespace

int main()
{
	const yrt::backend::metal::ExperimentalBackend backend;
	if (!backend.isAvailable())
	{
		std::cerr << "Metal experimental backend sample: FAIL "
		          << "(Metal device unavailable)\n";
		return 2;
	}
	if (!backend.isValid())
	{
		std::cerr << "Metal experimental backend sample: FAIL ("
		          << backend.errorMessage() << ")\n";
		return 1;
	}

	if (!runProjectionVectorSample(backend) || !runImageSample(backend) ||
	    !runPsfSample(backend))
	{
		std::cerr << "Metal experimental backend sample: FAIL\n";
		return 1;
	}

	std::cout << "Metal experimental backend sample: PASS\n";
	return 0;
}
