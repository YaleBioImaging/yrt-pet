/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>

namespace yrt::backend::metal
{

struct ImageShape
{
	std::uint32_t nx;
	std::uint32_t ny;
	std::uint32_t nz;
	std::uint32_t nt;

	std::size_t spatialVoxelCount() const;
	std::size_t voxelCount() const;
};

struct ImageThresholdParams
{
	float threshold;
	float valLeScale;
	float valLeOffset;
	float valGtScale;
	float valGtOffset;
};

bool launchImageFill(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& image, const ImageShape& shape,
    float value);
bool launchImageMultiplyScalar(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& image, const ImageShape& shape,
    float scalar);
bool launchImageAdd3DTo3D(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input3D, Buffer& output3D,
    const ImageShape& shape3D);
bool launchImageAdd3DTo4D(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input3D, Buffer& output4D,
    const ImageShape& shape4D);
bool launchImageApplyThreshold(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& image, const Buffer& mask,
    const ImageShape& shape3D, const ImageThresholdParams& params);
bool launchImageApplyThresholdBroadcast(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image4D,
    const Buffer& mask3D, const ImageShape& shape4D,
    const ImageThresholdParams& params);
bool launchImageUpdateEMStatic(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& update, Buffer& image,
    const Buffer& sensitivity, const ImageShape& shape, float threshold);
bool launchImageUpdateEMDynamic(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& update4D, Buffer& image4D,
    const Buffer& sensitivity3D, const ImageShape& shape4D, float threshold);
bool launchImageUpdateEMDynamicScaled(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& update4D, Buffer& image4D, const Buffer& sensitivity3D,
    const Buffer& sensitivityScaling, const ImageShape& shape4D,
    float threshold);
bool launchImageConvolve3DSeparableX(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape);
bool launchImageConvolve3DSeparableY(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape);
bool launchImageConvolve3DSeparableZ(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape);

}  // namespace yrt::backend::metal
