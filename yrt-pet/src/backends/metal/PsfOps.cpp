/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/PsfOps.hpp"

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"
#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"

#include <cstdint>
#include <limits>

namespace yrt::backend::metal
{
namespace
{

bool fitsUint32(std::int64_t value)
{
	return value > 0 &&
	       static_cast<std::uint64_t>(value) <=
	           std::numeric_limits<std::uint32_t>::max();
}

bool isKernelSupported(const std::vector<float>& kernel, int axisSize)
{
	return axisSize > 0 && !kernel.empty() && kernel.size() % 2 == 1 &&
	       kernel.size() <= static_cast<std::size_t>(axisSize) &&
	       kernel.size() <= std::numeric_limits<std::uint32_t>::max();
}

ImageShape makeShape(const ImageParams& params)
{
	return {static_cast<std::uint32_t>(params.nx),
	    static_cast<std::uint32_t>(params.ny),
	    static_cast<std::uint32_t>(params.nz),
	    static_cast<std::uint32_t>(params.nt)};
}

bool canUseHostImages(const Image& input, const Image& output)
{
	const ImageParams& inputParams = input.getParams();
	const ImageParams& outputParams = output.getParams();
	return input.isMemoryValid() && output.isMemoryValid() &&
	       inputParams.isSameDimensionsAs(outputParams) &&
	       inputParams.isSameNumFramesAs(outputParams) &&
	       fitsUint32(inputParams.nx) && fitsUint32(inputParams.ny) &&
	       fitsUint32(inputParams.nz) && fitsUint32(inputParams.nt);
}

}  // namespace

bool convolve3DSeparableHost(const Context& context, const Image& input,
                             Image& output,
                             const std::vector<float>& kernelX,
                             const std::vector<float>& kernelY,
                             const std::vector<float>& kernelZ)
{
	if (!context.isValid())
	{
		return false;
	}
	if (!canUseHostImages(input, output))
	{
		return false;
	}

	const ImageParams& params = input.getParams();
	if (!isKernelSupported(kernelX, params.nx) ||
	    !isKernelSupported(kernelY, params.ny) ||
	    !isKernelSupported(kernelZ, params.nz))
	{
		return false;
	}

	const ImageShape shape = makeShape(params);
	const std::size_t byteCount = sizeof(float) * shape.voxelCount();
	const Device& device = context.device();
	const Library& library = context.library();
	const CommandQueue& commandQueue = context.commandQueue();

	Buffer inputBuffer =
	    Buffer::copyFromHost(device, input.getRawPointer(), byteCount);
	Buffer tempA = Buffer::allocate(device, byteCount);
	Buffer tempB = Buffer::allocate(device, byteCount);
	Buffer kernelXBuffer = Buffer::copyFromHost(
	    device, kernelX.data(), sizeof(float) * kernelX.size());
	Buffer kernelYBuffer = Buffer::copyFromHost(
	    device, kernelY.data(), sizeof(float) * kernelY.size());
	Buffer kernelZBuffer = Buffer::copyFromHost(
	    device, kernelZ.data(), sizeof(float) * kernelZ.size());
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

	return tempA.copyToHost(output.getRawPointer(), byteCount);
}

bool convolve3DSeparableHost(const Image& input, Image& output,
                             const std::vector<float>& kernelX,
                             const std::vector<float>& kernelY,
                             const std::vector<float>& kernelZ)
{
	const Context context;
	return convolve3DSeparableHost(context, input, output, kernelX, kernelY,
	    kernelZ);
}

}  // namespace yrt::backend::metal
