/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ImageSpaceKernels.hpp"

namespace yrt::backend::metal
{
namespace
{

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
	std::uint32_t kernelSize;
};

bool isShapeValid(const ImageShape& shape)
{
	return shape.nx > 0 && shape.ny > 0 && shape.nz > 0 && shape.nt > 0;
}

bool coversFloatCount(const Buffer& buffer, std::size_t count)
{
	return buffer.isValid() && buffer.byteCount() >= sizeof(float) * count;
}

ImageThresholdKernelParams makeThresholdParams(const ImageShape& shape,
    const ImageThresholdParams& params)
{
	return {shape, params.threshold, params.valLeScale, params.valLeOffset,
	    params.valGtScale, params.valGtOffset};
}

bool isKernelValid(std::uint32_t kernelSize, std::uint32_t axisSize)
{
	return kernelSize > 0 && kernelSize % 2 == 1 && kernelSize <= axisSize;
}

bool launchImageConvolve3DSeparable(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const std::string& functionName,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape, std::uint32_t axisSize)
{
	const auto count = shape.voxelCount();
	const ImageConvolutionKernelParams params{shape, kernelSize};
	return isShapeValid(shape) && isKernelValid(kernelSize, axisSize) &&
	       coversFloatCount(input, count) && coversFloatCount(output, count) &&
	       coversFloatCount(kernel, kernelSize) &&
	       launchKernel1D(device, library, commandQueue, functionName,
	           {{&input, 0}, {&output, 1}, {&kernel, 2}},
	           {{&params, sizeof(params), 3}}, count);
}

}  // namespace

std::size_t ImageShape::spatialVoxelCount() const
{
	return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
	       static_cast<std::size_t>(nz);
}

std::size_t ImageShape::voxelCount() const
{
	return spatialVoxelCount() * static_cast<std::size_t>(nt);
}

bool launchImageFill(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& image, const ImageShape& shape,
    float value)
{
	const auto count = shape.voxelCount();
	const ImageScalarKernelParams params{shape, value};
	return isShapeValid(shape) && coversFloatCount(image, count) &&
	       launchKernel1D(device, library, commandQueue, "image_fill",
	           {{&image, 0}}, {{&params, sizeof(params), 1}}, count);
}

bool launchImageMultiplyScalar(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& image, const ImageShape& shape,
    float scalar)
{
	const auto count = shape.voxelCount();
	const ImageScalarKernelParams params{shape, scalar};
	return isShapeValid(shape) && coversFloatCount(image, count) &&
	       launchKernel1D(device, library, commandQueue,
	           "image_multiply_scalar", {{&image, 0}},
	           {{&params, sizeof(params), 1}}, count);
}

bool launchImageAdd3DTo3D(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input3D, Buffer& output3D,
    const ImageShape& shape3D)
{
	const auto count = shape3D.spatialVoxelCount();
	return isShapeValid(shape3D) && coversFloatCount(input3D, count) &&
	       coversFloatCount(output3D, count) &&
	       launchKernel1D(device, library, commandQueue, "image_add_3d_to_3d",
	           {{&input3D, 0}, {&output3D, 1}}, {}, count);
}

bool launchImageAdd3DTo4D(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input3D, Buffer& output4D,
    const ImageShape& shape4D)
{
	const auto spatialCount = shape4D.spatialVoxelCount();
	const auto count = shape4D.voxelCount();
	return isShapeValid(shape4D) && coversFloatCount(input3D, spatialCount) &&
	       coversFloatCount(output4D, count) &&
	       launchKernel1D(device, library, commandQueue, "image_add_3d_to_4d",
	           {{&input3D, 0}, {&output4D, 1}},
	           {{&shape4D, sizeof(shape4D), 2}}, count);
}

bool launchImageApplyThreshold(const Device& device, const Library& library,
    const CommandQueue& commandQueue, Buffer& image, const Buffer& mask,
    const ImageShape& shape3D, const ImageThresholdParams& params)
{
	const auto count = shape3D.spatialVoxelCount();
	const auto kernelParams = makeThresholdParams(shape3D, params);
	return isShapeValid(shape3D) && coversFloatCount(image, count) &&
	       coversFloatCount(mask, count) &&
	       launchKernel1D(device, library, commandQueue,
	           "image_apply_threshold", {{&image, 0}, {&mask, 1}},
	           {{&kernelParams, sizeof(kernelParams), 2}}, count);
}

bool launchImageApplyThresholdBroadcast(const Device& device,
    const Library& library, const CommandQueue& commandQueue, Buffer& image4D,
    const Buffer& mask3D, const ImageShape& shape4D,
    const ImageThresholdParams& params)
{
	const auto spatialCount = shape4D.spatialVoxelCount();
	const auto count = shape4D.voxelCount();
	const auto kernelParams = makeThresholdParams(shape4D, params);
	return isShapeValid(shape4D) && coversFloatCount(image4D, count) &&
	       coversFloatCount(mask3D, spatialCount) &&
	       launchKernel1D(device, library, commandQueue,
	           "image_apply_threshold_broadcast", {{&image4D, 0}, {&mask3D, 1}},
	           {{&kernelParams, sizeof(kernelParams), 2}}, count);
}

bool launchImageUpdateEMStatic(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& update, Buffer& image,
    const Buffer& sensitivity, const ImageShape& shape, float threshold)
{
	const auto count = shape.voxelCount();
	const ImageEMKernelParams params{shape, threshold};
	return isShapeValid(shape) && coversFloatCount(update, count) &&
	       coversFloatCount(image, count) && coversFloatCount(sensitivity, count) &&
	       launchKernel1D(device, library, commandQueue, "image_update_em_static",
	           {{&update, 0}, {&image, 1}, {&sensitivity, 2}},
	           {{&params, sizeof(params), 3}}, count);
}

bool launchImageUpdateEMDynamic(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& update4D, Buffer& image4D,
    const Buffer& sensitivity3D, const ImageShape& shape4D, float threshold)
{
	const auto spatialCount = shape4D.spatialVoxelCount();
	const auto count = shape4D.voxelCount();
	const ImageEMKernelParams params{shape4D, threshold};
	return isShapeValid(shape4D) && coversFloatCount(update4D, count) &&
	       coversFloatCount(image4D, count) &&
	       coversFloatCount(sensitivity3D, spatialCount) &&
	       launchKernel1D(device, library, commandQueue, "image_update_em_dynamic",
	           {{&update4D, 0}, {&image4D, 1}, {&sensitivity3D, 2}},
	           {{&params, sizeof(params), 3}}, count);
}

bool launchImageUpdateEMDynamicScaled(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& update4D, Buffer& image4D, const Buffer& sensitivity3D,
    const Buffer& sensitivityScaling, const ImageShape& shape4D,
    float threshold)
{
	const auto spatialCount = shape4D.spatialVoxelCount();
	const auto count = shape4D.voxelCount();
	const ImageEMKernelParams params{shape4D, threshold};
	return isShapeValid(shape4D) && coversFloatCount(update4D, count) &&
	       coversFloatCount(image4D, count) &&
	       coversFloatCount(sensitivity3D, spatialCount) &&
	       coversFloatCount(sensitivityScaling, shape4D.nt) &&
	       launchKernel1D(device, library, commandQueue,
	           "image_update_em_dynamic_scaled",
	           {{&update4D, 0},
	               {&image4D, 1},
	               {&sensitivity3D, 2},
	               {&sensitivityScaling, 3}},
	           {{&params, sizeof(params), 4}}, count);
}

bool launchImageConvolve3DSeparableX(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape)
{
	return launchImageConvolve3DSeparable(device, library, commandQueue,
	    "image_convolve3d_separable_x", input, output, kernel, kernelSize,
	    shape, shape.nx);
}

bool launchImageConvolve3DSeparableY(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape)
{
	return launchImageConvolve3DSeparable(device, library, commandQueue,
	    "image_convolve3d_separable_y", input, output, kernel, kernelSize,
	    shape, shape.ny);
}

bool launchImageConvolve3DSeparableZ(const Device& device,
    const Library& library, const CommandQueue& commandQueue,
    const Buffer& input, Buffer& output, const Buffer& kernel,
    std::uint32_t kernelSize, const ImageShape& shape)
{
	return launchImageConvolve3DSeparable(device, library, commandQueue,
	    "image_convolve3d_separable_z", input, output, kernel, kernelSize,
	    shape, shape.nz);
}

}  // namespace yrt::backend::metal
