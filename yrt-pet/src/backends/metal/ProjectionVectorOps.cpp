/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/ProjectionVectorOps.hpp"

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorKernels.hpp"

#include <cstddef>

namespace yrt::backend::metal
{
namespace
{

bool hasValues(const std::vector<float>& values)
{
	return !values.empty();
}

bool sameSize(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
	return lhs.size() == rhs.size();
}

std::size_t byteCount(const std::vector<float>& values)
{
	return sizeof(float) * values.size();
}

bool copyBufferToVector(const Buffer& buffer, std::vector<float>& values)
{
	return buffer.copyToHost(values.data(), byteCount(values));
}

bool canUseUnaryOutput(const Context& context, const std::vector<float>& values)
{
	return context.isValid() && hasValues(values);
}

bool canUseBinaryOutput(const Context& context, const std::vector<float>& input,
                        const std::vector<float>& output)
{
	return context.isValid() && hasValues(input) && sameSize(input, output);
}

}  // namespace

bool clear(const Context& context, std::vector<float>& values, float value)
{
	if (!canUseUnaryOutput(context, values))
	{
		return false;
	}

	Buffer valuesBuffer = Buffer::allocate(context.device(), byteCount(values));
	if (!valuesBuffer.isValid() ||
	    !launchProjectionClear(context.device(), context.library(),
	        context.commandQueue(), valuesBuffer, value, values.size()))
	{
		return false;
	}
	return copyBufferToVector(valuesBuffer, values);
}

bool add(const Context& context, const std::vector<float>& input,
         std::vector<float>& output)
{
	if (!canUseBinaryOutput(context, input, output))
	{
		return false;
	}

	Buffer inputBuffer =
	    Buffer::copyFromHost(context.device(), input.data(), byteCount(input));
	Buffer outputBuffer =
	    Buffer::copyFromHost(context.device(), output.data(), byteCount(output));
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchProjectionAdd(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, output.size()))
	{
		return false;
	}
	return copyBufferToVector(outputBuffer, output);
}

bool multiplyByScalar(const Context& context, std::vector<float>& values,
                      float scalar)
{
	if (!canUseUnaryOutput(context, values))
	{
		return false;
	}

	Buffer valuesBuffer =
	    Buffer::copyFromHost(context.device(), values.data(), byteCount(values));
	if (!valuesBuffer.isValid() ||
	    !launchProjectionMultiplyScalar(context.device(), context.library(),
	        context.commandQueue(), valuesBuffer, scalar, values.size()))
	{
		return false;
	}
	return copyBufferToVector(valuesBuffer, values);
}

bool multiplyElementwise(const Context& context,
                         const std::vector<float>& input,
                         std::vector<float>& output)
{
	if (!canUseBinaryOutput(context, input, output))
	{
		return false;
	}

	Buffer inputBuffer =
	    Buffer::copyFromHost(context.device(), input.data(), byteCount(input));
	Buffer outputBuffer =
	    Buffer::copyFromHost(context.device(), output.data(), byteCount(output));
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchProjectionMultiplyElementwise(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, output.size()))
	{
		return false;
	}
	return copyBufferToVector(outputBuffer, output);
}

bool divideMeasurements(const Context& context,
                        const std::vector<float>& measurements,
                        std::vector<float>& output)
{
	if (!canUseBinaryOutput(context, measurements, output))
	{
		return false;
	}

	Buffer measurementsBuffer = Buffer::copyFromHost(
	    context.device(), measurements.data(), byteCount(measurements));
	Buffer outputBuffer =
	    Buffer::copyFromHost(context.device(), output.data(), byteCount(output));
	if (!measurementsBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchProjectionDivideMeasurements(context.device(), context.library(),
	        context.commandQueue(), measurementsBuffer, outputBuffer,
	        output.size()))
	{
		return false;
	}
	return copyBufferToVector(outputBuffer, output);
}

bool invert(const Context& context, const std::vector<float>& input,
            std::vector<float>& output)
{
	if (!canUseBinaryOutput(context, input, output))
	{
		return false;
	}

	Buffer inputBuffer =
	    Buffer::copyFromHost(context.device(), input.data(), byteCount(input));
	Buffer outputBuffer = Buffer::allocate(context.device(), byteCount(output));
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchProjectionInvert(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, output.size()))
	{
		return false;
	}
	return copyBufferToVector(outputBuffer, output);
}

bool convertToACF(const Context& context, const std::vector<float>& input,
                  std::vector<float>& output, float unitFactor)
{
	if (!canUseBinaryOutput(context, input, output))
	{
		return false;
	}

	Buffer inputBuffer =
	    Buffer::copyFromHost(context.device(), input.data(), byteCount(input));
	Buffer outputBuffer = Buffer::allocate(context.device(), byteCount(output));
	if (!inputBuffer.isValid() || !outputBuffer.isValid() ||
	    !launchProjectionConvertToACF(context.device(), context.library(),
	        context.commandQueue(), inputBuffer, outputBuffer, unitFactor,
	        output.size()))
	{
		return false;
	}
	return copyBufferToVector(outputBuffer, output);
}

}  // namespace yrt::backend::metal
