/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define METALCPP_SYMBOL_VISIBILITY_HIDDEN
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <Metal.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "yrt-pet/backends/metal/MetalBackend.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace yrt::backend::metal
{
namespace
{

using AutoreleasePoolPtr = NS::SharedPtr<NS::AutoreleasePool>;

AutoreleasePoolPtr makeAutoreleasePool()
{
	return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

MTL::ResourceOptions sharedResourceOptions()
{
	return static_cast<MTL::ResourceOptions>(MTL::ResourceStorageModeShared);
}

NS::UInteger toUInteger(std::size_t value)
{
	return static_cast<NS::UInteger>(value);
}

}  // namespace

struct Device::Impl
{
	explicit Impl(NS::SharedPtr<MTL::Device> pp_device)
	    : p_device(std::move(pp_device))
	{
	}

	NS::SharedPtr<MTL::Device> p_device;
};

struct Library::Impl
{
	explicit Impl(NS::SharedPtr<MTL::Library> pp_library)
	    : p_library(std::move(pp_library))
	{
	}

	NS::SharedPtr<MTL::Library> p_library;
};

struct CommandQueue::Impl
{
	explicit Impl(NS::SharedPtr<MTL::CommandQueue> pp_commandQueue)
	    : p_commandQueue(std::move(pp_commandQueue))
	{
	}

	NS::SharedPtr<MTL::CommandQueue> p_commandQueue;
};

struct Buffer::Impl
{
	Impl(NS::SharedPtr<MTL::Buffer> pp_buffer, std::size_t p_byteCount)
	    : p_buffer(std::move(pp_buffer)), byteCount(p_byteCount)
	{
	}

	NS::SharedPtr<MTL::Buffer> p_buffer;
	std::size_t byteCount;
};

Device::Device() = default;

Device::Device(std::shared_ptr<Impl> pp_impl) : mp_impl(std::move(pp_impl)) {}

Device Device::createSystemDefault()
{
	const auto pool = makeAutoreleasePool();
	(void)pool;

	auto p_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!p_device)
	{
		return Device();
	}

	return Device(std::make_shared<Impl>(std::move(p_device)));
}

bool Device::isValid() const
{
	return mp_impl && mp_impl->p_device;
}

Library::Library() = default;

Library::Library(std::shared_ptr<Impl> pp_impl) : mp_impl(std::move(pp_impl)) {}

Library Library::loadFromFile(const Device& device, const std::string& path)
{
	if (!device.isValid() || path.empty())
	{
		return Library();
	}

	const auto pool = makeAutoreleasePool();
	(void)pool;

	auto p_path =
	    NS::TransferPtr(NS::String::alloc()->init(path.c_str(), NS::UTF8StringEncoding));
	if (!p_path)
	{
		return Library();
	}

	NS::Error* p_error = nullptr;
	auto p_library =
	    NS::TransferPtr(device.mp_impl->p_device->newLibrary(p_path.get(), &p_error));
	if (!p_library)
	{
		return Library();
	}

	return Library(std::make_shared<Impl>(std::move(p_library)));
}

bool Library::isValid() const
{
	return mp_impl && mp_impl->p_library;
}

CommandQueue::CommandQueue() = default;

CommandQueue::CommandQueue(std::shared_ptr<Impl> pp_impl)
    : mp_impl(std::move(pp_impl))
{
}

CommandQueue CommandQueue::create(const Device& device)
{
	if (!device.isValid())
	{
		return CommandQueue();
	}

	auto p_commandQueue =
	    NS::TransferPtr(device.mp_impl->p_device->newCommandQueue());
	if (!p_commandQueue)
	{
		return CommandQueue();
	}

	return CommandQueue(std::make_shared<Impl>(std::move(p_commandQueue)));
}

bool CommandQueue::isValid() const
{
	return mp_impl && mp_impl->p_commandQueue;
}

Buffer::Buffer() = default;

Buffer::Buffer(std::shared_ptr<Impl> pp_impl) : mp_impl(std::move(pp_impl)) {}

Buffer Buffer::allocate(const Device& device, std::size_t byteCount)
{
	if (!device.isValid() || byteCount == 0)
	{
		return Buffer();
	}

	auto p_buffer = NS::TransferPtr(
	    device.mp_impl->p_device->newBuffer(toUInteger(byteCount),
	        sharedResourceOptions()));
	if (!p_buffer)
	{
		return Buffer();
	}

	return Buffer(std::make_shared<Impl>(std::move(p_buffer), byteCount));
}

Buffer Buffer::copyFromHost(const Device& device, const void* source,
    std::size_t byteCount)
{
	if (source == nullptr || byteCount == 0 || !device.isValid())
	{
		return Buffer();
	}

	auto p_buffer = NS::TransferPtr(
	    device.mp_impl->p_device->newBuffer(source, toUInteger(byteCount),
	        sharedResourceOptions()));
	if (!p_buffer)
	{
		return Buffer();
	}

	return Buffer(std::make_shared<Impl>(std::move(p_buffer), byteCount));
}

bool Buffer::isValid() const
{
	return mp_impl && mp_impl->p_buffer;
}

std::size_t Buffer::byteCount() const
{
	return isValid() ? mp_impl->byteCount : 0;
}

bool Buffer::copyFromHost(const void* source, std::size_t p_byteCount)
{
	if (!isValid() || source == nullptr || p_byteCount > mp_impl->byteCount)
	{
		return false;
	}

	std::memcpy(mp_impl->p_buffer->contents(), source, p_byteCount);
	return true;
}

bool Buffer::copyToHost(void* destination, std::size_t p_byteCount) const
{
	if (!isValid() || destination == nullptr || p_byteCount > mp_impl->byteCount)
	{
		return false;
	}

	std::memcpy(destination, mp_impl->p_buffer->contents(), p_byteCount);
	return true;
}

bool isDeviceAvailable()
{
	return Device::createSystemDefault().isValid();
}

bool launchSmokeAddOne(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    std::size_t valueCount)
{
	return launchKernel1D(device, library, commandQueue, "smoke_add_one",
	    {{&input, 0}, {&output, 1}}, {}, valueCount);
}

bool launchKernel1D(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const std::string& functionName,
    const std::vector<BufferBinding>& buffers,
    const std::vector<BytesBinding>& bytes, std::size_t valueCount)
{
	if (!device.isValid() || !library.isValid() || !commandQueue.isValid() ||
	    functionName.empty())
	{
		return false;
	}
	if (valueCount == 0)
	{
		return true;
	}

	for (const BufferBinding& buffer : buffers)
	{
		if (buffer.buffer == nullptr || !buffer.buffer->isValid())
		{
			return false;
		}
	}
	for (const BytesBinding& byteBinding : bytes)
	{
		if (byteBinding.bytes == nullptr || byteBinding.byteCount == 0)
		{
			return false;
		}
	}

	const auto pool = makeAutoreleasePool();
	(void)pool;

	auto p_functionName = NS::TransferPtr(
	    NS::String::alloc()->init(functionName.c_str(), NS::UTF8StringEncoding));
	if (!p_functionName)
	{
		return false;
	}

	auto p_function =
	    NS::TransferPtr(library.mp_impl->p_library->newFunction(p_functionName.get()));
	if (!p_function)
	{
		return false;
	}

	NS::Error* p_error = nullptr;
	auto p_pipeline = NS::TransferPtr(
	    device.mp_impl->p_device->newComputePipelineState(p_function.get(),
	        &p_error));
	if (!p_pipeline)
	{
		return false;
	}

	MTL::CommandBuffer* p_rawCommandBuffer =
	    commandQueue.mp_impl->p_commandQueue->commandBuffer();
	if (p_rawCommandBuffer == nullptr)
	{
		return false;
	}
	auto p_commandBuffer = NS::RetainPtr(p_rawCommandBuffer);

	MTL::ComputeCommandEncoder* p_rawEncoder =
	    p_commandBuffer->computeCommandEncoder();
	if (p_rawEncoder == nullptr)
	{
		return false;
	}
	auto p_encoder = NS::RetainPtr(p_rawEncoder);

	p_encoder->setComputePipelineState(p_pipeline.get());
	for (const BufferBinding& buffer : buffers)
	{
		p_encoder->setBuffer(buffer.buffer->mp_impl->p_buffer.get(), 0,
		    toUInteger(buffer.index));
	}
	for (const BytesBinding& byteBinding : bytes)
	{
		p_encoder->setBytes(byteBinding.bytes, toUInteger(byteBinding.byteCount),
		    toUInteger(byteBinding.index));
	}

	const auto gridSize = MTL::Size::Make(toUInteger(valueCount), 1, 1);
	const auto threadExecutionWidth = p_pipeline->threadExecutionWidth();
	const auto threadsPerGroup =
	    std::max<NS::UInteger>(1,
	        std::min<NS::UInteger>(threadExecutionWidth, toUInteger(valueCount)));
	const auto threadgroupSize = MTL::Size::Make(threadsPerGroup, 1, 1);

	p_encoder->dispatchThreads(gridSize, threadgroupSize);
	p_encoder->endEncoding();

	p_commandBuffer->commit();
	p_commandBuffer->waitUntilCompleted();

	return p_commandBuffer->status() == MTL::CommandBufferStatusCompleted;
}

}  // namespace yrt::backend::metal
