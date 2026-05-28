/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace yrt::backend::metal
{

class Device;
class Library;
class CommandQueue;
class Buffer;

struct BufferBinding
{
	const Buffer* buffer;
	std::size_t index;
};

struct BytesBinding
{
	const void* bytes;
	std::size_t byteCount;
	std::size_t index;
};

bool launchKernel1D(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const std::string& functionName,
    const std::vector<BufferBinding>& buffers,
    const std::vector<BytesBinding>& bytes, std::size_t valueCount);

class Device
{
public:
	Device();

	static Device createSystemDefault();

	bool isValid() const;

private:
	struct Impl;

	explicit Device(std::shared_ptr<Impl> pp_impl);

	std::shared_ptr<Impl> mp_impl;

	friend class Library;
	friend class CommandQueue;
	friend class Buffer;
	friend bool launchKernel1D(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const std::string& functionName,
	    const std::vector<BufferBinding>& buffers,
	    const std::vector<BytesBinding>& bytes, std::size_t valueCount);
	friend bool launchSmokeAddOne(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
	    std::size_t valueCount);
};

class Library
{
public:
	Library();

	static Library loadFromFile(const Device& device, const std::string& path);

	bool isValid() const;

private:
	struct Impl;

	explicit Library(std::shared_ptr<Impl> pp_impl);

	std::shared_ptr<Impl> mp_impl;

	friend bool launchKernel1D(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const std::string& functionName,
	    const std::vector<BufferBinding>& buffers,
	    const std::vector<BytesBinding>& bytes, std::size_t valueCount);
	friend bool launchSmokeAddOne(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
	    std::size_t valueCount);
};

class CommandQueue
{
public:
	CommandQueue();

	static CommandQueue create(const Device& device);

	bool isValid() const;

private:
	struct Impl;

	explicit CommandQueue(std::shared_ptr<Impl> pp_impl);

	std::shared_ptr<Impl> mp_impl;

	friend bool launchKernel1D(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const std::string& functionName,
	    const std::vector<BufferBinding>& buffers,
	    const std::vector<BytesBinding>& bytes, std::size_t valueCount);
	friend bool launchSmokeAddOne(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
	    std::size_t valueCount);
	friend class Buffer;
};

class Buffer
{
public:
	Buffer();

	static Buffer allocate(const Device& device, std::size_t byteCount);
	static Buffer allocatePrivate(const Device& device, std::size_t byteCount);
	static Buffer copyFromHost(const Device& device, const void* source,
	    std::size_t byteCount);

	bool isValid() const;
	bool isHostVisible() const;
	std::size_t byteCount() const;
	bool copyFromHost(const void* source, std::size_t byteCount);
	bool copyToHost(void* destination, std::size_t byteCount) const;
	bool copyToHost(const CommandQueue& commandQueue, void* destination,
	                std::size_t byteCount) const;

private:
	struct Impl;

	explicit Buffer(std::shared_ptr<Impl> pp_impl);

	std::shared_ptr<Impl> mp_impl;

	friend bool launchKernel1D(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const std::string& functionName,
	    const std::vector<BufferBinding>& buffers,
	    const std::vector<BytesBinding>& bytes, std::size_t valueCount);
	friend bool launchSmokeAddOne(const Device& device, const Library& library,
	    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
	    std::size_t valueCount);
};

bool isDeviceAvailable();
bool launchSmokeAddOne(const Device& device, const Library& library,
    const CommandQueue& commandQueue, const Buffer& input, Buffer& output,
    std::size_t valueCount);

}  // namespace yrt::backend::metal
