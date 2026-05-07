/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/utils/GPUUtils.cuh"

#include "yrt-pet/utils/Assert.hpp"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace yrt
{

bool cudaCheckError()
{
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != 0)
	{
		std::cerr << "CUDA Error: " << cudaGetErrorString(cudaError)
		          << std::endl;
		return false;
	}
	return true;
}

size_t getDeviceInfo(bool verbose)
{
	selectDeviceWithMostFreeMemory(verbose);
	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	cudaCheckError();
	return freeMem;
}

size_t getCurrentDeviceInfo(bool verbose)
{
	int devicesNb = 0;
	cudaGetDeviceCount(&devicesNb);
	cudaCheckError();
	ASSERT_MSG(devicesNb > 0, "No CUDA device detected");

	int currentDevice = 0;
	cudaGetDevice(&currentDevice);
	cudaCheckError();

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, currentDevice);
	cudaCheckError();

	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	cudaCheckError();

	if (verbose)
	{
		std::cout << "\n"
		          << "*** GPU INFORMATION ***"
		          << "\n"
		          << std::endl;
		std::cout << "Current device id: " << currentDevice << std::endl;
		std::cout << "Device name: " << deviceProp.name << std::endl;
		std::cout << "Compute capability: " << deviceProp.major << "."
		          << deviceProp.minor << std::endl;
		std::cout << "Number of asynchronous engines: "
		          << deviceProp.asyncEngineCount << std::endl;
		std::cout << "Device memory - Total memory: "
		          << totalMem / static_cast<double>(1024 * 1024 * 1024)
		          << "GB - Available memory: "
		          << freeMem / static_cast<double>(1024 * 1024 * 1024)
		          << "GB \n"
		          << std::endl;
	}

	return freeMem;
}

int selectDeviceWithMostFreeMemory(bool verbose)
{
	int devicesNb = 0;
	cudaGetDeviceCount(&devicesNb);
	cudaCheckError();
	ASSERT_MSG(devicesNb > 0, "No CUDA device detected");

	std::cout << "\n"
	          << "*** GPUs INFORMATION ***"
	          << "\n"
	          << std::endl;
	std::cout << "Number of devices detected: " << devicesNb << std::endl;
	size_t freeMem = 0;
	size_t totalMem = 0;
	int gpu_id_toUse = 0;
	size_t maxDeviceMem = 0;
	for (int d_id = 0; d_id < devicesNb; d_id++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, d_id);
		cudaSetDevice(d_id);
		cudaMemGetInfo(&freeMem, &totalMem);
		if (verbose)
		{
			std::cout << "Device name: " << deviceProp.name << std::endl;
			std::cout << "Compute capability: " << deviceProp.major << "."
			          << deviceProp.minor << std::endl;
			std::cout << "Number of asynchronous engines: "
			          << deviceProp.asyncEngineCount << std::endl;
			std::cout << "Device memory - Total memory: "
			          << totalMem / static_cast<double>(1024 * 1024 * 1024)
			          << "GB - Available memory: "
			          << freeMem / static_cast<double>(1024 * 1024 * 1024)
			          << "GB \n"
			          << std::endl;
		}
		if (freeMem > maxDeviceMem)
		{
			maxDeviceMem = freeMem;
			gpu_id_toUse = d_id;
		}
	}
	std::cout << "Selected device id: " << gpu_id_toUse << "\n" << std::endl;
	setCUDADevice(gpu_id_toUse);
	return gpu_id_toUse;
}

void setCUDADevice(int deviceId)
{
	int devicesNb = 0;
	cudaGetDeviceCount(&devicesNb);
	cudaCheckError();
	if (deviceId < 0 || deviceId >= devicesNb)
	{
		throw std::runtime_error("CUDA device id " + std::to_string(deviceId) +
		                         " is out of range for " +
		                         std::to_string(devicesNb) +
		                         " visible device(s)");
	}
	cudaSetDevice(deviceId);
	ASSERT(cudaCheckError());
}

std::vector<int> parseGPUDeviceIds(const std::string& deviceList)
{
	std::vector<int> deviceIds;
	std::stringstream ss(deviceList);
	std::string token;
	while (std::getline(ss, token, ','))
	{
		if (token.empty())
		{
			continue;
		}
		deviceIds.push_back(std::stoi(token));
	}
	return deviceIds;
}

std::vector<int> getDefaultGPUDeviceIds()
{
	const char* envDeviceIds = std::getenv("YRT_CUDA_DEVICES");
	if (envDeviceIds != nullptr && envDeviceIds[0] != '\0')
	{
		return parseGPUDeviceIds(envDeviceIds);
	}
	return {selectDeviceWithMostFreeMemory(true)};
}

void gpuAssert(cudaError_t code, const char* file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
		        line);
		if (abort)
			exit(code);
	}
}

void gpuErrchk(cudaError_t code)
{
	gpuAssert(code, __FILE__, __LINE__);
}


}  // namespace yrt
