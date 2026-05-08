/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorDevice.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/operators/OperatorProjectorDD_GPU.cuh"
#include "yrt-pet/operators/OperatorProjectorSiddon_GPU.cuh"
#include "yrt-pet/utils/GPUUtils.cuh"
#include "yrt-pet/utils/GPUStream.cuh"
#include "yrt-pet/utils/Globals.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_operatorprojectordevice(py::module& m)
{
	auto c = py::class_<OperatorProjectorDevice, OperatorProjectorBase>(
	    m, "OperatorProjectorDevice");
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const ImageDevice* img,
	       ProjectionData* proj)
	    {
		    py::gil_scoped_release release;
		    self.applyA(img, proj);
	    },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const Image* img,
	       ProjectionData* proj)
	    {
		    py::gil_scoped_release release;
		    self.applyA(img, proj);
	    },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const ImageDevice* img,
	       ProjectionDataDevice* proj)
	    {
		    py::gil_scoped_release release;
		    self.applyA(img, proj);
	    },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const Image* img,
	       ProjectionDataDevice* proj)
	    {
		    py::gil_scoped_release release;
		    self.applyA(img, proj);
	    },
	    py::arg("img"), py::arg("proj"));

	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionData* proj,
	       Image* img)
	    {
		    py::gil_scoped_release release;
		    self.applyAH(proj, img);
	    },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionData* proj,
	       ImageDevice* img)
	    {
		    py::gil_scoped_release release;
		    self.applyAH(proj, img);
	    },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionDataDevice* proj,
	       Image* img)
	    {
		    py::gil_scoped_release release;
		    self.applyAH(proj, img);
	    },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionDataDevice* proj,
	       ImageDevice* img)
	    {
		    py::gil_scoped_release release;
		    self.applyAH(proj, img);
	    },
	    py::arg("proj"), py::arg("img"));
}
}  // namespace yrt

#endif

namespace yrt
{
namespace
{
class ScopedCUDADevice
{
public:
	explicit ScopedCUDADevice(int deviceId)
	{
		cudaGetDevice(&m_previousDevice);
		ASSERT(cudaCheckError());
		setCUDADevice(deviceId);
	}

	~ScopedCUDADevice()
	{
		cudaSetDevice(m_previousDevice);
	}

private:
	int m_previousDevice = 0;
};

class BinIteratorSlice : public BinIterator
{
public:
	BinIteratorSlice(const BinIterator& parent, size_t start, size_t count)
	    : mr_parent(parent), m_start(start), m_count(count)
	{
		ASSERT(start + count <= parent.size());
	}

	bin_t begin() const override
	{
		ASSERT(m_count > 0);
		return getSafe(0);
	}

	bin_t end() const override
	{
		ASSERT(m_count > 0);
		return getSafe(m_count - 1);
	}

	size_t size() const override
	{
		return m_count;
	}

private:
	bin_t getSafe(bin_t idx) const override
	{
		return mr_parent.get(m_start + idx);
	}

	const BinIterator& mr_parent;
	size_t m_start;
	size_t m_count;
};

std::vector<std::pair<size_t, size_t>> splitWork(size_t totalSize,
                                                 size_t numWorkers)
{
	std::vector<std::pair<size_t, size_t>> ranges;
	ranges.reserve(numWorkers);
	const size_t baseCount = totalSize / numWorkers;
	const size_t remainder = totalSize % numWorkers;
	size_t start = 0;
	for (size_t workerId = 0; workerId < numWorkers; workerId++)
	{
		const size_t count = baseCount + (workerId < remainder ? 1 : 0);
		ranges.emplace_back(start, count);
		start += count;
	}
	return ranges;
}

size_t getProjectionBatchOffset(const ProjectionDataDevice& dat, int batchId)
{
	return static_cast<size_t>(batchId) *
	       dat.getBatchSetup(0).getBatchSize(0);
}

double getEnvDouble(const char* envName, double defaultValue)
{
	const char* envValue = std::getenv(envName);
	if (envValue == nullptr || envValue[0] == '\0')
	{
		return defaultValue;
	}
	return std::stod(envValue);
}

size_t gibToBytes(double gib)
{
	return static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

std::string toLowerString(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c)
	               { return static_cast<char>(std::tolower(c)); });
	return value;
}

bool useMemoryAwareAliasProjectorSplit()
{
	const char* envValue = std::getenv("YRT_MULTI_GPU_ALIAS_SPLIT");
	if (envValue == nullptr || envValue[0] == '\0')
	{
		return true;
	}
	const std::string value(envValue);
	return value != "equal" && value != "EQUAL" && value != "0";
}

std::vector<size_t> getAliasProjectorWorkWeights(
    const std::vector<int>& deviceIds)
{
	std::vector<size_t> weights(deviceIds.size(), 0);
	const size_t primaryReserveBytes = gibToBytes(getEnvDouble(
	    "YRT_MULTI_GPU_ALIAS_PRIMARY_RESERVE_GB", 8.0));
	const size_t workerReserveBytes = gibToBytes(getEnvDouble(
	    "YRT_MULTI_GPU_ALIAS_WORKER_RESERVE_GB", 2.0));

	for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
	{
		ScopedCUDADevice guard(deviceIds.at(workerId));
		size_t freeMemory = 0;
		size_t totalMemory = 0;
		cudaMemGetInfo(&freeMemory, &totalMemory);
		ASSERT(cudaCheckError());
		const size_t reserveBytes =
		    workerId == 0 ? primaryReserveBytes : workerReserveBytes;
		weights.at(workerId) =
		    freeMemory > reserveBytes ? freeMemory - reserveBytes : 0;
	}
	return weights;
}

std::vector<std::pair<size_t, size_t>> splitWorkByWeights(
    size_t totalSize, const std::vector<size_t>& weights)
{
	size_t totalWeight = 0;
	for (size_t weight : weights)
	{
		totalWeight += weight;
	}
	if (totalWeight == 0)
	{
		return splitWork(totalSize, weights.size());
	}

	std::vector<std::pair<size_t, size_t>> ranges;
	ranges.reserve(weights.size());
	size_t start = 0;
	size_t cumulativeWeight = 0;
	for (size_t workerId = 0; workerId < weights.size(); workerId++)
	{
		if (workerId + 1 == weights.size())
		{
			ranges.emplace_back(start, totalSize - start);
			break;
		}
		cumulativeWeight += weights.at(workerId);
		const size_t end = std::min(
		    totalSize,
		    static_cast<size_t>(
		        (static_cast<long double>(totalSize) *
		         static_cast<long double>(cumulativeWeight)) /
		        static_cast<long double>(totalWeight)));
		ranges.emplace_back(start, end - start);
		start = end;
	}
	return ranges;
}

std::vector<std::pair<size_t, size_t>> splitAliasProjectorWork(
    size_t totalSize, const std::vector<int>& deviceIds)
{
	if (!useMemoryAwareAliasProjectorSplit())
	{
		return splitWork(totalSize, deviceIds.size());
	}
	return splitWorkByWeights(totalSize,
	                          getAliasProjectorWorkWeights(deviceIds));
}

void printAliasProjectorWorkSplit(
    const std::vector<int>& deviceIds,
    const std::vector<std::pair<size_t, size_t>>& ranges)
{
	std::cout << "YRT-PET multi-GPU alias projector work split:";
	for (size_t workerId = 0; workerId < ranges.size(); workerId++)
	{
		std::cout << " device " << deviceIds.at(workerId) << "="
		          << ranges.at(workerId).second << " LORs";
	}
	std::cout << std::endl;
}

bool isMultiGPUAliasProjectorEnabled()
{
	const char* envValue = std::getenv("YRT_MULTI_GPU_ALIAS_PROJECTOR");
	return envValue == nullptr || envValue[0] == '\0' ||
	       envValue[0] != '0';
}

bool isMultiGPUAliasHostStagingAllowed()
{
	const char* envValue =
	    std::getenv("YRT_MULTI_GPU_ALIAS_ALLOW_HOST_STAGING");
	return envValue != nullptr && envValue[0] != '\0' &&
	       envValue[0] != '0';
}

bool keepMultiGPUAliasLORsOnDevice()
{
	const char* envValue = std::getenv("YRT_CACHE_MULTI_GPU_ALIAS_LORS");
	if (envValue == nullptr || envValue[0] == '\0')
	{
		envValue = std::getenv("YRT_CACHE_MULTI_GPU_LORS");
	}
	if (envValue == nullptr || envValue[0] == '\0')
	{
		return false;
	}
	const std::string value = toLowerString(envValue);
	return value == "1" || value == "on" || value == "true" ||
	       value == "yes" || value == "device" || value == "gpu";
}

std::vector<int> getAliasProjectorRequestedDeviceIds()
{
	if (!globals::getCudaDeviceIds().empty())
	{
		return globals::getCudaDeviceIds();
	}

	const char* envDeviceIds = std::getenv("YRT_CUDA_DEVICES");
	if (envDeviceIds != nullptr && envDeviceIds[0] != '\0')
	{
		globals::setCudaDeviceIds(parseGPUDeviceIds(envDeviceIds));
		return globals::getCudaDeviceIds();
	}

	return {};
}

bool canAccessPeer(int destDeviceId, int sourceDeviceId)
{
	if (destDeviceId == sourceDeviceId)
	{
		return true;
	}
	int canAccess = 0;
	cudaDeviceCanAccessPeer(&canAccess, destDeviceId, sourceDeviceId);
	ASSERT(cudaCheckError());
	return canAccess != 0;
}

void logSkippedAliasProjectorDeviceOnce(int primaryDeviceId,
                                        int workerDeviceId)
{
	static std::set<std::pair<int, int>> s_loggedPairs;
	const auto key = std::make_pair(primaryDeviceId, workerDeviceId);
	if (s_loggedPairs.insert(key).second)
	{
		std::cout << "Skipping visible CUDA device " << workerDeviceId
		          << " for multi-GPU alias projection because it does not "
		             "have bidirectional P2P access with primary visible "
		             "CUDA device "
		          << primaryDeviceId
		          << ". Set YRT_MULTI_GPU_ALIAS_ALLOW_HOST_STAGING=1 to "
		             "force the slow host-staging fallback."
		          << std::endl;
	}
}

std::vector<int> getAliasProjectorUsableDeviceIds()
{
	const auto requestedDeviceIds = getAliasProjectorRequestedDeviceIds();
	if (requestedDeviceIds.size() <= 1 ||
	    isMultiGPUAliasHostStagingAllowed())
	{
		return requestedDeviceIds;
	}

	const int primaryDeviceId = requestedDeviceIds.front();
	std::vector<int> usableDeviceIds{primaryDeviceId};
	for (size_t i = 1; i < requestedDeviceIds.size(); i++)
	{
		const int workerDeviceId = requestedDeviceIds.at(i);
		if (canAccessPeer(primaryDeviceId, workerDeviceId) &&
		    canAccessPeer(workerDeviceId, primaryDeviceId))
		{
			usableDeviceIds.push_back(workerDeviceId);
		}
		else
		{
			logSkippedAliasProjectorDeviceOnce(primaryDeviceId,
			                                  workerDeviceId);
		}
	}
	return usableDeviceIds;
}

int getPointerDevice(const void* pointer, const char* pointerName)
{
	ASSERT_MSG(pointer != nullptr, "CUDA pointer is null");
	cudaPointerAttributes attributes{};
	const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
	if (status != cudaSuccess)
	{
		throw std::runtime_error(
		    std::string("Failed to query CUDA pointer attributes for ") +
		    pointerName + ": " + cudaGetErrorString(status));
	}
	return attributes.device;
}

void assertPointerOnPrimaryDevice(const void* pointer, const char* pointerName,
                                  int primaryDeviceId)
{
	const int pointerDevice = getPointerDevice(pointer, pointerName);
	if (pointerDevice != primaryDeviceId)
	{
		throw std::runtime_error(
		    std::string(pointerName) + " is on visible CUDA device " +
		    std::to_string(pointerDevice) +
		    ", but YRT multi-GPU alias projection expects it on primary "
		    "visible CUDA device " +
		    std::to_string(primaryDeviceId));
	}
}

void copyDeviceBufferAcrossGPUsAsync(const float* sourcePointer,
                                     int sourceDeviceId, float* destPointer,
                                     int destDeviceId, size_t numElements,
                                     int streamDeviceId,
                                     const cudaStream_t& stream,
                                     const char* bufferDescription)
{
	if (numElements == 0)
	{
		return;
	}
	ASSERT(sourcePointer != nullptr);
	ASSERT(destPointer != nullptr);
	ASSERT_MSG(streamDeviceId == sourceDeviceId ||
	               streamDeviceId == destDeviceId,
	           "Peer-copy stream must belong to source or destination device");

	const size_t byteCount = numElements * sizeof(float);
	{
		ScopedCUDADevice guard(streamDeviceId);
		if (sourceDeviceId == destDeviceId)
		{
			cudaMemcpyAsync(destPointer, sourcePointer, byteCount,
			                cudaMemcpyDeviceToDevice, stream);
			ASSERT(cudaCheckError());
			return;
		}

		if (ensurePeerAccess(destDeviceId, sourceDeviceId))
		{
			cudaMemcpyPeerAsync(destPointer, destDeviceId, sourcePointer,
			                    sourceDeviceId, byteCount, stream);
			ASSERT(cudaCheckError());
			return;
		}

		cudaStreamSynchronize(stream);
		ASSERT(cudaCheckError());
	}

	std::cout << "CUDA peer access is unavailable from visible device "
	          << destDeviceId << " to " << sourceDeviceId << "; staging "
	          << bufferDescription << " through host memory." << std::endl;
	std::vector<float> hostBuffer(numElements);
	{
		ScopedCUDADevice sourceGuard(sourceDeviceId);
		cudaMemcpy(hostBuffer.data(), sourcePointer, byteCount,
		           cudaMemcpyDeviceToHost);
		ASSERT(cudaCheckError());
	}
	{
		ScopedCUDADevice destGuard(destDeviceId);
		cudaMemcpy(destPointer, hostBuffer.data(), byteCount,
		           cudaMemcpyHostToDevice);
		ASSERT(cudaCheckError());
	}
}
}  // namespace

struct OperatorProjectorDevice::MultiGPUAliasProjectorCache
{
	struct Worker
	{
		struct Batch
		{
			size_t offset = 0;
			size_t count = 0;
			std::unique_ptr<BinIteratorSlice> sliceIterator;
			std::vector<const BinIterator*> binIterators;
			std::unique_ptr<ProjectionDataDeviceAlias>
			    primaryProjectionAlias;
			std::unique_ptr<ProjectionDataDeviceOwned>
			    workerProjectionBuffer;
		};

		int deviceId = -1;
		size_t start = 0;
		size_t count = 0;
		std::unique_ptr<GPUStream> mainStream;
		std::unique_ptr<GPUStream> auxStream;
		std::unique_ptr<BinIteratorSlice> sliceIterator;
		std::vector<const BinIterator*> binIterators;
		std::unique_ptr<OperatorProjectorDevice> projector;
		std::unique_ptr<ProjectionDataDeviceAlias> primaryProjectionAlias;
		std::unique_ptr<ProjectionDataDeviceOwned> workerProjectionBuffer;
		std::vector<Batch> batches;
		std::unique_ptr<ImageDeviceOwned> workerInputImage;
		std::unique_ptr<ImageDeviceOwned> workerPartialImage;
		std::unique_ptr<ImageDeviceOwned> primaryPartialImage;
	};

	~MultiGPUAliasProjectorCache()
	{
		const int primaryDeviceId =
		    deviceIds.empty() ? -1 : deviceIds.front();
		for (auto& worker : workers)
		{
			if (primaryDeviceId >= 0 && worker.primaryPartialImage != nullptr)
			{
				ScopedCUDADevice guard(primaryDeviceId);
				worker.primaryPartialImage.reset();
			}
			if (worker.deviceId >= 0)
			{
				ScopedCUDADevice guard(worker.deviceId);
				worker.batches.clear();
				worker.primaryProjectionAlias.reset();
				worker.workerProjectionBuffer.reset();
				worker.projector.reset();
				worker.workerInputImage.reset();
				worker.workerPartialImage.reset();
				worker.mainStream.reset();
				worker.auxStream.reset();
			}
		}
	}

	std::vector<int> deviceIds;
	const ProjectionData* reference = nullptr;
	const BinIterator* parentBinIter = nullptr;
	size_t parentSize = 0;
	ImageParams imageParams;
	bool imageBuffersInitialized = false;
	std::vector<Worker> workers;
};

OperatorProjectorDevice::OperatorProjectorDevice(
    const OperatorProjectorParams& pr_projParams,
    const cudaStream_t* pp_mainStream, const cudaStream_t* pp_auxStream)
    : OperatorProjectorBase{pr_projParams},
      DeviceSynchronized{pp_mainStream, pp_auxStream},
      m_projectorParams{pr_projParams}
{
	if (pr_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(pr_projParams.tofWidth_ps, pr_projParams.tofNumStd);
	}
	if (!pr_projParams.projPsf_fname.empty())
	{
		setupProjPsfManager(pr_projParams.projPsf_fname);
	}

	m_batchSize = 0ull;
}

OperatorProjectorDevice::~OperatorProjectorDevice() = default;

std::unique_ptr<OperatorProjectorDevice>
    OperatorProjectorDevice::createWorkerProjector(
        const BinIterator* workerBinIter, const cudaStream_t* mainStream,
        const cudaStream_t* auxStream) const
{
	OperatorProjectorParams workerParams(
	    workerBinIter, scanner, m_projectorParams.tofWidth_ps,
	    m_projectorParams.tofNumStd, m_projectorParams.projPsf_fname,
	    m_projectorParams.numRays);

	if (dynamic_cast<const OperatorProjectorSiddon_GPU*>(this) != nullptr)
	{
		return std::make_unique<OperatorProjectorSiddon_GPU>(
		    workerParams, mainStream, auxStream);
	}
	if (dynamic_cast<const OperatorProjectorDD_GPU*>(this) != nullptr)
	{
		return std::make_unique<OperatorProjectorDD_GPU>(
		    workerParams, mainStream, auxStream);
	}
	throw std::runtime_error(
	    "Unsupported device projector type for multi-GPU alias projection");
}

void OperatorProjectorDevice::ensureMultiGPUAliasProjectorCache(
    const ProjectionDataDevice& dat, const ImageParams& imageParams,
    const std::vector<int>& deviceIds)
{
	ASSERT_MSG(deviceIds.size() > 1,
	           "Multi-GPU alias projector requires multiple CUDA devices");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

	const ProjectionData* reference = dat.getReference();
	const bool cacheValid =
	    mp_multiGPUAliasCache != nullptr &&
	    mp_multiGPUAliasCache->deviceIds == deviceIds &&
	    mp_multiGPUAliasCache->reference == reference &&
	    mp_multiGPUAliasCache->parentBinIter == binIter &&
	    mp_multiGPUAliasCache->parentSize == binIter->size();

	if (!cacheValid)
	{
		mp_multiGPUAliasCache =
		    std::make_unique<MultiGPUAliasProjectorCache>();
		auto& cache = *mp_multiGPUAliasCache;
		cache.deviceIds = deviceIds;
		cache.reference = reference;
		cache.parentBinIter = binIter;
		cache.parentSize = binIter->size();
		cache.workers.resize(deviceIds.size());

		const auto ranges =
		    splitAliasProjectorWork(cache.parentSize, deviceIds);
		printAliasProjectorWorkSplit(deviceIds, ranges);
		const int primaryDeviceId = deviceIds.front();
		for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
		{
			const size_t start = ranges.at(workerId).first;
			const size_t count = ranges.at(workerId).second;
			auto& worker = cache.workers.at(workerId);
			worker.deviceId = deviceIds.at(workerId);
			worker.start = start;
			worker.count = count;
			if (count == 0)
			{
				continue;
			}

			ScopedCUDADevice guard(worker.deviceId);
			worker.mainStream = std::make_unique<GPUStream>();
			worker.auxStream = std::make_unique<GPUStream>();
			worker.sliceIterator =
			    std::make_unique<BinIteratorSlice>(*binIter, start, count);
			worker.binIterators = {worker.sliceIterator.get()};
			worker.projector = createWorkerProjector(
			    worker.sliceIterator.get(), &worker.mainStream->getStream(),
			    &worker.auxStream->getStream());

			ProjectionDataDeviceAlias sizingProjection(
			    scanner, reference, worker.binIterators);
			const auto& batchSetup = sizingProjection.getBatchSetup(0);
			const int numBatches = batchSetup.getNumBatches();
			std::cout << "Multi-GPU alias projector worker " << workerId
			          << " using visible CUDA device " << worker.deviceId
			          << " for " << count << " LORs in " << numBatches
			          << " cached LOR batches." << std::endl;
			worker.batches.reserve(numBatches);
			for (int batchId = 0; batchId < numBatches; batchId++)
			{
				MultiGPUAliasProjectorCache::Worker::Batch batch;
				batch.offset =
				    getProjectionBatchOffset(sizingProjection, batchId);
				batch.count = batchSetup.getBatchSize(batchId);
				batch.sliceIterator =
				    std::make_unique<BinIteratorSlice>(
				        *binIter, start + batch.offset, batch.count);
				batch.binIterators = {batch.sliceIterator.get()};
				if (worker.deviceId == primaryDeviceId)
				{
					batch.primaryProjectionAlias =
					    std::make_unique<ProjectionDataDeviceAlias>(
					        scanner, reference, batch.binIterators);
					ASSERT_MSG(
					    batch.primaryProjectionAlias->getNumBatches(0) == 1,
					    "Cached multi-GPU alias LOR batch does not fit "
					    "in one loaded batch on the primary GPU");
					batch.primaryProjectionAlias->precomputeBatchLORs(0, 0);
				}
				else
				{
					batch.workerProjectionBuffer =
					    std::make_unique<ProjectionDataDeviceOwned>(
					        scanner, reference, batch.binIterators);
					ASSERT_MSG(
					    batch.workerProjectionBuffer->getNumBatches(0) == 1,
					    "Cached multi-GPU alias LOR batch does not fit "
					    "in one loaded batch on a worker GPU");
					batch.workerProjectionBuffer->precomputeBatchLORs(0, 0);
				}
				worker.batches.push_back(std::move(batch));
			}
		}
	}

	auto& cache = *mp_multiGPUAliasCache;
	if (cache.imageBuffersInitialized &&
	    cache.imageParams.isSameAs(imageParams))
	{
		return;
	}

	const int primaryDeviceId = deviceIds.front();
	for (auto& worker : cache.workers)
	{
		if (worker.count == 0)
		{
			continue;
		}
		if (worker.workerInputImage != nullptr)
		{
			ScopedCUDADevice guard(worker.deviceId);
			worker.workerInputImage.reset();
		}
		if (worker.workerPartialImage != nullptr)
		{
			ScopedCUDADevice guard(worker.deviceId);
			worker.workerPartialImage.reset();
		}
		if (worker.primaryPartialImage != nullptr)
		{
			ScopedCUDADevice guard(primaryDeviceId);
			worker.primaryPartialImage.reset();
		}
	}

	for (auto& worker : cache.workers)
	{
		if (worker.count == 0)
		{
			continue;
		}
		if (worker.deviceId != primaryDeviceId)
		{
			ScopedCUDADevice workerGuard(worker.deviceId);
			worker.workerInputImage =
			    std::make_unique<ImageDeviceOwned>(
			        imageParams, &worker.mainStream->getStream());
			worker.workerInputImage->allocate(true, false);
			worker.workerPartialImage =
			    std::make_unique<ImageDeviceOwned>(
			        imageParams, &worker.mainStream->getStream());
			worker.workerPartialImage->allocate(true, true);
		}
		{
			ScopedCUDADevice primaryGuard(primaryDeviceId);
			worker.primaryPartialImage =
			    std::make_unique<ImageDeviceOwned>(imageParams);
			worker.primaryPartialImage->allocate(true, true);
		}
	}

	cache.imageParams = imageParams;
	cache.imageBuffersInitialized = true;
}

bool OperatorProjectorDevice::tryApplyAMultiGPUAlias(
	ImageDevice* imgIn, ProjectionDataDevice* datOut, bool synchronize)
{
	const auto deviceIds = getAliasProjectorUsableDeviceIds();
	if (deviceIds.size() <= 1 || !isMultiGPUAliasProjectorEnabled())
	{
		return false;
	}
	auto* imgAlias = dynamic_cast<ImageDeviceAlias*>(imgIn);
	auto* datAlias = dynamic_cast<ProjectionDataDeviceAlias*>(datOut);
	if (imgAlias == nullptr || datAlias == nullptr)
	{
		return false;
	}
	ASSERT_MSG(imgAlias->isDevicePointerSet(),
	           "ImageDeviceAlias pointer is not set");
	ASSERT_MSG(datAlias->isDevicePointerSet(),
	           "ProjectionDataDeviceAlias pointer is not set");

	const int primaryDeviceId = deviceIds.front();
	float* primaryImagePointer = imgAlias->getDevicePointer();
	float* primaryProjectionPointer = datAlias->getProjValuesDevicePointer();
	assertPointerOnPrimaryDevice(primaryImagePointer, "ImageDeviceAlias",
	                             primaryDeviceId);
	assertPointerOnPrimaryDevice(primaryProjectionPointer,
	                             "ProjectionDataDeviceAlias", primaryDeviceId);

	{
		ScopedCUDADevice guard(primaryDeviceId);
		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
		datAlias->releaseDeviceLORs({nullptr, true});
	}

	ensureMultiGPUAliasProjectorCache(*datAlias, imgAlias->getParams(),
	                                  deviceIds);
	auto& cache = *mp_multiGPUAliasCache;
	const bool keepLORsOnDevice = keepMultiGPUAliasLORsOnDevice();

	std::vector<std::exception_ptr> errors(cache.workers.size());
	std::vector<std::thread> workers;
	workers.reserve(cache.workers.size());

	for (size_t workerId = 0; workerId < cache.workers.size(); workerId++)
	{
		auto& worker = cache.workers.at(workerId);
		if (worker.count == 0)
		{
			continue;
		}
		workers.emplace_back(
		    [&, workerId]()
		    {
			    try
				    {
					    auto& worker = cache.workers.at(workerId);
					    ScopedCUDADevice guard(worker.deviceId);
					    GPUStream& mainStream = *worker.mainStream;
					    ImageDevice* imageForWorker = imgAlias;

					    if (worker.deviceId != primaryDeviceId)
					    {
						    ASSERT(worker.workerInputImage != nullptr);
						    copyDeviceBufferAcrossGPUsAsync(
						        primaryImagePointer, primaryDeviceId,
						        worker.workerInputImage->getDevicePointer(),
					        worker.deviceId, imgAlias->getImageSize(),
						        worker.deviceId, mainStream.getStream(),
						        "the alias input image");
						    imageForWorker = worker.workerInputImage.get();
					    }

					    for (auto& batch : worker.batches)
					    {
						    ProjectionDataDevice* projectionForBatch = nullptr;
						    if (worker.deviceId == primaryDeviceId)
						    {
							    ASSERT(batch.primaryProjectionAlias != nullptr);
							    batch.primaryProjectionAlias
							        ->setProjValuesDevicePointer(
							            primaryProjectionPointer +
							            worker.start + batch.offset);
							    projectionForBatch =
							        batch.primaryProjectionAlias.get();
						    }
						    else
						    {
							    ASSERT(batch.workerProjectionBuffer != nullptr);
							    batch.workerProjectionBuffer
							        ->allocateForProjValues(
							        {&mainStream.getStream(), false});
							    projectionForBatch =
							        batch.workerProjectionBuffer.get();
						    }
						    projectionForBatch->loadPrecomputedLORsToDevice(
						        {&mainStream.getStream(), false});

						    worker.projector->applyAOnLoadedBatch(
						        *imageForWorker, *projectionForBatch, false);

						    if (worker.deviceId != primaryDeviceId)
						    {
							    copyDeviceBufferAcrossGPUsAsync(
							        batch.workerProjectionBuffer
							            ->getProjValuesDevicePointer(),
							        worker.deviceId,
							        primaryProjectionPointer +
							            worker.start + batch.offset,
							        primaryDeviceId, batch.count,
						        worker.deviceId, mainStream.getStream(),
						        "one alias forward-projection batch");
						    }
						    if (!keepLORsOnDevice)
						    {
							    projectionForBatch->releaseDeviceLORs(
							        {&mainStream.getStream(), false});
						    }
					    }
					    cudaStreamSynchronize(mainStream.getStream());
				    ASSERT(cudaCheckError());
			    }
			    catch (...)
			    {
				    errors.at(workerId) = std::current_exception();
			    }
		    });
	}

	for (auto& worker : workers)
	{
		worker.join();
	}
	for (const auto& error : errors)
	{
		if (error != nullptr)
		{
			std::rethrow_exception(error);
		}
	}

	if (synchronize)
	{
		ScopedCUDADevice guard(primaryDeviceId);
		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
	}
	return true;
}

bool OperatorProjectorDevice::tryApplyAHMultiGPUAlias(
	ProjectionDataDevice* datIn, ImageDevice* imgOut, bool synchronize)
{
	const auto deviceIds = getAliasProjectorUsableDeviceIds();
	if (deviceIds.size() <= 1 || !isMultiGPUAliasProjectorEnabled())
	{
		return false;
	}
	auto* datAlias = dynamic_cast<ProjectionDataDeviceAlias*>(datIn);
	auto* imgAlias = dynamic_cast<ImageDeviceAlias*>(imgOut);
	if (datAlias == nullptr || imgAlias == nullptr)
	{
		return false;
	}
	ASSERT_MSG(datAlias->isDevicePointerSet(),
	           "ProjectionDataDeviceAlias pointer is not set");
	ASSERT_MSG(imgAlias->isDevicePointerSet(),
	           "ImageDeviceAlias pointer is not set");

	const int primaryDeviceId = deviceIds.front();
	float* primaryProjectionPointer = datAlias->getProjValuesDevicePointer();
	float* primaryImagePointer = imgAlias->getDevicePointer();
	assertPointerOnPrimaryDevice(primaryProjectionPointer,
	                             "ProjectionDataDeviceAlias", primaryDeviceId);
	assertPointerOnPrimaryDevice(primaryImagePointer, "ImageDeviceAlias",
	                             primaryDeviceId);

	{
		ScopedCUDADevice guard(primaryDeviceId);
		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
		datAlias->releaseDeviceLORs({nullptr, true});
	}

	ensureMultiGPUAliasProjectorCache(*datAlias, imgAlias->getParams(),
	                                  deviceIds);
	auto& cache = *mp_multiGPUAliasCache;
	const bool keepLORsOnDevice = keepMultiGPUAliasLORsOnDevice();

	std::vector<std::exception_ptr> errors(cache.workers.size());
	std::vector<std::thread> workers;
	workers.reserve(cache.workers.size());

	for (size_t workerId = 0; workerId < cache.workers.size(); workerId++)
	{
		auto& worker = cache.workers.at(workerId);
		if (worker.count == 0)
		{
			continue;
		}
		workers.emplace_back(
		    [&, workerId]()
		    {
			    try
				    {
					    auto& worker = cache.workers.at(workerId);
					    ScopedCUDADevice guard(worker.deviceId);
					    GPUStream& mainStream = *worker.mainStream;
					    ImageDevice* partialImageForWorker = nullptr;

					    if (worker.deviceId == primaryDeviceId)
					    {
						    ASSERT(worker.primaryPartialImage != nullptr);
						    partialImageForWorker =
						        worker.primaryPartialImage.get();
						    partialImageForWorker->setValueDevice(0.0f, true);
				    }
					    else
					    {
						    ASSERT(worker.workerPartialImage != nullptr);
						    partialImageForWorker =
						        worker.workerPartialImage.get();
						    partialImageForWorker->setValueDevice(0.0f, true);
					    }

					    for (auto& batch : worker.batches)
					    {
						    ProjectionDataDevice* projectionForBatch = nullptr;
						    if (worker.deviceId == primaryDeviceId)
						    {
							    ASSERT(batch.primaryProjectionAlias != nullptr);
							    batch.primaryProjectionAlias
							        ->setProjValuesDevicePointer(
							            primaryProjectionPointer +
							            worker.start + batch.offset);
							    projectionForBatch =
							        batch.primaryProjectionAlias.get();
						    }
						    else
						    {
							    ASSERT(batch.workerProjectionBuffer != nullptr);
							    batch.workerProjectionBuffer
							        ->allocateForProjValues(
							        {&mainStream.getStream(), false});
							    projectionForBatch =
							        batch.workerProjectionBuffer.get();
						    }
						    projectionForBatch->loadPrecomputedLORsToDevice(
						        {&mainStream.getStream(), false});

						    if (worker.deviceId != primaryDeviceId)
					    {
							    copyDeviceBufferAcrossGPUsAsync(
							        primaryProjectionPointer + worker.start +
							            batch.offset,
							        primaryDeviceId,
							        batch.workerProjectionBuffer
							            ->getProjValuesDevicePointer(),
							        worker.deviceId, batch.count,
							        worker.deviceId, mainStream.getStream(),
							        "one alias backprojection input batch");
						    }

						    worker.projector->applyAHOnLoadedBatch(
						        *projectionForBatch, *partialImageForWorker,
						        false);
						    if (!keepLORsOnDevice)
						    {
							    projectionForBatch->releaseDeviceLORs(
							        {&mainStream.getStream(), false});
						    }
					    }

					    if (worker.deviceId != primaryDeviceId)
				    {
					    ASSERT(worker.primaryPartialImage != nullptr);
					    copyDeviceBufferAcrossGPUsAsync(
					        worker.workerPartialImage->getDevicePointer(),
					        worker.deviceId,
					        worker.primaryPartialImage->getDevicePointer(),
					        primaryDeviceId, imgAlias->getImageSize(),
					        worker.deviceId, mainStream.getStream(),
					        "one alias backprojection partial image");
				    }
				    cudaStreamSynchronize(mainStream.getStream());
				    ASSERT(cudaCheckError());
			    }
			    catch (...)
			    {
				    errors.at(workerId) = std::current_exception();
			    }
		    });
	}

	for (auto& worker : workers)
	{
		worker.join();
	}
	for (const auto& error : errors)
	{
		if (error != nullptr)
		{
			std::rethrow_exception(error);
		}
	}

	{
		ScopedCUDADevice guard(primaryDeviceId);
		for (auto& worker : cache.workers)
		{
			if (worker.count == 0)
			{
				continue;
			}
			ASSERT(worker.primaryPartialImage != nullptr);
			worker.primaryPartialImage->addFirstImageToSecondDevice(
			    imgAlias, true);
		}
	}

	if (synchronize)
	{
		ScopedCUDADevice guard(primaryDeviceId);
		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
	}
	return true;
}

void OperatorProjectorDevice::applyA(const Variable* in, Variable* out)
{
	applyA(in, out, true);
}

void OperatorProjectorDevice::applyAH(const Variable* in, Variable* out)
{
	applyAH(in, out, true);
}

void OperatorProjectorDevice::applyA(const Variable* in, Variable* out,
                                     bool synchronize)
{
	auto* img_in_const = dynamic_cast<const ImageDevice*>(in);
	auto* dat_out = dynamic_cast<ProjectionDataDevice*>(out);

	// In case the user provided a host-side image
	std::unique_ptr<ImageDeviceOwned> deviceImg_out = nullptr;
	ImageDevice* img_in = nullptr;
	if (img_in_const == nullptr)
	{
		const auto* hostImg_in = dynamic_cast<const Image*>(in);
		ASSERT_MSG(
		    hostImg_in != nullptr,
		    "The image provided is not a ImageDevice nor a Image (host)");

		deviceImg_out = std::make_unique<ImageDeviceOwned>(
		    hostImg_in->getParams(), getAuxStream());
		deviceImg_out->allocate(false, false);
		deviceImg_out->transferToDeviceMemory(hostImg_in, true);

		// Use owned ImageDevice
		img_in = deviceImg_out.get();
	}
	else
	{
		img_in = const_cast<ImageDevice*>(img_in_const);
		ASSERT_MSG(img_in != nullptr, "ImageDevice is null. Cast failed");
	}

	// In case the user provided a Host-side ProjectionData
	bool isProjDataDeviceOwned = false;
	std::unique_ptr<ProjectionDataDeviceOwned> deviceDat_out = nullptr;
	ProjectionData* hostDat_out = nullptr;
	if (dat_out == nullptr)
	{
		hostDat_out = dynamic_cast<ProjectionData*>(out);
		ASSERT_MSG(hostDat_out != nullptr,
		           "The Projection Data provded is not a ProjectionDataDevice "
		           "nor a ProjectionData (host)");
		ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

		std::vector<const BinIterator*> binIterators;
		binIterators.push_back(binIter);  // We project only one subset
		deviceDat_out = std::make_unique<ProjectionDataDeviceOwned>(
		    getScanner(), hostDat_out, binIterators);

		// Use owned ProjectionDataDevice
		dat_out = deviceDat_out.get();
		isProjDataDeviceOwned = true;
	}

	if (!isProjDataDeviceOwned)
	{
		if (tryApplyAMultiGPUAlias(img_in, dat_out, synchronize))
		{
			return;
		}
		applyAOnLoadedBatch(*img_in, *dat_out, synchronize);
	}
	else
	{
		// Iterate over all the batches of the current subset
		const size_t numBatches = dat_out->getBatchSetup(0).getNumBatches();

		std::cout << "Loading batch 1/" << numBatches << "..." << std::endl;
		dat_out->precomputeBatchLORs(0, 0);
		deviceDat_out->allocateForProjValues({getMainStream(), false});

		for (size_t batchId = 0; batchId < numBatches; batchId++)
		{
			dat_out->loadPrecomputedLORsToDevice({getMainStream(), true});

			std::cout << "Forward projecting batch " << batchId + 1 << "/"
			          << numBatches << "..." << std::endl;
			dat_out->clearProjectionsDevice({getMainStream(), false});
			applyAOnLoadedBatch(*img_in, *dat_out, false);

			// If a future batch is due
			if (batchId < numBatches - 1)
			{
				std::cout << "Loading batch " << batchId + 2 << "/"
				          << numBatches << "..." << std::endl;
				dat_out->precomputeBatchLORs(0, batchId + 1);
			}
			std::cout << "Transferring batch to Host..." << std::endl;
			// This will force a necessary synchronization
			dat_out->transferProjValuesToHost(hostDat_out, getMainStream());
		}
	}
}

void OperatorProjectorDevice::applyAH(const Variable* in, Variable* out,
                                      bool synchronize)
{
	auto* dat_in_const = dynamic_cast<const ProjectionDataDevice*>(in);
	auto* img_out = dynamic_cast<ImageDevice*>(out);

	bool isImageDeviceOwned = false;

	// In case the user provided a host-side image
	std::unique_ptr<ImageDeviceOwned> deviceImg_out = nullptr;
	Image* hostImg_out = nullptr;
	if (img_out == nullptr)
	{
		hostImg_out = dynamic_cast<Image*>(out);
		ASSERT_MSG(
		    hostImg_out != nullptr,
		    "The image provided is not a ImageDevice nor a Image (host)");

		deviceImg_out = std::make_unique<ImageDeviceOwned>(
		    hostImg_out->getParams(), getAuxStream());
		deviceImg_out->allocate(false, false);
		deviceImg_out->transferToDeviceMemory(hostImg_out, true);

		// Use owned ImageDevice
		img_out = deviceImg_out.get();
		isImageDeviceOwned = true;
	}

	ProjectionDataDevice* dat_in = nullptr;
	bool isProjDataDeviceOwned = false;

	// In case the user provided a Host-side ProjectionData
	std::unique_ptr<ProjectionDataDeviceOwned> deviceDat_in = nullptr;
	if (dat_in_const == nullptr)
	{
		auto* hostDat_in = dynamic_cast<const ProjectionData*>(in);
		ASSERT_MSG(hostDat_in != nullptr,
		           "The Projection Data provded is not a ProjectionDataDevice "
		           "nor a ProjectionData (host)");
		ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

		std::vector<const BinIterator*> binIterators;
		binIterators.push_back(binIter);  // We project only one subset
		deviceDat_in = std::make_unique<ProjectionDataDeviceOwned>(
		    getScanner(), hostDat_in, binIterators);

		// Use owned ProjectionDataDevice
		dat_in = deviceDat_in.get();
		isProjDataDeviceOwned = true;
	}
	else
	{
		dat_in = const_cast<ProjectionDataDevice*>(dat_in_const);
		ASSERT_MSG(dat_in != nullptr,
		           "ProjectionDataDevice is null. Cast failed");
	}

	if (!isProjDataDeviceOwned)
	{
		if (tryApplyAHMultiGPUAlias(dat_in, img_out, synchronize))
		{
			return;
		}
		applyAHOnLoadedBatch(*dat_in, *img_out, synchronize);
	}
	else
	{
		// Iterate over all the batches of the current subset
		const size_t numBatches = dat_in->getBatchSetup(0).getNumBatches();
		const cudaStream_t* mainStream = getMainStream();

		std::cout << "Loading batch 1/" << numBatches << "..." << std::endl;
		dat_in->precomputeBatchLORs(0, 0);
		deviceDat_in->allocateForProjValues({mainStream, false});

		for (size_t batchId = 0; batchId < numBatches; batchId++)
		{
			deviceDat_in->loadPrecomputedLORsToDevice({mainStream, false});
			deviceDat_in->loadProjValuesFromReference({mainStream, false});
			std::cout << "Backprojecting batch " << batchId + 1 << "/"
			          << numBatches << "..." << std::endl;

			// Synchronize
			if (mainStream != nullptr)
			{
				cudaStreamSynchronize(*mainStream);
			}
			else
			{
				cudaDeviceSynchronize();
			}

			applyAHOnLoadedBatch(*dat_in, *img_out, false);

			if (batchId < numBatches - 1)
			{
				std::cout << "Loading batch " << batchId + 2 << "/"
				          << numBatches << "..." << std::endl;
				dat_in->precomputeBatchLORs(0, batchId + 1);
			}
		}

		// Synchronize before getting returning
		if (mainStream != nullptr)
		{
			cudaStreamSynchronize(*mainStream);
		}
		else
		{
			cudaDeviceSynchronize();
		}
	}

	if (isImageDeviceOwned)
	{
		// Need to transfer the generated image back to the host
		deviceImg_out->transferToHostMemory(hostImg_out, true);
	}
}

unsigned int OperatorProjectorDevice::getGridSize() const
{
	return m_launchParams.gridSize;
}

unsigned int OperatorProjectorDevice::getBlockSize() const
{
	return m_launchParams.blockSize;
}

void OperatorProjectorDevice::setBatchSize(size_t newBatchSize)
{
	m_batchSize = newBatchSize;
	m_launchParams = util::initiateDeviceParameters(m_batchSize);
}

size_t OperatorProjectorDevice::getBatchSize() const
{
	return m_batchSize;
}

void OperatorProjectorDevice::setupProjPsfManager(
    const std::string& psfFilename)
{
	mp_projPsfManager =
	    std::make_unique<ProjectionPsfManagerDevice>(psfFilename);
	ASSERT_MSG(mp_projPsfManager != nullptr,
	           "Error occured during the setup of ProjectionPsfManagerDevice");
}

void OperatorProjectorDevice::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<DeviceObject<TimeOfFlightHelper>>(
	    tofWidth_ps, tofNumStd);
}

const TimeOfFlightHelper*
    OperatorProjectorDevice::getTOFHelperDevicePointer() const
{
	if (mp_tofHelper != nullptr)
	{
		return mp_tofHelper->getDevicePointer();
	}
	return nullptr;
}

const float*
    OperatorProjectorDevice::getProjPsfKernelsDevicePointer(bool flipped) const
{
	if (mp_projPsfManager != nullptr)
	{
		if (!flipped)
		{
			return mp_projPsfManager->getKernelsDevicePointer();
		}
		return mp_projPsfManager->getFlippedKernelsDevicePointer();
	}
	return nullptr;
}
}  // namespace yrt
