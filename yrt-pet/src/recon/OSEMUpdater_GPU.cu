/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/recon/OSEMUpdater_GPU.cuh"
#include "yrt-pet/recon/OSEM_GPU.cuh"
#include "yrt-pet/utils/GPUStream.cuh"
#include "yrt-pet/utils/GPUUtils.cuh"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <omp.h>

namespace yrt
{
namespace
{
enum class MultiGPULORCacheMode
{
	Off,
	Host,
	Device
};

class ScopedCUDADevice
{
public:
	explicit ScopedCUDADevice(int deviceId)
	{
		cudaGetDevice(&m_previousDevice);
		setCUDADevice(deviceId);
	}

	~ScopedCUDADevice()
	{
		cudaSetDevice(m_previousDevice);
	}

private:
	int m_previousDevice = 0;
};

class ScopedOpenMPThreads
{
public:
	explicit ScopedOpenMPThreads(int numThreads)
	    : m_previousNumThreads(omp_get_max_threads())
	{
		omp_set_num_threads(numThreads);
	}

	~ScopedOpenMPThreads()
	{
		omp_set_num_threads(m_previousNumThreads);
	}

private:
	int m_previousNumThreads = 1;
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
	const size_t baseSize = totalSize / numWorkers;
	const size_t remainder = totalSize % numWorkers;
	size_t start = 0;
	for (size_t workerId = 0; workerId < numWorkers; workerId++)
	{
		const size_t count = baseSize + (workerId < remainder ? 1 : 0);
		ranges.emplace_back(start, count);
		start += count;
	}
	return ranges;
}

int getOpenMPThreadsPerGPUWorker(size_t numWorkers)
{
	const char* envThreads = std::getenv("YRT_OMP_THREADS_PER_GPU");
	if (envThreads != nullptr && envThreads[0] != '\0')
	{
		return std::max(1, std::stoi(envThreads));
	}

	const int maxThreads = omp_get_max_threads();
	return std::max(1, maxThreads / static_cast<int>(numWorkers));
}

bool isMultiGPUProfileEnabled()
{
	const char* envValue = std::getenv("YRT_MULTI_GPU_PROFILE");
	return envValue != nullptr && envValue[0] != '\0' &&
	       envValue[0] != '0';
}

bool isMultiGPUCUDAProfileEnabled()
{
	const char* envValue = std::getenv("YRT_MULTI_GPU_CUDA_PROFILE");
	return envValue != nullptr && envValue[0] != '\0' &&
	       envValue[0] != '0';
}

double secondsSince(std::chrono::steady_clock::time_point start)
{
	const std::chrono::duration<double> elapsed =
	    std::chrono::steady_clock::now() - start;
	return elapsed.count();
}

class ScopedProfileTimer
{
public:
	ScopedProfileTimer(bool enabled, double& destination)
	    : m_enabled(enabled),
	      mr_destination(destination),
	      m_start(enabled ? std::chrono::steady_clock::now() :
	                         std::chrono::steady_clock::time_point{})
	{
	}

	~ScopedProfileTimer()
	{
		if (m_enabled)
		{
			mr_destination += secondsSince(m_start);
		}
	}

private:
	bool m_enabled = false;
	double& mr_destination;
	std::chrono::steady_clock::time_point m_start;
};

class CUDAEventProfileTimer
{
public:
	CUDAEventProfileTimer(bool enabled, cudaStream_t stream)
	    : m_enabled(enabled), m_stream(stream)
	{
		if (!m_enabled)
		{
			return;
		}
		ASSERT(cudaEventCreate(&m_start) == cudaSuccess);
		ASSERT(cudaEventCreate(&m_stop) == cudaSuccess);
		ASSERT(cudaEventRecord(m_start, m_stream) == cudaSuccess);
	}

	CUDAEventProfileTimer(const CUDAEventProfileTimer&) = delete;
	CUDAEventProfileTimer& operator=(const CUDAEventProfileTimer&) = delete;

	CUDAEventProfileTimer(CUDAEventProfileTimer&& other) noexcept
	    : m_enabled(other.m_enabled),
	      m_stream(other.m_stream),
	      m_start(other.m_start),
	      m_stop(other.m_stop),
	      m_stopped(other.m_stopped)
	{
		other.m_enabled = false;
		other.m_start = nullptr;
		other.m_stop = nullptr;
		other.m_stopped = false;
	}

	~CUDAEventProfileTimer()
	{
		destroyEvents();
	}

	void stop()
	{
		if (!m_enabled || m_stopped)
		{
			return;
		}
		ASSERT(cudaEventRecord(m_stop, m_stream) == cudaSuccess);
		m_stopped = true;
	}

	void addElapsedTimeAfterSync(double& destination)
	{
		if (!m_enabled)
		{
			return;
		}
		stop();
		float milliseconds = 0.0f;
		ASSERT(cudaEventElapsedTime(&milliseconds, m_start, m_stop) ==
		       cudaSuccess);
		destination += static_cast<double>(milliseconds) / 1000.0;
		destroyEvents();
	}

private:
	void destroyEvents() noexcept
	{
		if (m_start != nullptr)
		{
			cudaEventDestroy(m_start);
			m_start = nullptr;
		}
		if (m_stop != nullptr)
		{
			cudaEventDestroy(m_stop);
			m_stop = nullptr;
		}
		m_enabled = false;
		m_stopped = false;
	}

	bool m_enabled = false;
	cudaStream_t m_stream = nullptr;
	cudaEvent_t m_start = nullptr;
	cudaEvent_t m_stop = nullptr;
	bool m_stopped = false;
};

struct MultiGPUStageProfile
{
	double total = 0.0;
	double copyInputImage = 0.0;
	double clearPartialImage = 0.0;
	double precomputeLORs = 0.0;
	double allocateProjectionValues = 0.0;
	double loadLORs = 0.0;
	double loadProjectionValues = 0.0;
	double forwardProjectEnqueue = 0.0;
	double correctionsEnqueue = 0.0;
	double ratioEnqueue = 0.0;
	double preBackprojectSync = 0.0;
	double backprojectEnqueue = 0.0;
	double finalSync = 0.0;
	double releaseLORs = 0.0;
	double copyPartialImage = 0.0;
	double cudaLoadLORs = 0.0;
	double cudaForwardProject = 0.0;
	double cudaCorrections = 0.0;
	double cudaRatio = 0.0;
	double cudaBackproject = 0.0;
};

void emitMultiGPUProfileMetric(const std::string& name, double seconds)
{
	if (!isMultiGPUProfileEnabled())
	{
		return;
	}
	std::cout << "YRT_MULTI_GPU_PROFILE " << name << " " << seconds
	          << std::endl;
}

void emitMultiGPUProfileAggregate(
    const std::string& prefix,
    const std::vector<MultiGPUStageProfile>& workerProfiles)
{
	if (!isMultiGPUProfileEnabled())
	{
		return;
	}

	struct Field
	{
		const char* name;
		double MultiGPUStageProfile::*value;
	};
	const Field fields[] = {
	    {"worker_total", &MultiGPUStageProfile::total},
	    {"copy_input_image", &MultiGPUStageProfile::copyInputImage},
	    {"clear_partial_image", &MultiGPUStageProfile::clearPartialImage},
	    {"precompute_lors", &MultiGPUStageProfile::precomputeLORs},
	    {"allocate_projection_values",
	     &MultiGPUStageProfile::allocateProjectionValues},
	    {"load_lors", &MultiGPUStageProfile::loadLORs},
	    {"load_projection_values",
	     &MultiGPUStageProfile::loadProjectionValues},
	    {"forward_project_enqueue",
	     &MultiGPUStageProfile::forwardProjectEnqueue},
	    {"corrections_enqueue", &MultiGPUStageProfile::correctionsEnqueue},
	    {"ratio_enqueue", &MultiGPUStageProfile::ratioEnqueue},
	    {"pre_backproject_sync",
	     &MultiGPUStageProfile::preBackprojectSync},
	    {"backproject_enqueue", &MultiGPUStageProfile::backprojectEnqueue},
	    {"final_sync", &MultiGPUStageProfile::finalSync},
	    {"release_lors", &MultiGPUStageProfile::releaseLORs},
	    {"copy_partial_image", &MultiGPUStageProfile::copyPartialImage},
	    {"cuda_load_lors", &MultiGPUStageProfile::cudaLoadLORs},
	    {"cuda_forward_project", &MultiGPUStageProfile::cudaForwardProject},
	    {"cuda_corrections", &MultiGPUStageProfile::cudaCorrections},
	    {"cuda_ratio", &MultiGPUStageProfile::cudaRatio},
	    {"cuda_backproject", &MultiGPUStageProfile::cudaBackproject},
	};

	for (const auto& field : fields)
	{
		double sum = 0.0;
		double maxValue = 0.0;
		size_t count = 0;
		for (const auto& profile : workerProfiles)
		{
			const double value = profile.*(field.value);
			if (value <= 0.0)
			{
				continue;
			}
			sum += value;
			maxValue = std::max(maxValue, value);
			count++;
		}
		if (count == 0)
		{
			continue;
		}
		emitMultiGPUProfileMetric(prefix + "_" + field.name + "_sum", sum);
		emitMultiGPUProfileMetric(prefix + "_" + field.name + "_max",
		                          maxValue);
	}
}

bool isMultiGPUSensitivityEnabled()
{
	const char* envValue = std::getenv("YRT_MULTI_GPU_SENSITIVITY");
	return envValue == nullptr || envValue[0] == '\0' ||
	       envValue[0] != '0';
}

bool isMultiGPULORPreloadEnabled()
{
	const char* envValue = std::getenv("YRT_PRELOAD_MULTI_GPU_LORS");
	return envValue != nullptr && envValue[0] != '\0' &&
	       envValue[0] != '0';
}

std::string toLower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c)
	               { return static_cast<char>(std::tolower(c)); });
	return value;
}

MultiGPULORCacheMode getMultiGPULORCacheModeFromEnv()
{
	const char* envValue = std::getenv("YRT_CACHE_MULTI_GPU_LORS");
	if (envValue == nullptr || envValue[0] == '\0')
	{
		return MultiGPULORCacheMode::Off;
	}

	const std::string value = toLower(envValue);
	if (value == "0" || value == "off" || value == "false" ||
	    value == "no")
	{
		return MultiGPULORCacheMode::Off;
	}
	if (value == "host" || value == "cpu" || value == "pinned")
	{
		return MultiGPULORCacheMode::Host;
	}
	if (value == "1" || value == "on" || value == "true" ||
	    value == "yes" || value == "device" || value == "gpu")
	{
		return MultiGPULORCacheMode::Device;
	}
	throw std::invalid_argument(
	    "YRT_CACHE_MULTI_GPU_LORS must be 0, 1, host, or device");
}

float getSensitivityProjectionMemoryShare(const Corrector_GPU& corrector)
{
	float memoryShare = ProjectionDataDevice::DefaultMemoryShare;
	if (corrector.hasHardwareAttenuation())
	{
		const size_t memoryUsagePerLOR =
		    corrector.getSensImgGenProjData()->hasTOF() ?
		        LORsDevice::MemoryUsagePerLORWithTOF :
		        LORsDevice::MemoryUsagePerLOR;
		const size_t baseMemoryUsagePerEvent =
		    memoryUsagePerLOR + sizeof(float);
		const size_t correctedMemoryUsagePerEvent =
		    memoryUsagePerLOR + 2 * sizeof(float);
		memoryShare *=
		    static_cast<float>(baseMemoryUsagePerEvent) /
		    static_cast<float>(correctedMemoryUsagePerEvent);
	}

	return memoryShare;
}

const char* getMultiGPULORCacheModeName(MultiGPULORCacheMode cacheMode)
{
	switch (cacheMode)
	{
	case MultiGPULORCacheMode::Off:
		return "off";
	case MultiGPULORCacheMode::Host:
		return "host";
	case MultiGPULORCacheMode::Device:
		return "device";
	}
	return "unknown";
}

bool shouldUseHostLORCache(MultiGPULORCacheMode cacheMode, int numBatches)
{
	return cacheMode == MultiGPULORCacheMode::Host ||
	       (cacheMode == MultiGPULORCacheMode::Device && numBatches > 1);
}

void configureLORCache(ProjectionDataDevice& projectionData,
                       MultiGPULORCacheMode cacheMode)
{
	const int numBatches = projectionData.getNumBatches(0);
	projectionData.setHostLORCacheEnabled(
	    shouldUseHostLORCache(cacheMode, numBatches));
}

const char* getLORCachePlacementName(MultiGPULORCacheMode cacheMode,
                                     int numBatches)
{
	if (cacheMode == MultiGPULORCacheMode::Off)
	{
		return "off";
	}
	if (shouldUseHostLORCache(cacheMode, numBatches))
	{
		return "host-precomputed";
	}
	return "device-resident";
}

void printMultiGPUSetup(const char* operationName,
                        const std::vector<int>& deviceIds)
{
	std::cout << "Computing " << operationName << " on " << deviceIds.size()
	          << " GPUs (visible device ids:";
	for (int deviceId : deviceIds)
	{
		std::cout << " " << deviceId;
	}
	std::cout << ")" << std::endl;

	const char* cudaVisibleDevices = std::getenv("CUDA_VISIBLE_DEVICES");
	const char* yrtCudaDevices = std::getenv("YRT_CUDA_DEVICES");
	const MultiGPULORCacheMode lorCacheMode =
	    getMultiGPULORCacheModeFromEnv();
	const int ompThreadsPerWorker =
	    getOpenMPThreadsPerGPUWorker(deviceIds.size());
	std::cout << "CUDA_VISIBLE_DEVICES="
	          << (cudaVisibleDevices != nullptr ? cudaVisibleDevices : "<unset>")
	          << ", YRT_CUDA_DEVICES="
	          << (yrtCudaDevices != nullptr ? yrtCudaDevices : "<unset>")
	          << ", OpenMP threads/GPU worker=" << ompThreadsPerWorker
	          << ", LOR cache="
	          << getMultiGPULORCacheModeName(lorCacheMode)
	          << std::endl;
}

void copyDeviceImageAcrossGPUsAsync(const ImageDevice& sourceImage,
                                    int sourceDeviceId, ImageDevice& destImage,
                                    int destDeviceId, int streamDeviceId,
                                    const cudaStream_t& stream,
                                    cudaEvent_t completionEvent,
                                    const char* imageDescription)
{
	ASSERT_MSG(destImage.getParams().isSameDimensionsAs(sourceImage.getParams()),
	           "Image dimensions mismatch");
	ASSERT_MSG(streamDeviceId == sourceDeviceId ||
	               streamDeviceId == destDeviceId,
	           "Peer-copy stream must belong to source or destination device");

	const size_t byteCount = sourceImage.getImageSize() * sizeof(float);
	{
		ScopedCUDADevice guard(streamDeviceId);
		if (sourceDeviceId == destDeviceId)
		{
			cudaMemcpyAsync(destImage.getDevicePointer(),
			                sourceImage.getDevicePointer(), byteCount,
			                cudaMemcpyDeviceToDevice, stream);
			ASSERT(cudaCheckError());
			if (completionEvent != nullptr)
			{
				cudaEventRecord(completionEvent, stream);
				ASSERT(cudaCheckError());
			}
			return;
		}

		if (ensurePeerAccess(destDeviceId, sourceDeviceId))
		{
			cudaMemcpyPeerAsync(destImage.getDevicePointer(), destDeviceId,
			                    sourceImage.getDevicePointer(), sourceDeviceId,
			                    byteCount, stream);
			ASSERT(cudaCheckError());
			if (completionEvent != nullptr)
			{
				cudaEventRecord(completionEvent, stream);
				ASSERT(cudaCheckError());
			}
			return;
		}

		cudaStreamSynchronize(stream);
		ASSERT(cudaCheckError());
	}

	std::cout << "CUDA peer access is unavailable from visible device "
	          << destDeviceId << " to " << sourceDeviceId << "; staging "
	          << imageDescription << " through host memory." << std::endl;
	auto hostImage = std::make_unique<ImageOwned>(sourceImage.getParams());
	hostImage->allocate();
	{
		ScopedCUDADevice sourceGuard(sourceDeviceId);
		sourceImage.transferToHostMemory(hostImage.get(), true);
	}
	{
		ScopedCUDADevice destGuard(destDeviceId);
		destImage.copyFromHostImage(hostImage.get(), true);
	}
	if (completionEvent != nullptr)
	{
		ScopedCUDADevice guard(streamDeviceId);
		cudaEventRecord(completionEvent, stream);
		ASSERT(cudaCheckError());
	}
}

void destroyCudaEvents(std::vector<cudaEvent_t>& events,
                       const std::vector<int>& deviceIds)
{
	for (size_t eventId = 0; eventId < events.size(); eventId++)
	{
		if (events.at(eventId) != nullptr)
		{
			ScopedCUDADevice guard(deviceIds.at(eventId));
			cudaEventDestroy(events.at(eventId));
			events.at(eventId) = nullptr;
		}
	}
}

void sumPrimaryPartialImagesToDevice(
    const std::vector<std::unique_ptr<ImageDeviceOwned>>& primaryPartialImages,
    const std::vector<std::unique_ptr<ImageDeviceOwned>>& workerPartialImages,
    const std::vector<size_t>& activeWorkerIds,
    const std::vector<cudaEvent_t>& partialReadyEvents,
    const std::vector<int>& deviceIds, ImageDevice& destImage,
    int primaryDeviceId, const cudaStream_t* primaryStream)
{
	ScopedCUDADevice guard(primaryDeviceId);
	for (size_t workerId : activeWorkerIds)
	{
		if (partialReadyEvents.at(workerId) != nullptr)
		{
			if (primaryStream != nullptr)
			{
				cudaStreamWaitEvent(*primaryStream,
				                    partialReadyEvents.at(workerId), 0);
			}
			else
			{
				cudaEventSynchronize(partialReadyEvents.at(workerId));
			}
			ASSERT(cudaCheckError());
		}
	}

	if (activeWorkerIds.empty())
	{
		destImage.setValueDevice(0.0f, true);
		return;
	}

	destImage.setValueDevice(0.0f, true);
	for (size_t workerId : activeWorkerIds)
	{
		const auto& image =
		    deviceIds.at(workerId) == primaryDeviceId ?
		        workerPartialImages.at(workerId) :
		        primaryPartialImages.at(workerId);
		ASSERT(image != nullptr);
		image->addFirstImageToSecondDevice(&destImage, true);
	}
	ASSERT(cudaCheckError());
}

void sumPrimaryPartialImagesToDevice(
    const std::vector<std::unique_ptr<ImageDeviceOwned>>& primaryPartialImages,
    const std::vector<std::unique_ptr<ImageDeviceOwned>>& workerPartialImages,
    const std::vector<size_t>& activeWorkerIds,
    std::vector<cudaEvent_t>& partialReadyEvents,
    const std::vector<int>& deviceIds, ImageDevice& destImage,
    int primaryDeviceId, const cudaStream_t* primaryStream)
{
	try
	{
		const std::vector<cudaEvent_t>& readOnlyEvents = partialReadyEvents;
		sumPrimaryPartialImagesToDevice(
		    primaryPartialImages, workerPartialImages, activeWorkerIds,
		    readOnlyEvents, deviceIds, destImage, primaryDeviceId,
		    primaryStream);
	}
	catch (...)
	{
		destroyCudaEvents(partialReadyEvents, deviceIds);
		throw;
	}
	destroyCudaEvents(partialReadyEvents, deviceIds);
}

void rethrowMultiGPUWorkerErrors(
    const std::vector<std::exception_ptr>& errors,
    std::vector<cudaEvent_t>& partialReadyEvents,
    const std::vector<int>& deviceIds)
{
	for (const auto& error : errors)
	{
		if (error != nullptr)
		{
			destroyCudaEvents(partialReadyEvents, deviceIds);
			std::rethrow_exception(error);
		}
	}
}

}  // namespace

struct OSEMUpdater_GPU::MultiGPUReconWorkerContext
{
	~MultiGPUReconWorkerContext()
	{
		if (deviceId >= 0)
		{
			ScopedCUDADevice guard(deviceId);
			projector.reset();
			corrector.reset();
			tmpBufferDevice.reset();
			measurementsDevice.reset();
			mainStream.reset();
			auxStream.reset();
		}
	}

	int deviceId = -1;
	size_t workerId = 0;
	size_t start = 0;
	size_t count = 0;
	std::unique_ptr<GPUStream> mainStream;
	std::unique_ptr<GPUStream> auxStream;
	std::unique_ptr<BinIteratorSlice> sliceIterator;
	std::vector<const BinIterator*> binIterators;
	std::unique_ptr<ProjectionDataDeviceOwned> measurementsDevice;
	std::unique_ptr<ProjectionDataDeviceOwned> tmpBufferDevice;
	std::unique_ptr<Corrector_GPU> corrector;
	const ProjectionDataDevice* correctorTempBuffer = nullptr;
	std::unique_ptr<OperatorProjectorDevice> projector;
};

struct OSEMUpdater_GPU::MultiGPUReconCache
{
	std::vector<int> deviceIds;
	const ProjectionData* dataInput = nullptr;
	int numSubsets = 0;
	std::vector<std::vector<std::unique_ptr<MultiGPUReconWorkerContext>>>
	    contexts;
};

OSEMUpdater_GPU::OSEMUpdater_GPU(OSEM_GPU* pp_osem) : mp_osem(pp_osem)
{
	ASSERT(mp_osem != nullptr);
}

OSEMUpdater_GPU::~OSEMUpdater_GPU()
{
	releaseMultiGPUImageBuffers(m_sensitivityBuffers);
	releaseMultiGPUImageBuffers(m_emBuffers);
	releaseMultiGPUReconCache();
}

bool OSEMUpdater_GPU::isMultiGPULORCacheEnabled() const
{
	return getMultiGPULORCacheModeFromEnv() != MultiGPULORCacheMode::Off;
}

void OSEMUpdater_GPU::releaseMultiGPUReconCache() const
{
	mp_reconCache.reset();
}

void OSEMUpdater_GPU::preloadMultiGPULORCacheIfRequested() const
{
	const MultiGPULORCacheMode lorCacheMode =
	    getMultiGPULORCacheModeFromEnv();
	if (!mp_osem->isMultiGPUEnabled() ||
	    lorCacheMode == MultiGPULORCacheMode::Off ||
	    !isMultiGPULORPreloadEnabled())
	{
		return;
	}

	const auto& deviceIds = mp_osem->getDeviceIds();
	std::cout << "Preloading multi-GPU LOR cache for "
	          << mp_osem->num_OSEM_subsets << " subsets on "
	          << deviceIds.size() << " GPUs (cache mode: "
	          << getMultiGPULORCacheModeName(lorCacheMode) << ")..."
	          << std::endl;

	const bool profileEnabled = isMultiGPUProfileEnabled();
	const auto startTime = std::chrono::steady_clock::now();
	double ensureReconCacheSeconds = 0.0;
	double workersWallSeconds = 0.0;
	std::vector<MultiGPUStageProfile> workerProfiles(deviceIds.size());

	std::vector<std::vector<std::pair<size_t, size_t>>> rangesBySubset;
	rangesBySubset.reserve(mp_osem->num_OSEM_subsets);
	for (int subsetId = 0; subsetId < mp_osem->num_OSEM_subsets; subsetId++)
	{
		const BinIterator& subsetIterator = *mp_osem->getBinIterator(subsetId);
		rangesBySubset.push_back(
		    splitWork(subsetIterator.size(), deviceIds.size()));
		{
			ScopedProfileTimer profileTimer(profileEnabled,
			                                ensureReconCacheSeconds);
			ensureMultiGPUReconCache(subsetId, subsetIterator,
			                         rangesBySubset.back());
		}
	}

	std::vector<std::exception_ptr> errors(deviceIds.size());
	std::vector<std::thread> workers;
	workers.reserve(deviceIds.size());
	const auto workersStartTime = std::chrono::steady_clock::now();

	for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
	{
		workers.emplace_back(
		    [&, workerId]()
		    {
			    try
			    {
				    auto& workerProfile = workerProfiles.at(workerId);
				    ScopedProfileTimer totalProfileTimer(profileEnabled,
				                                         workerProfile.total);
				    const int deviceId = deviceIds.at(workerId);
				    ScopedOpenMPThreads ompGuard(
				        getOpenMPThreadsPerGPUWorker(deviceIds.size()));
				    ScopedCUDADevice guard(deviceId);

				    for (int subsetId = 0;
				         subsetId < mp_osem->num_OSEM_subsets; subsetId++)
				    {
					    const auto& ranges = rangesBySubset.at(subsetId);
					    if (ranges.at(workerId).second == 0)
					    {
						    continue;
					    }

					    auto* context = mp_reconCache->contexts
					                        .at(subsetId)
					                        .at(workerId)
					                        .get();
					    ASSERT(context != nullptr);
					    auto* measurementsDevice =
					        context->measurementsDevice.get();
					    GPUStream& mainStream = *context->mainStream;
					    const int numBatches =
					        measurementsDevice->getNumBatches(0);

					    if (lorCacheMode == MultiGPULORCacheMode::Device &&
					        numBatches == 1)
					    {
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.precomputeLORs);
							    measurementsDevice->precomputeBatchLORs(0, 0);
						    }
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled, workerProfile.loadLORs);
							    measurementsDevice->loadPrecomputedLORsToDevice(
							        {&mainStream.getStream(), true});
						    }
						    continue;
					    }

					    std::cout
					        << "Precomputing host LOR cache for subset "
					        << subsetId + 1 << ", worker " << workerId
					        << " (" << numBatches << " batches)."
					        << std::endl;
					    for (int batch = 0; batch < numBatches; batch++)
					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled, workerProfile.precomputeLORs);
						    measurementsDevice->precomputeBatchLORs(0, batch);
					    }
				    }
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
	if (profileEnabled)
	{
		workersWallSeconds = secondsSince(workersStartTime);
	}
	for (const auto& error : errors)
	{
		if (error != nullptr)
		{
			std::rethrow_exception(error);
		}
	}

	const auto endTime = std::chrono::steady_clock::now();
	const std::chrono::duration<double> elapsed = endTime - startTime;
	std::cout << "Preloaded multi-GPU LOR cache in " << elapsed.count()
	          << " seconds." << std::endl;
	emitMultiGPUProfileMetric("preload_total", elapsed.count());
	emitMultiGPUProfileMetric("preload_ensure_recon_cache",
	                          ensureReconCacheSeconds);
	emitMultiGPUProfileMetric("preload_workers_wall", workersWallSeconds);
	emitMultiGPUProfileAggregate("preload", workerProfiles);
}

void OSEMUpdater_GPU::ensureMultiGPUReconCache(
    int subsetId, const BinIterator& subsetIterator,
    const std::vector<std::pair<size_t, size_t>>& ranges) const
{
	const auto& deviceIds = mp_osem->getDeviceIds();
	const ProjectionData* dataInput = mp_osem->getDataInput();
	const MultiGPULORCacheMode lorCacheMode =
	    getMultiGPULORCacheModeFromEnv();
	const bool cacheValid =
	    mp_reconCache != nullptr && mp_reconCache->deviceIds == deviceIds &&
	    mp_reconCache->dataInput == dataInput &&
	    mp_reconCache->numSubsets == mp_osem->num_OSEM_subsets;

	if (!cacheValid)
	{
		mp_reconCache = std::make_unique<MultiGPUReconCache>();
		mp_reconCache->deviceIds = deviceIds;
		mp_reconCache->dataInput = dataInput;
		mp_reconCache->numSubsets = mp_osem->num_OSEM_subsets;
		mp_reconCache->contexts.resize(mp_osem->num_OSEM_subsets);
		for (auto& subsetContexts : mp_reconCache->contexts)
		{
			subsetContexts.resize(deviceIds.size());
		}
	}

	ASSERT(subsetId >= 0);
	auto& subsetContexts = mp_reconCache->contexts.at(subsetId);
	for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
	{
		const size_t start = ranges.at(workerId).first;
		const size_t count = ranges.at(workerId).second;
		if (count == 0)
		{
			continue;
		}
		if (subsetContexts.at(workerId) != nullptr)
		{
			configureLORCache(*subsetContexts.at(workerId)->measurementsDevice,
			                  lorCacheMode);
			continue;
		}

		const int deviceId = deviceIds.at(workerId);
		ScopedCUDADevice guard(deviceId);
		auto context = std::make_unique<MultiGPUReconWorkerContext>();
		context->deviceId = deviceId;
		context->workerId = workerId;
		context->start = start;
		context->count = count;
		context->mainStream = std::make_unique<GPUStream>();
		context->auxStream = std::make_unique<GPUStream>();
		context->sliceIterator =
		    std::make_unique<BinIteratorSlice>(subsetIterator, start, count);
		context->binIterators = {context->sliceIterator.get()};
		context->measurementsDevice =
		    std::make_unique<ProjectionDataDeviceOwned>(
		        mp_osem->scanner, dataInput, context->binIterators);
		context->tmpBufferDevice =
		    std::make_unique<ProjectionDataDeviceOwned>(
		        context->measurementsDevice.get());
		configureLORCache(*context->measurementsDevice, lorCacheMode);
		context->corrector =
		    std::make_unique<Corrector_GPU>(mp_osem->getCorrector_GPU());
		context->corrector->initializeTemporaryDeviceBuffer(
		    context->measurementsDevice.get());
		context->correctorTempBuffer =
		    context->corrector->getTemporaryDeviceBuffer();
		context->projector = mp_osem->createDeviceProjector(
		    context->sliceIterator.get(), true,
		    &context->mainStream->getStream(),
		    &context->auxStream->getStream());

		subsetContexts.at(workerId) = std::move(context);
	}
}

void OSEMUpdater_GPU::releaseMultiGPUImageBuffers(
    MultiGPUImageBuffers& buffers) const
{
	if (!buffers.primaryPartialImages.empty())
	{
		if (!buffers.deviceIds.empty())
		{
			ScopedCUDADevice guard(buffers.deviceIds.front());
			for (auto& image : buffers.primaryPartialImages)
			{
				image.reset();
			}
		}
	}

	for (size_t workerId = 0; workerId < buffers.workerInputImages.size();
	     workerId++)
	{
		if (buffers.workerInputImages.at(workerId) != nullptr &&
		    workerId < buffers.deviceIds.size())
		{
			ScopedCUDADevice guard(buffers.deviceIds.at(workerId));
			buffers.workerInputImages.at(workerId).reset();
		}
	}
	for (size_t workerId = 0; workerId < buffers.workerPartialImages.size();
	     workerId++)
	{
		if (buffers.workerPartialImages.at(workerId) != nullptr &&
		    workerId < buffers.deviceIds.size())
		{
			ScopedCUDADevice guard(buffers.deviceIds.at(workerId));
			buffers.workerPartialImages.at(workerId).reset();
		}
	}

	buffers.primaryPartialImages.clear();
	buffers.workerInputImages.clear();
	buffers.workerPartialImages.clear();
	buffers.deviceIds.clear();
	buffers.initialized = false;
}

void OSEMUpdater_GPU::ensureMultiGPUImageBuffers(
    MultiGPUImageBuffers& buffers, const ImageParams& params,
    bool allocateInputImages) const
{
	const auto& deviceIds = mp_osem->getDeviceIds();
	const bool cacheValid =
	    buffers.initialized && buffers.params.isSameAs(params) &&
	    buffers.deviceIds == deviceIds &&
	    buffers.primaryPartialImages.size() == deviceIds.size() &&
	    buffers.workerPartialImages.size() == deviceIds.size() &&
	    (!allocateInputImages ||
	     buffers.workerInputImages.size() == deviceIds.size());

	if (cacheValid)
	{
		return;
	}

	releaseMultiGPUImageBuffers(buffers);
	buffers.params = params;
	buffers.deviceIds = deviceIds;
	buffers.primaryPartialImages.resize(deviceIds.size());
	buffers.workerPartialImages.resize(deviceIds.size());
	if (allocateInputImages)
	{
		buffers.workerInputImages.resize(deviceIds.size());
	}

	const int primaryDeviceId = mp_osem->getPrimaryDeviceId();
	{
		ScopedCUDADevice guard(primaryDeviceId);
		for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
		{
			if (deviceIds.at(workerId) != primaryDeviceId)
			{
				buffers.primaryPartialImages.at(workerId) =
				    std::make_unique<ImageDeviceOwned>(params);
				buffers.primaryPartialImages.at(workerId)->allocate(true);
			}
		}
	}

	for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
	{
		ScopedCUDADevice guard(deviceIds.at(workerId));
		buffers.workerPartialImages.at(workerId) =
		    std::make_unique<ImageDeviceOwned>(params);
		buffers.workerPartialImages.at(workerId)->allocate(true);

		if (allocateInputImages)
		{
			if (deviceIds.at(workerId) != primaryDeviceId)
			{
				buffers.workerInputImages.at(workerId) =
				    std::make_unique<ImageDeviceOwned>(params);
				buffers.workerInputImages.at(workerId)->allocate(true);
			}
		}
	}

	buffers.initialized = true;
}

void OSEMUpdater_GPU::computeSensitivityImage(ImageDevice& destImage) const
{
	if (mp_osem->isMultiGPUEnabled() && isMultiGPUSensitivityEnabled())
	{
		computeSensitivityImageMultiGPU(destImage);
		return;
	}

	OperatorProjectorDevice* projector = mp_osem->getProjector();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	Corrector_GPU& corrector = mp_osem->getCorrector_GPU();

	const cudaStream_t* mainStream = mp_osem->getMainStream();
	const cudaStream_t* auxStream = mp_osem->getAuxStream();

	ProjectionDataDeviceOwned* sensDataBuffer =
	    mp_osem->getSensitivityDataDeviceBuffer();
	const int numBatchesInCurrentSubset =
	    sensDataBuffer->getNumBatches(currentSubset);

	bool loadGlobalScalingFactor = !corrector.hasMultiplicativeCorrection();

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Loading batch " << batch + 1 << "/"
		          << numBatchesInCurrentSubset << "..." << std::endl;

		sensDataBuffer->precomputeBatchLORs(currentSubset, batch);

		// Allocate for the projection values
		const bool hasReallocated =
		    sensDataBuffer->allocateForProjValues({mainStream, false});

		sensDataBuffer->loadPrecomputedLORsToDevice({mainStream, false});

		// Load the projection values to backproject
		// This will either load projection values from sensitivity histogram,
		//  from ACF histogram, or it will load "ones" from a uniform histogram
		sensDataBuffer->loadProjValuesFromReference({mainStream, false});

		// Load the projection values to the device buffer depending on the
		//  situation
		if (corrector.hasSensitivityHistogram())
		{
			// Apply global scaling factor if it's not 1.0
			if (corrector.hasGlobalScalingFactor())
			{
				sensDataBuffer->multiplyProjValues(
				    corrector.getGlobalScalingFactor(), {mainStream, false});
			}

			// Invert sensitivity if needed
			if (corrector.mustInvertSensitivity())
			{
				sensDataBuffer->invertProjValuesDevice({mainStream, false});
			}
		}
		if (corrector.hasHardwareAttenuationImage())
		{
			// TODO: it would be faster if this was done on the auxiliary stream
			//  so that a backprojection is done at the same time as a forward
			corrector
			    .applyHardwareAttenuationToGivenDeviceBufferFromAttenuationImage(
			        sensDataBuffer, projector, {mainStream, false});
		}
		else if (corrector.doesHardwareACFComeFromHistogram())
		{
			corrector
			    .applyHardwareAttenuationToGivenDeviceBufferFromACFHistogram(
			        sensDataBuffer, {mainStream, false});
		}

		if (!corrector.hasMultiplicativeCorrection() &&
		    (loadGlobalScalingFactor || hasReallocated))
		{
			// Need to set all bins to the global scaling factor value, but only
			//  do it the first time (unless a reallocation has occured)
			sensDataBuffer->clearProjections(
			    corrector.getGlobalScalingFactor());
			loadGlobalScalingFactor = false;
		}

		if (mainStream != nullptr)
		{
			cudaStreamSynchronize(*mainStream);
		}

		std::cout << "Backprojecting batch " << batch + 1 << "/"
		          << numBatchesInCurrentSubset << "..." << std::endl;

		// Backproject values
		projector->applyAH(sensDataBuffer, &destImage, false);
	}

	if (mainStream != nullptr)
	{
		cudaStreamSynchronize(*mainStream);
	}
}

void OSEMUpdater_GPU::computeEMUpdateImage(const ImageDevice& inputImage,
                                           ImageDevice& destImage) const
{
	if (mp_osem->isMultiGPUEnabled())
	{
		computeEMUpdateImageMultiGPU(inputImage, destImage);
		return;
	}

	OperatorProjectorDevice* projector = mp_osem->getProjector();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	Corrector_GPU& corrector = mp_osem->getCorrector_GPU();

	const cudaStream_t* mainStream = mp_osem->getMainStream();
	const cudaStream_t* auxStream = mp_osem->getAuxStream();

	ProjectionDataDeviceOwned* measurementsDevice =
	    mp_osem->getMLEMDataDeviceBuffer();
	ProjectionDataDeviceOwned* tmpBufferDevice =
	    mp_osem->getMLEMDataTmpDeviceBuffer();
	const ProjectionDataDevice* correctorTempBuffer =
	    corrector.getTemporaryDeviceBuffer();
	const bool measurementsAreUniform = mp_osem->getDataInput()->isUniform();

	ASSERT(projector != nullptr);
	ASSERT(measurementsDevice != nullptr);
	ASSERT(tmpBufferDevice != nullptr);
	ASSERT(destImage.isMemoryValid());

	const int numBatchesInCurrentSubset =
	    measurementsDevice->getNumBatches(currentSubset);

	for (int batch = 0; batch < numBatchesInCurrentSubset; batch++)
	{
		std::cout << "Batch " << batch + 1 << "/" << numBatchesInCurrentSubset
		          << "..." << std::endl;
		measurementsDevice->precomputeBatchLORs(currentSubset, batch);

		measurementsDevice->loadPrecomputedLORsToDevice({mainStream, false});
		if (!measurementsAreUniform)
		{
			measurementsDevice->allocateForProjValues({mainStream, false});
			measurementsDevice->loadProjValuesFromReference({mainStream, false});
		}

		tmpBufferDevice->allocateForProjValues({mainStream, false});

		projector->applyA(&inputImage, tmpBufferDevice, false);

		if (corrector.hasAdditiveCorrection(*measurementsDevice))
		{
			corrector.loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
			    {mainStream, false});

			// We need to synchronize the stream here if we want to apply both
			// the additive corrections AND the pre-correction. This is because
			// they both use the same buffer. If this is not done, and the GPU
			// forward projection takes longer than both
			// `loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(...)` and
			// `loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(...)`, the
			// code would end up treating the precorrection factors as additive
			// factors...
			const bool synchronize = corrector.hasInVivoAttenuation();

			tmpBufferDevice->addProjValues(correctorTempBuffer,
			                               {mainStream, synchronize});
		}
		if (corrector.hasInVivoAttenuation())
		{
			corrector.loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
			    {mainStream, false});
			tmpBufferDevice->multiplyProjValues(correctorTempBuffer,
			                                    {mainStream, false});
		}

		if (measurementsAreUniform)
		{
			tmpBufferDevice->invertProjValuesDevice({mainStream, false});
		}
		else
		{
			tmpBufferDevice->divideMeasurementsDevice(measurementsDevice,
			                                          {mainStream, false});
		}

		if (mainStream != nullptr)
		{
			cudaStreamSynchronize(*mainStream);
		}

		projector->applyAH(tmpBufferDevice, &destImage, false);
	}

	if (mainStream != nullptr)
	{
		cudaStreamSynchronize(*mainStream);
	}
}

void OSEMUpdater_GPU::computeSensitivityImageMultiGPU(
    ImageDevice& destImage) const
{
	const auto& deviceIds = mp_osem->getDeviceIds();
	const BinIterator& subsetIterator = *mp_osem->getCurrentBinIterator();
	const auto ranges = splitWork(subsetIterator.size(), deviceIds.size());
	const int primaryDeviceId = mp_osem->getPrimaryDeviceId();

	const bool profileEnabled = isMultiGPUProfileEnabled();
	const auto totalStartTime = std::chrono::steady_clock::now();
	double ensureImageBuffersSeconds = 0.0;
	double workersWallSeconds = 0.0;
	double reducePartialImagesSeconds = 0.0;

	printMultiGPUSetup("sensitivity image", deviceIds);
	{
		ScopedProfileTimer profileTimer(profileEnabled,
		                                ensureImageBuffersSeconds);
		ensureMultiGPUImageBuffers(m_sensitivityBuffers, destImage.getParams(),
		                           false);
	}

	std::vector<std::exception_ptr> errors(deviceIds.size());
	std::vector<cudaEvent_t> partialReadyEvents(deviceIds.size(), nullptr);
	std::vector<size_t> activeWorkerIds;
	activeWorkerIds.reserve(deviceIds.size());
	std::vector<std::thread> workers;
	workers.reserve(deviceIds.size());
	std::vector<MultiGPUStageProfile> workerProfiles(deviceIds.size());
	const auto workersStartTime = std::chrono::steady_clock::now();

	for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
	{
		const size_t start = ranges.at(workerId).first;
		const size_t count = ranges.at(workerId).second;
		if (count == 0)
		{
			continue;
		}
		activeWorkerIds.push_back(workerId);

		workers.emplace_back(
		    [&, workerId, start, count]()
			    {
				    cudaEvent_t copyDoneEvent = nullptr;
				    try
				    {
					    auto& workerProfile = workerProfiles.at(workerId);
					    ScopedProfileTimer totalProfileTimer(
					        profileEnabled, workerProfile.total);
					    const int deviceId = deviceIds.at(workerId);
					    ScopedOpenMPThreads ompGuard(
					        getOpenMPThreadsPerGPUWorker(deviceIds.size()));
				    ScopedCUDADevice guard(deviceId);
				    GPUStream mainStream;
				    GPUStream auxStream;

				    BinIteratorSlice sliceIterator(subsetIterator, start,
				                                   count);
				    std::vector<const BinIterator*> binIterators{
				        &sliceIterator};

				    Corrector_GPU corrector(mp_osem->getCorrector_GPU());
				    if (corrector.hasHardwareAttenuationImage())
				    {
					    // Reserve the attenuation image before GPU batch sizing.
					    // Otherwise the per-worker ProjectionDataDevice can claim
					    // nearly all free VRAM and leave no room for this image.
					    corrector.initializeTemporaryDeviceImageIfNeeded(
					        corrector.getHardwareAttenuationImage(),
					        {&mainStream.getStream(), true});
				    }

				    auto sensDataBuffer =
				        std::make_unique<ProjectionDataDeviceOwned>(
				            mp_osem->scanner,
				            corrector.getSensImgGenProjData(), binIterators,
				            getSensitivityProjectionMemoryShare(corrector));
				    corrector.initializeTemporaryDeviceBuffer(
				        sensDataBuffer.get());

				    auto projector = mp_osem->createDeviceProjector(
				        &sliceIterator, false, &mainStream.getStream(),
				        &auxStream.getStream());

					    ImageDeviceOwned* partialImageDevice =
					        m_sensitivityBuffers.workerPartialImages.at(
					            workerId)
					            .get();
					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled,
						        workerProfile.clearPartialImage);
						    partialImageDevice->setValueDevice(0.0f, true);
					    }

				    const int numBatches = sensDataBuffer->getNumBatches(0);
				    std::cout << "Multi-GPU sensitivity worker " << workerId
				              << " using visible CUDA device " << deviceId
				              << " for " << count << " LORs in "
				              << numBatches << " batches." << std::endl;

				    bool loadGlobalScalingFactor =
				        !corrector.hasMultiplicativeCorrection();

					    for (int batch = 0; batch < numBatches; batch++)
					    {
						    bool hasReallocated = false;
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.precomputeLORs);
							    sensDataBuffer->precomputeBatchLORs(0, batch);
						    }
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.allocateProjectionValues);
							    hasReallocated =
							        sensDataBuffer->allocateForProjValues(
							            {&mainStream.getStream(), false});
						    }
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled, workerProfile.loadLORs);
							    sensDataBuffer->loadPrecomputedLORsToDevice(
							        {&mainStream.getStream(), false});
						    }
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.loadProjectionValues);
							    sensDataBuffer->loadProjValuesFromReference(
							        {&mainStream.getStream(), false});
						    }

						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.correctionsEnqueue);
							    if (corrector.hasSensitivityHistogram())
							    {
								    if (corrector.hasGlobalScalingFactor())
								    {
									    sensDataBuffer->multiplyProjValues(
									        corrector.getGlobalScalingFactor(),
									        {&mainStream.getStream(), false});
								    }
								    if (corrector.mustInvertSensitivity())
								    {
									    sensDataBuffer->invertProjValuesDevice(
									        {&mainStream.getStream(), false});
								    }
							    }
							    if (corrector.hasHardwareAttenuationImage())
							    {
								    corrector
								        .applyHardwareAttenuationToGivenDeviceBufferFromAttenuationImage(
								            sensDataBuffer.get(),
								            projector.get(),
								            {&mainStream.getStream(), false});
							    }
							    else if (
							        corrector
							            .doesHardwareACFComeFromHistogram())
							    {
								    corrector
								        .applyHardwareAttenuationToGivenDeviceBufferFromACFHistogram(
								            sensDataBuffer.get(),
								            {&mainStream.getStream(), false});
							    }

							    if (!corrector.hasMultiplicativeCorrection() &&
							        (loadGlobalScalingFactor ||
							         hasReallocated))
							    {
								    sensDataBuffer->clearProjections(
								        corrector.getGlobalScalingFactor());
								    loadGlobalScalingFactor = false;
							    }
						    }

						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.preBackprojectSync);
							    cudaStreamSynchronize(mainStream.getStream());
						    }
						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.backprojectEnqueue);
							    projector->applyAH(sensDataBuffer.get(),
							                       partialImageDevice, false);
						    }
					    }

					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled, workerProfile.finalSync);
						    cudaStreamSynchronize(mainStream.getStream());
					    }
					    ASSERT(cudaCheckError());
					    cudaEventCreateWithFlags(&copyDoneEvent,
					                             cudaEventDisableTiming);
				    ASSERT(cudaCheckError());
				    if (deviceId == primaryDeviceId)
				    {
					    cudaEventRecord(copyDoneEvent, mainStream.getStream());
					    ASSERT(cudaCheckError());
				    }
					    else
					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled,
						        workerProfile.copyPartialImage);
						    copyDeviceImageAcrossGPUsAsync(
						        *partialImageDevice, deviceId,
						        *m_sensitivityBuffers.primaryPartialImages.at(
						            workerId),
						        primaryDeviceId, deviceId,
						        mainStream.getStream(), copyDoneEvent,
						        "one sensitivity partial image");
					    }
				    partialReadyEvents.at(workerId) = copyDoneEvent;
				    copyDoneEvent = nullptr;
			    }
			    catch (...)
			    {
				    if (copyDoneEvent != nullptr)
				    {
					    ScopedCUDADevice guard(deviceIds.at(workerId));
					    cudaEventDestroy(copyDoneEvent);
				    }
				    errors.at(workerId) = std::current_exception();
			    }
		    });
	}

	for (auto& worker : workers)
	{
		worker.join();
	}
	if (profileEnabled)
	{
		workersWallSeconds = secondsSince(workersStartTime);
	}
	rethrowMultiGPUWorkerErrors(errors, partialReadyEvents, deviceIds);

	{
		ScopedProfileTimer profileTimer(profileEnabled,
		                                reducePartialImagesSeconds);
		sumPrimaryPartialImagesToDevice(
		    m_sensitivityBuffers.primaryPartialImages,
		    m_sensitivityBuffers.workerPartialImages, activeWorkerIds,
		    partialReadyEvents, deviceIds, destImage, primaryDeviceId,
		    mp_osem->getMainStream());
	}
	emitMultiGPUProfileMetric("sensitivity_total",
	                          secondsSince(totalStartTime));
	emitMultiGPUProfileMetric("sensitivity_ensure_image_buffers",
	                          ensureImageBuffersSeconds);
	emitMultiGPUProfileMetric("sensitivity_workers_wall",
	                          workersWallSeconds);
	emitMultiGPUProfileMetric("sensitivity_reduce_partials",
	                          reducePartialImagesSeconds);
	emitMultiGPUProfileAggregate("sensitivity", workerProfiles);
}

void OSEMUpdater_GPU::computeEMUpdateImageMultiGPU(
    const ImageDevice& inputImage, ImageDevice& destImage) const
{
	const auto& deviceIds = mp_osem->getDeviceIds();
	const BinIterator& subsetIterator = *mp_osem->getCurrentBinIterator();
	const auto ranges = splitWork(subsetIterator.size(), deviceIds.size());
	const int primaryDeviceId = mp_osem->getPrimaryDeviceId();
	const int currentSubset = mp_osem->getCurrentOSEMSubset();
	const MultiGPULORCacheMode lorCacheMode =
	    getMultiGPULORCacheModeFromEnv();
	const bool useLORCache = lorCacheMode != MultiGPULORCacheMode::Off;
	const bool measurementsAreUniform = mp_osem->getDataInput()->isUniform();

	const bool profileEnabled = isMultiGPUProfileEnabled();
	const bool cudaProfileEnabled =
	    profileEnabled && isMultiGPUCUDAProfileEnabled();
	const auto totalStartTime = std::chrono::steady_clock::now();
	double syncPrimarySeconds = 0.0;
	double ensureImageBuffersSeconds = 0.0;
	double ensureReconCacheSeconds = 0.0;
	double workersWallSeconds = 0.0;
	double reducePartialImagesSeconds = 0.0;

	printMultiGPUSetup("EM update image", deviceIds);
	{
		ScopedProfileTimer profileTimer(profileEnabled, syncPrimarySeconds);
		ScopedCUDADevice primaryGuard(primaryDeviceId);
		cudaDeviceSynchronize();
		ASSERT(cudaCheckError());
	}
	{
		ScopedProfileTimer profileTimer(profileEnabled,
		                                ensureImageBuffersSeconds);
		ensureMultiGPUImageBuffers(m_emBuffers, destImage.getParams(), true);
	}
	if (useLORCache)
	{
		ScopedProfileTimer profileTimer(profileEnabled,
		                                ensureReconCacheSeconds);
		ensureMultiGPUReconCache(currentSubset, subsetIterator, ranges);
	}

	std::vector<std::exception_ptr> errors(deviceIds.size());
	std::vector<cudaEvent_t> partialReadyEvents(deviceIds.size(), nullptr);
	std::vector<size_t> activeWorkerIds;
	activeWorkerIds.reserve(deviceIds.size());
	std::vector<std::thread> workers;
	workers.reserve(deviceIds.size());
	std::vector<MultiGPUStageProfile> workerProfiles(deviceIds.size());
	const auto workersStartTime = std::chrono::steady_clock::now();

	for (size_t workerId = 0; workerId < deviceIds.size(); workerId++)
	{
		const size_t start = ranges.at(workerId).first;
		const size_t count = ranges.at(workerId).second;
		if (count == 0)
		{
			continue;
		}
		activeWorkerIds.push_back(workerId);

		workers.emplace_back(
		    [&, workerId, start, count]()
			    {
				    cudaEvent_t copyDoneEvent = nullptr;
				    try
				    {
					    auto& workerProfile = workerProfiles.at(workerId);
					    ScopedProfileTimer totalProfileTimer(
					        profileEnabled, workerProfile.total);
					    const int deviceId = deviceIds.at(workerId);
					    ScopedOpenMPThreads ompGuard(
					        getOpenMPThreadsPerGPUWorker(deviceIds.size()));
				    ScopedCUDADevice guard(deviceId);

				    std::unique_ptr<GPUStream> localMainStream;
				    std::unique_ptr<GPUStream> localAuxStream;
				    std::unique_ptr<BinIteratorSlice> localSliceIterator;
				    std::vector<const BinIterator*> localBinIterators;
				    std::unique_ptr<ProjectionDataDeviceOwned>
				        localMeasurementsDevice;
				    std::unique_ptr<ProjectionDataDeviceOwned>
				        localTmpBufferDevice;
				    std::unique_ptr<Corrector_GPU> localCorrector;
				    std::unique_ptr<OperatorProjectorDevice> localProjector;

				    GPUStream* mainStreamPtr = nullptr;
				    ProjectionDataDeviceOwned* measurementsDevice = nullptr;
				    ProjectionDataDeviceOwned* tmpBufferDevice = nullptr;
				    Corrector_GPU* corrector = nullptr;
				    const ProjectionDataDevice* correctorTempBuffer = nullptr;
				    OperatorProjectorDevice* projector = nullptr;

				    if (useLORCache)
				    {
					    auto* context = mp_reconCache->contexts
					                        .at(currentSubset)
					                        .at(workerId)
					                        .get();
					    ASSERT(context != nullptr);
					    mainStreamPtr = context->mainStream.get();
					    measurementsDevice = context->measurementsDevice.get();
					    tmpBufferDevice = context->tmpBufferDevice.get();
					    corrector = context->corrector.get();
					    correctorTempBuffer = context->correctorTempBuffer;
					    projector = context->projector.get();
				    }
				    else
				    {
					    localMainStream = std::make_unique<GPUStream>();
					    localAuxStream = std::make_unique<GPUStream>();
					    localSliceIterator =
					        std::make_unique<BinIteratorSlice>(
					            subsetIterator, start, count);
					    localBinIterators = {localSliceIterator.get()};
					    localMeasurementsDevice =
					        std::make_unique<ProjectionDataDeviceOwned>(
					            mp_osem->scanner, mp_osem->getDataInput(),
					            localBinIterators);
					    localTmpBufferDevice =
					        std::make_unique<ProjectionDataDeviceOwned>(
					            localMeasurementsDevice.get());
					    localCorrector = std::make_unique<Corrector_GPU>(
					        mp_osem->getCorrector_GPU());
					    localCorrector->initializeTemporaryDeviceBuffer(
					        localMeasurementsDevice.get());
					    localProjector = mp_osem->createDeviceProjector(
					        localSliceIterator.get(), true,
					        &localMainStream->getStream(),
					        &localAuxStream->getStream());

					    mainStreamPtr = localMainStream.get();
					    measurementsDevice = localMeasurementsDevice.get();
					    tmpBufferDevice = localTmpBufferDevice.get();
					    corrector = localCorrector.get();
					    correctorTempBuffer =
					        localCorrector->getTemporaryDeviceBuffer();
					    projector = localProjector.get();
				    }
				    GPUStream& mainStream = *mainStreamPtr;

				    const ImageDevice* inputImageForWorker = &inputImage;
				    if (deviceId != primaryDeviceId)
				    {
						    ImageDeviceOwned* inputImageDevice =
						        m_emBuffers.workerInputImages.at(workerId).get();
						    ASSERT(inputImageDevice != nullptr);
						    ScopedProfileTimer profileTimer(
						        profileEnabled, workerProfile.copyInputImage);
						    copyDeviceImageAcrossGPUsAsync(
						        inputImage, primaryDeviceId, *inputImageDevice,
						        deviceId, deviceId, mainStream.getStream(),
						        nullptr, "the EM input image");
						    inputImageForWorker = inputImageDevice;
					    }

					    ImageDeviceOwned* partialImageDevice =
					        m_emBuffers.workerPartialImages.at(workerId).get();
					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled,
						        workerProfile.clearPartialImage);
						    partialImageDevice->setValueDevice(0.0f, true);
					    }

				    const int numBatches =
				        measurementsDevice->getNumBatches(0);
					    std::cout << "Multi-GPU EM worker " << workerId
					              << " using visible CUDA device " << deviceId
					              << " for " << count << " LORs in "
					              << numBatches << " batches (LOR cache: "
					              << getLORCachePlacementName(lorCacheMode,
					                                          numBatches)
					              << ")." << std::endl;

						    std::vector<CUDAEventProfileTimer>
						        backprojectCudaTimers;
						    backprojectCudaTimers.reserve(numBatches);
						    for (int batch = 0; batch < numBatches; batch++)
						    {
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
							        workerProfile.precomputeLORs);
								    measurementsDevice->precomputeBatchLORs(
								        0, batch);
							    }
							    CUDAEventProfileTimer loadLORsCudaTimer(
							        cudaProfileEnabled, mainStream.getStream());
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled, workerProfile.loadLORs);
								    measurementsDevice
								        ->loadPrecomputedLORsToDevice(
								            {&mainStream.getStream(), false});
							    }
							    loadLORsCudaTimer.stop();
							    if (!measurementsAreUniform)
							    {
								    {
									    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile
								            .allocateProjectionValues);
								    measurementsDevice->allocateForProjValues(
								        {&mainStream.getStream(), false});
							    }
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile.loadProjectionValues);
								    measurementsDevice
								        ->loadProjValuesFromReference(
								            {&mainStream.getStream(), false});
							    }
						    }

						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled,
							        workerProfile.allocateProjectionValues);
								    tmpBufferDevice->allocateForProjValues(
								        {&mainStream.getStream(), false});
							    }
							    CUDAEventProfileTimer forwardProjectCudaTimer(
							        cudaProfileEnabled, mainStream.getStream());
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile.forwardProjectEnqueue);
								    projector->applyA(inputImageForWorker,
								                      tmpBufferDevice, false);
							    }
							    forwardProjectCudaTimer.stop();

							    CUDAEventProfileTimer correctionsCudaTimer(
							        cudaProfileEnabled, mainStream.getStream());
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile.correctionsEnqueue);
							    if (corrector->hasAdditiveCorrection(
							            *measurementsDevice))
							    {
								    corrector
								        ->loadAdditiveCorrectionFactorsToTemporaryDeviceBuffer(
								            {&mainStream.getStream(), false});
								    const bool synchronize =
								        corrector->hasInVivoAttenuation();
								    tmpBufferDevice->addProjValues(
								        correctorTempBuffer,
								        {&mainStream.getStream(),
								         synchronize});
							    }
							    if (corrector->hasInVivoAttenuation())
							    {
								    corrector
								        ->loadInVivoAttenuationFactorsToTemporaryDeviceBuffer(
								            {&mainStream.getStream(), false});
								    tmpBufferDevice->multiplyProjValues(
								        correctorTempBuffer,
									        {&mainStream.getStream(), false});
								    }
							    }
							    correctionsCudaTimer.stop();

							    CUDAEventProfileTimer ratioCudaTimer(
							        cudaProfileEnabled, mainStream.getStream());
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile.ratioEnqueue);
							    if (measurementsAreUniform)
							    {
								    tmpBufferDevice->invertProjValuesDevice(
								        {&mainStream.getStream(), false});
							    }
							    else
							    {
								    tmpBufferDevice->divideMeasurementsDevice(
								        measurementsDevice,
									        {&mainStream.getStream(), false});
								    }
							    }
							    ratioCudaTimer.stop();

							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile.preBackprojectSync);
								    cudaStreamSynchronize(mainStream.getStream());
							    }
							    loadLORsCudaTimer.addElapsedTimeAfterSync(
							        workerProfile.cudaLoadLORs);
							    forwardProjectCudaTimer.addElapsedTimeAfterSync(
							        workerProfile.cudaForwardProject);
							    correctionsCudaTimer.addElapsedTimeAfterSync(
							        workerProfile.cudaCorrections);
							    ratioCudaTimer.addElapsedTimeAfterSync(
							        workerProfile.cudaRatio);
							    const size_t backprojectTimerIndex =
							        backprojectCudaTimers.size();
							    backprojectCudaTimers.emplace_back(
							        cudaProfileEnabled, mainStream.getStream());
							    {
								    ScopedProfileTimer profileTimer(
								        profileEnabled,
								        workerProfile.backprojectEnqueue);
								    projector->applyAH(tmpBufferDevice,
								                       partialImageDevice, false);
							    }
							    backprojectCudaTimers.at(backprojectTimerIndex)
							        .stop();
						    }

						    {
							    ScopedProfileTimer profileTimer(
							        profileEnabled, workerProfile.finalSync);
							    cudaStreamSynchronize(mainStream.getStream());
						    }
						    for (auto& timer : backprojectCudaTimers)
						    {
							    timer.addElapsedTimeAfterSync(
							        workerProfile.cudaBackproject);
						    }
					    ASSERT(cudaCheckError());
					    if (lorCacheMode == MultiGPULORCacheMode::Host)
					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled, workerProfile.releaseLORs);
						    measurementsDevice->releaseDeviceLORs(
						        {&mainStream.getStream(), true});
					    }
				    cudaEventCreateWithFlags(&copyDoneEvent,
				                             cudaEventDisableTiming);
				    ASSERT(cudaCheckError());
				    if (deviceId == primaryDeviceId)
				    {
					    cudaEventRecord(copyDoneEvent, mainStream.getStream());
					    ASSERT(cudaCheckError());
				    }
					    else
					    {
						    ScopedProfileTimer profileTimer(
						        profileEnabled,
						        workerProfile.copyPartialImage);
						    copyDeviceImageAcrossGPUsAsync(
						        *partialImageDevice, deviceId,
						        *m_emBuffers.primaryPartialImages.at(workerId),
						        primaryDeviceId, deviceId,
						        mainStream.getStream(), copyDoneEvent,
						        "one EM partial image");
					    }
				    partialReadyEvents.at(workerId) = copyDoneEvent;
				    copyDoneEvent = nullptr;
			    }
			    catch (...)
			    {
				    if (copyDoneEvent != nullptr)
				    {
					    ScopedCUDADevice guard(deviceIds.at(workerId));
					    cudaEventDestroy(copyDoneEvent);
				    }
				    errors.at(workerId) = std::current_exception();
			    }
		    });
	}

	for (auto& worker : workers)
	{
		worker.join();
	}
	if (profileEnabled)
	{
		workersWallSeconds = secondsSince(workersStartTime);
	}
	rethrowMultiGPUWorkerErrors(errors, partialReadyEvents, deviceIds);

	{
		ScopedProfileTimer profileTimer(profileEnabled,
		                                reducePartialImagesSeconds);
		sumPrimaryPartialImagesToDevice(
		    m_emBuffers.primaryPartialImages, m_emBuffers.workerPartialImages,
		    activeWorkerIds, partialReadyEvents, deviceIds, destImage,
		    primaryDeviceId, mp_osem->getMainStream());
	}
	emitMultiGPUProfileMetric("em_total", secondsSince(totalStartTime));
	emitMultiGPUProfileMetric("em_sync_primary", syncPrimarySeconds);
	emitMultiGPUProfileMetric("em_ensure_image_buffers",
	                          ensureImageBuffersSeconds);
	emitMultiGPUProfileMetric("em_ensure_recon_cache",
	                          ensureReconCacheSeconds);
	emitMultiGPUProfileMetric("em_workers_wall", workersWallSeconds);
	emitMultiGPUProfileMetric("em_reduce_partials",
	                          reducePartialImagesSeconds);
	emitMultiGPUProfileAggregate("em", workerProfiles);
}
}  // namespace yrt
