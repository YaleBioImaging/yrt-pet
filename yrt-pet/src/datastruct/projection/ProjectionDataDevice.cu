/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/ProjectionDataDevice.cuh"
#include "yrt-pet/datastruct/projection/ProjectionSpaceKernels.cuh"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"

#include <utility>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{

void py_setup_projectiondatadevice(py::module& m)
{
	auto c = py::class_<ProjectionDataDevice, ProjectionList>(
	    m, "ProjectionDataDevice");
	c.def(
	    "prepareBatchLORs",
	    [](ProjectionDataDevice& self, size_t subsetId, size_t batchId,
	       BinFilter& binFilter) {
		    self.prepareBatchLORs(subsetId, batchId, {nullptr, true},
		                          binFilter);
	    },
	    "Load the LORs of a specific batch in a specific subset", "subsetId"_a,
	    "batchId"_a, "binFilter"_a);

	c.def("loadProjValuesFromReference", [](ProjectionDataDeviceOwned& self)
	      { self.loadProjValuesFromReference({nullptr, true}); });
	c.def("loadProjValuesFromHost",
	      [](ProjectionDataDevice& self, const ProjectionData* src)
	      { self.loadProjValuesFromHost(src, {nullptr, true}); });
	c.def("loadProjValuesFromHostRandoms",
	      [](ProjectionDataDevice& self, const ProjectionData* src)
	      { self.loadProjValuesFromHostRandoms(src, {nullptr, true}); });
	c.def("loadProjValuesFromHostHistogram",
	      [](ProjectionDataDevice& self, const Histogram* histo)
	      { self.loadProjValuesFromHostHistogram(histo, {nullptr, true}); });
	c.def("loadProjValuesFromHost",
	      [](ProjectionDataDevice& self, const Histogram* histo)
	      { self.loadProjValuesFromHostHistogram(histo, {nullptr, true}); });

	c.def("transferProjValuesToHost",
	      [](const ProjectionDataDevice& self, ProjectionData* dest)
	      { self.transferProjValuesToHost(dest, nullptr); });

	c.def("getLoadedBatchSize", &ProjectionDataDevice::getLoadedBatchSize);
	c.def("getLoadedBatchId", &ProjectionDataDevice::getLoadedBatchId);
	c.def("getLoadedSubsetId", &ProjectionDataDevice::getLoadedSubsetId);
	c.def("getNumBatches", &ProjectionDataDevice::getNumBatches);
	c.def("areLORsGathered", &ProjectionDataDevice::areLORsGathered);

	auto c_owned = py::class_<ProjectionDataDeviceOwned, ProjectionDataDevice>(
	    m, "ProjectionDataDeviceOwned");
	c_owned.def(
	    py::init<const Scanner&, const ProjectionData*, size_t, size_t, int>(),
	    "Create a ProjectionDataDevice. This constructor will also store "
	    "the Scanner LUT in the device",
	    "scanner"_a, "reference"_a, "memory_usage_per_LOR"_a, "mem_available"_a,
	    "num_OSEM_subsets"_a = 1);
	c_owned.def(py::init<const ProjectionDataDevice*>(),
	            "Create a ProjectionDataDevice from an existing one. They will "
	            "share the LORs",
	            "orig"_a);
	c_owned.def("allocateForProjValues", [](ProjectionDataDeviceOwned& self)
	            { self.allocateForProjValues({nullptr, true}); });

	auto c_alias = py::class_<ProjectionDataDeviceAlias, ProjectionDataDevice>(
	    m, "ProjectionDataDeviceAlias");
	c_alias.def(
	    py::init<const Scanner&, const ProjectionData*, size_t, size_t, int>(),
	    "Create a ProjectionDataDeviceAlias. This constructor will also "
	    "store "
	    "the Scanner LUT in the device",
	    "scanner"_a, "reference"_a, "memory_usage_per_LOR"_a, "mem_available"_a,
	    "num_OSEM_subsets"_a = 1);
	c_alias.def(
	    py::init<const ProjectionDataDevice*>(),
	    "Create a ProjectionDataDeviceAlias from an existing one. They will "
	    "share the LORs",
	    "orig"_a);
	c_alias.def("getProjValuesDevicePointer",
	            &ProjectionDataDeviceAlias::getProjValuesDevicePointerInULL);
	c_alias.def("setProjValuesDevicePointer",
	            static_cast<void (ProjectionDataDeviceAlias::*)(size_t)>(
	                &ProjectionDataDeviceAlias::setProjValuesDevicePointer),
	            "Set a device address for the projection values array. For "
	            "usage with PyTorch, use \'myArray.data_ptr()\'",
	            "data_ptr"_a);
	c_alias.def("isDevicePointerSet",
	            &ProjectionDataDeviceAlias::isDevicePointerSet,
	            "Returns true if the device pointer is not null");
}

}  // namespace yrt

#endif  // if BUILD_PYBIND11

namespace yrt
{

ProjectionDataDevice::ProjectionDataDevice(const ProjectionDataDevice* orig)
    : ProjectionList(orig->mp_reference),
      mp_binIteratorList(orig->mp_binIteratorList),
      mp_LORs(orig->mp_LORs),
      mr_scanner(orig->mr_scanner)
{
	for (const auto& origBatchSetup : orig->m_batchSetups)
	{
		m_batchSetups.push_back(origBatchSetup);
	}
}

ProjectionDataDevice::ProjectionDataDevice(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    size_t memoryUsagePerLOR, size_t memAvailable)
    : ProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mr_scanner(pr_scanner)
{
	mp_LORs = std::make_unique<LORsDevice>();
	createBatchSetups(memoryUsagePerLOR, memAvailable);
}

ProjectionDataDevice::ProjectionDataDevice(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    size_t memoryUsagePerLOR, size_t memAvailable)
    : ProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mp_LORs(std::move(pp_LORs)),
      mr_scanner(pp_reference->getScanner())
{
	createBatchSetups(memoryUsagePerLOR, memAvailable);
}

ProjectionDataDevice::ProjectionDataDevice(std::shared_ptr<LORsDevice> pp_LORs,
                                           const ProjectionData* pp_reference,
                                           size_t memoryUsagePerLOR,
                                           size_t memAvailable,
                                           int num_OSEM_subsets)
    : ProjectionList(pp_reference),
      mp_LORs(std::move(pp_LORs)),
      mr_scanner(pp_reference->getScanner())
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(memoryUsagePerLOR, memAvailable);
}

ProjectionDataDevice::ProjectionDataDevice(const Scanner& pr_scanner,
                                           const ProjectionData* pp_reference,
                                           size_t memoryUsagePerLOR,
                                           size_t memAvailable,
                                           int num_OSEM_subsets)
    : ProjectionList(pp_reference), mr_scanner(pr_scanner)
{
	mp_LORs = std::make_unique<LORsDevice>();
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(memoryUsagePerLOR, memAvailable);
}

void ProjectionDataDevice::createBinIterators(int num_OSEM_subsets)
{
	m_binIterators.reserve(num_OSEM_subsets);
	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		m_binIterators.push_back(
		    mp_reference->getBinIter(num_OSEM_subsets, subsetId));
		mp_binIteratorList.push_back(m_binIterators.at(subsetId).get());
	}
}

void ProjectionDataDevice::createBatchSetups(size_t memoryUsagePerLOR,
                                             size_t memAvailable)
{
	const size_t memoryUsagePerEvent = memoryUsagePerLOR + sizeof(float);

	const size_t possibleEventsPerBatch =
	    memAvailable / (globals::ThreadsPerBlockData * memoryUsagePerEvent) *
	    globals::ThreadsPerBlockData;

	const size_t numSubsets = mp_binIteratorList.size();
	m_batchSetups.reserve(numSubsets);
	for (size_t subsetId = 0; subsetId < numSubsets; subsetId++)
	{
		m_batchSetups.emplace_back(mp_binIteratorList.at(subsetId)->size(),
		                           possibleEventsPerBatch);
	}
}

void ProjectionDataDevice::prepareBatchLORs(int subsetId, int batchId,
                                            GPULaunchConfig launchConfig,
                                            const BinFilter& binFilter)
{
	precomputeBatchLORs(subsetId, batchId, binFilter);

	// Necessary bottleneck
	// Must wait until previous operation using the device buffers is
	// finished before loading another batch
	if (launchConfig.stream != nullptr)
	{
		cudaStreamSynchronize(*launchConfig.stream);
	}
	else
	{
		cudaDeviceSynchronize();
	}
	loadPrecomputedLORsToDevice(launchConfig);
}

void ProjectionDataDevice::precomputeBatchLORs(int subsetId, int batchId,
                                               const BinFilter& binFilter)
{
	mp_LORs->precomputeBatchLORs(*mp_binIteratorList.at(subsetId),
	                             m_batchSetups.at(subsetId), subsetId, batchId,
	                             *mp_reference, binFilter);
}

void ProjectionDataDevice::loadPrecomputedLORsToDevice(
    GPULaunchConfig launchConfig)
{
	mp_LORs->loadPrecomputedLORsToDevice(launchConfig);
}

void ProjectionDataDevice::loadProjValuesFromReference(
    GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(getReference(), nullptr, false,
	                               launchConfig);
}

void ProjectionDataDevice::loadProjValuesFromHost(const ProjectionData* src,
                                                  GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(src, nullptr, false, launchConfig);
}

void ProjectionDataDevice::loadProjValuesFromHostRandoms(
    const ProjectionData* src, GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(src, nullptr, true, launchConfig);
}

void ProjectionDataDevice::loadProjValuesFromHostHistogram(
    const Histogram* histo, GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(getReference(), histo, false, launchConfig);
}

void ProjectionDataDevice::loadProjValuesFromHostInternal(
    const ProjectionData* src, const Histogram* histo, bool gatherRandoms,
    GPULaunchConfig launchConfig)
{
	ASSERT(src != nullptr);
	if (gatherRandoms)
	{
		ASSERT_MSG(src->hasRandomsEstimates(),
		           "Source projection data has no randoms estimates");
		loadProjValuesFromHostInternal<true>(src, histo, launchConfig);
	}
	else
	{
		loadProjValuesFromHostInternal<false>(src, histo, launchConfig);
	}
}

template <bool GatherRandoms>
void ProjectionDataDevice::loadProjValuesFromHostInternal(
    const ProjectionData* src, const Histogram* histo,
    GPULaunchConfig launchConfig)
{
	if (src->isUniform() && histo == nullptr)
	{
		// No need to "getProjectionValue" everywhere, just fill the buffer with
		// the same value
		clearProjectionsDevice(src->getProjectionValue(0), launchConfig);
	}
	else
	{
		const size_t batchSize = getPrecomputedBatchSize();
		ASSERT_MSG(batchSize > 0,
		           "The Batch size is 0. You didn't load the LORs "
		           "before loading the projection values");

		m_tempBuffer.reAllocateIfNeeded(batchSize);
		float* projValuesBuffer = m_tempBuffer.getPointer();

		auto* binIter = mp_binIteratorList.at(getPrecomputedSubsetId());
		const size_t firstBatchSize =
		    getBatchSetup(getPrecomputedSubsetId()).getBatchSize(0);
		const size_t offset = getPrecomputedBatchId() * firstBatchSize;

		if (histo == nullptr)
		{
			// TODO: Add optimization if loading from ProjectionList (since its
			//  memory is contiguous)

			// Fill the buffer using the source directly
			util::parallelForChunked(batchSize, globals::getNumThreads(),
			                         [offset, &binIter, projValuesBuffer, src,
			                          batchSize](bin_t binIdx, size_t /*tid*/)
			                         {
				                         bin_t binId =
				                             binIter->get(binIdx + offset);
				                         if constexpr (GatherRandoms)
				                         {
					                         projValuesBuffer[binIdx] =
					                             src->getRandomsEstimate(binId);
				                         }
				                         else
				                         {
					                         projValuesBuffer[binIdx] =
					                             src->getProjectionValue(binId);
				                         }
			                         });
		}
		else
		{
			// Fill the buffer using the corresponding value in the histogram
			util::parallelForChunked(
			    batchSize, globals::getNumThreads(),
			    [offset, binIter, projValuesBuffer, src, batchSize,
			     histo](bin_t binIdx, size_t /*tid*/)
			    {
				    bin_t binId = binIter->get(binIdx + offset);
				    histo_bin_t histoBin = src->getHistogramBin(binId);
				    projValuesBuffer[binIdx] =
				        histo->getProjectionValueFromHistogramBin(histoBin);
			    });
		}

		util::copyHostToDevice(getProjValuesDevicePointer(), projValuesBuffer,
		                       batchSize, launchConfig);
	}
}
template void ProjectionDataDevice::loadProjValuesFromHostInternal<true>(
    const ProjectionData*, const Histogram*, GPULaunchConfig);
template void ProjectionDataDevice::loadProjValuesFromHostInternal<false>(
    const ProjectionData*, const Histogram*, GPULaunchConfig);

void ProjectionDataDevice::transferProjValuesToHost(
    ProjectionData* projDataDest, const cudaStream_t* stream) const
{
	const size_t batchSize = getLoadedBatchSize();
	ASSERT_MSG(batchSize > 0, "The Batch size is 0. You didn't load the LORs "
	                          "before loading the projection values");

	m_tempBuffer.reAllocateIfNeeded(batchSize);
	float* projValuesBuffer = m_tempBuffer.getPointer();
	util::copyDeviceToHost(projValuesBuffer, getProjValuesDevicePointer(),
	                       batchSize, {stream, true});

	auto* binIter = mp_binIteratorList.at(getLoadedSubsetId());
	const size_t firstBatchSize =
	    m_batchSetups.at(getLoadedSubsetId()).getBatchSize(0);
	const size_t offset = getLoadedBatchId() * firstBatchSize;

	util::parallelForChunked(batchSize, globals::getNumThreads(),
	                         [binIter, offset, projDataDest,
	                          projValuesBuffer](bin_t binIdx, size_t /*tid*/)
	                         {
		                         bin_t binId = binIter->get(binIdx + offset);
		                         projDataDest->setProjectionValue(
		                             binId, projValuesBuffer[binIdx]);
	                         });
}

size_t ProjectionDataDevice::getPrecomputedBatchSize() const
{
	return mp_LORs->getPrecomputedBatchSize();
}

size_t ProjectionDataDevice::getPrecomputedBatchId() const
{
	return mp_LORs->getPrecomputedBatchId();
}

size_t ProjectionDataDevice::getPrecomputedSubsetId() const
{
	return mp_LORs->getPrecomputedSubsetId();
}

size_t ProjectionDataDevice::getLoadedBatchSize() const
{
	return mp_LORs->getLoadedBatchSize();
}

size_t ProjectionDataDevice::getLoadedBatchId() const
{
	return mp_LORs->getLoadedBatchId();
}

size_t ProjectionDataDevice::getLoadedSubsetId() const
{
	return mp_LORs->getLoadedSubsetId();
}

const char* ProjectionDataDevice::getProjectionPropertiesDevicePointer() const
{
	return mp_LORs->getProjectionPropertiesDevicePointer();
}

float ProjectionDataDevice::getProjectionValue(bin_t id) const
{
	(void)id;
	throw std::logic_error("Disabled function in Device-side class");
}

void ProjectionDataDevice::setProjectionValue(bin_t id, float val)
{
	(void)id;
	(void)val;
	throw std::logic_error("Disabled function in Device-side class");
}

void ProjectionDataDevice::clearProjections(float value)
{
	clearProjectionsDevice(value, {nullptr, true});
}

void ProjectionDataDevice::clearProjectionsDevice(float value,
                                                  GPULaunchConfig launchConfig)
{
	if (value == 0.0f)
	{
		clearProjectionsDevice(launchConfig);
		return;
	}
	const size_t batchSize = getPrecomputedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		clearProjections_kernel<<<launchParams.gridSize, launchParams.blockSize,
		                          0, *launchConfig.stream>>>(
		    getProjValuesDevicePointer(), value, static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		clearProjections_kernel<<<launchParams.gridSize,
		                          launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), value, static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::clearProjectionsDevice(GPULaunchConfig launchConfig)
{
	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		cudaMemsetAsync(getProjValuesDevicePointer(), 0,
		                sizeof(float) * getLoadedBatchSize(),
		                *launchConfig.stream);
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		cudaMemset(getProjValuesDevicePointer(), 0,
		           sizeof(float) * getLoadedBatchSize());
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::divideMeasurements(
    const ProjectionData* measurements, const BinIterator* binIter)
{
	(void)binIter;  // Not needed as this class has its own BinIterators
	divideMeasurementsDevice(measurements, {nullptr, true});
}

void ProjectionDataDevice::divideMeasurementsDevice(
    const ProjectionData* measurements, GPULaunchConfig launchConfig)
{
	const auto* measurements_device =
	    dynamic_cast<const ProjectionDataDevice*>(measurements);
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);
	ASSERT(measurements_device->getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		divideMeasurements_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize, 0,
		                            *launchConfig.stream>>>(
		    measurements_device->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		divideMeasurements_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize>>>(
		    measurements_device->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::invertProjValuesDevice(GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		invertProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize,
		                          0, *launchConfig.stream>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(),
		    static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		invertProjValues_kernel<<<launchParams.gridSize,
		                          launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(),
		    static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::addProjValues(const ProjectionDataDevice* projValues,
                                         GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(projValues->getProjValuesDevicePointer() != nullptr);
	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize, 0,
		                       *launchConfig.stream>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::convertToACFsDevice(GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		convertToACFs_kernel<<<launchParams.gridSize, launchParams.blockSize, 0,
		                       *launchConfig.stream>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(), 0.1f,
		    static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		convertToACFs_kernel<<<launchParams.gridSize, launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(), 0.1f,
		    static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::multiplyProjValues(
    const ProjectionDataDevice* projValues, GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		multiplyProjValues_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize, 0,
		                            *launchConfig.stream>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		multiplyProjValues_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionDataDevice::multiplyProjValues(float scalar,
                                              GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		multiplyProjValues_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize, 0,
		                            *launchConfig.stream>>>(
		    scalar, getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		multiplyProjValues_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize>>>(
		    scalar, getProjValuesDevicePointer(), static_cast<int>(batchSize));
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

const GPUBatchSetup& ProjectionDataDevice::getBatchSetup(size_t subsetId) const
{
	return m_batchSetups.at(subsetId);
}

size_t ProjectionDataDevice::getNumBatches(size_t subsetId) const
{
	return m_batchSetups.at(subsetId).getNumBatches();
}

bool ProjectionDataDevice::areLORsGathered() const
{
	return mp_LORs->areLORsGathered();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    size_t memoryUsagePerLOR, size_t memAvailable)
    : ProjectionDataDevice(pr_scanner, pp_reference,
                           std::move(pp_binIteratorList), memoryUsagePerLOR,
                           memAvailable)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    size_t memoryUsagePerLOR, size_t memAvailable, int num_OSEM_subsets)
    : ProjectionDataDevice(pr_scanner, pp_reference, memoryUsagePerLOR,
                           memAvailable, num_OSEM_subsets)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    size_t memoryUsagePerLOR, size_t memAvailable, int num_OSEM_subsets)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference, memoryUsagePerLOR,
                           memAvailable, num_OSEM_subsets)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    size_t memoryUsagePerLOR, size_t memAvailable)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference,
                           std::move(pp_binIteratorList), memoryUsagePerLOR,
                           memAvailable)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    const ProjectionDataDevice* orig)
    : ProjectionDataDevice(orig)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

float* ProjectionDataDeviceOwned::getProjValuesDevicePointer()
{
	return mp_projValues->getDevicePointer();
}

const float* ProjectionDataDeviceOwned::getProjValuesDevicePointer() const
{
	return mp_projValues->getDevicePointer();
}

bool ProjectionDataDeviceOwned::allocateForProjValues(
    GPULaunchConfig launchConfig)
{
	// Allocate projection value buffers based on the latest precomputed batch
	//  size
	return mp_projValues->allocate(getPrecomputedBatchSize(), launchConfig);
}

void ProjectionDataDeviceOwned::loadProjValuesFromHostInternal(
    const ProjectionData* src, const Histogram* histo, bool gatherRandoms,
    GPULaunchConfig launchConfig)
{
	if (!mp_projValues->isAllocated())
	{
		allocateForProjValues(launchConfig);
	}
	ProjectionDataDevice::loadProjValuesFromHostInternal(src, histo, false,
	                                                     launchConfig);
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    size_t memoryUsagePerLOR, size_t memAvailable)
    : ProjectionDataDevice(pr_scanner, pp_reference,
                           std::move(pp_binIteratorList), memoryUsagePerLOR,
                           memAvailable),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    size_t memoryUsagePerLOR, size_t memAvailable, int num_OSEM_subsets)
    : ProjectionDataDevice(pr_scanner, pp_reference, memoryUsagePerLOR,
                           memAvailable, num_OSEM_subsets),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    size_t memoryUsagePerLOR, size_t memAvailable, int num_OSEM_subsets)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference, memoryUsagePerLOR,
                           memAvailable, num_OSEM_subsets),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    size_t memoryUsagePerLOR, size_t memAvailable)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference,
                           std::move(pp_binIteratorList), memoryUsagePerLOR,
                           memAvailable),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    const ProjectionDataDevice* orig)
    : ProjectionDataDevice(orig), mpd_devicePointer(nullptr)
{
}

float* ProjectionDataDeviceAlias::getProjValuesDevicePointer()
{
	return mpd_devicePointer;
}

const float* ProjectionDataDeviceAlias::getProjValuesDevicePointer() const
{
	return mpd_devicePointer;
}

size_t ProjectionDataDeviceAlias::getProjValuesDevicePointerInULL() const
{
	return reinterpret_cast<size_t>(mpd_devicePointer);
}

void ProjectionDataDeviceAlias::setProjValuesDevicePointer(
    float* ppd_devicePointer)
{
	mpd_devicePointer = ppd_devicePointer;
}

void ProjectionDataDeviceAlias::setProjValuesDevicePointer(
    size_t ppd_pointerInULL)
{
	mpd_devicePointer = reinterpret_cast<float*>(ppd_pointerInULL);
}

bool ProjectionDataDeviceAlias::isDevicePointerSet() const
{
	return mpd_devicePointer != nullptr;
}

}  // namespace yrt
