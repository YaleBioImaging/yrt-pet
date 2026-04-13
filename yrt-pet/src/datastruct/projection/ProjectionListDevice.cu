/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/ProjectionListDevice.cuh"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/operators/ProjectionSpaceKernels.cuh"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"

#include <utility>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{

void py_setup_projectionlistdevice(py::module& m)
{
	auto c = py::class_<ProjectionListDevice, ProjectionList>(
	    m, "ProjectionListDevice");
	c.def(
	    "prepareBatchLORs",
	    [](ProjectionListDevice& self, size_t subsetId, size_t batchId)
	    { self.prepareBatchLORs(subsetId, batchId, {nullptr, true}); },
	    "Load the LORs of a specific batch in a specific subset", "subset_id"_a,
	    "batch_id"_a);

	c.def("loadProjValuesFromReference", [](ProjectionListDeviceOwned& self)
	      { self.loadProjValuesFromReference({nullptr, true}); });
	c.def(
	    "loadProjValuesFromHost",
	    [](ProjectionListDevice& self, const ProjectionData* src)
	    {
		    const Histogram* srcAsHisto = dynamic_cast<const Histogram*>(src);
		    if (srcAsHisto == nullptr)
		    {
			    // Use the one from non-histograms
			    self.loadProjValuesFromHost(src, {nullptr, true});
		    }
		    else
		    {
			    std::cout << "Warning: loadProjValuesFromHost using a "
			                 "histogram input is deprecated, use "
			                 "loadProjValuesFromHostHistogram"
			              << std::endl;
		    }
	    },
	    "src"_a,
	    "Load the projection values of the given \'src\' ProjectionData object "
	    "into the projection values buffer.");
	c.def(
	    "loadProjValuesFromHostRandoms",
	    [](ProjectionListDevice& self, const ProjectionData* src)
	    { self.loadProjValuesFromHostRandoms(src, {nullptr, true}); }, "src"_a,
	    "Load the randoms estimates of the given \'src\' ProjectionData object "
	    "into the projection values buffer.");
	c.def(
	    "loadProjValuesFromHostHistogram",
	    [](ProjectionListDevice& self, const Histogram* histo)
	    { self.loadProjValuesFromHostHistogram(histo, {nullptr, true}); },
	    "src_histo"_a,
	    "Load the projection values of the given \'src_histo\' Histogram "
	    "object into the projection values buffer. If the reference is a "
	    "list-mode, this function will gather the values from the histogram "
	    "using the histogram bin of each event");

	c.def(
	    "transferProjValuesToHost",
	    [](const ProjectionListDevice& self, ProjectionData* dest)
	    { self.transferProjValuesToHost(dest, nullptr); }, "dest"_a,
	    "Transfer the currently-loaded batch of projection values to a "
	    "host-side projection-space buffer 'dest'");

	c.def("getLoadedBatchSize", &ProjectionListDevice::getLoadedBatchSize);
	c.def("getLoadedBatchId", &ProjectionListDevice::getLoadedBatchId);
	c.def("getLoadedSubsetId", &ProjectionListDevice::getLoadedSubsetId);
	c.def("getNumBatches", &ProjectionListDevice::getNumBatches, "subset_id"_a);
	c.def("getMemoryUsagePerEvent",
	      &ProjectionListDevice::getMemoryUsagePerEvent);
	c.def("areLORsGathered", &ProjectionListDevice::areLORsGathered);

	auto c_owned = py::class_<ProjectionListDeviceOwned, ProjectionListDevice>(
	    m, "ProjectionListDeviceOwned");
	// Constructor without constraints
	c_owned.def(
	    py::init(
	        [](const Scanner& pr_scanner, const ProjectionData* pp_reference,
	           const std::set<ProjectionPropertyType>& properties,
	           size_t memAvailable, int num_OSEM_subsets)
	        {
		        std::vector<Constraint*> noConstraints = {};
		        return std::make_unique<ProjectionListDeviceOwned>(
		            pr_scanner, pp_reference, noConstraints, properties,
		            memAvailable, num_OSEM_subsets);
	        }),
	    "Create a ProjectionListDeviceOwned. This constructor will initialize "
	    "its own LORsDevice, use the reference to create a bin iterator for "
	    "every subset, and compute its own batch setups",
	    "scanner"_a, "reference"_a, "properties"_a, "mem_available"_a,
	    "num_OSEM_subsets"_a = 1);
	c_owned.def(
	    py::init<const ProjectionListDevice*>(),
	    "Create a ProjectionListDeviceOwned. This constructor will share the "
	    "given LORsDevice (instead of initializing its own), use the given "
	    "list of bin iterators, and compute its own batch setups",
	    "orig"_a);
	c_owned.def("allocateForProjValuesIfNeeded",
	            [](ProjectionListDeviceOwned& self)
	            { self.allocateForProjValuesIfNeeded({nullptr, true}); });

	auto c_alias = py::class_<ProjectionListDeviceAlias, ProjectionListDevice>(
	    m, "ProjectionListDeviceAlias");
	c_alias.def(
	    py::init(
	        [](const Scanner& pr_scanner, const ProjectionData* pp_reference,
	           const std::set<ProjectionPropertyType>& properties,
	           size_t memAvailable, int num_OSEM_subsets)
	        {
		        std::vector<Constraint*> noConstraints = {};
		        return std::make_unique<ProjectionListDeviceAlias>(
		            pr_scanner, pp_reference, noConstraints, properties,
		            memAvailable, num_OSEM_subsets);
	        }),
	    "Create a ProjectionListDeviceAlias. This constructor will initialize "
	    "its own LORsDevice, use the reference to create a bin iterator for "
	    "every subset, and compute its own batch setups",
	    "scanner"_a, "reference"_a, "properties"_a, "mem_available"_a,
	    "num_OSEM_subsets"_a = 1);
	c_alias.def(
	    py::init<const ProjectionListDevice*>(),
	    "Create a ProjectionListDeviceAlias. This constructor will share the "
	    "given LORsDevice (instead of initializing its own), use the given "
	    "list of bin iterators, and compute its own batch setups",
	    "orig"_a);
	c_alias.def("getProjValuesDevicePointer",
	            &ProjectionListDeviceAlias::getProjValuesDevicePointerInULL);
	c_alias.def("setProjValuesDevicePointer",
	            static_cast<void (ProjectionListDeviceAlias::*)(size_t)>(
	                &ProjectionListDeviceAlias::setProjValuesDevicePointer),
	            "Set a device address for the projection values array. For "
	            "usage with PyTorch, use \'myArray.data_ptr()\'",
	            "data_ptr"_a);
	c_alias.def("isDevicePointerSet",
	            &ProjectionListDeviceAlias::isDevicePointerSet,
	            "Returns true if the device pointer is not null");
}

}  // namespace yrt

#endif  // if BUILD_PYBIND11

namespace yrt
{

ProjectionListDevice::ProjectionListDevice(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& properties, size_t memAvailable)
    : ProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mr_scanner(pr_scanner)
{
	initLORsDevice(constraints, properties);
	createBatchSetups(memAvailable);
}

ProjectionListDevice::ProjectionListDevice(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& properties, size_t memAvailable,
    int num_OSEM_subsets)
    : ProjectionList(pp_reference), mr_scanner(pr_scanner)
{
	initLORsDevice(constraints, properties);
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(memAvailable);
}

ProjectionListDevice::ProjectionListDevice(std::shared_ptr<LORsDevice> pp_LORs,
                                           const ProjectionData* pp_reference,
                                           size_t memAvailable,
                                           int num_OSEM_subsets)
    : ProjectionList(pp_reference),
      mp_LORs(std::move(pp_LORs)),
      mr_scanner(pp_reference->getScanner())
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(memAvailable);
}

ProjectionListDevice::ProjectionListDevice(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList, size_t memAvailable)
    : ProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mp_LORs(std::move(pp_LORs)),
      mr_scanner(pp_reference->getScanner())
{
	createBatchSetups(memAvailable);
}

ProjectionListDevice::ProjectionListDevice(const ProjectionListDevice* orig)
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

void ProjectionListDevice::prepareBatchLORs(int subsetId, int batchId,
                                            GPULaunchConfig launchConfig)
{
	precomputeBatchLORs(subsetId, batchId);

	// Necessary bottleneck:
	//  Wait for whatever previous device operation before loading a new batch
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

void ProjectionListDevice::precomputeBatchLORs(int subsetId, int batchId)
{
	mp_LORs->precomputeBatchLORs(*mp_binIteratorList.at(subsetId),
	                             m_batchSetups.at(subsetId), subsetId, batchId,
	                             *mp_reference);
}

void ProjectionListDevice::loadPrecomputedLORsToDevice(
    GPULaunchConfig launchConfig)
{
	mp_LORs->loadPrecomputedLORsToDevice(launchConfig);
}

void ProjectionListDevice::loadProjValuesFromReference(
    GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(getReference(), nullptr, false,
	                               launchConfig);
}

void ProjectionListDevice::loadProjValuesFromHost(const ProjectionData* src,
                                                  GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(src, nullptr, false, launchConfig);
}

void ProjectionListDevice::loadProjValuesFromHostRandoms(
    const ProjectionData* src, GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(src, nullptr, true, launchConfig);
}

void ProjectionListDevice::loadProjValuesFromHostHistogram(
    const Histogram* histo, GPULaunchConfig launchConfig)
{
	loadProjValuesFromHostInternal(getReference(), histo, false, launchConfig);
}

void ProjectionListDevice::loadProjValuesFromHostInternal(
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

void ProjectionListDevice::createBinIterators(int num_OSEM_subsets)
{
	m_binIterators.reserve(num_OSEM_subsets);
	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		m_binIterators.push_back(
		    mp_reference->getBinIter(num_OSEM_subsets, subsetId));
		mp_binIteratorList.push_back(m_binIterators.at(subsetId).get());
	}
}

void ProjectionListDevice::createBatchSetups(size_t memAvailable)
{
	const size_t memoryUsagePerEvent = getMemoryUsagePerEvent();

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

template <bool GatherRandoms>
void ProjectionListDevice::loadProjValuesFromHostInternal(
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
				    const bin_t projBin = binIter->get(binIdx + offset);
				    const histo_bin_t histoBin = src->getHistogramBin(projBin);
				    projValuesBuffer[binIdx] =
				        histo->getProjectionValueFromHistogramBin(histoBin);
			    });
		}

		util::copyHostToDevice(getProjValuesDevicePointer(), projValuesBuffer,
		                       batchSize, launchConfig);
	}
}
template void ProjectionListDevice::loadProjValuesFromHostInternal<true>(
    const ProjectionData*, const Histogram*, GPULaunchConfig);
template void ProjectionListDevice::loadProjValuesFromHostInternal<false>(
    const ProjectionData*, const Histogram*, GPULaunchConfig);

void ProjectionListDevice::transferProjValuesToHost(
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
		                         bin_t bin = binIter->get(binIdx + offset);
		                         projDataDest->setProjectionValue(
		                             bin, projValuesBuffer[binIdx]);
	                         });
}

size_t ProjectionListDevice::getPrecomputedBatchSize() const
{
	return mp_LORs->getPrecomputedBatchSize();
}

size_t ProjectionListDevice::getPrecomputedBatchId() const
{
	return mp_LORs->getPrecomputedBatchId();
}

size_t ProjectionListDevice::getPrecomputedSubsetId() const
{
	return mp_LORs->getPrecomputedSubsetId();
}

size_t ProjectionListDevice::getLoadedBatchSize() const
{
	return mp_LORs->getLoadedBatchSize();
}

size_t ProjectionListDevice::getLoadedBatchId() const
{
	return mp_LORs->getLoadedBatchId();
}

size_t ProjectionListDevice::getLoadedSubsetId() const
{
	return mp_LORs->getLoadedSubsetId();
}

size_t ProjectionListDevice::getMemoryUsagePerEvent() const
{
	// Memory used by on row of the LORsDevice object + 1 float for the
	//  projection value
	return mp_LORs->getElementSize() + sizeof(float);
}

const ProjectionPropertyManager*
    ProjectionListDevice::getProjectionPropertyManagerDevicePointer() const
{
	return mp_LORs->getProjectionPropertyManagerDevicePointer();
}

const PropertyUnit*
    ProjectionListDevice::getProjectionPropertiesDevicePointer() const
{
	return mp_LORs->getProjectionPropertiesDevicePointer();
}

float ProjectionListDevice::getProjectionValue(bin_t /*id*/) const
{
	throw std::logic_error("Disabled function in Device-side class");
}

void ProjectionListDevice::setProjectionValue(bin_t /*id*/, float /*val*/)
{
	throw std::logic_error("Disabled function in Device-side class");
}

void ProjectionListDevice::clearProjections(float value)
{
	clearProjectionsDevice(value, {nullptr, true});
}

void ProjectionListDevice::clearProjectionsDevice(float value,
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
		    getProjValuesDevicePointer(), value, batchSize);
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		clearProjections_kernel<<<launchParams.gridSize,
		                          launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), value, batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionListDevice::clearProjectionsDevice(GPULaunchConfig launchConfig)
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

void ProjectionListDevice::divideMeasurements(
    const ProjectionData* measurements, const BinIterator* /*binIter*/)
{
	// "binIter" is not needed as this class has its own BinIterators
	divideMeasurementsDevice(measurements, {nullptr, true});
}

void ProjectionListDevice::divideMeasurementsDevice(
    const ProjectionData* measurements, GPULaunchConfig launchConfig)
{
	const auto* measurements_device =
	    dynamic_cast<const ProjectionListDevice*>(measurements);
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
		    getProjValuesDevicePointer(), batchSize);
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
		    getProjValuesDevicePointer(), batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionListDevice::invertProjValuesDevice(GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		invertProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize,
		                          0, *launchConfig.stream>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(),
		    batchSize);
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
		    batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionListDevice::addProjValues(const ProjectionListDevice* projValues,
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
		    getProjValuesDevicePointer(), batchSize);
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionListDevice::convertToACFsDevice(GPULaunchConfig launchConfig)
{
	const size_t batchSize = getLoadedBatchSize();
	const auto launchParams = util::initiateDeviceParameters(batchSize);

	ASSERT(getProjValuesDevicePointer() != nullptr);

	if (launchConfig.stream != nullptr)
	{
		convertToACFs_kernel<<<launchParams.gridSize, launchParams.blockSize, 0,
		                       *launchConfig.stream>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(), 0.1f,
		    batchSize);
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		convertToACFs_kernel<<<launchParams.gridSize, launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), getProjValuesDevicePointer(), 0.1f,
		    batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionListDevice::multiplyProjValues(
    const ProjectionListDevice* projValues, GPULaunchConfig launchConfig)
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
		    getProjValuesDevicePointer(), batchSize);
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
		    getProjValuesDevicePointer(), batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

void ProjectionListDevice::multiplyProjValues(float scalar,
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
		    scalar, getProjValuesDevicePointer(), batchSize);
		if (launchConfig.synchronize)
		{
			cudaStreamSynchronize(*launchConfig.stream);
		}
	}
	else
	{
		multiplyProjValues_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize>>>(
		    scalar, getProjValuesDevicePointer(), batchSize);
		if (launchConfig.synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

const GPUBatchSetup& ProjectionListDevice::getBatchSetup(size_t subsetId) const
{
	return m_batchSetups.at(subsetId);
}

size_t ProjectionListDevice::getNumBatches(size_t subsetId) const
{
	return m_batchSetups.at(subsetId).getNumBatches();
}

bool ProjectionListDevice::areLORsGathered() const
{
	return mp_LORs->areLORsGathered();
}

void ProjectionListDevice::initLORsDevice(
    const std::vector<Constraint*>& constraints,
    std::set<ProjectionPropertyType> properties)
{
	mp_LORs = std::make_unique<LORsDevice>(constraints, properties);
}

ProjectionListDeviceOwned::ProjectionListDeviceOwned(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& properties, size_t memAvailable)
    : ProjectionListDevice(pr_scanner, pp_reference,
                           std::move(pp_binIteratorList), constraints,
                           properties, memAvailable)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionListDeviceOwned::ProjectionListDeviceOwned(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& properties, size_t memAvailable,
    int num_OSEM_subsets)
    : ProjectionListDevice(pr_scanner, pp_reference, constraints, properties,
                           memAvailable, num_OSEM_subsets)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionListDeviceOwned::ProjectionListDeviceOwned(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    size_t memAvailable, int num_OSEM_subsets)
    : ProjectionListDevice(std::move(pp_LORs), pp_reference, memAvailable,
                           num_OSEM_subsets)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionListDeviceOwned::ProjectionListDeviceOwned(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList, size_t memAvailable)
    : ProjectionListDevice(std::move(pp_LORs), pp_reference,
                           std::move(pp_binIteratorList), memAvailable)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionListDeviceOwned::ProjectionListDeviceOwned(
    const ProjectionListDevice* orig)
    : ProjectionListDevice(orig)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

bool ProjectionListDeviceOwned::allocateForProjValuesIfNeeded(
    GPULaunchConfig launchConfig)
{
	// Allocate projection value buffers based on the latest precomputed batch
	//  size
	return mp_projValues->allocate(getPrecomputedBatchSize(), launchConfig);
}

float* ProjectionListDeviceOwned::getProjValuesDevicePointer()
{
	return mp_projValues->getDevicePointer();
}

const float* ProjectionListDeviceOwned::getProjValuesDevicePointer() const
{
	return mp_projValues->getDevicePointer();
}

void ProjectionListDeviceOwned::loadProjValuesFromHostInternal(
    const ProjectionData* src, const Histogram* histo, bool gatherRandoms,
    GPULaunchConfig launchConfig)
{
	if (!mp_projValues->isAllocated())
	{
		allocateForProjValuesIfNeeded(launchConfig);
	}
	ProjectionListDevice::loadProjValuesFromHostInternal(src, histo, false,
	                                                     launchConfig);
}

ProjectionListDeviceAlias::ProjectionListDeviceAlias(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& properties, size_t memAvailable)
    : ProjectionListDevice(pr_scanner, pp_reference,
                           std::move(pp_binIteratorList), constraints,
                           properties, memAvailable),
      mpd_devicePointer(nullptr)
{
}

ProjectionListDeviceAlias::ProjectionListDeviceAlias(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    const std::vector<Constraint*>& constraints,
    const std::set<ProjectionPropertyType>& properties, size_t memAvailable,
    int num_OSEM_subsets)
    : ProjectionListDevice(pr_scanner, pp_reference, constraints, properties,
                           memAvailable, num_OSEM_subsets),
      mpd_devicePointer(nullptr)
{
}

ProjectionListDeviceAlias::ProjectionListDeviceAlias(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    size_t memAvailable, int num_OSEM_subsets)
    : ProjectionListDevice(std::move(pp_LORs), pp_reference, memAvailable,
                           num_OSEM_subsets),
      mpd_devicePointer(nullptr)
{
}

ProjectionListDeviceAlias::ProjectionListDeviceAlias(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList, size_t memAvailable)
    : ProjectionListDevice(std::move(pp_LORs), pp_reference,
                           std::move(pp_binIteratorList), memAvailable),
      mpd_devicePointer(nullptr)
{
}

ProjectionListDeviceAlias::ProjectionListDeviceAlias(
    const ProjectionListDevice* orig)
    : ProjectionListDevice(orig), mpd_devicePointer(nullptr)
{
}

float* ProjectionListDeviceAlias::getProjValuesDevicePointer()
{
	return mpd_devicePointer;
}

const float* ProjectionListDeviceAlias::getProjValuesDevicePointer() const
{
	return mpd_devicePointer;
}

size_t ProjectionListDeviceAlias::getProjValuesDevicePointerInULL() const
{
	return reinterpret_cast<size_t>(mpd_devicePointer);
}

void ProjectionListDeviceAlias::setProjValuesDevicePointer(
    float* ppd_devicePointer)
{
	mpd_devicePointer = ppd_devicePointer;
}

void ProjectionListDeviceAlias::setProjValuesDevicePointer(
    size_t ppd_pointerInULL)
{
	mpd_devicePointer = reinterpret_cast<float*>(ppd_pointerInULL);
}

bool ProjectionListDeviceAlias::isDevicePointerSet() const
{
	return mpd_devicePointer != nullptr;
}

}  // namespace yrt
