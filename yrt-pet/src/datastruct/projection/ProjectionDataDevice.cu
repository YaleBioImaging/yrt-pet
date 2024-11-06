/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ListMode.hpp"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "datastruct/projection/ProjectionSpaceKernels.cuh"
#include "datastruct/projection/UniformHistogram.hpp"
#include "operators/OperatorDevice.cuh"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"

#include "omp.h"
#include <utility>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_projectiondatadevice(py::module& m)
{
	auto c = py::class_<ProjectionDataDevice, ProjectionList>(
	    m, "ProjectionDataDevice");
	c.def(
	    "loadEventLORs",
	    [](ProjectionDataDevice& self, size_t subsetId, size_t batchId,
	       const ImageParams& imgParams)
	    { self.loadEventLORs(subsetId, batchId, imgParams); },
	    "Load the LORs of a specific batch in a specific subset", "subsetId"_a,
	    "batchId"_a, "imgParams"_a);
	c.def("transferProjValuesToHost",
	      [](ProjectionDataDevice& self, ProjectionData* dest)
	      { self.transferProjValuesToHost(dest); });
	c.def("loadProjValuesFromHost",
	      [](ProjectionDataDevice& self, const ProjectionData* src)
	      { self.loadProjValuesFromHost(src, nullptr); });
	c.def("loadProjValuesFromHost",
	      [](ProjectionDataDevice& self, const ProjectionData* src,
	         const Histogram* histo)
	      { self.loadProjValuesFromHost(src, histo, nullptr); });
	c.def("loadProjValuesFromReference", [](ProjectionDataDeviceOwned& self)
	      { self.loadProjValuesFromReference(); });
	c.def("getCurrentBatchSize", &ProjectionDataDevice::getCurrentBatchSize);
	c.def("getCurrentBatchId", &ProjectionDataDevice::getCurrentBatchId);
	c.def("getCurrentSubsetId", &ProjectionDataDevice::getCurrentSubsetId);
	c.def("getNumBatches", &ProjectionDataDevice::getNumBatches);
	c.def("areLORsGathered", &ProjectionDataDevice::areLORsGathered);

	auto c_owned = py::class_<ProjectionDataDeviceOwned, ProjectionDataDevice>(
	    m, "ProjectionDataDeviceOwned");
	c_owned.def(
	    py::init<const Scanner&, const ProjectionData*, int, float>(),
	    "Create a ProjectionDataDevice. This constructor will also store "
	    "the Scanner LUT in the device",
	    "scanner"_a, "reference"_a, "num_OSEM_subsets"_a = 1,
	    "shareOfMemoryToUse"_a = ProjectionDataDevice::DefaultMemoryShare);
	c_owned.def(py::init<const ProjectionDataDevice*>(),
	            "Create a ProjectionDataDevice from an existing one. They will "
	            "share the LORs",
	            "orig"_a);
	c_owned.def("allocateForProjValues", [](ProjectionDataDeviceOwned& self)
	            { self.allocateForProjValues(); });

	auto c_alias = py::class_<ProjectionDataDeviceAlias, ProjectionDataDevice>(
	    m, "ProjectionDataDeviceAlias");
	c_alias.def(
	    py::init<const Scanner&, const ProjectionData*, int, float>(),
	    "Create a ProjectionDataDeviceAlias. This constructor will also "
	    "store "
	    "the Scanner LUT in the device",
	    "scanner"_a, "reference"_a, "num_OSEM_subsets"_a = 1,
	    "shareOfMemoryToUse"_a = ProjectionDataDevice::DefaultMemoryShare);
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

#endif  // if BUILD_PYBIND11

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
    float shareOfMemoryToUse)
    : ProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mr_scanner(pr_scanner)
{
	mp_LORs = std::make_unique<LORsDevice>(mr_scanner);
	createBatchSetups(shareOfMemoryToUse);
}

ProjectionDataDevice::ProjectionDataDevice(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : ProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mp_LORs(std::move(pp_LORs)),
      mr_scanner(mp_LORs->getScanner())
{
	createBatchSetups(shareOfMemoryToUse);
}

ProjectionDataDevice::ProjectionDataDevice(std::shared_ptr<LORsDevice> pp_LORs,
                                           const ProjectionData* pp_reference,
                                           int num_OSEM_subsets,
                                           float shareOfMemoryToUse)
    : ProjectionList(pp_reference),
      mp_LORs(std::move(pp_LORs)),
      mr_scanner(mp_LORs->getScanner())
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(shareOfMemoryToUse);
}

ProjectionDataDevice::ProjectionDataDevice(
    std::shared_ptr<ScannerDevice> pp_scannerDevice,
    const ProjectionData* pp_reference, int num_OSEM_subsets,
    float shareOfMemoryToUse)
    : ProjectionList(pp_reference), mr_scanner(pp_scannerDevice->getScanner())
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(shareOfMemoryToUse);
	mp_LORs = std::make_unique<LORsDevice>(std::move(pp_scannerDevice));
}

ProjectionDataDevice::ProjectionDataDevice(const Scanner& pr_scanner,
                                           const ProjectionData* pp_reference,
                                           int num_OSEM_subsets,
                                           float shareOfMemoryToUse)
    : ProjectionList(pp_reference), mr_scanner(pr_scanner)
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(shareOfMemoryToUse);

	mp_LORs = std::make_unique<LORsDevice>(mr_scanner);
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

void ProjectionDataDevice::createBatchSetups(float shareOfMemoryToUse)
{
	size_t memAvailable = getDeviceInfo(true);
	// Shrink memory according to the portion we want to use
	memAvailable = static_cast<size_t>(static_cast<float>(memAvailable) *
	                                   shareOfMemoryToUse);

	const size_t memoryUsagePerLOR = getReference()->hasTOF() ?
	                                     LORsDevice::MemoryUsagePerLORWithTOF :
	                                     LORsDevice::MemoryUsagePerLOR;
	const size_t memoryUsagePerEvent = memoryUsagePerLOR + sizeof(float);

	const size_t possibleEventsPerBatch =
	    memAvailable /
	    (GlobalsCuda::ThreadsPerBlockData * memoryUsagePerEvent) *
	    GlobalsCuda::ThreadsPerBlockData;

	const size_t numSubsets = mp_binIteratorList.size();
	m_batchSetups.reserve(numSubsets);
	for (size_t subsetId = 0; subsetId < numSubsets; subsetId++)
	{
		m_batchSetups.emplace_back(mp_binIteratorList.at(subsetId)->size(),
		                           possibleEventsPerBatch);
	}
}

void ProjectionDataDevice::loadEventLORs(size_t subsetId, size_t batchId,
                                         const ImageParams& imgParams,
                                         const cudaStream_t* stream)
{
	mp_LORs->loadEventLORs(*mp_binIteratorList.at(subsetId),
	                       m_batchSetups.at(subsetId), subsetId, batchId,
	                       *mp_reference, imgParams, stream);
}

void ProjectionDataDevice::loadProjValuesFromReference(
    const cudaStream_t* stream)
{
	loadProjValuesFromHostInternal(getReference(), nullptr, stream);
}

void ProjectionDataDevice::loadProjValuesFromHost(const ProjectionData* src,
                                                  const cudaStream_t* stream)
{
	loadProjValuesFromHostInternal(src, nullptr, stream);
}

void ProjectionDataDevice::loadProjValuesFromHost(const ProjectionData* src,
                                                  const Histogram* histo,
                                                  const cudaStream_t* stream)
{
	loadProjValuesFromHostInternal(src, histo, stream);
}

void ProjectionDataDevice::loadProjValuesFromHostInternal(
    const ProjectionData* src, const Histogram* histo,
    const cudaStream_t* stream)
{
	if (src->isUniform() && histo == nullptr)
	{
		// No need to "getProjectionValue" everywhere, just fill the buffer with
		// the same value
		clearProjections(getReference()->getProjectionValue(0), stream);
	}
	else
	{
		const size_t batchSize = getCurrentBatchSize();
		ASSERT_MSG(batchSize > 0,
		           "The Batch size is 0. You didn't load the LORs "
		           "before loading the projection values");

		m_tempBuffer.reAllocateIfNeeded(batchSize);
		float* projValuesBuffer = m_tempBuffer.getPointer();

		auto* binIter = mp_binIteratorList.at(getCurrentSubsetId());
		const size_t firstBatchSize =
		    getBatchSetup(getCurrentSubsetId()).getBatchSize(0);
		const size_t offset = getCurrentBatchId() * firstBatchSize;

		size_t binIdx;
		bin_t binId;
		if (histo == nullptr)
		{
			// Fill the buffer using the source directly
#pragma omp parallel for default(none) private(binIdx, binId) \
    firstprivate(offset, binIter, projValuesBuffer, src, batchSize)
			for (binIdx = 0; binIdx < batchSize; binIdx++)
			{
				binId = binIter->get(binIdx + offset);
				projValuesBuffer[binIdx] = src->getProjectionValue(binId);
			}
		}
		else
		{
			// Fill the buffer using the corresponding value in the histogram
			histo_bin_t histoBin;
#pragma omp parallel for default(none) private(binIdx, binId, histoBin) \
    firstprivate(offset, binIter, projValuesBuffer, src, batchSize, histo)
			for (binIdx = 0; binIdx < batchSize; binIdx++)
			{
				binId = binIter->get(binIdx + offset);
				histoBin = src->getHistogramBin(binId);
				projValuesBuffer[binIdx] =
				    histo->getProjectionValueFromHistogramBin(histoBin);
			}
		}

		Util::copyHostToDevice(getProjValuesDevicePointer(), projValuesBuffer,
		                       batchSize, stream, true);
	}
}

void ProjectionDataDevice::transferProjValuesToHost(
    ProjectionData* projDataDest, const cudaStream_t* stream) const
{
	const size_t batchSize = getCurrentBatchSize();
	ASSERT_MSG(batchSize > 0, "The Batch size is 0. You didn't load the LORs "
	                          "before loading the projection values");

	m_tempBuffer.reAllocateIfNeeded(batchSize);
	float* projValuesBuffer = m_tempBuffer.getPointer();
	Util::copyDeviceToHost(projValuesBuffer, getProjValuesDevicePointer(),
	                       batchSize, stream, true);

	auto* binIter = mp_binIteratorList.at(getCurrentSubsetId());
	const size_t firstBatchSize =
	    m_batchSetups.at(getCurrentSubsetId()).getBatchSize(0);
	const size_t offset = getCurrentBatchId() * firstBatchSize;

	size_t binIdx;
	bin_t binId;
#pragma omp parallel for default(none) private(binIdx, binId) \
    firstprivate(offset, binIter, projValuesBuffer, projDataDest, batchSize)
	for (binIdx = 0; binIdx < batchSize; binIdx++)
	{
		binId = binIter->get(binIdx + offset);
		projDataDest->setProjectionValue(binId, projValuesBuffer[binIdx]);
	}
}

std::shared_ptr<ScannerDevice> ProjectionDataDevice::getScannerDevice() const
{
	return mp_LORs->getScannerDevice();
}

size_t ProjectionDataDevice::getCurrentBatchSize() const
{
	return mp_LORs->getLoadedBatchSize();
}

size_t ProjectionDataDevice::getCurrentBatchId() const
{
	return mp_LORs->getLoadedBatchId();
}

size_t ProjectionDataDevice::getCurrentSubsetId() const
{
	return mp_LORs->getLoadedSubsetId();
}

const float4* ProjectionDataDevice::getLorDet1PosDevicePointer() const
{
	return mp_LORs->getLorDet1PosDevicePointer();
}

const float4* ProjectionDataDevice::getLorDet1OrientDevicePointer() const
{
	return mp_LORs->getLorDet1OrientDevicePointer();
}

const float4* ProjectionDataDevice::getLorDet2PosDevicePointer() const
{
	return mp_LORs->getLorDet2PosDevicePointer();
}

const float4* ProjectionDataDevice::getLorDet2OrientDevicePointer() const
{
	return mp_LORs->getLorDet2OrientDevicePointer();
}

const float* ProjectionDataDevice::getLorTOFValueDevicePointer() const
{
	return mp_LORs->getLorTOFValueDevicePointer();
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
	clearProjections(value, nullptr);
}

void ProjectionDataDevice::clearProjections(float value,
                                            const cudaStream_t* stream)
{
	if (value == 0.0f)
	{
		clearProjections(stream);
		return;
	}
	const size_t batchSize = getCurrentBatchSize();
	const auto launchParams = Util::initiateDeviceParameters(batchSize);

	if (stream != nullptr)
	{
		clearProjections_kernel<<<launchParams.gridSize, launchParams.blockSize,
		                          0, *stream>>>(
		    getProjValuesDevicePointer(), value, static_cast<int>(batchSize));
		cudaStreamSynchronize(*stream);
	}
	else
	{
		clearProjections_kernel<<<launchParams.gridSize,
		                          launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), value, static_cast<int>(batchSize));
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void ProjectionDataDevice::clearProjections(const cudaStream_t* stream)
{
	if (stream != nullptr)
	{
		cudaMemsetAsync(getProjValuesDevicePointer(), 0,
		                sizeof(float) * getCurrentBatchSize(), *stream);
		cudaStreamSynchronize(*stream);
	}
	else
	{
		cudaMemset(getProjValuesDevicePointer(), 0,
		           sizeof(float) * getCurrentBatchSize());
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void ProjectionDataDevice::divideMeasurements(
    const ProjectionData* measurements, const BinIterator* binIter)
{
	divideMeasurements(measurements, binIter, nullptr);
}

void ProjectionDataDevice::divideMeasurements(
    const ProjectionData* measurements, const BinIterator* binIter,
    const cudaStream_t* stream)
{
	(void)binIter;  // Not needed as this class has its own BinIterators
	const auto* measurements_device =
	    dynamic_cast<const ProjectionDataDevice*>(measurements);
	const size_t batchSize = getCurrentBatchSize();
	const auto launchParams = Util::initiateDeviceParameters(batchSize);

	if (stream != nullptr)
	{
		divideMeasurements_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize, 0, *stream>>>(
		    measurements_device->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaStreamSynchronize(*stream);
	}
	else
	{
		divideMeasurements_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize>>>(
		    measurements_device->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void ProjectionDataDevice::addProjValues(const ProjectionDataDevice* projValues,
                                         const cudaStream_t* stream)
{
	const size_t batchSize = getCurrentBatchSize();
	const auto launchParams = Util::initiateDeviceParameters(batchSize);

	if (stream != nullptr)
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize, 0,
		                       *stream>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaStreamSynchronize(*stream);
	}
	else
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaDeviceSynchronize();
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
    float shareOfMemoryToUse)
    : ProjectionDataDevice(pr_scanner, pp_reference,
                           std::move(pp_binIteratorList), shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : ProjectionDataDevice(pr_scanner, pp_reference, num_OSEM_subsets,
                           shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    std::shared_ptr<ScannerDevice> pp_scannerDevice,
    const ProjectionData* pp_reference, int num_OSEM_subsets,
    float shareOfMemoryToUse)
    : ProjectionDataDevice(std::move(pp_scannerDevice), pp_reference,
                           num_OSEM_subsets, shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference, num_OSEM_subsets,
                           shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<DeviceArray<float>>();
}

ProjectionDataDeviceOwned::ProjectionDataDeviceOwned(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference,
                           std::move(pp_binIteratorList), shareOfMemoryToUse)
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

void ProjectionDataDeviceOwned::allocateForProjValues(
    const cudaStream_t* stream)
{
	mp_projValues->allocate(getCurrentBatchSize(), stream);
}

void ProjectionDataDeviceOwned::loadProjValuesFromHostInternal(
    const ProjectionData* src, const Histogram* histo,
    const cudaStream_t* stream)
{
	if (!mp_projValues->isAllocated())
	{
		allocateForProjValues(stream);
	}
	ProjectionDataDevice::loadProjValuesFromHostInternal(src, histo, stream);
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : ProjectionDataDevice(pr_scanner, pp_reference,
                           std::move(pp_binIteratorList), shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    const Scanner& pr_scanner, const ProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : ProjectionDataDevice(pr_scanner, pp_reference, num_OSEM_subsets,
                           shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    std::shared_ptr<ScannerDevice> pp_scannerDevice,
    const ProjectionData* pp_reference, int num_OSEM_subsets,
    float shareOfMemoryToUse)
    : ProjectionDataDevice(std::move(pp_scannerDevice), pp_reference,
                           num_OSEM_subsets, shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference, num_OSEM_subsets,
                           shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

ProjectionDataDeviceAlias::ProjectionDataDeviceAlias(
    std::shared_ptr<LORsDevice> pp_LORs, const ProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : ProjectionDataDevice(std::move(pp_LORs), pp_reference,
                           std::move(pp_binIteratorList), shareOfMemoryToUse),
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