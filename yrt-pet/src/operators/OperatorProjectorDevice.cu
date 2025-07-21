/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/OperatorProjectorDevice.cuh"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/GPUUtils.cuh"

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
	       ProjectionData* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const Image* img,
	       ProjectionData* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const ImageDevice* img,
	       ProjectionDataDevice* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](OperatorProjectorDevice& self, const Image* img,
	       ProjectionDataDevice* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));

	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionData* proj,
	       Image* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionData* proj,
	       ImageDevice* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionDataDevice* proj,
	       Image* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](OperatorProjectorDevice& self, const ProjectionDataDevice* proj,
	       ImageDevice* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
}
}  // namespace yrt

#endif

namespace yrt
{
OperatorProjectorDevice::OperatorProjectorDevice(
    const OperatorProjectorParams& pr_projParams,
    const cudaStream_t* pp_mainStream, const cudaStream_t* pp_auxStream)
    : OperatorProjectorBase{pr_projParams},
      DeviceSynchronized{pp_mainStream, pp_auxStream}
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
