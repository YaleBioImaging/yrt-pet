/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/LREM_GPU.cuh"

#include "yrt-pet/operators/OperatorProjectorDevice.cuh"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_lrem_gpu(pybind11::module& m)
{
	auto c = py::class_<LREM_GPU, LREM, OSEM_GPU>(m, "LREM_GPU");
}
}  // namespace yrt
#endif

namespace yrt
{

LREM_GPU::LREM_GPU(const Scanner& pr_scanner) : OSEM_GPU(pr_scanner), LREM()
{
	std::cout << "Using Low-Rank approximation..." << std::endl;
	projectorParams.updaterType = UpdaterType::LR;
}

void LREM_GPU::prepareBuffersForRecon()
{
	OSEM_GPU::prepareBuffersForRecon();

	const bool dualUpdate = isDualUpdate();

	// Check LR Updater
	ProjectorUpdaterDeviceWrapper* lr = &m_updaterContainer;

	// YN: Check with YD
	// TODO: HBasis should already be set at creation of
	//  OperatorProjectorDevice (in the updater wrapper, in
	//  initUpdater). We can do it again here just in case (if code
	//  changes in wrapper or updater), or trust we will remember to do
	//  it here again if not initialized during projector creation
	std::cout << "Setting HBasis for ProjectorUpdaterDeviceLR..."
			  << std::endl;
	lr->setHBasis(projectorParams.HBasis);
	if (lr->getUpdateH() != projectorParams.updateH)
	{
		throw std::logic_error(
			"member updateH of ProjectorUpdaterLR is "
			"different than input updateH in projectorParams");
	}
	mp_HNumerator->fill(0.f);
	lr->setHBasisWrite(*mp_HNumerator);

	// HBasis is rank x T
	const std::array<size_t, 2> dims = projectorParams.HBasis.getDims();
	const auto rank = static_cast<int>(dims[0]);

	if (!projectorParams.updateH || dualUpdate)
	{
		m_cWUpdate.resize(rank, 0.f);
		generateWUpdateSensScaling();
		sync_cWUpdateHostToDevice();
	}
	if (projectorParams.updateH || dualUpdate)
	{
		m_cHUpdate.resize(rank, 0.f);
		// TODO: This is suboptimal as the HUpdateSensScaling could be done
		//  on GPU
		generateHUpdateSensScalingInternal();
		// Is this necessary ?
	}
}

void LREM_GPU::resetEMUpdateImage()
{
	OSEM_GPU::resetEMUpdateImage();

	if (isLowRank() && projectorParams.updateH)
	{
		mp_HNumerator->fill(0.f);
		syncHostToDeviceHBasisWrite();
	}
}

void LREM_GPU::applyImageUpdate()
{
	const bool dualUpdate = isDualUpdate();

	if (!projectorParams.updateH || dualUpdate)
	{
		mpd_mlemImage->updateEMThresholdDynamic(mpd_tmpImage1.get(),
		                                        mpd_sensImageBuffer.get(),
		                                        m_cWUpdate, EPS_FLT);
	}
	if (projectorParams.updateH ||
	    (dualUpdate && getCurrentMLEMIteration() > 0))
	{
		// TODO: This is suboptimal as the update could be done on GPU
		//  instead of copying to Device to do it (YN: Did you mean Host ?)
		syncDeviceToHostHBasisWrite();
		applyHUpdate();
		syncHostToDeviceHBasis();
	}
}

void LREM_GPU::completeSubset()
{
	if (isDualUpdate())
	{
		std::cout << "\nUpdating LR Sensitivity image scaling..." << std::endl;

		// Sync with Device side values
		sync_cWUpdateDeviceToHost();

		resetSensScaling();
		generateWUpdateSensScaling();
		generateHUpdateSensScalingInternal();

		// Sync back new Host side values to device
		sync_cWUpdateHostToDevice();
	}
}

void LREM_GPU::setupForDynamicRecon()
{
	OSEM_GPU::setupForDynamicRecon();

	const bool lowRank = isLowRank();
	const bool dualUpdate = isDualUpdate();

	if (dualUpdate)
	{
		projectorParams.updateH = true;
	}

	if (lowRank)
	{
		const Array2DAlias<float> HBasis = projectorParams.HBasis;
		ASSERT_MSG(HBasis.isMemoryValid(),
				   "The H matrix is not defined. Impossible to do Low-Rank "
				   "reconstruction.");

		const std::array<size_t, 2> dims = HBasis.getDims();
		const auto rank = static_cast<int>(dims[0]);

		allocateHBasisTmpBuffer();

		// Ensure that the image parameters used to allocate the
		//  reconstruction image(s) use the rank as the fourth dimension
		//  (instead of the number of dynamic frames)
		imageParams.nt = rank;
	}
}

void LREM_GPU::saveForCurrentIteration()
{
	const int numDigitsInFilename = util::numberOfDigits(num_MLEM_iterations);
	const bool dualUpdate = isDualUpdate();

	if (dualUpdate || !projectorParams.updateH)
	{
		OSEM::saveForCurrentIteration();
	}
	if (dualUpdate || projectorParams.updateH)
	{
		saveHBasisBinary(saveIterPath, getCurrentMLEMIteration() + 1,
		                 numDigitsInFilename);
	}
}


std::unique_ptr<Image> LREM_GPU::reconstructAlternating(
    int numWUpdatesPerIteration, int numHUpdatesPerIteration,
    const std::string& out_fname)
{
	safetyChecksBeforeRecon();
	ASSERT_MSG(numWUpdatesPerIteration > 0,
			   "Number of W updates per iteration must be > 0");
	ASSERT_MSG(numHUpdatesPerIteration > 0,
			   "Number of H updates per iteration must be > 0");

	// Initialize in W-update mode
	setUpdateH(false);
	initializeForRecon();

	for (int outerIter = 0; outerIter < num_MLEM_iterations; outerIter++)
	{
		setCurrentMLEMIteration(outerIter);

		std::cout << "\n=== Alternating iteration " << outerIter + 1 << "/"
		          << num_MLEM_iterations << " ===" << std::endl;

		// W-update mode. On the first outer iteration the projector is already
		// in W mode from initializeForRecon(). On subsequent iterations,
		// prepareForUpdatePhase(false) was called at the end of the previous H
		// mode, so the state is already correct.
		std::cout << "--- W-update mode (" << numWUpdatesPerIteration
		          << " iteration(s)) ---" << std::endl;
		for (int wIter = 0; wIter < numWUpdatesPerIteration; wIter++)
		{
			std::cout << "W-update " << wIter + 1 << "/"
			          << numWUpdatesPerIteration << std::endl;
			iterate();
			completeMLEMIteration();
		}

		// H-update mode
		std::cout << "--- H-update mode (" << numHUpdatesPerIteration
		          << " iteration(s)) ---" << std::endl;
		prepareForUpdatePhase(true);
		for (int hIter = 0; hIter < numHUpdatesPerIteration; hIter++)
		{
			std::cout << "H-update " << hIter + 1 << "/"
			          << numHUpdatesPerIteration << std::endl;
			iterate();
			completeMLEMIteration();
		}

		// Switch back to W mode; recomputes cWUpdate with the updated H
		prepareForUpdatePhase(false);

		// Save W and H if this outer iteration is in the save list
		if (saveIterRanges.isIn(outerIter + 1))
		{
			const int numDigits = util::numberOfDigits(num_MLEM_iterations);
			// Save W via the base OSEM implementation, bypassing the
			// updateH-dependent branching in LREM_GPU::saveForCurrentIteration
			OSEM::saveForCurrentIteration();
			saveHBasisBinary(saveIterPath, outerIter + 1, numDigits);
		}
	}

	endRecon();
	clearCopiedSensImage();

	if (!out_fname.empty())
	{
		std::cout << "Saving image..." << std::endl;
		outImage->writeToFile(out_fname);
	}

	return std::move(outImage);
}

void LREM_GPU::prepareForUpdatePhase(bool updateH)
{
	projectorParams.updateH = updateH;

	// Recreate the device-side projector updater with the new updateH flag.
	// This also re-uploads the current HBasis to the device, which is important
	// after an H-update mode may have modified it via applyHUpdate().
	setupProjectorUpdater(projectorParams);

	if (!updateH)
	{
		// Switching to W-update mode.
		// Recompute the W sensitivity scaling in case H changed during the
		// preceding H-update mode.
		generateWUpdateSensScaling();
		sync_cWUpdateHostToDevice();
	}
	else
	{
		// Switching to H-update mode.
		// Recompute the H denominator scaling in case W changed during the
		// preceding W-update mode, and reset the H numerator buffer.
		mp_HNumerator->fill(0.f);
		m_updaterContainer.setHBasisWrite(*mp_HNumerator);
		generateHUpdateSensScalingInternal();
	}
}


ProjectorParams& LREM_GPU::getProjectorParams()
{
	return projectorParams;
}
ImageBase* LREM_GPU::getEMUpdateImageBuffer()
{
	// Usual case
	if (!projectorParams.updateH || isDualUpdate())
	{
		return OSEM_GPU::getEMUpdateImageBuffer();
	}
	// When updating H, destImage must be the actual image (and not
	//  a zeroed buffer) to retrieve the value from the image during
	//  backupdate
	return mpd_mlemImage.get();
}

void LREM_GPU::setupProjectorUpdater(const ProjectorParams& params)
{
	if (params.updaterType == UpdaterType::LR)
	{
		if (params.HBasis.getSizeTotal() == 0)
		{
			throw std::invalid_argument(
			    "LR updater was requested but HBasis is empty");
		}

		m_updaterContainer.initUpdater(params.updaterType, params.HBasis,
		                               params.updateH);
	}
	else if (params.updaterType == UpdaterType::LRDUALUPDATE)
	{
		if (params.HBasis.getSizeTotal() == 0)
		{
			throw std::invalid_argument(
			    "LRDUALUPDATE updater was requested but HBasis is empty");
		}
		throw std::invalid_argument(
		    "LRDUALUPDATE updater is not implemented on GPU yet");
	}
	else
	{
		// For updaters requiring no arguments
		OSEM_GPU::setupProjectorUpdater(params);
	}
}

void LREM_GPU::generateHUpdateSensScalingInternal()
{
	const ImageDevice* W_imgDevice = mpd_mlemImage.get();
	ASSERT(W_imgDevice != nullptr);

	// copy to host
	auto W_img = std::make_unique<ImageOwned>(W_imgDevice->getParams());
	W_img->allocate();
	W_imgDevice->transferToHostMemory(W_img.get(), true);

	const ImageDevice* s_imgDevice = mpd_sensImageBuffer.get();
	ASSERT(s_imgDevice != nullptr);

	// copy to host
	auto s_img = std::make_unique<ImageOwned>(s_imgDevice->getParams());
	s_img->allocate();
	s_imgDevice->transferToHostMemory(s_img.get(), true);

	generateHUpdateSensScaling(*W_img, *s_img);
}

void LREM_GPU::sync_cWUpdateDeviceToHost()
{
	if (m_cWUpdateDevice.isAllocated())
	{
		const GPULaunchConfig launchConfig{nullptr, true};
		m_cWUpdateDevice.copyToHost(m_cWUpdate.data(), m_cWUpdate.size(),
		                            launchConfig);
	}
	else
	{
		throw std::logic_error("m_cWUpdateDevice is not allocated");
	}
}

void LREM_GPU::sync_cWUpdateHostToDevice()
{
	const GPULaunchConfig launchConfig{nullptr, true};
	if (!m_cWUpdateDevice.isAllocated())
	{
		m_cWUpdateDevice.allocate(m_cWUpdate.size(), launchConfig);
	}
	m_cWUpdateDevice.copyFromHost(m_cWUpdate.data(), m_cWUpdate.size(),
	                              launchConfig);
}

void LREM_GPU::syncHostToDeviceHBasis()
{
	m_updaterContainer.syncHostToDeviceHBasis();
}

void LREM_GPU::syncDeviceToHostHBasis()
{
	m_updaterContainer.syncDeviceToHostHBasis();
}

void LREM_GPU::syncHostToDeviceHBasisWrite()
{
	m_updaterContainer.syncHostToDeviceHBasisWrite();
}

void LREM_GPU::syncDeviceToHostHBasisWrite()
{
	m_updaterContainer.syncDeviceToHostHBasisWrite();
}

}  // namespace yrt
