/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/LREM_CPU.hpp"

#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace yrt
{
void py_setup_lrem_cpu(pybind11::module& m)
{
	auto c = py::class_<LREM_CPU, LREM, OSEM_CPU>(m, "LREM_CPU");
}
}  // namespace yrt
#endif

namespace yrt
{

LREM_CPU::LREM_CPU(const Scanner& pr_scanner) : OSEM_CPU(pr_scanner), LREM()
{
	std::cout << "Using Low-Rank approximation..." << std::endl;
	projectorParams.updaterType = UpdaterType::LR;
}

void LREM_CPU::setupProjectorForRecon()
{
	OSEM_CPU::setupProjectorForRecon();

	const bool dualUpdate = isDualUpdate();

	// Check LR Updater
	Projector* proj = mp_projector.get();
	ASSERT_MSG(proj != nullptr, "Projector undefined");
	ProjectorUpdaterLR* lr =
	    dynamic_cast<ProjectorUpdaterLR*>(proj->getUpdater());
	ASSERT_MSG(lr != nullptr, "proj->getUpdater could not be cast "
	                          "to ProjectorUpdaterLR");
	ASSERT_MSG(lr->getUpdateH() == projectorParams.updateH,
	           "Member updateH of ProjectorUpdaterLR is "
	           "different than input updateH in projectorParams");

	// HBasis shape is [rank, T]
	const std::array<size_t, 2> dims = projectorParams.HBasis.getDims();
	const auto rank = static_cast<int>(dims[0]);

	if (!projectorParams.updateH || dualUpdate)
	{
		m_cWUpdate.resize(rank, 0.f);
		generateWUpdateSensScaling();
	}
	if (projectorParams.updateH || dualUpdate)
	{
		m_cHUpdate.resize(rank, 0.f);
		const Image& sensImage =
		    *dynamic_cast<const Image*>(getSensImageBuffer());
		generateHUpdateSensScaling(*outImage, sensImage);
		mp_HNumerator->fill(0.f);
		if (!dualUpdate)
		{
			// switch to H accumulation mode
			lr->setUpdateH(projectorParams.updateH);
		}
		std::cout << "Setting HBasis for "
		             "ProjectorUpdaterDeviceLR..."
		          << std::endl;
		lr->setHBasis(projectorParams.HBasis);
		lr->setHBasisWrite(*mp_HNumerator);  // write into mp_HWrite
		// TODO NOW: Make sure this makes sense even with PSF
		lr->setCurrentImgBuffer(flagImagePSF ? mp_imageTmpPsf.get() :
		                                       outImage.get());
	}
}

void LREM_CPU::resetEMUpdateImage()
{
	OSEM_CPU::resetEMUpdateImage();

	if ((isLowRank() && projectorParams.updateH) || isDualUpdate())
	{
		mp_HNumerator->fill(0.f);
	}
}

void LREM_CPU::applyImageUpdate()
{
	// This is because the CPU implementation accumulated the H value for
	//  every thread individually. We therefore need to re-accumulate all the H
	//  matrices of every thread into one matrix
	if (isLowRank())
	{
		ASSERT(mp_projector != nullptr);
		auto updater =
		    dynamic_cast<ProjectorUpdaterLR*>(mp_projector->getUpdater());
		ASSERT(updater != nullptr);
		if (updater->getUpdateH())
		{
			updater->accumulateH();
		}
	}

	const bool dualUpdate = isDualUpdate();

	if (!projectorParams.updateH || dualUpdate)
	{
		outImage->updateEMThresholdDynamic(mp_mlemImageTmpEMRatio.get(),
		                                   getSensImageBuffer(), m_cWUpdate,
		                                   EPS_FLT);
	}
	if (projectorParams.updateH ||
	    (dualUpdate && getCurrentMLEMIteration() > 0))
	{
		applyHUpdate();
	}
}

void LREM_CPU::completeSubset()
{
	if (isDualUpdate())
	{
		std::cout << "\nUpdating LR Sensitivity image scaling..." << std::endl;

		const Image& sensImage =
		    *dynamic_cast<const Image*>(getSensImageBuffer());

		resetSensScaling();
		generateWUpdateSensScaling();
		generateHUpdateSensScaling(*outImage, sensImage);
	}
}

void LREM_CPU::setupForDynamicRecon()
{
	OSEM_CPU::setupForDynamicRecon();

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

void LREM_CPU::saveForCurrentIteration()
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

ProjectorParams& LREM_CPU::getProjectorParams()
{
	return projectorParams;
}

ImageBase* LREM_CPU::getEMUpdateImageBuffer()
{
	if (!projectorParams.updateH || isDualUpdate())
	{
		// Normal situation
		return OSEM_CPU::getEMUpdateImageBuffer();
	}
	// When updating H, destImage must be the actual image (and not
	//  a zeroed buffer) to retrieve the value from the image during
	//  backupdate
	return outImage.get();
}

}  // namespace yrt
