/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/operators/Projector.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/recon/OSEM.hpp"

namespace yrt
{
class OSEM_CPU : public OSEM
{
public:
	explicit OSEM_CPU(const Scanner& pr_scanner);
	~OSEM_CPU() override = default;

	void addImagePSF(const std::string& p_imagePsf_fname,
	                 ImagePSFMode p_imagePSFMode) override;

protected:
	// Sens Image generator driver
	void setupProjectorForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image> generateSensitivityImageForCurrentSubset() override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupForDynamicRecon() override;
	void setupProjectorForRecon() override;
	void allocateForRecon() override;
	void loadCurrentSubset(bool forRecon) override;
	void resetEMUpdateImage() override;
	void computeEMUpdateImage() override;
	void applyImageUpdate() override;
	void completeSubset() override;
	void completeMLEMIteration() override;
	void endRecon() override;

	// Abstract Getters
	ImageBase* getSensImageBuffer() override;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase* getEMUpdateImageBuffer() override;
	const Corrector& getCorrector() const override;
	Corrector& getCorrector() override;
	const Corrector_CPU& getCorrector_CPU() const;

	std::unique_ptr<Projector> mp_projector;
	std::unique_ptr<BinLoader> mp_binLoader;
	// For sensitivity image generation
	std::unique_ptr<Image> mp_tempSensImageBuffer;
	// For reconstruction
	std::unique_ptr<Image> mp_mlemImageTmpEMRatio;
	std::unique_ptr<Image> mp_imageTmpPsf;
	// Corrector
	std::unique_ptr<Corrector_CPU> mp_corrector;

private:
	void initBinLoaderIfNeeded(bool forRecon);
	std::set<ProjectionPropertyType> getNeededProperties(bool forRecon) const;

};
}  // namespace yrt
