/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/operators/OperatorProjectorBase.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/recon/OSEM.hpp"
#include "yrt-pet/recon/OSEMUpdater_CPU.hpp"

namespace yrt
{
class OSEM_CPU : public OSEM
{
public:
	explicit OSEM_CPU(const Scanner& pr_scanner);
	~OSEM_CPU() override;

	// Getters for internal objects
	const Image* getOutputImage() const;
	const Corrector& getCorrector() const override;
	Corrector& getCorrector() override;
	const Corrector_CPU& getCorrector_CPU() const;

	// Getters for operators
	const OperatorProjector* getProjector() const;
	void setupProjectorUpdater() override;

protected:
	// Sens Image generator driver
	void setupOperatorsForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image>
	    getLatestSensitivityImage(bool isLastSubset) override;
	void computeSensitivityImage(ImageBase& destImage) override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupOperatorsForRecon() override;
	void allocateForRecon() override;
	void endRecon() override;
	void completeMLEMIteration() override;

	// Internal getters
	ImageBase* getSensImageBuffer() override;
	const ProjectionData* getSensitivityBuffer() const;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase* getImageTmpBuffer(TemporaryImageSpaceBufferType type) override;
	const ProjectionData* getMLEMDataBuffer() override;
	ProjectionData* getMLEMDataTmpBuffer() override;

	// Common methods
	void loadSubset(int subsetId, bool forRecon) override;
	void computeEMUpdateImage(const ImageBase& inputImage,
	                          ImageBase& destImage) override;

	// LR methods
	void generateWUpdateSensScaling(float* c) override;
	void generateHUpdateSensScaling(float* c) override;
	void setupForDynamicRecon(int& rank, int& T) override;
	void applyImageUpdate(ImageBase* destImage, ImageBase* numerator,
	                      const ImageBase* norm, const float eps,
	                      bool isDynamic) override;
	void SyncHostToDeviceHBasisWrite() override;
	void SyncDeviceToHostHBasisWrite() override;


private:
	// For sensitivity image generation
	std::unique_ptr<Image> mp_tempSensImageBuffer;
	// For reconstruction
	std::unique_ptr<Image> mp_mlemImageTmpEMRatio;
	std::unique_ptr<Image> mp_imageTmpPsf;
	std::unique_ptr<ProjectionData> mp_datTmp;

	std::unique_ptr<Corrector_CPU> mp_corrector;
	std::unique_ptr<OSEMUpdater_CPU> mp_updater;

	int m_current_OSEM_subset;
};
}  // namespace yrt
