/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/Corrector_CPU.hpp"
#include "recon/OSEM.hpp"
#include "recon/OSEMUpdater_CPU.hpp"

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
	OperatorPsf* getOperatorPsf() const;  // Image-space PSF

protected:
	// Sens Image generator driver
	void setupOperatorsForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image>
	    getLatestSensitivityImage(bool isLastSubset) override;
	void endSensImgGen() override;

	// Reconstruction driver
	void setupOperatorsForRecon() override;
	void allocateForRecon() override;
	void endRecon() override;
	void completeMLEMIteration() override;

	// Internal getters
	ImageBase* getSensImageBuffer() override;
	const ProjectionData* getSensitivityBuffer() override;
	ImageBase* getMLEMImageBuffer() override;
	ImageBase*
	    getMLEMImageTmpBuffer(TemporaryImageSpaceBufferType type) override;
	const ProjectionData* getMLEMDataBuffer() override;
	ProjectionData* getMLEMDataTmpBuffer() override;

	// Common methods
	void loadBatch(int batchId, bool forRecon) override;
	void loadSubset(int subsetId, bool forRecon) override;
	void computeEMUpdateImage(const ImageBase& inputImage,
	                          ImageBase& destImage) override;

private:
	// For sensitivity image generation
	std::unique_ptr<Image> mp_tempSensImageBuffer;
	// For reconstruction
	std::unique_ptr<Image> mp_mlemImageTmp;
	std::unique_ptr<Image> mp_mlemImageTmpPsf;
	std::unique_ptr<ProjectionData>
	    mp_datTmp;  // TODO NOW: Will need to be removed

	std::unique_ptr<Corrector_CPU> mp_corrector;
	std::unique_ptr<OSEMUpdater_CPU> mp_updater;

	int m_current_OSEM_subset;
};
