/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "operators/OperatorProjector.hpp"
#include "operators/OperatorPsf.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#endif

class ImageWarperTemplate;

class OSEM
{
public:
	// ---------- Constants ----------
	static constexpr int DEFAULT_NUM_ITERATIONS = 10;
	static constexpr float DEFAULT_HARD_THRESHOLD = 1.0f;
	static constexpr float INITIAL_VALUE_MLEM = 0.1f;
	// ---------- Public methods ----------
	explicit OSEM(const Scanner& pr_scanner);
	virtual ~OSEM() = default;
	OSEM(const OSEM&) = delete;
	OSEM& operator=(const OSEM&) = delete;

	// Sensitivity image generation
	void generateSensitivityImages(const std::string& out_fname);
	void generateSensitivityImages(
	    std::vector<std::shared_ptr<Image>>& sensImages,
	    const std::string& out_fname);
	bool validateSensImagesAmount(int size) const;

	// In case the sensitivity images were already generated
	void setSensitivityImages(
	    const std::vector<std::shared_ptr<Image>>& sensImages);
#if BUILD_PYBIND11
	void setSensitivityImages(pybind11::list& imageList);
#endif

	// OSEM Reconstruction
	std::shared_ptr<Image> reconstruct(const std::string& out_fname);
	std::shared_ptr<Image>
	    reconstructWithWarperMotion(const std::string& out_fname);

	// Prints a summary of the parameters
	void summary() const;

	// ---------- Getters and setters ----------
	ProjectionData* getSensDataInput() { return sensDataInput; }
	const ProjectionData* getSensDataInput() const { return sensDataInput; }
	void setSensDataInput(ProjectionData* p_sensDataInput);
	ProjectionData* getDataInput() { return dataInput; }
	const ProjectionData* getDataInput() const { return dataInput; }
	void setDataInput(ProjectionData* p_dataInput);
	void addTOF(float p_tofWidth_ps, int p_tofNumStd);
	void addProjPSF(const std::string& p_projSpacePsf_fname);
	void addImagePSF(OperatorPsf* p_imageSpacePsf);
	void setSaveSteps(int p_saveSteps, const std::string& p_saveStepsPath);
	void setListModeEnabled(bool enabled);
	void setProjector(const std::string& projectorName);  // Helper
	bool isListModeEnabled() const;
	void enableNeedToMakeCopyOfSensImage();
	ImageParams getImageParams() const;
	void setImageParams(const ImageParams& params);

	// ---------- Public members ----------
	int num_MLEM_iterations;
	int num_OSEM_subsets;
	float hardThreshold;
	int numRays;  // For Siddon only
	OperatorProjector::ProjectorType projectorType;
	const Scanner& scanner;
	const Image* maskImage;
	const Image* attenuationImage;
	const Image* attenuationImageForBackprojection;
	const Histogram* addHis;
	ImageWarperTemplate* warper;  // For MLEM with Warper only

protected:
	// ---------- Internal Getters ----------
	auto& getBinIterators() { return m_binIterators; }
	const auto& getBinIterators() const { return m_binIterators; }

	auto& getProjector() { return mp_projector; }
	const auto& getProjector() const { return mp_projector; }

	const Image* getSensitivityImage(int subsetId) const;
	Image* getSensitivityImage(int subsetId);

	// ---------- Protected members ----------
	bool flagImagePSF;
	OperatorPsf* imageSpacePsf;
	bool flagProjPSF;
	std::string projSpacePsf_fname;
	bool flagProjTOF;
	float tofWidth_ps;
	int tofNumStd;
	int saveSteps;
	std::string saveStepsPath;
	bool usingListModeInput;  // true => ListMode, false => Histogram
	std::unique_ptr<OperatorProjectorBase> mp_projector;
	bool needToMakeCopyOfSensImage;
	ImageParams imageParams;
	std::shared_ptr<ImageOwned> outImage;  // Note: This is a host image

	// ---------- Virtual pure functions ----------

	// Sens Image generator driver
	virtual void setupOperatorsForSensImgGen() = 0;
	virtual void allocateForSensImgGen() = 0;
	virtual std::shared_ptr<Image>
	    getLatestSensitivityImage(bool isLastSubset) = 0;
	virtual void endSensImgGen() = 0;

	// Reconstruction driver
	virtual void setupOperatorsForRecon() = 0;
	virtual void allocateForRecon() = 0;
	virtual void endRecon() = 0;
	virtual void completeMLEMIteration() = 0;

	// Abstract Getters
	virtual ImageBase* getSensImageBuffer() = 0;
	virtual ProjectionData* getSensDataInputBuffer() = 0;
	virtual ImageBase* getMLEMImageBuffer() = 0;
	virtual ImageBase* getMLEMImageTmpBuffer() = 0;
	virtual ProjectionData* getMLEMDataBuffer() = 0;
	virtual ProjectionData* getMLEMDataTmpBuffer() = 0;
	virtual int getNumBatches(int subsetId, bool forRecon) const;

	// Common methods
	virtual void loadBatch(int p_batchId, bool p_forRecon) = 0;
	virtual void loadSubset(int p_subsetId, bool p_forRecon) = 0;

private:
	void loadSubsetInternal(int p_subsetId, bool p_forRecon);
	void initializeForSensImgGen();
	void generateSensitivityImageForSubset(int subsetId);
	void generateSensitivityImagesCore(
	    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
	    std::vector<std::shared_ptr<Image>>& sensImages);
	void initializeForRecon();

	std::vector<std::unique_ptr<BinIterator>> m_binIterators;

	ProjectionData* sensDataInput;
	ProjectionData* dataInput;

	std::vector<std::shared_ptr<Image>> sensitivityImages;
	// In the specific case of ListMode reconstructions launched from Python
	std::unique_ptr<ImageOwned> copiedSensitivityImage;
};
