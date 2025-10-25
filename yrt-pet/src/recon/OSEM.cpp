/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/OSEM.hpp"

#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/projection/ProjectionList.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/OperatorProjectorDD.hpp"
#include "yrt-pet/operators/OperatorProjectorSiddon.hpp"
#include "yrt-pet/operators/OperatorPsf.hpp"
#include "yrt-pet/operators/OperatorVarPsf.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace pybind11::literals;

namespace yrt
{
void py_setup_osem(pybind11::module& m)
{
	py::enum_<ImagePSFMode>(m, "ImagePSFMode")
	    .value("UNIFORM", ImagePSFMode::UNIFORM)
	    .value("VARIANT", ImagePSFMode::VARIANT)
	    .export_values();

	auto c = py::class_<OSEM>(m, "OSEM");

	// This returns a python list of the sensitivity images
	c.def(
	    "generateSensitivityImages",
	    [](OSEM& self, const std::string& out_fname,
	       bool saveToMemory) -> py::list
	    {
		    py::list pySensImagesList;
		    if (!saveToMemory)
		    {
			    self.generateSensitivityImages(out_fname);
			    return pySensImagesList;
		    }

		    std::vector<std::unique_ptr<Image>> sensImages;
		    self.generateSensitivityImages(sensImages, out_fname);
		    for (size_t i = 0; i < sensImages.size(); i++)
		    {
			    pySensImagesList.append(std::move(sensImages[i]));
		    }
		    return pySensImagesList;
	    },
	    "out_fname"_a = "", "saveToMemory"_a = true);

	c.def("getExpectedSensImagesAmount", &OSEM::getExpectedSensImagesAmount);

	c.def("setSensitivityImage", &OSEM::setSensitivityImage, "sens_image"_a,
	      "subset"_a = 0);
	c.def("setSensitivityImages",
	      static_cast<void (OSEM::*)(const pybind11::list& pySensImgList)>(
	          &OSEM::setSensitivityImages));

	c.def("reconstruct", &OSEM::reconstruct, "out_fname"_a = "");
	c.def("summary", &OSEM::summary);

	c.def("setSensitivityHistogram", &OSEM::setSensitivityHistogram,
	      "sens_his"_a);
	c.def("getSensitivityHistogram", &OSEM::getSensitivityHistogram);
	c.def("setGlobalScalingFactor", &OSEM::setGlobalScalingFactor,
	      "global_scale"_a);
	c.def("setInvertSensitivity", &OSEM::setInvertSensitivity,
	      "invert"_a = true);
	c.def("getDataInput", &OSEM::getDataInput);
	c.def("setDataInput", &OSEM::setDataInput, "proj_data"_a);
	c.def("addTOF", &OSEM::addTOF, "tof_width_ps"_a, "tof_num_std"_a);
	c.def("addProjPSF", &OSEM::addProjPSF, "proj_psf_fname"_a);
	c.def("addImagePSF", &OSEM::addImagePSF, "image_psf_fname"_a,
	      "image_psf_mode"_a = ImagePSFMode::UNIFORM);
	c.def("setSaveIterRanges", &OSEM::setSaveIterRanges, "range_list"_a,
	      "path"_a);
	c.def("setListModeEnabled", &OSEM::setListModeEnabled, "enabled"_a);
	c.def("setProjector", &OSEM::setProjector, "projector_name"_a);
	c.def("setImageParams", &OSEM::setImageParams, "params"_a);
	c.def("getImageParams", &OSEM::getImageParams);
	c.def("isListModeEnabled", &OSEM::isListModeEnabled);
	c.def("setRandomsHistogram", &OSEM::setRandomsHistogram, "randoms_his"_a);
	c.def("setScatterHistogram", &OSEM::setScatterHistogram, "scatter_his"_a);
	c.def("setAttenuationImage", &OSEM::setAttenuationImage, "att_image"_a);
	c.def("setACFHistogram", &OSEM::setACFHistogram, "acf_his"_a);
	c.def("setHardwareAttenuationImage", &OSEM::setHardwareAttenuationImage,
	      "att_hardware"_a);
	c.def("setHardwareACFHistogram", &OSEM::setHardwareACFHistogram,
	      "acf_hardware_his"_a);
	c.def("setInVivoAttenuationImage", &OSEM::setInVivoAttenuationImage,
	      "att_invivo"_a);
	c.def("setInVivoACFHistogram", &OSEM::setInVivoACFHistogram,
	      "acf_invivo_his"_a);

	// New projParams getters and setters
	c.def("getProjectorParams",&OSEM::getProjectorParams,
	     py::return_value_policy::reference_internal);
	c.def("getHBasis", &OSEM::getHBasis,
	      py::return_value_policy::reference_internal);
	c.def("getUpdateH", &OSEM::getUpdateH);
	c.def("getProjectorUpdaterType",&OSEM::getProjectorUpdaterType);
	c.def("getNumRays",&OSEM::getNumRays);

	c.def("setHBasis", &OSEM::setHBasis, py::arg("HBasisAlias"));
	c.def("setUpdateH", &OSEM::setUpdateH, py::arg("updateH"));
	c.def("setProjectorUpdaterType",&OSEM::setProjectorUpdaterType,
	      py::arg("projectorUpdaterType"));
	c.def("setNumRays",&OSEM::setNumRays);

	c.def(
	    "setHBasisFromNumpy",
	    [](OSEM& self, py::buffer& np_data) {
		    py::buffer_info buffer = np_data.request();

		    if (buffer.ndim != 2)
			    throw std::invalid_argument("HBasis must be 2D (rank x time).");

		    if (buffer.format != py::format_descriptor<float>::format())
			    throw std::invalid_argument("HBasis must be float32.");

		    auto* ptr = reinterpret_cast<float*>(buffer.ptr);
		    const size_t rank = static_cast<size_t>(buffer.shape[0]);
		    const size_t T    = static_cast<size_t>(buffer.shape[1]);

		    self.projectorParams.HBasis.bind(ptr, rank, T);
	    },
	    py::arg("HBasis"),
	    py::keep_alive<1, 2>()  // keep the buffer owner alive
	);

	c.def("getHBasisNumpy",
	      [](OSEM& self) {
		      const auto& H = self.getHBasis();               // Array2DAlias<float>
		      auto dims = H.getDims();                        // {rank, T}
		      int R = static_cast<int>(dims[0]);
		      int T = static_cast<int>(dims[1]);

		      py::array_t<float> arr({R, T});                 // C-contiguous
		      std::memcpy(arr.mutable_data(),                 // copy all at once
		                  H.getRawPointer(),
		                  static_cast<size_t>(R*T) * sizeof(float));
		      return arr;                                     // copy
	      });

	c.def_readwrite("num_MLEM_iterations", &OSEM::num_MLEM_iterations);
	c.def_readwrite("num_OSEM_subsets", &OSEM::num_OSEM_subsets);
	c.def_readwrite("hardThreshold", &OSEM::hardThreshold);
	c.def_readwrite("projectorType", &OSEM::projectorType);
	c.def_readwrite("maskImage", &OSEM::maskImage);
	c.def_readwrite("initialEstimate", &OSEM::initialEstimate);
}
}  // namespace yrt

#endif

namespace yrt
{

OSEM::OSEM(const Scanner& pr_scanner)
    : num_MLEM_iterations(DEFAULT_NUM_ITERATIONS),
      num_OSEM_subsets(1),
      hardThreshold(DEFAULT_HARD_THRESHOLD),
	  projectorType(OperatorProjector::SIDDON),
      projectorParams(
          /*binIter*/        nullptr,
          /*scanner*/        pr_scanner),
      scanner(pr_scanner),
      maskImage(nullptr),
      initialEstimate(nullptr),
      flagImagePSF(false),
      imagePsf(nullptr),
      saveIterRanges(),
      usingListModeInput(false),
      needToMakeCopyOfSensImage(false),
      outImage(nullptr),
      mp_dataInput(nullptr),
      mp_copiedSensitivityImage(nullptr)
{
}

const OperatorProjectorParams& OSEM::getProjectorParams() const {
	return projectorParams;
}

const Array2DAlias<float>& OSEM::getHBasis() const {
	const auto& H = projectorParams.HBasis;
	auto dims = H.getDims();
	if (dims[0] == 0 || dims[1] == 0) {
		throw std::runtime_error("HBasis not set (empty alias)");
	}
	return H;
}

bool OSEM::getUpdateH() const
{
	return projectorParams.updateH;
}

void OSEM::setUpdateH(bool p_updateH)
{
	projectorParams.updateH = p_updateH;
}

OperatorProjectorParams::ProjectorUpdaterType OSEM::getProjectorUpdaterType() const
{
	return projectorParams.projectorUpdaterType;
}

int OSEM::getNumRays() const
{
	return projectorParams.numRays;
}

void OSEM::setHBasis(const Array2DAlias<float>& HBasisAlias) {
	// Rebind our alias to the provided alias's storage
	projectorParams.HBasis.bind(HBasisAlias);
}

void OSEM::setProjectorUpdaterType(
    OperatorProjectorParams::ProjectorUpdaterType projectorUpdaterType)
{
	projectorParams.projectorUpdaterType = projectorUpdaterType;
}

void OSEM::setNumRays(int p_numRays)
{
	projectorParams.numRays = p_numRays;
}

void OSEM::generateSensitivityImages(const std::string& out_fname)
{
	std::vector<std::unique_ptr<Image>> dummy;
	generateSensitivityImagesCore(true, out_fname, false, dummy);
}

void OSEM::generateSensitivityImages(
    std::vector<std::unique_ptr<Image>>& sensImages,
    const std::string& out_fname)
{
	if (out_fname.empty())
	{
		generateSensitivityImagesCore(false, "", true, sensImages);
	}
	else
	{
		generateSensitivityImagesCore(true, out_fname, true, sensImages);
	}
}

void OSEM::generateSensitivityImageForLoadedSubset()
{
	getSensImageBuffer()->setValue(0.0);

	computeSensitivityImage(*getSensImageBuffer());

	if (flagImagePSF)
	{
		ImageBase* tmpPSFImage =
		    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF);
		imagePsf->applyAH(getSensImageBuffer(), tmpPSFImage);
		getSensImageBuffer()->copyFromImage(tmpPSFImage);
	}

	std::cout << "Applying threshold..." << std::endl;
	getSensImageBuffer()->applyThreshold(getSensImageBuffer(), hardThreshold,
	                                     0.0, 0.0, 1.0, 0.0);
}

void OSEM::generateSensitivityImagesCore(
    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
    std::vector<std::unique_ptr<Image>>& sensImages)
{
	ASSERT_MSG(imageParams.isValid(), "Image parameters not valid/set");
	ASSERT_MSG(num_OSEM_subsets > 0, "Not enough OSEM subsets");

	Corrector& corrector = getCorrector();

	// This is done to make sure we only make one sensitivity image if we're on
	// ListMode
	const int originalNumOSEMSubsets = num_OSEM_subsets;
	if (usingListModeInput)
	{
		num_OSEM_subsets = 1;
	}

	corrector.setup();
	initializeForSensImgGen();

	sensImages.clear();

	const int numDigitsInFilename =
	    num_OSEM_subsets > 1 ? util::numberOfDigits(num_OSEM_subsets - 1) : 1;

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		std::cout << "OSEM subset " << subsetId + 1 << "/" << num_OSEM_subsets
		          << "..." << std::endl;

		// Load subset
		loadSubsetInternal(subsetId, false);

		// Generate sensitivity image for loaded subset
		generateSensitivityImageForLoadedSubset();

		// Save sensitivity image
		auto generatedImage =
		    getLatestSensitivityImage(subsetId == num_OSEM_subsets - 1);

		if (saveOnDisk)
		{
			std::cout << "Saving image to disk..." << std::endl;
			std::string outFileName = out_fname;
			if (num_OSEM_subsets != 1)
			{
				outFileName = util::addBeforeExtension(
				    out_fname,
				    std::string("_subset") +
				        util::padZeros(subsetId, numDigitsInFilename));
			}
			generatedImage->writeToFile(outFileName);
		}

		if (saveOnMemory)
		{
			sensImages.push_back(std::move(generatedImage));
		}
	}

	endSensImgGen();

	// Restore original value
	num_OSEM_subsets = originalNumOSEMSubsets;
}

int OSEM::getExpectedSensImagesAmount() const
{
	if (usingListModeInput)
	{
		return 1;
	}
	return num_OSEM_subsets;
}

void OSEM::setSensitivityImages(const std::vector<Image*>& sensImages)
{
	ImageParams imageSensParams;
	m_sensitivityImages.clear();

	for (size_t i = 0; i < sensImages.size(); i++)
	{
		auto sensImage = sensImages[i];

		ASSERT(sensImage != nullptr);
		ASSERT_MSG(sensImage->getParams().isValid(),
		           "Invalid image parameters");

		if (i == 0)
		{
			imageSensParams = sensImage->getParams();
		}
		else
		{
			ASSERT_MSG(sensImage->getParams().isSameAs(imageSensParams),
			           "Image parameters mismatch");
		}
		m_sensitivityImages.push_back(sensImage);
	}
	imageSensParams.num_frames = imageParams.num_frames;
	setImageParams(imageSensParams);
}

void OSEM::setSensitivityImages(
    const std::vector<std::unique_ptr<Image>>& sensImages)
{
	std::vector<Image*> sensImages_raw;
	for (size_t i = 0; i < sensImages.size(); i++)
	{
		sensImages_raw.push_back(sensImages[i].get());
	}
	setSensitivityImages(sensImages_raw);
}

#if BUILD_PYBIND11
void OSEM::setSensitivityImages(const pybind11::list& pySensImgList)
{
	std::vector<Image*> sensImages_raw;
	for (size_t i = 0; i < pySensImgList.size(); i++)
	{
		sensImages_raw.push_back(pySensImgList[i].cast<Image*>());
	}
	setSensitivityImages(sensImages_raw);
}
#endif

void OSEM::setSensitivityImage(Image* sensImage, int subset)
{
	if (usingListModeInput)
	{
		ASSERT_MSG(subset == 0, "In List-Mode reconstruction, only one "
		                        "sensitivity image is needed");
	}
	else if (subset >= num_OSEM_subsets)
	{
		std::string errorMessage = "Subset index too high. The expected number "
		                           "of sensitivity images is ";
		errorMessage += std::to_string(num_OSEM_subsets) + ". Subset " +
		                std::to_string(subset) + " does not exist.";
		ASSERT_MSG(false, errorMessage.c_str());
	}
	const size_t expectedSize = usingListModeInput ? 1 : num_OSEM_subsets;
	if (m_sensitivityImages.size() != expectedSize)
	{
		if (subset != 0)
		{
			m_sensitivityImages.resize(expectedSize);
		}
		else
		{
			m_sensitivityImages.resize(1);
		}
	}

	ASSERT(sensImage != nullptr);
	ASSERT_MSG(sensImage->getParams().isValid(), "Invalid image parameters");

	if (imageParams.isValid())
	{
		ASSERT_MSG(sensImage->getParams().isSameAs(imageParams),
		           "Image parameters mismatch");
	}
	else
	{
		setImageParams(sensImage->getParams());
	}

	m_sensitivityImages[subset] = sensImage;
}

void OSEM::loadSubsetInternal(int p_subsetId, bool p_forRecon)
{
	mp_projector->setBinIter(getBinIterators()[p_subsetId].get());
	loadSubset(p_subsetId, p_forRecon);
}

void OSEM::initializeForSensImgGen()
{
	setupOperatorsForSensImgGen();
	allocateForSensImgGen();
}

void OSEM::initializeForRecon()
{
	setupOperatorsForRecon();
	allocateForRecon();
}

void OSEM::setSensitivityHistogram(const Histogram* pp_sensitivityData)
{
	getCorrector().setSensitivityHistogram(pp_sensitivityData);
}

const Histogram* OSEM::getSensitivityHistogram() const
{
	return getCorrector().getSensitivityHistogram();
}

const ProjectionData* OSEM::getDataInput() const
{
	return mp_dataInput;
}

void OSEM::setDataInput(const ProjectionData* pp_dataInput)
{
	mp_dataInput = pp_dataInput;
	if (dynamic_cast<const ListMode*>(mp_dataInput))
	{
		usingListModeInput = true;
	}
	else
	{
		usingListModeInput = false;
	}
}

void OSEM::addTOF(float p_tofWidth_ps, int p_tofNumStd)
{
	projectorParams.tofWidth_ps = p_tofWidth_ps;
	projectorParams.tofNumStd = p_tofNumStd;
	projectorParams.flagProjTOF = true;
}

void OSEM::addProjPSF(const std::string& pr_projPsf_fname)
{
	projectorParams.projPsf_fname = pr_projPsf_fname;
}

void OSEM::addImagePSF(const std::string& p_imagePsf_fname,
                       ImagePSFMode p_imagePSFMode)
{
	ASSERT_MSG(!p_imagePsf_fname.empty(), "Empty filename for Image-space PSF");
	if (p_imagePSFMode == UNIFORM)
	{
		imagePsf = std::make_unique<OperatorPsf>(p_imagePsf_fname);
	}
	else
	{
		ASSERT_MSG(imageParams.isValid(),
		           "For spatially variant PSF, image parameters have to be set "
		           "before calling addImagePSF");
		imagePsf =
		    std::make_unique<OperatorVarPsf>(p_imagePsf_fname, imageParams);
	}
	flagImagePSF = true;
}

void OSEM::setSaveIterRanges(util::RangeList p_saveIterList,
                             const std::string& p_saveIterPath)
{
	saveIterRanges = p_saveIterList;
	saveIterPath = p_saveIterPath;
}

void OSEM::setListModeEnabled(bool enabled)
{
	usingListModeInput = enabled;
}

void OSEM::setProjector(const std::string& projectorName)
{
	projectorType = io::getProjector(projectorName);
}

bool OSEM::isListModeEnabled() const
{
	return usingListModeInput;
}

bool OSEM::hasImagePSF() const
{
	return flagImagePSF;
}

void OSEM::enableNeedToMakeCopyOfSensImage()
{
	needToMakeCopyOfSensImage = true;
}

ImageParams OSEM::getImageParams() const
{
	return imageParams;
}

void OSEM::setImageParams(const ImageParams& params)
{
	imageParams = params;
}

void OSEM::setRandomsHistogram(const Histogram* pp_randoms)
{
	getCorrector().setRandomsHistogram(pp_randoms);
}

void OSEM::setScatterHistogram(const Histogram* pp_scatter)
{
	getCorrector().setScatterHistogram(pp_scatter);
}

void OSEM::setGlobalScalingFactor(float globalScalingFactor)
{
	getCorrector().setGlobalScalingFactor(globalScalingFactor);
}

void OSEM::setAttenuationImage(const Image* pp_attenuationImage)
{
	getCorrector().setAttenuationImage(pp_attenuationImage);
}

void OSEM::setACFHistogram(const Histogram* pp_acf)
{
	getCorrector().setACFHistogram(pp_acf);
}

void OSEM::setHardwareAttenuationImage(const Image* pp_hardwareAttenuationImage)
{
	getCorrector().setHardwareAttenuationImage(pp_hardwareAttenuationImage);
}

void OSEM::setHardwareACFHistogram(const Histogram* pp_hardwareAcf)
{
	getCorrector().setHardwareACFHistogram(pp_hardwareAcf);
}

void OSEM::setInVivoAttenuationImage(const Image* pp_inVivoAttenuationImage)
{
	getCorrector().setInVivoAttenuationImage(pp_inVivoAttenuationImage);
}

void OSEM::setInVivoACFHistogram(const Histogram* pp_inVivoAcf)
{
	getCorrector().setInVivoACFHistogram(pp_inVivoAcf);
}

void OSEM::setInvertSensitivity(bool invert)
{
	getCorrector().setInvertSensitivity(invert);
}

const Image* OSEM::getSensitivityImage(int subsetId) const
{
	if (mp_copiedSensitivityImage != nullptr)
	{
		return mp_copiedSensitivityImage.get();
	}
	return m_sensitivityImages.at(subsetId);
}

Image* OSEM::getSensitivityImage(int subsetId)
{
	if (mp_copiedSensitivityImage != nullptr)
	{
		return mp_copiedSensitivityImage.get();
	}
	return m_sensitivityImages.at(subsetId);
}

void OSEM::dumpHBasis(const Array2DAlias<float>& H, const char* tag) {
	auto dims = H.getDims();                 // {R, T}
	const size_t R = dims[0], T = dims[1];
	const size_t showR = std::min<size_t>(R, 8);   // limit output
	const size_t showT = std::min<size_t>(T, 30);

	std::printf("HBasis%s%s (R=%zu, T=%zu)\n",
				tag ? " " : "", tag ? tag : "", R, T);

	for (size_t r = 0; r < showR; ++r) {
		std::printf("r=%zu:", r);
		for (size_t t = 0; t < showT; ++t) {
			std::printf(" %.6g", H[r][t]);
		}
		if (showT < T) std::printf(" ...");
		std::printf("\n");
	}
	if (showR < R) std::printf("...\n");
	std::fflush(stdout);
}

template<bool IS_DYNAMIC>
void apply_update(ImageBase*, ImageBase*, const ImageBase*,
                  const float*, float);

std::unique_ptr<ImageOwned> OSEM::reconstruct(const std::string& out_fname)
{
	ASSERT_MSG(mp_dataInput != nullptr, "Data input unspecified");
	ASSERT_MSG(!m_sensitivityImages.empty(), "Sensitivity image(s) not set");
	ASSERT_MSG(num_OSEM_subsets > 0, "Not enough OSEM subsets");
	ASSERT_MSG(num_MLEM_iterations > 0, "Not enough MLEM iterations");

	if (!imageParams.isValid())
	{
		imageParams = m_sensitivityImages[0]->getParams();
		printf("Getting ImageParams from Sensitivity Image...");
	}

	printf("\n before sens im \n");
	const int expectedNumberOfSensImages = getExpectedSensImagesAmount();
	if (expectedNumberOfSensImages !=
	    static_cast<int>(m_sensitivityImages.size()))
	{
		throw std::logic_error("The number of sensitivity images provided does "
		                       "not match the number of subsets. Expected " +
		                       std::to_string(expectedNumberOfSensImages) +
		                       " but received " +
		                       std::to_string(m_sensitivityImages.size()));
	}

	outImage = std::make_unique<ImageOwned>(imageParams);
	outImage->allocate();

	if (usingListModeInput)
	{
		if (needToMakeCopyOfSensImage)
		{
			std::cout << "Arranging sensitivity image scaling for ListMode in "
			             "separate copy..."
			          << std::endl;
			// This is for the specific case of doing a list-mode reconstruction
			// from Python
			auto imageSensParams = m_sensitivityImages[0]->getParams();
			// imageSensParams.num_frames = 1;
			mp_copiedSensitivityImage =
			    std::make_unique<ImageOwned>(imageSensParams);
			mp_copiedSensitivityImage->allocate();
			mp_copiedSensitivityImage->copyFromImage(m_sensitivityImages.at(0));
			mp_copiedSensitivityImage->multWithScalar(
			    1.0f / (static_cast<float>(num_OSEM_subsets)));
		}
		else if (num_OSEM_subsets != 1)
		{
			std::cout << "Arranging sensitivity image scaling for ListMode..."
			          << std::endl;
			m_sensitivityImages[0]->multWithScalar(
			    1.0f / (static_cast<float>(num_OSEM_subsets)));
		}
	}
	printf("\n after sens im, before getCorrector \n");
	getCorrector().setup();
	printf("\n after getCorrector, before initializeForRecon \n");
	initializeForRecon();
	printf("\n after initializeForRecon \n");

	// Calculate factor to use for sensitivity image DEFAULT4D and LR
	const bool IS_DYNAMIC =
		(projectorParams.projectorUpdaterType != OperatorProjectorParams::DEFAULT3D);
	const bool isLowRank =
		(projectorParams.projectorUpdaterType == OperatorProjectorParams::LR ||
			projectorParams.projectorUpdaterType == OperatorProjectorParams::LRDUALUPDATE);
	const bool dualUpdate =
		(projectorParams.projectorUpdaterType == OperatorProjectorParams::LRDUALUPDATE);

	std::vector<float> c_WUpdate_r;
	std::vector<float> c_Hupdate_r;
	int rank;
	int T = 0;
	allocateHBasisTmpBuffer();
	auto* HBuffer = dynamic_cast<Array2D<float>*>(getHBasisTmpBuffer());
	// std::unique_ptr<Array2D<float>> HBuffer = std::make_unique<Array2D<float>>();

	if (isLowRank) {
		// Check LR Updater
		if (auto* proj = dynamic_cast<OperatorProjector*>(mp_projector.get())) {
			if (auto* lr = dynamic_cast<OperatorProjectorUpdaterLR*>(proj->getUpdater())) {
				if (lr->getUpdateH() != projectorParams.updateH)
				{
					throw std::logic_error("member updateH of OperatorProjectorUpdaterLR is "
							   "different than input updateH in projectorParams");
				}
			}
			else
			{
				throw std::runtime_error("proj->getUpdater could not be cast to OperatorProjectorUpdaterLR");
			}
		}

		// HBasis is rank x T
		const auto dims = projectorParams.HBasis.getDims();   // std::array<size_t,2>
		rank = static_cast<int>(dims[0]);
		T = static_cast<int>(dims[1]);

		if (!projectorParams.updateH || dualUpdate)
		{
			c_WUpdate_r.resize(rank, 0.f);
			generateWUpdateSensScaling(c_WUpdate_r.data());
		}
		if (projectorParams.updateH || dualUpdate)
		{
			c_Hupdate_r.resize(rank, 0.f);
			generateHUpdateSensScaling(c_Hupdate_r.data());
			HBuffer->fill(0.f);
			if (auto* proj = reinterpret_cast<OperatorProjector*>(mp_projector.get())) {
				if (auto* lr = dynamic_cast<OperatorProjectorUpdaterLR*>(proj->getUpdater()))
				{
					if (!dualUpdate)
					{
						lr->setUpdateH(projectorParams.updateH);            // switch to H accumulation mode
					}
					lr->setHBasis(projectorParams.HBasis);
					lr->setHBasisWrite(*HBuffer);       // write into mp_HWrite
					lr->setCurrentImgBuffer(outImage.get());
					// todo: remove outImage to direct towards mlemImage_rp in case PSF is used
					printf("set HBasisWrite for OperatorProjectorUpdaterLR");
				}
				else
				{
					throw std::runtime_error("proj->getUpdater could not be cast to OperatorProjectorUpdaterLR");
				}
			}
			else
			{
				throw std::runtime_error("mp_projector could not be cast to OperatorProjector");
			}
		}
	}
	else
	{
		// 4D dynamic case
		T = imageParams.num_frames;
		c_WUpdate_r.resize(T, 1.f);
	}

	const int numDigitsInFilename = util::numberOfDigits(num_MLEM_iterations);

	// MLEM iterations
	for (int iter = 0; iter < num_MLEM_iterations; iter++)
	{
		std::cout << "\n"
		          << "MLEM iteration " << iter + 1 << "/" << num_MLEM_iterations
		          << "..." << std::endl;
		// OSEM subsets
		for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
		{
			std::cout << "OSEM subset " << subsetId + 1 << "/"
			          << num_OSEM_subsets << "..." << std::endl;

			loadSubsetInternal(subsetId, true);

			// SET TMP VARIABLES TO 0
			getImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO)
			    ->setValue(0.0);
			if ((isLowRank && projectorParams.updateH) || dualUpdate) {
				printf("\n initialize HBasisTmpBuffer with zeros.\n");
				HBuffer->fill(0.f);
				printf("\n HBasisTmpBuffer initialized with zeros.\n");
			}

			ImageBase* mlemImage_rp;
			if (flagImagePSF)
			{
				getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
				    ->setValue(0.0);
				// PSF
				imagePsf->applyA(
				    getMLEMImageBuffer(),
				    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
				mlemImage_rp =
				    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF);
			}
			else
			{
				mlemImage_rp = getMLEMImageBuffer();
			}

			if (!projectorParams.updateH || dualUpdate)
			{
				printf("\n Compute EM (1).\n");
				computeEMUpdateImage(
					*mlemImage_rp,
					*getImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO));
			}
			else
			{
				// When updating H, destImage must be the actual image (and not
				// a zeroed buffer) to retrieve the value from the image during backudpate
				printf("\n Compute EM (2).\n");
				computeEMUpdateImage(*mlemImage_rp,
					*mlemImage_rp);
			}

			// PSF
			if (flagImagePSF)
			{
				getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
				    ->setValue(0.0);
				imagePsf->applyAH(
				    getImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO),
				    getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF));
			}

			// UPDATE
			if (!projectorParams.updateH || dualUpdate)
			{
				ImageBase* updateImage =
				    flagImagePSF
				        ? getImageTmpBuffer(TemporaryImageSpaceBufferType::PSF)
				        : getImageTmpBuffer(TemporaryImageSpaceBufferType::EM_RATIO);
				printf("\n Apply EM W Update \n");
				if (IS_DYNAMIC) {
					apply_update<true>(getMLEMImageBuffer(), updateImage, getSensImageBuffer(),
									   c_WUpdate_r.data(), EPS_FLT);
				} else {
					apply_update<false>(getMLEMImageBuffer(), updateImage, getSensImageBuffer(),
										nullptr, EPS_FLT);
				}
			}
			if (projectorParams.updateH || (dualUpdate && iter > 0))
			{
				printf("\n Apply EM H Update \n");
				float*       H_old_ptr = projectorParams.HBasis.getRawPointer(); // current H
				const float* Hnum_ptr  = HBuffer->getRawPointer();               // numerator accumulated this subset

				// shapes: rank x T
				const int    R = rank;
				const int    T_ = T;

				double min_ratio = 1e30, max_ratio = -1e30, mean_ratio = 0.0;
				double sum_num = 0.0, sum_den = 0.0;

				for (int r = 0; r < R; ++r) {
					const double den = std::max<double>(c_Hupdate_r[r], EPS_FLT);
					sum_den += den;
					for (int t = 0; t < T_; ++t) {
						const double num = Hnum_ptr[r*T_ + t];
						sum_num += num;
						const double ratio = num / den;
						min_ratio = std::min(min_ratio, ratio);
						max_ratio = std::max(max_ratio, ratio);
						mean_ratio += ratio;
					}
				}
				mean_ratio /= (R * T_);

				printf("\nH update stats: sum_num=%.6g sum_den=%.6g  ratio[min,mean,max]=[%.3g, %.3g, %.3g]\n",
					   sum_num, sum_den, min_ratio, mean_ratio, max_ratio);

				printf("\n --- Before Update --- \n");
				double sum = 0.0;
				for (int i = 0; i < rank*T; ++i) sum += H_old_ptr[i];
				printf("sum(H)=%.6g, mean(H)=%.6g\n", sum, sum / (rank*T));


				// H_new := H_old * (Hnum / c_r)
				util::parallelForChunked(
					T, globals::getNumThreads(),
					[rank, T, c_Hupdate_r, H_old_ptr, Hnum_ptr](int t, int /*tid*/)
					{
						for (int r = 0; r < rank; ++r)
						{
							const float denom = std::max(c_Hupdate_r[r], EPS_FLT);
							const float inv   = 1.0f / denom;
							float*       Hr  = H_old_ptr + r * T;
							const float* Nr  = Hnum_ptr  + r * T;
							Hr[t] = Hr[t] * (Nr[t] * inv); // write the *new H* back over H_old
						}
					});

				printf("\n --- After Update --- \n");
				double sum_after = 0.0;
				for (int i = 0; i < rank * T; ++i) sum_after += H_old_ptr[i];
				printf("sum(H)=%.6g, mean(H)=%.6g\n", sum_after, sum_after / (rank*T));

			}

			if (dualUpdate)
			{
				printf("\n Updating LR Sensitivity image scaling...\n");
				float sum_c_w = 0.0;
				float sum_c_h = 0.0;
				for (int i = 0; i < rank; ++i)
				{
					sum_c_w += c_WUpdate_r[i];
					sum_c_h += c_Hupdate_r[i];
				}
				printf("Before: sum(c_W)=%.6g, sum(c_H)=%.6g\n", sum_c_w, sum_c_h);
				std::fill(c_WUpdate_r.begin(), c_WUpdate_r.end(), 0.0f);
				std::fill(c_Hupdate_r.begin(), c_Hupdate_r.end(), 0.0f);
				generateWUpdateSensScaling(c_WUpdate_r.data());
				generateHUpdateSensScaling(c_Hupdate_r.data());
				float sum_c_w_2 = 0.0;
				float sum_c_h_2 = 0.0;
				for (int i = 0; i < rank; ++i)
				{
					sum_c_w_2 += c_WUpdate_r[i];
					sum_c_h_2 += c_Hupdate_r[i];
				}
				printf("After: sum(c_W)=%.6g, sum(c_H)=%.6g\n", sum_c_w_2, sum_c_h_2);
			}
		}
		if (saveIterRanges.isIn(iter + 1))
		{
			std::string iteration_name =
			    util::padZeros(iter + 1, numDigitsInFilename);
			std::string outIteration_fname = util::addBeforeExtension(
			    saveIterPath, std::string("_iteration") + iteration_name);
			getMLEMImageBuffer()->writeToFile(outIteration_fname);
		}
		completeMLEMIteration();
	}

	// restore H Basis
	// if (isLowRank && projectorParams.updateH) {
	// 	if (auto* proj = dynamic_cast<OperatorProjector*>(mp_projector.get())) {
	// 		if (auto* lr = dynamic_cast<OperatorProjectorUpdaterLR*>(proj->getUpdater())) {
	// 			lr->setHBasis(projectorParams.HBasis); // point back to real H
	// 		}
	// 	}
	// }

	endRecon();

	// Deallocate the copied sensitivity image if it was allocated
	mp_copiedSensitivityImage = nullptr;

	if (!out_fname.empty())
	{
		std::cout << "Saving image..." << std::endl;
		outImage->writeToFile(out_fname);
	}

	return std::move(outImage);
}

void OSEM::summary() const
{
	std::cout << "Number of iterations: " << num_MLEM_iterations << std::endl;
	std::cout << "Number of subsets: " << num_OSEM_subsets << std::endl;
	if (usingListModeInput)
	{
		std::cout << "Uses List-Mode data as input" << std::endl;
	}

	int numberOfSensImagesSet = 0;
	for (size_t i = 0; i < m_sensitivityImages.size(); i++)
	{
		if (m_sensitivityImages[i] != nullptr)
		{
			numberOfSensImagesSet++;
		}
	}
	std::cout << "Number of sensitivity images set: " << numberOfSensImagesSet
	          << std::endl;

	std::cout << "Hard threshold: " << hardThreshold << std::endl;
	if (projectorType == OperatorProjector::SIDDON)
	{
		std::cout << "Projector type: Siddon" << std::endl;
		std::cout << "Number of Siddon rays: " << projectorParams.numRays << std::endl;
	}
	else if (projectorType == OperatorProjector::DD)
	{
		std::cout << "Projector type: Distance-Driven" << std::endl;
	}

	std::cout << "Number of threads used: " << globals::getNumThreads()
	          << std::endl;
	std::cout << "Scanner name: " << scanner.scannerName << std::endl;

	if (flagImagePSF)
	{
		if (dynamic_cast<OperatorVarPsf*>(imagePsf.get()) == nullptr)
		{
			std::cout << "Uses Image-space PSF" << std::endl;
		}
		else
		{
			std::cout << "Uses Image-space variant PSF" << std::endl;
		}
	}
	if (projectorParams.projPsf_fname.empty())
	{
		std::cout << "Uses Projection-space PSF" << std::endl;
	}
	if (projectorParams.flagProjTOF)
	{
		std::cout << "Uses Time-of-flight with " << std::endl;
	}

	std::cout << "Saved iterations list: " << saveIterRanges << std::endl;
	if (!saveIterRanges.empty())
	{
		std::cout << "Saved image files prefix name: " << saveIterPath
		          << std::endl;
	}
}

template<bool IS_DYNAMIC>
void apply_update(ImageBase* destImage,
                  ImageBase* numerator,
                  const ImageBase* norm,
                  const float* c_r,  // sum_t H[r,t] (null if !IS_DYNAMIC)
                  const float eps)
{
	if constexpr (IS_DYNAMIC) {
		destImage->updateEMThresholdRankScaled(
			numerator, norm, c_r, eps);
	} else {
		destImage->updateEMThreshold(
		    numerator, norm, eps);
	}
}




}  // namespace yrt
