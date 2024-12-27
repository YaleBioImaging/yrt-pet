/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "motion/ImageWarperTemplate.hpp"

#include "utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_imagewarpertemplate(py::module& m)
{
	auto c = py::class_<ImageWarperTemplate>(m, "ImageWarperTemplate");
	c.def("setImageHyperParam",
	      static_cast<void (ImageWarperTemplate::*)(
	          const std::vector<int>& imDim, const std::vector<float>& imSize)>(
	          &ImageWarperTemplate::setImageHyperParam));
	c.def("setImageHyperParam", static_cast<void (ImageWarperTemplate::*)(
	                                const ImageParams& img_params)>(
	                                &ImageWarperTemplate::setImageHyperParam));
	c.def("setMotionHyperParam", &ImageWarperTemplate::setMotionHyperParam);
	c.def("setFramesParamFromFile",
	      &ImageWarperTemplate::setFramesParamFromFile);
	c.def("initParamContainer", &ImageWarperTemplate::initParamContainer);
	c.def("setRefImage", &ImageWarperTemplate::setRefImage);
	c.def("warpRefImage", &ImageWarperTemplate::warpRefImage);
	c.def("warpImageToRefFrame", &ImageWarperTemplate::warpImageToRefFrame);
	c.def("computeGlobalWarpToRefFrame",
	      &ImageWarperTemplate::computeGlobalWarpToRefFrame);
	c.def("deactivateFrame", &ImageWarperTemplate::deactivateFrame);
	c.def("setReferenceFrameParam",
	      &ImageWarperTemplate::setReferenceFrameParam);
	c.def("setFrameParam", &ImageWarperTemplate::setFrameParam);
	c.def("reset", &ImageWarperTemplate::reset);
	c.def("isFrameUsed", &ImageWarperTemplate::isFrameUsed);
	c.def("getImNbVoxel", &ImageWarperTemplate::getImNbVoxel);
	c.def("getReferenceFrameId", &ImageWarperTemplate::getReferenceFrameId);
	c.def("getNumberOfFrame", &ImageWarperTemplate::getNumberOfFrame);
	c.def("getFrameStartTime", &ImageWarperTemplate::getFrameStartTime);

	auto c_operator =
	    py::class_<OperatorWarpRefImage>(m, "OperatorWarpRefImage");
	c_operator.def(py::init<int>());
	c_operator.def("setFrameId", &OperatorWarpRefImage::setFrameId);
	c_operator.def("applyA", &OperatorWarpRefImage::applyA);
	c_operator.def("applyAH", &OperatorWarpRefImage::applyAH);
}
#endif


ImageWarperTemplate::ImageWarperTemplate()
{
	m_warpMode = "";

	// Class states boolean.
	m_imHyperInit = false;
	m_motionHyperInit = false;
	m_warperParamContainerInit = false;
	m_refFrameDefined = false;
}


ImageWarperTemplate::~ImageWarperTemplate() = default;


void ImageWarperTemplate::setImageHyperParam(const std::vector<int>& imDim,
                                             const std::vector<float>& imSize)
{
	if (m_imHyperInit == true)
	{
		std::cerr << "setImageHyperParam() has already been called. Exiting,"
		          << std::endl;
		exit(EXIT_FAILURE);
	}

	// TODO: Add validations.
	m_imNbVoxel = imDim;
	m_imSize = imSize;

	m_imHyperInit = true;
}

void ImageWarperTemplate::setImageHyperParam(const ImageParams& img_params)
{
	const std::vector<int> imDim({img_params.nx, img_params.ny, img_params.nz});
	const std::vector<float> imSize({static_cast<float>(img_params.length_x),
	                                 static_cast<float>(img_params.length_y),
	                                 static_cast<float>(img_params.length_z)});
	setImageHyperParam(imDim, imSize);
}


void ImageWarperTemplate::setMotionHyperParam(int numberOfFrame)
{
	if (m_motionHyperInit == true)
	{
		std::cerr << "setMotionHyperParam() has already been called. Exiting,"
		          << std::endl;
		exit(EXIT_FAILURE);
	}

	// TODO: Add validations check.
	m_numberOfFrame = numberOfFrame;

	m_motionHyperInit = true;
}


void ImageWarperTemplate::setFramesParamFromFile(std::string paramFileName)
{
	size_t referenceFrameId;
	std::vector<std::vector<double>> extractedFrameParam;

	referenceFrameId =
	    extractFramesParamFromFile(paramFileName, extractedFrameParam);

	setMotionHyperParam(extractedFrameParam.size());
	initParamContainer();

	setReferenceFrameParam(referenceFrameId,
	                       extractedFrameParam[referenceFrameId][0],
	                       extractedFrameParam[referenceFrameId][1]);

	std::vector<double> cFrameWarpParam;
	// Number of parameters was validated in extractFramesParamFromFile().
	cFrameWarpParam.resize(7);
	for (size_t m = 0; m < extractedFrameParam.size(); m++)
	{
		if (m != referenceFrameId)
		{
			for (int i = 0; i < 7; i++)
			{
				cFrameWarpParam[i] = extractedFrameParam[m][i + 2];
			}
			setFrameParam(m, cFrameWarpParam, (float)extractedFrameParam[m][0],
			              (float)extractedFrameParam[m][1]);
		}
	}
}


int ImageWarperTemplate::extractFramesParamFromFile(
    std::string paramFileName, std::vector<std::vector<double>>& frameParam)
{
	std::ifstream infile(paramFileName);
	if (!infile.good())
	{
		std::cerr << "The warp parameters file " << paramFileName
		          << " does not exist." << std::endl;
		exit(EXIT_FAILURE);
	}

	int referenceFrameId = -1;

	std::vector<double> currLineParam;
	std::string delimiter = " ";
	std::string line;
	// Parse each line.
	while (getline(infile, line))
	{
		std::istringstream iss(line);
		if (line.rfind("Reference frame Id: ", 0) == 0)
		{
			if (referenceFrameId == -1)
			{
				referenceFrameId = stoi(line.substr(line.find(":") + 1));
			}
			else
			{
				std::cerr
				    << "The reference frame was defined more than one time."
				    << std::endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (line.rfind("#", 0) != 0)
		{
			currLineParam = splitStringIntoVectorOfDouble(line, delimiter);
			if (currLineParam.size() != 9)
			{
				std::cerr
				    << "Invalid number of parameters given for the current "
				       "line "
				       "parsed. It shoud have been 9, not "
				    << currLineParam.size() << std::endl;
				exit(EXIT_FAILURE);
			}
			else
			{
				frameParam.push_back(currLineParam);
			}
		}
	}

	return referenceFrameId;
}


void ImageWarperTemplate::initParamContainer()
{
	if (m_warperParamContainerInit == true)
	{
		std::cerr << "initParamContainer() has already been called. Exiting,"
		          << std::endl;
		exit(EXIT_FAILURE);
	}

	if ((m_imHyperInit != true) or (m_motionHyperInit != true))
	{
		std::cerr
		    << "Image or motion hyper parameters were not defined. Exiting"
		    << std::endl;
		exit(EXIT_FAILURE);
	}

	// Fow now, it seems that the rotation center will always be defined
	// relative to
	// the center of the image space.
	m_rotCenter.resize(m_imNbOfDimension);
	m_rotCenter = {0.0, 0.0, 0.0};

	// By default, all motion frame should be used.
	m_motionFrameUsed.resize(m_numberOfFrame);
	fill(m_motionFrameUsed.begin(), m_motionFrameUsed.end(), true);

	// No default values.
	m_weightOfFrame.resize(m_numberOfFrame);
	m_frameTimeBinStart.resize(m_numberOfFrame);

	// By default, we want this object to be mostly mute.
	m_verboseLevel = 0;

	// By default, the weight of the frame is not apply when warping or inverse
	// warping.
	// Currently, the frame weightinh is only applied in
	// computeGlobalWarpToRefFrame().
	m_applyWeightToWarp = false;

	// Let the child initialize what he needs.
	initWarpModeSpecificParameters();

	// Explicit that the warp parameters of each frame were not defined.
	m_frameWarpParamDefined.resize(m_numberOfFrame);
	fill(m_motionFrameUsed.begin(), m_motionFrameUsed.end(), false);

	// Initializaton of warper parameters contrainer was successful.
	m_warperParamContainerInit = true;
}


void ImageWarperTemplate::reset()
{
	m_imNbVoxel.clear();
	m_imSize.clear();
	m_rotCenter.clear();
	m_referenceFrameId = -1;
	m_numberOfFrame = 0;
	m_motionFrameUsed.clear();
	m_frameTimeBinStart.clear();
	mp_refImage = nullptr;
	m_warpMode = "";

	// Set the state parameters to false.
	m_imHyperInit = false;
	m_motionHyperInit = false;
	m_warperParamContainerInit = false;
	m_frameWarpParamDefined.clear();
}

void ImageWarperTemplate::setRefImage(const Image* image)
{
	mp_refImage = image;
}


void ImageWarperTemplate::warpRefImage(Image* image, int frameId) const
{
	if (frameId != m_referenceFrameId)
	{
		warp(image, frameId);
	}
	else
	{
		image->copyFromImage(mp_refImage);
		if (m_applyWeightToWarp == true)
		{
			image->multWithScalar(m_weightOfFrame[frameId]);
		}
	}
}

void OperatorWarpRefImage::applyA(const Variable* warp, Variable* out)
{
	const ImageWarperTemplate* warper =
	    dynamic_cast<const ImageWarperTemplate*>(warp);
	Image* img = dynamic_cast<Image*>(out);
	ASSERT(img != nullptr);
	ASSERT(warper != nullptr);
	warper->warpRefImage(img, m_frameId);
}
void OperatorWarpRefImage::applyAH(const Variable* warp, Variable* out)
{
	const ImageWarperTemplate* warper =
	    dynamic_cast<const ImageWarperTemplate*>(warp);
	Image* img = dynamic_cast<Image*>(out);
	ASSERT(img != nullptr);
	ASSERT(warper != nullptr);
	warper->warpImageToRefFrame(img, m_frameId);
}

void ImageWarperTemplate::warpImageToRefFrame(Image* image, int frameId) const
{
	// If the motion frame specified is the reference one, the image is not
	// deformed
	// since all motion state transformation are relative to this one.
	// TODO: Might not be the case with some interpolation scheme. Keep in mind.
	if (frameId != m_referenceFrameId)
	{
		inverseWarp(image, frameId);
	}
	else if (m_applyWeightToWarp == true)
	{
		image->multWithScalar(m_weightOfFrame[frameId]);
	}
}


void ImageWarperTemplate::computeGlobalWarpToRefFrame(Image* image,
                                                      bool writeFileSteps)
{
	// Warping methods of this class modify the given image so we need a copy to
	// warp it multiple time.
	auto tmpCopyOfGivenImage = std::make_unique<ImageOwned>(image->getParams());
	tmpCopyOfGivenImage->allocate();
	// Temporary container for the results to which each warp results will be
	// added.
	auto tmpGlobalWarpResult = std::make_unique<ImageOwned>(ImageParams(
	    m_imNbVoxel[0], m_imNbVoxel[1], m_imNbVoxel[2], (double)m_imSize[0],
	    (double)m_imSize[1], (double)m_imSize[2]));
	tmpGlobalWarpResult->allocate();
	// If m_applyWeightToWarp is false, set it temporarily to true so that
	// inverseWarp() is weighted.
	bool flipBackApplyWeightToWarp = false;
	if (m_applyWeightToWarp == false)
	{
		m_applyWeightToWarp = true;
		flipBackApplyWeightToWarp = true;
	}
	for (int m = 0; m < m_numberOfFrame; m++)
	{
		// While copying, we also apply the weight of the current frame.
		tmpCopyOfGivenImage->copyFromImage(image);

		if (writeFileSteps)
			tmpCopyOfGivenImage->writeToFile("tmpCopyOfGivenImage_inverse" +
			                                 std::to_string(m) + ".nii");

		inverseWarp(tmpCopyOfGivenImage.get(), m);

		if (writeFileSteps)
			tmpCopyOfGivenImage->writeToFile("tmpCopyOfGivenImage" +
			                                 std::to_string(m) + ".nii");

		tmpCopyOfGivenImage->addFirstImageToSecond(tmpGlobalWarpResult.get());
	}
	// If the m_applyWeightToWarp was flipped, reverit it back.
	if (flipBackApplyWeightToWarp == true)
	{
		m_applyWeightToWarp = false;
	}

	// The final results is copied into the container provided.
	image->copyFromImage(tmpGlobalWarpResult.get());
}


void ImageWarperTemplate::deactivateFrame(int frameId)
{
	m_motionFrameUsed[frameId] = false;
}


void ImageWarperTemplate::setFrameParam(int frameId,
                                        const std::vector<double>& warpParam,
                                        float frameTimeBinStart,
                                        float frameWeight)
{
	isFrameIdValidToDefine(frameId);

	setFrameWeight(frameWeight, frameId);

	setFrameTimeBinStart(frameTimeBinStart, frameId);

	setFrameWarpParameters(frameId, warpParam);

	m_frameWarpParamDefined[frameId] = true;
}


void ImageWarperTemplate::setReferenceFrameParam(int frameId,
                                                 float frameTimeBinStart,
                                                 float frameWeight)
{
	bool isRefFrame = true;
	isFrameIdValidToDefine(frameId, isRefFrame);

	setFrameWeight(frameWeight, frameId);

	setFrameTimeBinStart(frameTimeBinStart, frameId);

	m_referenceFrameId = frameId;

	m_frameWarpParamDefined[frameId] = true;
	m_refFrameDefined = true;
}


void ImageWarperTemplate::isFrameIdValidToDefine(int frameId, bool isRef)
{
	// Check if the frame Id is in the valid range.
	if ((frameId < 0) || ((frameId >= m_numberOfFrame)))
	{
		std::cerr << "A frame Id of " << frameId
		          << " is invalid. Should be between 0 and "
		          << m_numberOfFrame - 1 << std::endl;
		exit(EXIT_FAILURE);
	}

	// Check if the frame Id was not previously defined.
	// TODO: Currently, there is a loop-hole where the reference frame Id could
	// be
	// affected warp parameters. Right now, this should be impossible with the
	// following
	// check but it could be circumvented by a try-catch (I think...) so it is
	// not
	// fail-safe.
	if (m_frameWarpParamDefined[frameId] == true)
	{
		std::cerr << "The frame Id of " << frameId
		          << " was already defined previously. "
		             "Exiting."
		          << std::endl;
		exit(EXIT_FAILURE);
	}

	// Check if the reference frame Id has already been defined.
	if ((isRef == true) && (m_refFrameDefined == true))
	{
		std::cerr
		    << "Attempt to set the reference frame parameters a second time. "
		       "Exiting."
		    << std::endl;
		exit(EXIT_FAILURE);
	}
}


void ImageWarperTemplate::setFrameWeight(float weight, int frameId)
{
	if (weight > 0.0)
	{
		m_weightOfFrame[frameId] = weight;
	}
	else
	{
		std::cerr << "A frame weight of " << weight
		          << " was attributed to frame " << frameId
		          << "which is invalid. It should be positive." << std::endl;
		exit(EXIT_FAILURE);
	}
}


void ImageWarperTemplate::setFrameTimeBinStart(float timeBinStart, int frameId)
{
	// TODO: Add somewhere a check to ensure that they are in chronological
	// order.
	if (timeBinStart >= 0.0)
	{
		m_frameTimeBinStart[frameId] = 1000 * timeBinStart;  // Times are in ms
	}
	else
	{
		std::cerr << "A frame time bins starting time of " << timeBinStart
		          << " was attributed to frame " << frameId
		          << " which is invalid. It should be positive." << std::endl;
		exit(EXIT_FAILURE);
	}
}


std::vector<std::string>
    ImageWarperTemplate::splitStringIntoVector(std::string stringTpParse,
                                               std::string delimiter)
{
	std::vector<std::string> result;
	while (stringTpParse.size())
	{
		size_t index = stringTpParse.find(delimiter);
		if (index != std::string::npos)
		{
			if (stringTpParse.substr(0, index) != "")
			{
				result.push_back(stringTpParse.substr(0, index));
			}
			stringTpParse = stringTpParse.substr(index + delimiter.size());
		}
		else
		{
			if (stringTpParse != "")
			{
				result.push_back(stringTpParse);
			}
			stringTpParse = "";
		}
	}
	return result;
}


std::vector<double> ImageWarperTemplate::splitStringIntoVectorOfDouble(
    std::string stringToParse, std::string delimiter)
{
	std::vector<double> result;
	while (stringToParse.size())
	{
		size_t index = stringToParse.find(delimiter);
		if (index != std::string::npos)
		{
			if (stringToParse.substr(0, index) != "")
			{
				result.push_back(stof(stringToParse.substr(0, index)));
			}
			stringToParse = stringToParse.substr(index + delimiter.size());
		}
		else
		{
			if (stringToParse != "")
			{
				result.push_back(stod(stringToParse));
			}
			stringToParse = "";
		}
	}
	return result;
}


std::vector<float>
    ImageWarperTemplate::createRotationMatrix(float _angle,
                                              std::vector<float> rotationAxis)
{
	std::vector<float> rotMatrix;
	rotMatrix.resize(9);

	// Convert to radians.
	float theta = M_PI * (_angle / 180.0);
	// TODO: Check the validity of rotation axis?
	float uX = rotationAxis[0];
	float uY = rotationAxis[1];
	float uZ = rotationAxis[2];

	rotMatrix[0] = cos(theta) + pow(uX, 2) * (1.0 - cos(theta));
	rotMatrix[1] = uX * uY * (1.0 - cos(theta)) - uZ * sin(theta);
	rotMatrix[2] = uX * uZ * (1.0 - cos(theta)) + uY * sin(theta);

	rotMatrix[3] = uY * uX * (1.0 - cos(theta)) + uZ * sin(theta);
	rotMatrix[4] = cos(theta) + pow(uY, 2) * (1.0 - cos(theta));
	rotMatrix[5] = uY * uZ * (1.0 - cos(theta)) - uX * sin(theta);

	rotMatrix[6] = uZ * uX * (1.0 - cos(theta)) - uY * sin(theta);
	rotMatrix[7] = uZ * uY * (1.0 - cos(theta)) + uX * sin(theta);
	rotMatrix[8] = cos(theta) + pow(uZ, 2) * (1.0 - cos(theta));

	return rotMatrix;
}


std::vector<double> ImageWarperTemplate::convertQuaternionToRotationMatrix(
    const std::vector<double> quaternion)
{
	std::vector<double> result;
	result.resize(9);

	// Ensure it is normalized.
	std::vector<double> nQuat;
	nQuat = quaternion;
	double norm = 0.0;
	for (int i = 0; i < 4; i++)
	{
		norm += nQuat[i] * nQuat[i];
	}
	for (int i = 0; i < 4; i++)
	{
		nQuat[i] /= sqrt(norm);
	}

	// Create the rotation matrix.
	// First line.
	result[0] = 1.0 - 2.0 * nQuat[2] * nQuat[2] - 2.0 * nQuat[3] * nQuat[3];
	result[1] = 2.0 * nQuat[1] * nQuat[2] - 2.0 * nQuat[3] * nQuat[0];
	result[2] = 2.0 * nQuat[1] * nQuat[3] + 2.0 * nQuat[2] * nQuat[0];
	// Second line.
	result[3] = 2.0 * nQuat[1] * nQuat[2] + 2.0 * nQuat[3] * nQuat[0];
	result[4] = 1 - 2.0 * nQuat[1] * nQuat[1] - 2.0 * nQuat[3] * nQuat[3];
	result[5] = 2.0 * nQuat[2] * nQuat[3] - 2.0 * nQuat[1] * nQuat[0];
	// Third line.
	result[6] = 2.0 * nQuat[1] * nQuat[3] - 2.0 * nQuat[2] * nQuat[0];
	result[7] = 2.0 * nQuat[2] * nQuat[3] + 2.0 * nQuat[1] * nQuat[0];
	result[8] = 1.0 - 2.0 * nQuat[1] * nQuat[1] - 2.0 * nQuat[2] * nQuat[2];

	return result;
}


bool ImageWarperTemplate::isFrameUsed(int frameId)
{
	return m_motionFrameUsed[frameId];
}


std::vector<int> ImageWarperTemplate::getImNbVoxel()
{
	return m_imNbVoxel;
}


unsigned int ImageWarperTemplate::getReferenceFrameId()
{
	return m_referenceFrameId;
}


int ImageWarperTemplate::getNumberOfFrame()
{
	return m_numberOfFrame;
}


float ImageWarperTemplate::getFrameStartTime(int frameId)
{
	if (frameId != m_numberOfFrame)
	{
		return m_frameTimeBinStart[frameId];
	}
	else
	{
		return std::numeric_limits<float>::infinity();
	}
}
