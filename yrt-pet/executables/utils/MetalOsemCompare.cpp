/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/recon/OSEM_CPU.hpp"
#include "yrt-pet/utils/Tools.hpp"
#include "yrt-pet/utils/Version.hpp"

#include <cxxopts.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace
{

struct CompareStats
{
	float maxAbsDiff = 0.0f;
	float maxRelDiff = 0.0f;
	std::size_t mismatchCount = 0;
};

struct LorCsvData
{
	std::vector<yrt::Line3D> lines;
	std::vector<float> values;
	std::vector<yrt::frame_t> frames;
};

std::size_t imageVoxelCount(const yrt::ImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

yrt::Scanner makeDefaultScanner()
{
	return yrt::Scanner("metal_osem_compare", 25.0f, 5.0f, 3.0f, 10.0f,
	                    300.0f, 256, 5, 1, 4, 2, 8);
}

std::vector<double> parseCsvValues(std::string line)
{
	const std::size_t commentPos = line.find('#');
	if (commentPos != std::string::npos)
	{
		line.erase(commentPos);
	}
	std::replace(line.begin(), line.end(), ',', ' ');
	std::stringstream stream(line);

	std::vector<double> values;
	double value = 0.0;
	while (stream >> value)
	{
		values.push_back(value);
	}
	return values;
}

LorCsvData readLorCsv(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file)
	{
		throw std::filesystem::filesystem_error(
		    "The LOR CSV file \"" + filename + "\" could not be opened",
		    std::make_error_code(std::errc::no_such_file_or_directory));
	}

	LorCsvData data;
	bool hasFrames = false;
	bool hasSeenDataLine = false;
	std::string line;
	std::size_t lineNumber = 0;
	while (std::getline(file, line))
	{
		++lineNumber;
		const std::vector<double> row = parseCsvValues(line);
		if (row.empty())
		{
			continue;
		}
		if (row.size() != 6 && row.size() != 7 && row.size() != 8)
		{
			throw std::runtime_error(
			    "LOR CSV line " + std::to_string(lineNumber) +
			    " must have 6, 7, or 8 numeric columns");
		}
		if (!hasSeenDataLine)
		{
			hasFrames = row.size() == 8;
			hasSeenDataLine = true;
		}
		else if (hasFrames != (row.size() == 8))
		{
			throw std::runtime_error(
			    "LOR CSV must either provide a frame column on every row "
			    "or on no rows");
		}

		yrt::Line3D line3d;
		line3d.point1.x = static_cast<float>(row[0]);
		line3d.point1.y = static_cast<float>(row[1]);
		line3d.point1.z = static_cast<float>(row[2]);
		line3d.point2.x = static_cast<float>(row[3]);
		line3d.point2.y = static_cast<float>(row[4]);
		line3d.point2.z = static_cast<float>(row[5]);
		data.lines.push_back(line3d);
		data.values.push_back(row.size() >= 7 ? static_cast<float>(row[6])
		                                      : 1.0f);
		if (hasFrames)
		{
			data.frames.push_back(static_cast<yrt::frame_t>(std::lround(row[7])));
		}
	}

	if (data.lines.empty())
	{
		throw std::runtime_error("LOR CSV file is empty");
	}
	return data;
}

void validateDynamicFrames(const LorCsvData& lors,
                           const yrt::ImageParams& params)
{
	if (lors.frames.empty())
	{
		if (params.nt != 1)
		{
			throw std::runtime_error(
			    "4D images require an eighth dynamic-frame column in the "
			    "LOR CSV");
		}
		return;
	}

	yrt::frame_t maxFrame = 0;
	for (const yrt::frame_t frame : lors.frames)
	{
		if (frame >= 0)
		{
			maxFrame = std::max(maxFrame, frame);
		}
	}
	if (static_cast<int>(maxFrame + 1) != params.nt)
	{
		throw std::runtime_error(
		    "The LOR CSV dynamic-frame range does not match the image nt");
	}
}

yrt::size_t getNumDynamicFrames(const LorCsvData& lors)
{
	if (lors.frames.empty())
	{
		return 1;
	}

	yrt::frame_t maxFrame = 0;
	for (const yrt::frame_t frame : lors.frames)
	{
		if (frame >= 0)
		{
			maxFrame = std::max(maxFrame, frame);
		}
	}
	return static_cast<yrt::size_t>(maxFrame) + 1;
}

class CsvProjectionData final : public yrt::ProjectionData
{
public:
	CsvProjectionData(const yrt::Scanner& scanner, const LorCsvData& lorData)
	    : ProjectionData(scanner),
	      m_lines{lorData.lines},
	      m_values{lorData.values},
	      m_frames{lorData.frames}
	{
	}

	yrt::size_t count() const override
	{
		return m_lines.size();
	}

	float getProjectionValue(yrt::bin_t id) const override
	{
		return m_values[static_cast<std::size_t>(id)];
	}

	void setProjectionValue(yrt::bin_t id, float val) override
	{
		m_values[static_cast<std::size_t>(id)] = val;
	}

	yrt::det_id_t getDetector1(yrt::bin_t /*id*/) const override
	{
		return 0;
	}

	yrt::det_id_t getDetector2(yrt::bin_t /*id*/) const override
	{
		return 1;
	}

	std::unique_ptr<yrt::BinIterator>
	    getBinIter(int numSubsets, int idxSubset) const override
	{
		if (numSubsets == 1)
		{
			return std::make_unique<yrt::BinIteratorRange>(
			    static_cast<yrt::bin_t>(m_lines.size()));
		}
		return std::make_unique<yrt::BinIteratorRange>(
		    static_cast<yrt::bin_t>(idxSubset),
		    static_cast<yrt::bin_t>(m_lines.size()),
		    static_cast<yrt::bin_t>(numSubsets));
	}

	bool hasDynamicFraming() const override
	{
		return !m_frames.empty();
	}

	yrt::frame_t getDynamicFrame(yrt::bin_t id) const override
	{
		return hasDynamicFraming() ? m_frames[static_cast<std::size_t>(id)] : 0;
	}

	yrt::size_t getNumDynamicFrames() const override
	{
		if (!hasDynamicFraming())
		{
			return 1;
		}

		yrt::frame_t maxFrame = 0;
		for (const yrt::frame_t frame : m_frames)
		{
			if (frame >= 0)
			{
				maxFrame = std::max(maxFrame, frame);
			}
		}
		return static_cast<yrt::size_t>(maxFrame) + 1;
	}

	bool hasArbitraryLORs() const override
	{
		return true;
	}

	yrt::Line3D getArbitraryLOR(yrt::bin_t id) const override
	{
		return m_lines[static_cast<std::size_t>(id)];
	}

	std::set<yrt::ProjectionPropertyType>
	    getProjectionPropertyTypes() const override
	{
		auto projectionProperties = ProjectionData::getProjectionPropertyTypes();
		if (hasDynamicFraming())
		{
			projectionProperties.insert(
			    yrt::ProjectionPropertyType::DYNAMIC_FRAME);
		}
		return projectionProperties;
	}

private:
	std::vector<yrt::Line3D> m_lines;
	std::vector<float> m_values;
	std::vector<yrt::frame_t> m_frames;
};

void copyImageValues(const yrt::Image& source, yrt::Image& dest)
{
	const yrt::ImageParams& params = source.getParams();
	const std::size_t count = imageVoxelCount(params);
	const float* sourcePtr = source.getRawPointer();
	float* destPtr = dest.getRawPointer();
	std::copy(sourcePtr, sourcePtr + count, destPtr);
}

void fillImage(yrt::Image& image, float value)
{
	const std::size_t count = imageVoxelCount(image.getParams());
	float* ptr = image.getRawPointer();
	std::fill(ptr, ptr + count, value);
}

CompareStats compareImages(const yrt::Image& expected, const yrt::Image& actual,
                           yrt::Image& diffImage, float absTolerance,
                           float relTolerance)
{
	CompareStats stats;
	const std::size_t count = imageVoxelCount(expected.getParams());
	const float* expectedPtr = expected.getRawPointer();
	const float* actualPtr = actual.getRawPointer();
	float* diffPtr = diffImage.getRawPointer();

	for (std::size_t i = 0; i < count; ++i)
	{
		const float diff = actualPtr[i] - expectedPtr[i];
		diffPtr[i] = diff;
		const float absDiff = std::fabs(diff);
		const float scale = std::max(1.0f, std::fabs(expectedPtr[i]));
		stats.maxAbsDiff = std::max(stats.maxAbsDiff, absDiff);
		stats.maxRelDiff = std::max(stats.maxRelDiff, absDiff / scale);
		if (absDiff > absTolerance + relTolerance * scale)
		{
			++stats.mismatchCount;
		}
	}
	return stats;
}

std::unique_ptr<yrt::Image> runOsem(const yrt::Scanner& scanner,
                                    yrt::ProjectionData& dataInput,
                                    yrt::Image& initialEstimate,
                                    yrt::Image& sensitivityImage,
                                    const std::string& imagePsfFilename,
                                    int numIterations, int numSubsets,
                                    bool useExperimentalMetalProjector,
                                    bool& didMetalRun)
{
	yrt::OSEM_CPU osem(scanner);
	osem.num_MLEM_iterations = numIterations;
	osem.num_OSEM_subsets = numSubsets;
	osem.setProjector(yrt::ProjectorType::SIDDON);
	osem.setNumRays(1);
	osem.setDataInput(&dataInput);
	osem.setImageParams(initialEstimate.getParams());
	osem.setInitialEstimate(&initialEstimate);
	osem.setSensitivityImage(&sensitivityImage);
	if (!imagePsfFilename.empty())
	{
		osem.addImagePSF(imagePsfFilename, yrt::ImagePSFMode::UNIFORM);
	}
	osem.setExperimentalMetalProjectorEnabled(useExperimentalMetalProjector);
	if (useExperimentalMetalProjector && !imagePsfFilename.empty())
	{
		osem.setExperimentalMetalProjectorImagePsfEnabled(true);
	}
	auto output = osem.reconstruct("");
	didMetalRun = osem.didLastExperimentalMetalProjectorRun();
	return output;
}

void writeImageIfRequested(const yrt::Image& image,
                           const std::string& filename)
{
	if (!filename.empty())
	{
		image.writeToFile(filename);
	}
}

}  // namespace

int main(int argc, char** argv)
{
	try
	{
		std::string initialFilename;
		std::string sensitivityFilename;
		std::string lorFilename;
		std::string scannerFilename;
		std::string imagePsfFilename;
		std::string cpuOutFilename;
		std::string metalOutFilename;
		std::string diffOutFilename;
		float initialValue = 0.125f;
		float absTolerance = 1.0e-4f;
		float relTolerance = 1.0e-4f;
		int numIterations = 1;
		int numSubsets = 1;

		cxxopts::Options options(
		    argv[0],
		    "Compare CPU OSEM_CPU with explicit opt-in Metal Siddon projector");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("l,lors", "LOR CSV: x1,y1,z1,x2,y2,z2[,measurement][,dynamic_frame]",
		 cxxopts::value<std::string>(lorFilename))
		("s,sensitivity", "Sensitivity image file",
		 cxxopts::value<std::string>(sensitivityFilename))
		("i,initial", "Optional initial estimate image. If omitted, an image with sensitivity geometry is filled with --initial-value.",
		 cxxopts::value<std::string>(initialFilename))
		("initial-value", "Initial estimate value when --initial is omitted",
		 cxxopts::value<float>(initialValue)->default_value("0.125"))
		("scanner", "Optional scanner JSON file. A synthetic scanner is used when omitted.",
		 cxxopts::value<std::string>(scannerFilename))
		("p,psf", "Optional uniform image-space PSF CSV file",
		 cxxopts::value<std::string>(imagePsfFilename))
		("iterations", "Number of MLEM iterations",
		 cxxopts::value<int>(numIterations)->default_value("1"))
		("subsets", "Number of OSEM subsets",
		 cxxopts::value<int>(numSubsets)->default_value("1"))
		("cpu-out", "Optional CPU reconstruction output image",
		 cxxopts::value<std::string>(cpuOutFilename))
		("metal-out", "Optional Metal opt-in reconstruction output image",
		 cxxopts::value<std::string>(metalOutFilename))
		("diff-out", "Optional signed difference output image (Metal - CPU)",
		 cxxopts::value<std::string>(diffOutFilename))
		("atol", "Absolute comparison tolerance",
		 cxxopts::value<float>(absTolerance)->default_value("1e-4"))
		("rtol", "Relative comparison tolerance",
		 cxxopts::value<float>(relTolerance)->default_value("1e-4"))
		("version", "Print version information")
		("h,help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
		if (result.count("version"))
		{
			yrt::version::printVersion();
			return 0;
		}
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> requiredParams = {"lors", "sensitivity"};
		bool missingArgs = false;
		for (const auto& param : requiredParams)
		{
			if (result.count(param) == 0)
			{
				std::cerr << "Argument '" << param << "' missing\n";
				missingArgs = true;
			}
		}
		if (missingArgs)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}
		if (numIterations <= 0 || numSubsets <= 0)
		{
			throw std::runtime_error(
			    "--iterations and --subsets must both be positive");
		}

		const yrt::backend::metal::Context context;
		if (!context.isValid())
		{
			std::cerr << "Metal OSEM compare: FAIL ("
			          << context.errorMessage() << ")\n";
			return 2;
		}

		yrt::Scanner scanner =
		    scannerFilename.empty() ? makeDefaultScanner()
		                            : yrt::Scanner(scannerFilename);
		yrt::ImageOwned sensitivityImage(sensitivityFilename);
		const LorCsvData lorData = readLorCsv(lorFilename);

		std::unique_ptr<yrt::ImageOwned> initialFromFile;
		yrt::ImageParams initialParams = sensitivityImage.getParams();
		if (!initialFilename.empty())
		{
			initialFromFile = std::make_unique<yrt::ImageOwned>(
			    initialFilename);
			initialParams = initialFromFile->getParams();
		}
		else
		{
			initialParams.nt = static_cast<int>(getNumDynamicFrames(lorData));
		}

		yrt::ImageOwned initialEstimate(initialParams);
		initialEstimate.allocate();
		if (initialFromFile != nullptr)
		{
			initialEstimate.copyFromImage(initialFromFile.get());
		}
		else
		{
			fillImage(initialEstimate, initialValue);
		}

		validateDynamicFrames(lorData, initialParams);

		yrt::ImageOwned metalInitial(initialEstimate.getParams());
		yrt::ImageOwned metalSensitivity(sensitivityImage.getParams());
		metalInitial.allocate();
		metalSensitivity.allocate();
		copyImageValues(initialEstimate, metalInitial);
		copyImageValues(sensitivityImage, metalSensitivity);

		CsvProjectionData cpuData(scanner, lorData);
		CsvProjectionData metalData(scanner, lorData);

		bool cpuMetalRan = false;
		bool optInMetalRan = false;
		auto cpuOutput = runOsem(scanner, cpuData, initialEstimate,
		                         sensitivityImage, imagePsfFilename,
		                         numIterations, numSubsets, false,
		                         cpuMetalRan);
		auto metalOutput = runOsem(scanner, metalData, metalInitial,
		                           metalSensitivity, imagePsfFilename,
		                           numIterations, numSubsets, true,
		                           optInMetalRan);
		if (cpuMetalRan)
		{
			std::cerr << "Metal OSEM compare: FAIL "
			          << "(CPU reference unexpectedly ran Metal)\n";
			return 1;
		}
		if (!optInMetalRan)
		{
			std::cerr << "Metal OSEM compare: FAIL "
			          << "(experimental Metal projector did not run)\n";
			return 1;
		}

		yrt::ImageOwned diffImage(cpuOutput->getParams());
		diffImage.allocate();
		const CompareStats stats =
		    compareImages(*cpuOutput, *metalOutput, diffImage, absTolerance,
		                  relTolerance);

		writeImageIfRequested(*cpuOutput, cpuOutFilename);
		writeImageIfRequested(*metalOutput, metalOutFilename);
		writeImageIfRequested(diffImage, diffOutFilename);

		std::cout << "Metal OSEM compare: metal_projector_ran=yes"
		          << " max_abs_diff=" << stats.maxAbsDiff
		          << " max_rel_diff=" << stats.maxRelDiff
		          << " mismatches=" << stats.mismatchCount << '\n';
		if (stats.mismatchCount != 0)
		{
			std::cerr << "Metal OSEM compare: FAIL\n";
			return 1;
		}

		std::cout << "Metal OSEM compare: PASS\n";
		return 0;
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e)
	{
		yrt::util::printExceptionMessage(e);
		return -1;
	}
}
