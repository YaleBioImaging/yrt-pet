/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"
#include "yrt-pet/utils/Assert.hpp"
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
	return yrt::Scanner("metal_projector_compare", 25.0f, 5.0f, 3.0f,
	                    10.0f, 300.0f, 256, 5, 1, 4, 2, 8);
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
	    getBinIter(int /*numSubsets*/, int /*idxSubset*/) const override
	{
		return std::make_unique<yrt::BinIteratorRange>(
		    static_cast<yrt::bin_t>(m_lines.size()));
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

	const std::vector<float>& values() const
	{
		return m_values;
	}

private:
	std::vector<yrt::Line3D> m_lines;
	std::vector<float> m_values;
	std::vector<yrt::frame_t> m_frames;
};

CompareStats compareValues(const std::vector<float>& expected,
                           const std::vector<float>& actual,
                           std::vector<float>& diff, float absTolerance,
                           float relTolerance)
{
	CompareStats stats;
	diff.resize(expected.size(), 0.0f);
	if (expected.size() != actual.size())
	{
		stats.mismatchCount = std::max(expected.size(), actual.size());
		return stats;
	}

	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		diff[i] = actual[i] - expected[i];
		const float absDiff = std::fabs(diff[i]);
		const float scale = std::max(1.0f, std::fabs(expected[i]));
		stats.maxAbsDiff = std::max(stats.maxAbsDiff, absDiff);
		stats.maxRelDiff = std::max(stats.maxRelDiff, absDiff / scale);
		if (absDiff > absTolerance + relTolerance * scale)
		{
			++stats.mismatchCount;
		}
	}
	return stats;
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

void writeProjectionValuesIfRequested(const std::vector<float>& values,
                                      const std::string& filename)
{
	if (filename.empty())
	{
		return;
	}

	std::ofstream file(filename);
	if (!file)
	{
		throw std::filesystem::filesystem_error(
		    "The output file \"" + filename + "\" could not be opened",
		    std::make_error_code(std::errc::io_error));
	}
	for (float value : values)
	{
		file << value << '\n';
	}
}

void writeImageIfRequested(const yrt::Image& image,
                           const std::string& filename)
{
	if (!filename.empty())
	{
		image.writeToFile(filename);
	}
}

bool statsPass(const CompareStats& stats)
{
	return stats.mismatchCount == 0;
}

}  // namespace

int main(int argc, char** argv)
{
	try
	{
		std::string inputFilename;
		std::string lorFilename;
		std::string scannerFilename;
		std::string cpuOutFilename;
		std::string metalOutFilename;
		std::string diffOutFilename;
		float absTolerance = 1.0e-4f;
		float relTolerance = 1.0e-4f;
		bool applyAdjoint = false;

		cxxopts::Options options(
		    argv[0],
		    "Compare CPU OperatorProjector with explicit opt-in Metal Siddon");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("i,input", "Input image file. In adjoint mode, this image provides the output image shape and initial values.",
		 cxxopts::value<std::string>(inputFilename))
		("l,lors", "LOR CSV: x1,y1,z1,x2,y2,z2[,projection_value][,dynamic_frame]",
		 cxxopts::value<std::string>(lorFilename))
		("scanner", "Optional scanner JSON file. A synthetic scanner is used when omitted.",
		 cxxopts::value<std::string>(scannerFilename))
		("cpu-out", "Optional CPU output file. Forward mode writes projection values CSV; adjoint mode writes an image.",
		 cxxopts::value<std::string>(cpuOutFilename))
		("metal-out", "Optional Metal opt-in output file. Forward mode writes projection values CSV; adjoint mode writes an image.",
		 cxxopts::value<std::string>(metalOutFilename))
		("diff-out", "Optional signed difference output file (Metal opt-in - CPU). Forward mode writes CSV; adjoint mode writes an image.",
		 cxxopts::value<std::string>(diffOutFilename))
		("atol", "Absolute comparison tolerance",
		 cxxopts::value<float>(absTolerance)->default_value("1e-4"))
		("rtol", "Relative comparison tolerance",
		 cxxopts::value<float>(relTolerance)->default_value("1e-4"))
		("adjoint", "Compare adjoint projector (AH) instead of forward projector (A)",
		 cxxopts::value<bool>(applyAdjoint)->default_value("false"))
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

		std::vector<std::string> requiredParams = {"input", "lors"};
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

		const yrt::backend::metal::Context context;
		if (!context.isValid())
		{
			std::cerr << "Metal projector compare: FAIL ("
			          << context.errorMessage() << ")\n";
			return 2;
		}

		yrt::Scanner scanner =
		    scannerFilename.empty() ? makeDefaultScanner()
		                            : yrt::Scanner(scannerFilename);
		yrt::ImageOwned inputImage(inputFilename);
		const LorCsvData lorData = readLorCsv(lorFilename);
		validateDynamicFrames(lorData, inputImage.getParams());

		yrt::ProjectorParams projectorParams(scanner);
		projectorParams.projectorType = yrt::ProjectorType::SIDDON;
		projectorParams.numRays = 1;

		CsvProjectionData supportData(scanner, lorData);
		auto supportBinIterator = supportData.getBinIter(1, 0);
		yrt::OperatorProjector supportProjector(projectorParams,
		                                        supportBinIterator.get());
		const yrt::backend::metal::OperatorProjectorMetalBridge bridge(context);
		const auto support = bridge.canRunSiddon(supportProjector);
		if (!support.supported)
		{
			std::cerr << "Metal projector compare: FAIL (unsupported: "
			          << support.reason << ")\n";
			return 1;
		}

		if (applyAdjoint)
		{
			yrt::ImageOwned cpuOutput(inputImage.getParams());
			yrt::ImageOwned directMetalOutput(inputImage.getParams());
			yrt::ImageOwned optInMetalOutput(inputImage.getParams());
			yrt::ImageOwned directDiff(inputImage.getParams());
			yrt::ImageOwned optInDiff(inputImage.getParams());
			cpuOutput.allocate();
			directMetalOutput.allocate();
			optInMetalOutput.allocate();
			directDiff.allocate();
			optInDiff.allocate();
			cpuOutput.copyFromImage(&inputImage);
			directMetalOutput.copyFromImage(&inputImage);
			optInMetalOutput.copyFromImage(&inputImage);

			CsvProjectionData cpuData(scanner, lorData);
			auto cpuBinIterator = cpuData.getBinIter(1, 0);
			yrt::OperatorProjector cpuProjector(projectorParams,
			                                    cpuBinIterator.get());
			cpuProjector.applyAH(&cpuData, &cpuOutput);

			CsvProjectionData directData(scanner, lorData);
			auto directBinIterator = directData.getBinIter(1, 0);
			yrt::OperatorProjector directProjector(projectorParams,
			                                       directBinIterator.get());
			if (!bridge.applyAH(directProjector, directData, directMetalOutput,
			        *directBinIterator, *directProjector.getBinLoader()))
			{
				std::cerr << "Metal projector compare: FAIL "
				          << "(direct Metal bridge adjoint failed)\n";
				return 1;
			}

			CsvProjectionData optInData(scanner, lorData);
			auto optInBinIterator = optInData.getBinIter(1, 0);
			yrt::OperatorProjector optInProjector(projectorParams,
			                                      optInBinIterator.get());
			optInProjector.setExperimentalMetalProjectorEnabled(true);
			optInProjector.applyAH(&optInData, &optInMetalOutput);

			const CompareStats directStats =
			    compareImages(cpuOutput, directMetalOutput, directDiff,
			                  absTolerance, relTolerance);
			const CompareStats optInStats =
			    compareImages(cpuOutput, optInMetalOutput, optInDiff,
			                  absTolerance, relTolerance);
			writeImageIfRequested(cpuOutput, cpuOutFilename);
			writeImageIfRequested(optInMetalOutput, metalOutFilename);
			writeImageIfRequested(optInDiff, diffOutFilename);

			std::cout << "Metal projector compare: direct_max_abs_diff="
			          << directStats.maxAbsDiff << " direct_max_rel_diff="
			          << directStats.maxRelDiff
			          << " direct_mismatches=" << directStats.mismatchCount
			          << " opt_in_max_abs_diff=" << optInStats.maxAbsDiff
			          << " opt_in_max_rel_diff=" << optInStats.maxRelDiff
			          << " opt_in_mismatches=" << optInStats.mismatchCount
			          << '\n';
			if (!statsPass(directStats) || !statsPass(optInStats))
			{
				std::cerr << "Metal projector compare: FAIL\n";
				return 1;
			}
		}
		else
		{
			CsvProjectionData cpuData(scanner, lorData);
			auto cpuBinIterator = cpuData.getBinIter(1, 0);
			yrt::OperatorProjector cpuProjector(projectorParams,
			                                    cpuBinIterator.get());
			cpuProjector.applyA(&inputImage, &cpuData);

			CsvProjectionData directData(scanner, lorData);
			auto directBinIterator = directData.getBinIter(1, 0);
			yrt::OperatorProjector directProjector(projectorParams,
			                                       directBinIterator.get());
			if (!bridge.applyA(directProjector, inputImage, directData,
			        *directBinIterator, *directProjector.getBinLoader()))
			{
				std::cerr << "Metal projector compare: FAIL "
				          << "(direct Metal bridge forward failed)\n";
				return 1;
			}

			CsvProjectionData optInData(scanner, lorData);
			auto optInBinIterator = optInData.getBinIter(1, 0);
			yrt::OperatorProjector optInProjector(projectorParams,
			                                      optInBinIterator.get());
			optInProjector.setExperimentalMetalProjectorEnabled(true);
			optInProjector.applyA(&inputImage, &optInData);

			std::vector<float> directDiff;
			std::vector<float> optInDiff;
			const CompareStats directStats =
			    compareValues(cpuData.values(), directData.values(), directDiff,
			                  absTolerance, relTolerance);
			const CompareStats optInStats =
			    compareValues(cpuData.values(), optInData.values(), optInDiff,
			                  absTolerance, relTolerance);
			writeProjectionValuesIfRequested(cpuData.values(), cpuOutFilename);
			writeProjectionValuesIfRequested(optInData.values(),
			                                 metalOutFilename);
			writeProjectionValuesIfRequested(optInDiff, diffOutFilename);

			std::cout << "Metal projector compare: direct_max_abs_diff="
			          << directStats.maxAbsDiff << " direct_max_rel_diff="
			          << directStats.maxRelDiff
			          << " direct_mismatches=" << directStats.mismatchCount
			          << " opt_in_max_abs_diff=" << optInStats.maxAbsDiff
			          << " opt_in_max_rel_diff=" << optInStats.maxRelDiff
			          << " opt_in_mismatches=" << optInStats.mismatchCount
			          << '\n';
			if (!statsPass(directStats) || !statsPass(optInStats))
			{
				std::cerr << "Metal projector compare: FAIL\n";
				return 1;
			}
		}

		std::cout << "Metal projector compare: PASS\n";
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
