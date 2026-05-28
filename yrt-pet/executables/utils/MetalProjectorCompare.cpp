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
#include "yrt-pet/operators/ProjectorSiddon.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Version.hpp"

#include <cxxopts.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

struct DiffRecord
{
	std::size_t index = 0;
	float expected = 0.0f;
	float actual = 0.0f;
	float diff = 0.0f;
	float absDiff = 0.0f;
	float relDiff = 0.0f;
};

struct RayDiffRecord
{
	std::size_t index = 0;
	CompareStats stats;
};

struct TraceEntry
{
	std::size_t offset = 0;
	yrt::frame_t frame = 0;
	float weight = 0.0f;
	float value = 0.0f;
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

	for (const yrt::frame_t frame : lors.frames)
	{
		if (frame >= params.nt)
		{
			throw std::runtime_error(
			    "A LOR CSV dynamic frame is outside the image nt range");
		}
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

class TraceProjectorUpdater final : public yrt::ProjectorUpdater
{
public:
	float forwardUpdate(float weight, float* curImgPtr, yrt::size_t offset,
	                    yrt::frame_t dynamicFrame,
	                    yrt::size_t numVoxelsPerFrame) const override
	{
		return weight *
		       curImgPtr[static_cast<std::size_t>(dynamicFrame) *
		                     numVoxelsPerFrame +
		                 offset];
	}

	void backUpdate(float value, float weight, float* curImgPtr,
	                yrt::size_t offset, yrt::frame_t dynamicFrame,
	                yrt::size_t numVoxelsPerFrame,
	                int /*tid*/) override
	{
		const std::size_t flatOffset =
		    static_cast<std::size_t>(dynamicFrame) * numVoxelsPerFrame +
		    offset;
		const float update = value * weight;
		curImgPtr[flatOffset] += update;
		entries.push_back({offset, dynamicFrame, weight, update});
	}

	std::vector<TraceEntry> entries;
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

void considerTopDiff(std::vector<DiffRecord>& records,
                     const DiffRecord& candidate, std::size_t topK)
{
	if (topK == 0 || candidate.absDiff == 0.0f)
	{
		return;
	}
	if (records.size() < topK)
	{
		records.push_back(candidate);
		return;
	}

	auto minIt = std::min_element(records.begin(), records.end(),
	    [](const DiffRecord& lhs, const DiffRecord& rhs)
	    { return lhs.absDiff < rhs.absDiff; });
	if (minIt != records.end() && candidate.absDiff > minIt->absDiff)
	{
		*minIt = candidate;
	}
}

void sortTopDiffs(std::vector<DiffRecord>& records)
{
	std::sort(records.begin(), records.end(),
	    [](const DiffRecord& lhs, const DiffRecord& rhs)
	    { return lhs.absDiff > rhs.absDiff; });
}

std::vector<DiffRecord> collectTopValueDiffs(
    const std::vector<float>& expected, const std::vector<float>& actual,
    std::size_t topK)
{
	std::vector<DiffRecord> records;
	if (expected.size() != actual.size())
	{
		return records;
	}

	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		const float diff = actual[i] - expected[i];
		const float absDiff = std::fabs(diff);
		const float scale = std::max(1.0f, std::fabs(expected[i]));
		considerTopDiff(records,
		    {i, expected[i], actual[i], diff, absDiff, absDiff / scale},
		    topK);
	}
	sortTopDiffs(records);
	return records;
}

std::vector<DiffRecord> collectTopImageDiffs(const yrt::Image& expected,
                                             const yrt::Image& actual,
                                             std::size_t topK)
{
	std::vector<DiffRecord> records;
	const std::size_t count = imageVoxelCount(expected.getParams());
	const float* expectedPtr = expected.getRawPointer();
	const float* actualPtr = actual.getRawPointer();
	for (std::size_t i = 0; i < count; ++i)
	{
		const float diff = actualPtr[i] - expectedPtr[i];
		const float absDiff = std::fabs(diff);
		const float scale = std::max(1.0f, std::fabs(expectedPtr[i]));
		considerTopDiff(records,
		    {i, expectedPtr[i], actualPtr[i], diff, absDiff, absDiff / scale},
		    topK);
	}
	sortTopDiffs(records);
	return records;
}

std::string imageIndexLabel(std::size_t flatIndex,
                            const yrt::ImageParams& params)
{
	const std::size_t nx = static_cast<std::size_t>(params.nx);
	const std::size_t ny = static_cast<std::size_t>(params.ny);
	const std::size_t nz = static_cast<std::size_t>(params.nz);
	const std::size_t spatialCount = nx * ny * nz;
	const std::size_t frame = spatialCount > 0 ? flatIndex / spatialCount : 0;
	std::size_t spatialIndex =
	    spatialCount > 0 ? flatIndex % spatialCount : flatIndex;
	const std::size_t z = nx * ny > 0 ? spatialIndex / (nx * ny) : 0;
	spatialIndex = nx * ny > 0 ? spatialIndex % (nx * ny) : spatialIndex;
	const std::size_t y = nx > 0 ? spatialIndex / nx : 0;
	const std::size_t x = nx > 0 ? spatialIndex % nx : spatialIndex;
	return std::to_string(frame) + "x" + std::to_string(z) + "x" +
	       std::to_string(y) + "x" + std::to_string(x);
}

void printTopProjectionDiffs(const std::vector<DiffRecord>& records,
                             const LorCsvData& lorData)
{
	if (records.empty())
	{
		return;
	}

	std::cout << "top_projection_diff_rank,bin,cpu,metal,diff,abs_diff,"
	             "rel_diff,input_value,dynamic_frame,x1,y1,z1,x2,y2,z2\n";
	for (std::size_t rank = 0; rank < records.size(); ++rank)
	{
		const DiffRecord& record = records[rank];
		const yrt::Line3D& line = lorData.lines[record.index];
		const yrt::frame_t frame =
		    lorData.frames.empty() ? 0 : lorData.frames[record.index];
		std::cout << (rank + 1) << ',' << record.index << ','
		          << record.expected << ',' << record.actual << ','
		          << record.diff << ',' << record.absDiff << ','
		          << record.relDiff << ',' << lorData.values[record.index]
		          << ',' << frame << ',' << line.point1.x << ','
		          << line.point1.y << ',' << line.point1.z << ','
		          << line.point2.x << ',' << line.point2.y << ','
		          << line.point2.z << '\n';
	}
}

void printTopImageDiffs(const std::vector<DiffRecord>& records,
                        const yrt::ImageParams& params)
{
	if (records.empty())
	{
		return;
	}

	std::cout << "top_image_diff_rank,index,flat_index,cpu,metal,diff,"
	             "abs_diff,rel_diff\n";
	for (std::size_t rank = 0; rank < records.size(); ++rank)
	{
		const DiffRecord& record = records[rank];
		std::cout << (rank + 1) << ','
		          << imageIndexLabel(record.index, params) << ','
		          << record.index << ',' << record.expected << ','
		          << record.actual << ',' << record.diff << ','
		          << record.absDiff << ',' << record.relDiff << '\n';
	}
}

void considerTopRayDiff(std::vector<RayDiffRecord>& records,
                        const RayDiffRecord& candidate, std::size_t topK)
{
	if (topK == 0 || candidate.stats.maxAbsDiff == 0.0f)
	{
		return;
	}
	if (records.size() < topK)
	{
		records.push_back(candidate);
		return;
	}

	auto minIt = std::min_element(records.begin(), records.end(),
	    [](const RayDiffRecord& lhs, const RayDiffRecord& rhs)
	    { return lhs.stats.maxAbsDiff < rhs.stats.maxAbsDiff; });
	if (minIt != records.end() &&
	    candidate.stats.maxAbsDiff > minIt->stats.maxAbsDiff)
	{
		*minIt = candidate;
	}
}

void sortTopRayDiffs(std::vector<RayDiffRecord>& records)
{
	std::sort(records.begin(), records.end(),
	    [](const RayDiffRecord& lhs, const RayDiffRecord& rhs)
	    { return lhs.stats.maxAbsDiff > rhs.stats.maxAbsDiff; });
}

LorCsvData makeSingleLorData(const LorCsvData& lorData, std::size_t index)
{
	LorCsvData single;
	single.lines.push_back(lorData.lines[index]);
	single.values.push_back(lorData.values[index]);
	if (!lorData.frames.empty())
	{
		single.frames.push_back(lorData.frames[index]);
	}
	return single;
}

CompareStats compareSingleAdjointRay(const yrt::Scanner& scanner,
                                     const yrt::ProjectorParams& params,
                                     const yrt::Image& inputImage,
                                     const LorCsvData& lorData,
                                     std::size_t index,
                                     const yrt::backend::metal::
                                         OperatorProjectorMetalBridge& bridge)
{
	const LorCsvData singleLorData = makeSingleLorData(lorData, index);

	yrt::ImageOwned cpuOutput(inputImage.getParams());
	yrt::ImageOwned metalOutput(inputImage.getParams());
	yrt::ImageOwned diffImage(inputImage.getParams());
	cpuOutput.allocate();
	metalOutput.allocate();
	diffImage.allocate();
	cpuOutput.copyFromImage(&inputImage);
	metalOutput.copyFromImage(&inputImage);

	CsvProjectionData cpuData(scanner, singleLorData);
	auto cpuBinIterator = cpuData.getBinIter(1, 0);
	yrt::OperatorProjector cpuProjector(params, cpuBinIterator.get());
	cpuProjector.applyAH(&cpuData, &cpuOutput);

	CsvProjectionData metalData(scanner, singleLorData);
	auto metalBinIterator = metalData.getBinIter(1, 0);
	yrt::OperatorProjector metalProjector(params, metalBinIterator.get());
	if (!bridge.applyAH(metalProjector, metalData, metalOutput,
	        *metalBinIterator, *metalProjector.getBinLoader()))
	{
		throw std::runtime_error("direct Metal bridge adjoint failed for LOR " +
		                         std::to_string(index));
	}

	return compareImages(cpuOutput, metalOutput, diffImage, 0.0f, 0.0f);
}

std::vector<RayDiffRecord> collectTopAdjointRayDiffs(
    const yrt::Scanner& scanner, const yrt::ProjectorParams& params,
    const yrt::Image& inputImage, const LorCsvData& lorData,
    const yrt::backend::metal::OperatorProjectorMetalBridge& bridge,
    std::size_t topK)
{
	std::vector<RayDiffRecord> records;
	for (std::size_t i = 0; i < lorData.lines.size(); ++i)
	{
		const CompareStats stats = compareSingleAdjointRay(
		    scanner, params, inputImage, lorData, i, bridge);
		considerTopRayDiff(records, {i, stats}, topK);
	}
	sortTopRayDiffs(records);
	return records;
}

void printTopAdjointRayDiffs(const std::vector<RayDiffRecord>& records,
                             const LorCsvData& lorData)
{
	if (records.empty())
	{
		return;
	}

	std::cout << "top_adjoint_ray_diff_rank,bin,max_abs_diff,max_rel_diff,"
	             "mismatches,input_value,dynamic_frame,x1,y1,z1,x2,y2,z2\n";
	for (std::size_t rank = 0; rank < records.size(); ++rank)
	{
		const RayDiffRecord& record = records[rank];
		const yrt::Line3D& line = lorData.lines[record.index];
		const yrt::frame_t frame =
		    lorData.frames.empty() ? 0 : lorData.frames[record.index];
		std::cout << (rank + 1) << ',' << record.index << ','
		          << record.stats.maxAbsDiff << ','
		          << record.stats.maxRelDiff << ','
		          << record.stats.mismatchCount << ','
		          << lorData.values[record.index] << ',' << frame << ','
		          << line.point1.x << ',' << line.point1.y << ','
		          << line.point1.z << ',' << line.point2.x << ','
		          << line.point2.y << ',' << line.point2.z << '\n';
	}
}

yrt::Line3D makeCenteredLineForImage(const yrt::Line3D& line,
                                     const yrt::ImageParams& params)
{
	yrt::Line3D centered = line;
	const yrt::Vector3D offsetVec{params.off_x, params.off_y, params.off_z};
	centered.point1 = centered.point1 - offsetVec;
	centered.point2 = centered.point2 - offsetVec;
	return centered;
}

std::unique_ptr<yrt::ImageOwned> makeZeroImage(const yrt::ImageParams& params)
{
	auto image = std::make_unique<yrt::ImageOwned>(params);
	image->allocate();
	std::fill(image->getRawPointer(),
	          image->getRawPointer() + imageVoxelCount(params), 0.0f);
	return image;
}

void traceSelectedAdjointLor(
    const yrt::Scanner& scanner, const yrt::ProjectorParams& params,
    const yrt::Image& inputImage, const LorCsvData& lorData,
    const yrt::backend::metal::OperatorProjectorMetalBridge& bridge,
    std::size_t topK)
{
	if (lorData.lines.size() != 1)
	{
		throw std::runtime_error("--trace-selected-lor requires --select-lor");
	}

	const yrt::ImageParams& imageParams = inputImage.getParams();
	const std::size_t spatialCount = static_cast<std::size_t>(imageParams.nx) *
	                                 static_cast<std::size_t>(imageParams.ny) *
	                                 static_cast<std::size_t>(imageParams.nz);
	const yrt::frame_t frame = lorData.frames.empty() ? 0 : lorData.frames[0];
	const float projectionValue = lorData.values[0];

	auto cpuTraceImage = makeZeroImage(imageParams);
	TraceProjectorUpdater updater;
	float cpuProjectionValue = projectionValue;
	yrt::Line3D centeredLine =
	    makeCenteredLineForImage(lorData.lines[0], imageParams);
	yrt::ProjectorSiddon::project_helper<false, true, false, true>(
	    cpuTraceImage.get(), centeredLine, cpuProjectionValue, &updater, frame, 0,
	    nullptr, 0.0f);

	auto metalTraceImage = makeZeroImage(imageParams);
	CsvProjectionData metalData(scanner, lorData);
	auto metalBinIterator = metalData.getBinIter(1, 0);
	yrt::OperatorProjector metalProjector(params, metalBinIterator.get());
	if (!bridge.applyAH(metalProjector, metalData, *metalTraceImage,
	        *metalBinIterator, *metalProjector.getBinLoader()))
	{
		throw std::runtime_error("direct Metal bridge adjoint trace failed");
	}

	yrt::ImageOwned diffImage(imageParams);
	diffImage.allocate();
	const CompareStats traceStats = compareImages(
	    *cpuTraceImage, *metalTraceImage, diffImage, 0.0f, 0.0f);
	std::vector<DiffRecord> topDiffs =
	    collectTopImageDiffs(*cpuTraceImage, *metalTraceImage, topK);

	std::cout << "trace_selected_lor_segments=" << updater.entries.size()
	          << " trace_max_abs_diff=" << traceStats.maxAbsDiff
	          << " trace_max_rel_diff=" << traceStats.maxRelDiff
	          << " trace_mismatches=" << traceStats.mismatchCount << '\n';
	printTopImageDiffs(topDiffs, imageParams);

	std::map<std::size_t, float> cpuWeights;
	for (const TraceEntry& entry : updater.entries)
	{
		const std::size_t flatOffset =
		    static_cast<std::size_t>(entry.frame) * spatialCount +
		    entry.offset;
		cpuWeights[flatOffset] += entry.value;
	}

	std::map<std::size_t, float> metalWeights;
	const float* metalPtr = metalTraceImage->getRawPointer();
	const std::size_t voxelCount = imageVoxelCount(imageParams);
	for (std::size_t i = 0; i < voxelCount; ++i)
	{
		if (metalPtr[i] != 0.0f)
		{
			metalWeights[i] = metalPtr[i];
		}
	}

	std::vector<DiffRecord> records;
	std::map<std::size_t, float> allOffsets = cpuWeights;
	for (const auto& [offset, value] : metalWeights)
	{
		allOffsets.try_emplace(offset, 0.0f);
		(void)value;
	}

	std::size_t sharedCount = 0;
	std::size_t cpuOnlyCount = 0;
	std::size_t metalOnlyCount = 0;
	double cpuSum = 0.0;
	double metalSum = 0.0;
	for (const auto& [offset, value] : cpuWeights)
	{
		(void)offset;
		cpuSum += static_cast<double>(value);
	}
	for (const auto& [offset, value] : metalWeights)
	{
		(void)offset;
		metalSum += static_cast<double>(value);
	}
	for (const auto& [offset, unused] : allOffsets)
	{
		(void)unused;
		const bool hasCpu = cpuWeights.count(offset) != 0;
		const bool hasMetal = metalWeights.count(offset) != 0;
		if (hasCpu && hasMetal)
		{
			++sharedCount;
		}
		else if (hasCpu)
		{
			++cpuOnlyCount;
		}
		else if (hasMetal)
		{
			++metalOnlyCount;
		}
	}
	std::cout << "trace_weight_sets,cpu_voxels,metal_voxels,shared_voxels,"
	             "cpu_only_voxels,metal_only_voxels,cpu_sum,metal_sum,sum_diff\n"
	          << cpuWeights.size() << ',' << metalWeights.size() << ','
	          << sharedCount << ',' << cpuOnlyCount << ',' << metalOnlyCount
	          << ',' << cpuSum << ',' << metalSum << ','
	          << (metalSum - cpuSum) << '\n';

	for (const auto& [offset, unused] : allOffsets)
	{
		(void)unused;
		const float cpuValue =
		    cpuWeights.count(offset) != 0 ? cpuWeights[offset] : 0.0f;
		const float metalValue =
		    metalWeights.count(offset) != 0 ? metalWeights[offset] : 0.0f;
		const float diff = metalValue - cpuValue;
		const float absDiff = std::fabs(diff);
		const float scale = std::max(1.0f, std::fabs(cpuValue));
		considerTopDiff(records,
		    {offset, cpuValue, metalValue, diff, absDiff, absDiff / scale},
		    topK);
	}
	sortTopDiffs(records);

	std::cout << "top_trace_weight_diff_rank,index,flat_index,cpu_weight,"
	             "metal_weight,diff,abs_diff,rel_diff\n";
	for (std::size_t rank = 0; rank < records.size(); ++rank)
	{
		const DiffRecord& record = records[rank];
		std::cout << (rank + 1) << ','
		          << imageIndexLabel(record.index, imageParams) << ','
		          << record.index << ',' << record.expected << ','
		          << record.actual << ',' << record.diff << ','
		          << record.absDiff << ',' << record.relDiff << '\n';
	}
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
		bool diagnoseAdjointRays = false;
		bool traceSelectedLor = false;
		int topK = 0;
		int selectLor = -1;

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
		("diagnose-adjoint-rays", "In adjoint mode, run one LOR at a time and print the top ray-level differences",
		 cxxopts::value<bool>(diagnoseAdjointRays)->default_value("false"))
		("trace-selected-lor", "In adjoint mode with --select-lor, print CPU-vs-Metal per-voxel path weights for that one LOR",
		 cxxopts::value<bool>(traceSelectedLor)->default_value("false"))
		("select-lor", "Use only this zero-based row from the LOR CSV for focused diagnostics",
		 cxxopts::value<int>(selectLor)->default_value("-1"))
		("top-k", "Print the N largest projection or image differences",
		 cxxopts::value<int>(topK)->default_value("0"))
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
		if (topK < 0)
		{
			std::cerr << "--top-k must be non-negative\n";
			return -1;
		}
		if (selectLor < -1)
		{
			std::cerr << "--select-lor must be -1 or a non-negative row index\n";
			return -1;
		}
		const std::size_t topKCount = static_cast<std::size_t>(topK);

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
		LorCsvData lorData = readLorCsv(lorFilename);
		if (selectLor >= 0)
		{
			const std::size_t selectedIndex = static_cast<std::size_t>(selectLor);
			if (selectedIndex >= lorData.lines.size())
			{
				std::cerr << "--select-lor is outside the LOR CSV row range\n";
				return -1;
			}
			const yrt::Line3D selectedLine = lorData.lines[selectedIndex];
			const float selectedValue = lorData.values[selectedIndex];
			const yrt::frame_t selectedFrame =
			    lorData.frames.empty() ? 0 : lorData.frames[selectedIndex];
			lorData = makeSingleLorData(lorData, selectedIndex);
			std::cout << "selected_lor_source_row=" << selectedIndex
			          << " input_value=" << selectedValue
			          << " dynamic_frame=" << selectedFrame << " x1="
			          << selectedLine.point1.x << " y1=" << selectedLine.point1.y
			          << " z1=" << selectedLine.point1.z << " x2="
			          << selectedLine.point2.x << " y2=" << selectedLine.point2.y
			          << " z2=" << selectedLine.point2.z << '\n';
		}
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
			const std::vector<DiffRecord> optInTopDiffs =
			    collectTopImageDiffs(cpuOutput, optInMetalOutput, topKCount);
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
			printTopImageDiffs(optInTopDiffs, inputImage.getParams());
			if (diagnoseAdjointRays)
			{
				const std::size_t rayTopK = topKCount == 0 ? 10 : topKCount;
				const std::vector<RayDiffRecord> topRayDiffs =
				    collectTopAdjointRayDiffs(scanner, projectorParams,
				        inputImage, lorData, bridge, rayTopK);
				printTopAdjointRayDiffs(topRayDiffs, lorData);
			}
			if (traceSelectedLor)
			{
				const std::size_t traceTopK = topKCount == 0 ? 20 : topKCount;
				traceSelectedAdjointLor(scanner, projectorParams, inputImage,
				                        lorData, bridge, traceTopK);
			}
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
			const std::vector<DiffRecord> optInTopDiffs =
			    collectTopValueDiffs(cpuData.values(), optInData.values(),
			                         topKCount);
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
			printTopProjectionDiffs(optInTopDiffs, lorData);
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
