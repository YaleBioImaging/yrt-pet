/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/backends/metal/OperatorProjectorMetalBridge.hpp"

#include "yrt-pet/backends/metal/JosephProjectorOps.hpp"
#include "yrt-pet/backends/metal/ProjectionBatchMetal.hpp"
#include "yrt-pet/backends/metal/ProjectionVectorKernels.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorMetal.hpp"
#include "yrt-pet/backends/metal/SiddonProjectorOps.hpp"
#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/BinIterator.hpp"
#include "yrt-pet/datastruct/projection/BinLoader.hpp"
#include "yrt-pet/datastruct/projection/ListModeLUT.hpp"
#include "yrt-pet/datastruct/projection/ProjectionData.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/geometry/Line3D.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"
#include "yrt-pet/recon/Corrector_CPU.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace yrt::backend::metal
{
namespace
{

using Clock = std::chrono::steady_clock;

constexpr std::size_t kDefaultCacheMaxBytes =
    static_cast<std::size_t>(1024) * 1024 * 1024;
constexpr std::size_t kDefaultMaxBatchEvents = 1000000;
constexpr std::uint32_t kNoJosephAxis = 3;

struct BridgeEvent
{
	bin_t bin = 0;
	ProjectionLineEndpoints line;
	frame_t frame = 0;
	float projectionValue = 0.0f;
};

struct BridgeFrameBatch
{
	frame_t frame = 0;
	std::vector<ProjectionLineEndpoints> lines;
	std::vector<bin_t> bins;
};

enum class AdjointEventOrder
{
	None,
	MajorAxis,
	LineHash,
	TileRoundRobin
};

enum class JosephAxisSpecialization
{
	None,
	Forward,
	Both
};

struct CachedOsemCorrections
{
	bool valid = false;
	std::size_t valueCount = 0;
	bool hasSensitivity = false;
	bool hasAttenuation = false;
	bool hasScatterEstimates = false;
	bool hasRandomsEstimates = false;
	bool hasInVivoAttenuation = false;
	const ProjectionData* measurements = nullptr;
	const Corrector_CPU* corrector = nullptr;
	std::vector<float> measurementValues;
	std::vector<float> multiplicativeValues;
	std::vector<float> additiveValues;
	std::vector<float> inVivoAttenuationValues;
	Buffer measurementBuffer;
	Buffer multiplicativeBuffer;
	Buffer additiveBuffer;
	Buffer inVivoAttenuationBuffer;
};

struct CachedCorrectionBatch
{
	frame_t frame = 0;
	std::vector<bin_t> bins;
	CachedOsemCorrections osemCorrections;
};

struct CachedFrameBatch
{
	frame_t frame = 0;
	std::vector<bin_t> bins;
	ProjectionBatchMetal batch;
	CachedOsemCorrections osemCorrections;
	std::uint32_t josephAxis = kNoJosephAxis;
};

struct CachedIteratorSegment
{
	std::size_t offset = 0;
	std::size_t spanSize = 0;
	std::size_t eventCount = 0;
	std::size_t byteCount = 0;
	bool cached = false;
	bool correctionOnlyCached = false;
	bool correctionOnlyCacheBuilt = false;
	std::vector<bin_t> zeroProjectionBins;
	std::vector<CachedFrameBatch> frameBatches;
	std::vector<CachedCorrectionBatch> correctionBatches;
};

struct CachedBinIteratorEntry
{
	std::vector<CachedIteratorSegment> segments;
	std::size_t eventCount = 0;
	std::size_t cachedEventCount = 0;
	std::size_t byteCount = 0;
};

struct OsemRatioBuffers
{
	Buffer measurements;
	Buffer sensitivity;
	Buffer attenuation;
	Buffer randoms;
	Buffer scatter;
	Buffer inVivoAttenuation;
};

struct ForwardImageResources
{
	ForwardImageResources(const Image& pp_image, const Buffer& pp_imageBuffer)
	    : image(pp_image), imageBuffer(pp_imageBuffer)
	{
	}

	const Image& image;
	const Buffer& imageBuffer;
	Sampler sampler;
	std::map<frame_t, Texture3D> josephTexturesByFrame;
};

double getElapsedSeconds(Clock::time_point start, Clock::time_point end)
{
	return std::chrono::duration<double>(end - start).count();
}

bool usePrivateUpdateImageBuffer()
{
	const char* value = std::getenv("YRTPET_METAL_USE_PRIVATE_UPDATE_BUFFER");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool profileRatioNonzeroDiagnostic()
{
	const char* value = std::getenv("YRTPET_METAL_PROFILE_RATIO_NONZERO");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool useDirectHostRatioFrameBatches()
{
	const char* value = std::getenv("YRTPET_METAL_DIRECT_FRAME_BATCHES");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool useDirectHostRatioFrameBatches(
    const OperatorProjectorMetalOsemConfig& config)
{
	if (config.directFrameBatchesExplicit)
	{
		return config.directFrameBatches;
	}
	return useDirectHostRatioFrameBatches();
}

ProjectorKernelOptions makeProjectorKernelOptions(
    const OperatorProjectorMetalOsemConfig* config)
{
	ProjectorKernelOptions options;
	if (config == nullptr)
	{
		return options;
	}

	options.nativeFloatAtomicsExplicit =
	    config->nativeFloatAtomicsExplicit;
	options.nativeFloatAtomics = config->nativeFloatAtomics;
	options.josephAdjointAxisSwitchOnceExplicit =
	    config->josephAdjointAxisSwitchOnceExplicit;
	options.josephAdjointAxisSwitchOnce =
	    config->josephAdjointAxisSwitchOnce;
	options.launchOptions.threadsPerThreadgroupExplicit =
	    config->threadsPerThreadgroupExplicit;
	options.launchOptions.threadsPerThreadgroup =
	    config->threadsPerThreadgroup;
	return options;
}

JosephAxisSpecialization getJosephAxisSpecialization()
{
	const char* value = std::getenv("YRTPET_METAL_JOSEPH_AXIS_SPECIALIZED");
	if (value == nullptr || value[0] == '\0' || value[0] == '0')
	{
		return JosephAxisSpecialization::None;
	}

	std::string mode(value);
	std::transform(mode.begin(), mode.end(), mode.begin(),
	               [](unsigned char c)
	               {
		               return static_cast<char>(std::tolower(c));
	               });
	std::replace(mode.begin(), mode.end(), '_', '-');

	if (mode == "forward" || mode == "forward-only")
	{
		return JosephAxisSpecialization::Forward;
	}
	if (mode == "both" || mode == "1" || mode == "true" || mode == "yes" ||
	    mode == "on")
	{
		return JosephAxisSpecialization::Both;
	}
	return JosephAxisSpecialization::None;
}

AdjointEventOrder getAdjointEventOrder()
{
	const char* value = std::getenv("YRTPET_METAL_ADJOINT_EVENT_ORDER");
	if (value == nullptr || value[0] == '\0')
	{
		return AdjointEventOrder::None;
	}

	std::string order(value);
	std::transform(order.begin(), order.end(), order.begin(),
	               [](unsigned char c)
	               {
		               return static_cast<char>(std::tolower(c));
	               });
	std::replace(order.begin(), order.end(), '_', '-');

	if (order == "major-axis" || order == "axis")
	{
		return AdjointEventOrder::MajorAxis;
	}
	if (order == "line-hash" || order == "hash")
	{
		return AdjointEventOrder::LineHash;
	}
	if (order == "tile-round-robin" || order == "tile-rr" ||
	    order == "tile")
	{
		return AdjointEventOrder::TileRoundRobin;
	}
	return AdjointEventOrder::None;
}

std::uint32_t getAdjointTileSize()
{
	const char* value = std::getenv("YRTPET_METAL_ADJOINT_TILE_SIZE");
	if (value == nullptr || value[0] == '\0' || value[0] == '-')
	{
		return 8;
	}

	char* end = nullptr;
	const unsigned long parsed = std::strtoul(value, &end, 10);
	if (end == value || parsed == 0)
	{
		return 8;
	}
	return static_cast<std::uint32_t>(
	    std::min<unsigned long>(parsed, 256));
}

std::size_t getAdjointDiagnosticMaxBatches()
{
	const char* value =
	    std::getenv("YRTPET_METAL_ADJOINT_DIAGNOSTIC_MAX_BATCHES");
	if (value == nullptr || value[0] == '\0' || value[0] == '-')
	{
		return 0;
	}

	char* end = nullptr;
	const unsigned long long parsed = std::strtoull(value, &end, 10);
	if (end == value)
	{
		return 0;
	}
	return static_cast<std::size_t>(
	    std::min<unsigned long long>(
	        parsed, static_cast<unsigned long long>(
	                    std::numeric_limits<std::size_t>::max())));
}

std::size_t getAdjointDiagnosticStride()
{
	const char* value =
	    std::getenv("YRTPET_METAL_ADJOINT_DIAGNOSTIC_STRIDE");
	if (value == nullptr || value[0] == '\0' || value[0] == '-')
	{
		return 1;
	}

	char* end = nullptr;
	const unsigned long long parsed = std::strtoull(value, &end, 10);
	if (end == value || parsed <= 1)
	{
		return 1;
	}
	return static_cast<std::size_t>(
	    std::min<unsigned long long>(
	        parsed, static_cast<unsigned long long>(
	                    std::numeric_limits<std::size_t>::max())));
}

std::uint64_t mix64(std::uint64_t value)
{
	value ^= value >> 33;
	value *= UINT64_C(0xff51afd7ed558ccd);
	value ^= value >> 33;
	value *= UINT64_C(0xc4ceb9fe1a85ec53);
	value ^= value >> 33;
	return value;
}

void hashCombine(std::uint64_t& seed, std::uint64_t value)
{
	seed ^= mix64(value + UINT64_C(0x9e3779b97f4a7c15) + (seed << 6) +
	              (seed >> 2));
}

std::uint64_t hashCoordinate(float value)
{
	const auto quantized =
	    static_cast<std::int64_t>(std::llround(static_cast<double>(value) * 16.0));
	return static_cast<std::uint64_t>(quantized);
}

std::uint64_t hashLine(const ProjectionLineEndpoints& line)
{
	std::uint64_t seed = UINT64_C(0x6d2b79f5aa9fdfd7);
	hashCombine(seed, hashCoordinate(line.p1x));
	hashCombine(seed, hashCoordinate(line.p1y));
	hashCombine(seed, hashCoordinate(line.p1z));
	hashCombine(seed, hashCoordinate(line.p2x));
	hashCombine(seed, hashCoordinate(line.p2y));
	hashCombine(seed, hashCoordinate(line.p2z));
	return seed;
}

std::uint32_t majorAxis(const ProjectionLineEndpoints& line)
{
	const float dx = std::abs(line.p2x - line.p1x);
	const float dy = std::abs(line.p2y - line.p1y);
	const float dz = std::abs(line.p2z - line.p1z);
	if (dx >= dy && dx >= dz)
	{
		return 0;
	}
	if (dy >= dz)
	{
		return 1;
	}
	return 2;
}

std::uint32_t josephMajorAxis(const ProjectionLineEndpoints& line,
                              const SiddonForwardImageParams& params)
{
	const float sx = std::abs(line.p2x - line.p1x) * params.invVoxelX;
	const float sy = std::abs(line.p2y - line.p1y) * params.invVoxelY;
	const float sz = std::abs(line.p2z - line.p1z) * params.invVoxelZ;
	if (sx >= sy && sx >= sz)
	{
		return 0;
	}
	return sy >= sz ? 1 : 2;
}

bool updateLineBoxAlpha(float start, float delta, float halfLength,
                        float& alphaMin, float& alphaMax)
{
	constexpr float kEpsilon = 1.0e-6f;
	if (std::abs(delta) < kEpsilon)
	{
		return start >= -halfLength && start <= halfLength;
	}

	float a0 = (-halfLength - start) / delta;
	float a1 = (halfLength - start) / delta;
	if (a0 > a1)
	{
		std::swap(a0, a1);
	}
	alphaMin = std::max(alphaMin, a0);
	alphaMax = std::min(alphaMax, a1);
	return alphaMin <= alphaMax;
}

float representativeAlphaInImageBox(const ProjectionLineEndpoints& line,
                                    const SiddonForwardImageParams& params)
{
	float alphaMin = 0.0f;
	float alphaMax = 1.0f;
	const bool intersects =
	    updateLineBoxAlpha(line.p1x, line.p2x - line.p1x,
	        params.halfLengthX, alphaMin, alphaMax) &&
	    updateLineBoxAlpha(line.p1y, line.p2y - line.p1y,
	        params.halfLengthY, alphaMin, alphaMax) &&
	    updateLineBoxAlpha(line.p1z, line.p2z - line.p1z,
	        params.halfLengthZ, alphaMin, alphaMax);
	return intersects ? 0.5f * (alphaMin + alphaMax) : 0.5f;
}

std::uint32_t tileCoord(float coord, float halfLength, float invVoxel,
                        std::uint32_t voxelCount, std::uint32_t tileSize)
{
	if (voxelCount == 0)
	{
		return 0;
	}
	const float voxelCoord = (coord + halfLength) * invVoxel;
	const auto voxel = static_cast<std::uint32_t>(
	    std::min<float>(
	        static_cast<float>(voxelCount - 1),
	        std::max<float>(0.0f, std::floor(voxelCoord))));
	return voxel / std::max<std::uint32_t>(1, tileSize);
}

std::uint32_t representativeTile(const ProjectionLineEndpoints& line,
                                 const SiddonForwardImageParams& params,
                                 std::uint32_t tileSize)
{
	const float alpha = representativeAlphaInImageBox(line, params);
	const float x = line.p1x + alpha * (line.p2x - line.p1x);
	const float y = line.p1y + alpha * (line.p2y - line.p1y);
	const float z = line.p1z + alpha * (line.p2z - line.p1z);
	const std::uint32_t nxTiles =
	    (params.nx + tileSize - 1) / tileSize;
	const std::uint32_t nyTiles =
	    (params.ny + tileSize - 1) / tileSize;
	const std::uint32_t tx =
	    tileCoord(x, params.halfLengthX, params.invVoxelX, params.nx,
	        tileSize);
	const std::uint32_t ty =
	    tileCoord(y, params.halfLengthY, params.invVoxelY, params.ny,
	        tileSize);
	const std::uint32_t tz =
	    tileCoord(z, params.halfLengthZ, params.invVoxelZ, params.nz,
	        tileSize);
	return tx + nxTiles * (ty + nyTiles * tz);
}

std::vector<std::size_t> makeTileRoundRobinAdjointIndices(
    const std::vector<BridgeEvent>& events,
    const std::vector<std::size_t>& indices,
    const SiddonForwardImageParams& params)
{
	const std::uint32_t tileSize = getAdjointTileSize();
	const std::size_t nxTiles = (params.nx + tileSize - 1) / tileSize;
	const std::size_t nyTiles = (params.ny + tileSize - 1) / tileSize;
	const std::size_t nzTiles = (params.nz + tileSize - 1) / tileSize;
	const std::size_t tileCount = nxTiles * nyTiles * nzTiles;
	if (tileCount == 0)
	{
		return indices;
	}

	std::vector<std::size_t> counts(tileCount, 0);
	for (const std::size_t index : indices)
	{
		counts[representativeTile(events[index].line, params, tileSize)] += 1;
	}

	std::vector<std::size_t> offsets(tileCount, 0);
	std::vector<std::size_t> activeBuckets;
	activeBuckets.reserve(tileCount);
	std::size_t offset = 0;
	for (std::size_t tile = 0; tile < tileCount; ++tile)
	{
		offsets[tile] = offset;
		if (counts[tile] != 0)
		{
			activeBuckets.push_back(tile);
		}
		offset += counts[tile];
	}

	std::vector<std::size_t> writePositions(offsets);
	std::vector<std::size_t> grouped(indices.size());
	for (const std::size_t index : indices)
	{
		const std::size_t tile =
		    representativeTile(events[index].line, params, tileSize);
		grouped[writePositions[tile]++] = index;
	}

	std::vector<std::size_t> readPositions(offsets);
	std::vector<std::size_t> ordered;
	ordered.reserve(indices.size());
	while (!activeBuckets.empty())
	{
		std::size_t nextActive = 0;
		for (const std::size_t tile : activeBuckets)
		{
			ordered.push_back(grouped[readPositions[tile]++]);
			if (readPositions[tile] < offsets[tile] + counts[tile])
			{
				activeBuckets[nextActive++] = tile;
			}
		}
		activeBuckets.resize(nextActive);
	}
	return ordered;
}

std::vector<std::size_t> makeOrderedAdjointIndices(
    const std::vector<BridgeEvent>& events,
    const std::vector<std::size_t>& indices,
    const SiddonForwardImageParams* params = nullptr)
{
	const AdjointEventOrder order = getAdjointEventOrder();
	if (order == AdjointEventOrder::None || indices.size() < 2)
	{
		return indices;
	}

	if (order == AdjointEventOrder::MajorAxis)
	{
		std::size_t axisCounts[3] = {0, 0, 0};
		for (const std::size_t index : indices)
		{
			axisCounts[majorAxis(events[index].line)] += 1;
		}

		std::size_t axisPositions[3] = {
		    0,
		    axisCounts[0],
		    axisCounts[0] + axisCounts[1],
		};
		std::vector<std::size_t> ordered(indices.size());
		for (const std::size_t index : indices)
		{
			const std::uint32_t axis = majorAxis(events[index].line);
			ordered[axisPositions[axis]++] = index;
		}
		return ordered;
	}

	if (order == AdjointEventOrder::TileRoundRobin && params != nullptr)
	{
		return makeTileRoundRobinAdjointIndices(events, indices, *params);
	}
	if (order == AdjointEventOrder::TileRoundRobin)
	{
		return indices;
	}

	std::vector<std::size_t> ordered(indices);
	std::sort(ordered.begin(), ordered.end(),
	          [&](std::size_t lhs, std::size_t rhs)
	          {
		          const auto lhsHash = hashLine(events[lhs].line);
		          const auto rhsHash = hashLine(events[rhs].line);
		          return lhsHash == rhsHash ? lhs < rhs : lhsHash < rhsHash;
	          });
	return ordered;
}

bool canUseJosephAxisSpecializedForward(OperatorProjectorMetalKernel kernel)
{
	const JosephAxisSpecialization mode = getJosephAxisSpecialization();
	return kernel == OperatorProjectorMetalKernel::Joseph &&
	       (mode == JosephAxisSpecialization::Forward ||
	           mode == JosephAxisSpecialization::Both);
}

bool canUseJosephAxisSpecializedAdjoint(OperatorProjectorMetalKernel kernel)
{
	return kernel == OperatorProjectorMetalKernel::Joseph &&
	       getJosephAxisSpecialization() == JosephAxisSpecialization::Both;
}

bool shouldSplitJosephAxisBatches(OperatorProjectorMetalKernel kernel)
{
	return canUseJosephAxisSpecializedForward(kernel) ||
	       canUseJosephAxisSpecializedAdjoint(kernel);
}

bool forwardProjectSingleRay(OperatorProjectorMetalKernel kernel,
                             const Context& context, const Buffer& imageBuffer,
                             ProjectionBatchMetal& batch,
                             const SiddonForwardImageParams& params,
                             SiddonProjectorKernelProfile* profile,
                             std::uint32_t josephAxis = kNoJosephAxis,
                             const ProjectorKernelOptions* options = nullptr)
{
	if (kernel == OperatorProjectorMetalKernel::Joseph)
	{
		if (canUseJosephAxisSpecializedForward(kernel) &&
		    josephAxis < kNoJosephAxis)
		{
			return forwardProjectJosephSingleRayAxis(
			    context, imageBuffer, batch, params, josephAxis, profile,
			    options);
		}
		return forwardProjectJosephSingleRay(context, imageBuffer, batch,
		                                     params, profile, options);
	}
	return forwardProjectSiddonSingleRay(context, imageBuffer, batch, params,
	                                     profile, options);
}

bool forwardProjectSingleRay(OperatorProjectorMetalKernel kernel,
                             const Context& context,
                             ForwardImageResources& imageResources,
                             ProjectionBatchMetal& batch,
                             const SiddonForwardImageParams& params,
                             SiddonProjectorKernelProfile* profile,
                             std::uint32_t josephAxis = kNoJosephAxis,
                             const ProjectorKernelOptions* options = nullptr)
{
	if (kernel != OperatorProjectorMetalKernel::JosephTextureForward)
	{
		return forwardProjectSingleRay(kernel, context,
		                               imageResources.imageBuffer, batch,
		                               params, profile, josephAxis, options);
	}

	auto textureIt = imageResources.josephTexturesByFrame.find(params.frame);
	if (textureIt == imageResources.josephTexturesByFrame.end())
	{
		Texture3D texture;
		if (!uploadJosephImageFrameTexture(context, imageResources.image,
		                                   params.frame, texture,
		                                   imageResources.sampler, profile))
		{
			return false;
		}
		textureIt = imageResources.josephTexturesByFrame
		                .emplace(params.frame, std::move(texture))
		                .first;
	}

	return forwardProjectJosephSingleRayTexture(context, textureIt->second,
	                                            imageResources.sampler, batch,
	                                            params, profile, options);
}

bool backProjectSingleRay(OperatorProjectorMetalKernel kernel,
                          const Context& context,
                          const ProjectionBatchMetal& batch,
                          Buffer& imageBuffer,
                          const SiddonForwardImageParams& params,
                          SiddonProjectorKernelProfile* profile,
                          std::uint32_t josephAxis = kNoJosephAxis,
                          const ProjectorKernelOptions* options = nullptr)
{
	if (kernel == OperatorProjectorMetalKernel::Joseph ||
	    kernel == OperatorProjectorMetalKernel::JosephTextureForward)
	{
		if (canUseJosephAxisSpecializedAdjoint(kernel) &&
		    josephAxis < kNoJosephAxis)
		{
			return backProjectJosephSingleRayAxis(
			    context, batch, imageBuffer, params, josephAxis, profile,
			    options);
		}
		return backProjectJosephSingleRay(context, batch, imageBuffer, params,
		                                  profile, options);
	}
	return backProjectSiddonSingleRay(context, batch, imageBuffer, params,
	                                  profile, options);
}

std::size_t estimateCorrectionBytesPerEvent(
    const OperatorProjectorMetalOsemConfig* config)
{
	if (config == nullptr || !config->cacheCorrectionFactors)
	{
		return 0;
	}

	std::size_t valuesPerEvent = 3;
	if (config->hasInVivoAttenuation)
	{
		valuesPerEvent += 1;
	}
	return sizeof(float) * valuesPerEvent;
}

std::size_t estimateCachedBytes(
    std::size_t eventCount,
    const OperatorProjectorMetalOsemConfig* config = nullptr)
{
	return eventCount *
	       (sizeof(ProjectionLineEndpoints) * 2 + sizeof(float) +
	        sizeof(bin_t) + estimateCorrectionBytesPerEvent(config));
}

std::size_t estimateCorrectionOnlyCachedBytes(
    std::size_t eventCount, const OperatorProjectorMetalOsemConfig* config)
{
	return eventCount * estimateCorrectionBytesPerEvent(config);
}

std::size_t floatByteCount(std::size_t count)
{
	return sizeof(float) * count;
}

std::size_t imageFloatCount(const SiddonForwardImageParams& params)
{
	return static_cast<std::size_t>(params.nx) *
	       static_cast<std::size_t>(params.ny) *
	       static_cast<std::size_t>(params.nz) *
	       static_cast<std::size_t>(params.nt);
}

Buffer uploadFloatVector(const Context& context,
                         const std::vector<float>& values)
{
	if (values.empty())
	{
		return {};
	}
	return Buffer::copyFromHost(context.device(), values.data(),
	                            floatByteCount(values.size()));
}

Buffer uploadDummyFloat(const Context& context, float value)
{
	return Buffer::copyFromHost(context.device(), &value, sizeof(value));
}

Buffer uploadOptionalFloatVector(const Context& context,
                                 const std::vector<float>& values,
                                 bool useValues, float dummyValue)
{
	return useValues ? uploadFloatVector(context, values) :
	                   uploadDummyFloat(context, dummyValue);
}

void releaseFloatVectorStorage(std::vector<float>& values)
{
	std::vector<float>{}.swap(values);
}

bool allocateClearedSiddonImageBuffer(const Context& context,
                                      const Image& image, Buffer& imageBuffer,
                                      float value,
                                      SiddonProjectorKernelProfile* profile)
{
	SiddonForwardImageParams params{};
	if (!context.isValid() || !makeSiddonForwardImageParams(image, 0, params))
	{
		return false;
	}

	const std::size_t count = imageFloatCount(params);
	const auto clearStart = Clock::now();
	imageBuffer =
	    usePrivateUpdateImageBuffer() ?
	        Buffer::allocatePrivate(context.device(), floatByteCount(count)) :
	        Buffer::allocate(context.device(), floatByteCount(count));
	const bool didClear =
	    imageBuffer.isValid() &&
	    launchProjectionClear(context.device(), context.library(),
	                          context.commandQueue(), imageBuffer, value,
	                          count);
	if (profile != nullptr)
	{
		profile->imageUploadSeconds +=
		    getElapsedSeconds(clearStart, Clock::now());
	}
	return didClear;
}

OperatorProjectorMetalSupport supported()
{
	return {true, {}};
}

OperatorProjectorMetalSupport unsupported(std::string reason)
{
	return {false, std::move(reason)};
}

bool hasOnlySupportedProjectionProperties(
    const std::set<ProjectionPropertyType>& projectionProperties)
{
	for (const ProjectionPropertyType property : projectionProperties)
	{
		if (property != ProjectionPropertyType::LOR &&
		    property != ProjectionPropertyType::DYNAMIC_FRAME)
		{
			return false;
		}
	}
	return true;
}

bool hasConstraints(const BinLoader& binLoader)
{
	return binLoader.getConstraintManager().getTypeID() != 0;
}

void addGatherProfile(OperatorProjectorMetalProfile* profile,
                      const BinLoader& binLoader, bool profileAsForward,
                      bool cacheBuild, double elapsed)
{
	if (profile == nullptr)
	{
		return;
	}

	double& total = profileAsForward ? profile->forwardGatherSeconds :
	                                   profile->adjointGatherSeconds;
	double& cacheBuildTotal = profileAsForward ?
	                              profile->forwardGatherCacheBuildSeconds :
	                              profile->adjointGatherCacheBuildSeconds;
	double& uncachedTotal = profileAsForward ?
	                            profile->forwardGatherUncachedSeconds :
	                            profile->adjointGatherUncachedSeconds;
	double& directTotal = profileAsForward ?
	                          profile->forwardGatherDirectSeconds :
	                          profile->adjointGatherDirectSeconds;
	double& constrainedTotal = profileAsForward ?
	                               profile->forwardGatherConstrainedSeconds :
	                               profile->adjointGatherConstrainedSeconds;

	total += elapsed;
	if (cacheBuild)
	{
		cacheBuildTotal += elapsed;
	}
	else
	{
		uncachedTotal += elapsed;
	}
	if (hasConstraints(binLoader))
	{
		constrainedTotal += elapsed;
	}
	else
	{
		directTotal += elapsed;
	}
}

void addPackProfile(OperatorProjectorMetalProfile* profile,
                    bool profileAsForward, bool cacheBuild, double elapsed)
{
	if (profile == nullptr)
	{
		return;
	}

	double& total = profileAsForward ? profile->forwardPackSeconds :
	                                   profile->adjointPackSeconds;
	double& cacheBuildTotal = profileAsForward ?
	                              profile->forwardPackCacheBuildSeconds :
	                              profile->adjointPackCacheBuildSeconds;
	double& uncachedTotal = profileAsForward ?
	                            profile->forwardPackUncachedSeconds :
	                            profile->adjointPackUncachedSeconds;

	total += elapsed;
	if (cacheBuild)
	{
		cacheBuildTotal += elapsed;
	}
	else
	{
		uncachedTotal += elapsed;
	}
}

void addBatchUploadProfile(OperatorProjectorMetalProfile* profile,
                           bool profileAsForward, bool cacheBuild,
                           double elapsed)
{
	if (profile == nullptr)
	{
		return;
	}

	double& total = profileAsForward ? profile->forwardBatchUploadSeconds :
	                                   profile->adjointBatchUploadSeconds;
	double& cacheBuildTotal = profileAsForward ?
	                              profile->forwardBatchUploadCacheBuildSeconds :
	                              profile->adjointBatchUploadCacheBuildSeconds;
	double& uncachedTotal = profileAsForward ?
	                            profile->forwardBatchUploadUncachedSeconds :
	                            profile->adjointBatchUploadUncachedSeconds;

	total += elapsed;
	if (cacheBuild)
	{
		cacheBuildTotal += elapsed;
	}
	else
	{
		uncachedTotal += elapsed;
	}
}

SiddonProjectorKernelProfile
    makeSiddonKernelProfile(const OperatorProjectorMetalProfile* profile)
{
	SiddonProjectorKernelProfile kernelProfile;
	(void)profile;
	return kernelProfile;
}

SiddonProjectorKernelProfile
    makeAdjointSiddonKernelProfile(OperatorProjectorMetalProfile* profile)
{
	SiddonProjectorKernelProfile kernelProfile;
	if (profile == nullptr ||
	    (!profile->diagnoseAdjointUpdateCounts &&
	        !profile->diagnoseAdjointVoxelHits))
	{
		return kernelProfile;
	}

	const std::size_t maxBatches = getAdjointDiagnosticMaxBatches();
	const std::size_t stride = getAdjointDiagnosticStride();
	profile->adjointDiagnosticMaxBatches = maxBatches;
	profile->adjointDiagnosticStride = stride;
	profile->adjointDiagnosticBatchesSeen += 1;

	const std::size_t batchIndex = profile->adjointDiagnosticBatchesSeen - 1;
	const bool strideSelected = stride <= 1 || batchIndex % stride == 0;
	const bool underLimit =
	    maxBatches == 0 ||
	    profile->adjointDiagnosticBatchesProfiled < maxBatches;
	if (strideSelected && underLimit)
	{
		kernelProfile.diagnoseAdjointUpdateCounts =
		    profile->diagnoseAdjointUpdateCounts;
		kernelProfile.diagnoseAdjointVoxelHits =
		    profile->diagnoseAdjointVoxelHits;
		profile->adjointDiagnosticBatchesProfiled += 1;
	}
	else
	{
		profile->adjointDiagnosticBatchesSkipped += 1;
	}
	return kernelProfile;
}

void addForwardKernelProfile(OperatorProjectorMetalProfile* profile,
                             const SiddonProjectorKernelProfile& kernelProfile)
{
	if (profile == nullptr)
	{
		return;
	}
	profile->forwardKernelSeconds += kernelProfile.kernelSeconds;
}

void addAdjointKernelProfile(OperatorProjectorMetalProfile* profile,
                             const SiddonProjectorKernelProfile& kernelProfile)
{
	if (profile == nullptr)
	{
		return;
	}
	profile->adjointKernelSeconds += kernelProfile.kernelSeconds;
	profile->adjointUpdateCountSeconds +=
	    kernelProfile.adjointUpdateCountSeconds;
	profile->adjointVoxelHitCountSeconds +=
	    kernelProfile.adjointVoxelHitCountSeconds;
	profile->adjointVoxelUpdates += kernelProfile.adjointVoxelUpdates;
	profile->adjointRaysWithUpdates += kernelProfile.adjointRaysWithUpdates;
	profile->adjointMaxUpdatesPerRay =
	    std::max(profile->adjointMaxUpdatesPerRay,
	             kernelProfile.adjointMaxUpdatesPerRay);
	profile->adjointVoxelHitMaps += kernelProfile.adjointVoxelHitMaps;
	profile->adjointBatchHitVoxels += kernelProfile.adjointBatchHitVoxels;
	profile->adjointVoxelHitTotalUpdates +=
	    kernelProfile.adjointVoxelHitTotalUpdates;
	profile->adjointMaxVoxelHits = std::max(profile->adjointMaxVoxelHits,
	                                        kernelProfile.adjointMaxVoxelHits);
	profile->adjointMaxBatchP50VoxelHits =
	    std::max(profile->adjointMaxBatchP50VoxelHits,
	             kernelProfile.adjointMaxBatchP50VoxelHits);
	profile->adjointMaxBatchP90VoxelHits =
	    std::max(profile->adjointMaxBatchP90VoxelHits,
	             kernelProfile.adjointMaxBatchP90VoxelHits);
	profile->adjointMaxBatchP95VoxelHits =
	    std::max(profile->adjointMaxBatchP95VoxelHits,
	             kernelProfile.adjointMaxBatchP95VoxelHits);
	profile->adjointMaxBatchP99VoxelHits =
	    std::max(profile->adjointMaxBatchP99VoxelHits,
	             kernelProfile.adjointMaxBatchP99VoxelHits);
	profile->adjointMaxBatchP999VoxelHits =
	    std::max(profile->adjointMaxBatchP999VoxelHits,
	             kernelProfile.adjointMaxBatchP999VoxelHits);
	profile->adjointMaxBatchMeanVoxelHits =
	    std::max(profile->adjointMaxBatchMeanVoxelHits,
	             kernelProfile.adjointMaxBatchMeanVoxelHits);
	profile->adjointMaxBatchTop1PctVoxelHitFraction =
	    std::max(profile->adjointMaxBatchTop1PctVoxelHitFraction,
	             kernelProfile.adjointMaxBatchTop1PctVoxelHitFraction);
	profile->adjointMaxBatchTop01PctVoxelHitFraction =
	    std::max(profile->adjointMaxBatchTop01PctVoxelHitFraction,
	             kernelProfile.adjointMaxBatchTop01PctVoxelHitFraction);
	profile->adjointTileSize =
	    std::max(profile->adjointTileSize, kernelProfile.adjointTileSize);
	profile->adjointVoxelHitTiles += kernelProfile.adjointVoxelHitTiles;
	profile->adjointVoxelHitTileTotalUpdates +=
	    kernelProfile.adjointVoxelHitTileTotalUpdates;
	profile->adjointMaxTileHits =
	    std::max(profile->adjointMaxTileHits, kernelProfile.adjointMaxTileHits);
	profile->adjointMaxBatchP95TileHits =
	    std::max(profile->adjointMaxBatchP95TileHits,
	             kernelProfile.adjointMaxBatchP95TileHits);
	profile->adjointMaxBatchP99TileHits =
	    std::max(profile->adjointMaxBatchP99TileHits,
	             kernelProfile.adjointMaxBatchP99TileHits);
	profile->adjointMaxBatchMeanTileHits =
	    std::max(profile->adjointMaxBatchMeanTileHits,
	             kernelProfile.adjointMaxBatchMeanTileHits);
	profile->adjointMaxBatchTop1PctTileHitFraction =
	    std::max(profile->adjointMaxBatchTop1PctTileHitFraction,
	             kernelProfile.adjointMaxBatchTop1PctTileHitFraction);
	profile->adjointMaxBatchTop01PctTileHitFraction =
	    std::max(profile->adjointMaxBatchTop01PctTileHitFraction,
	             kernelProfile.adjointMaxBatchTop01PctTileHitFraction);
}

ProjectionLineEndpoints makeCenteredLine(const Line3D& line,
                                         const ImageParams& params)
{
	return {line.point1.x - params.off_x, line.point1.y - params.off_y,
	        line.point1.z - params.off_z, line.point2.x - params.off_x,
	        line.point2.y - params.off_y, line.point2.z - params.off_z};
}

Vector3D applyTransform(const Vector3D& point, const transform_t& transform)
{
	return {transform.r00 * point.x + transform.r01 * point.y +
	            transform.r02 * point.z + transform.tx,
	        transform.r10 * point.x + transform.r11 * point.y +
	            transform.r12 * point.z + transform.ty,
	        transform.r20 * point.x + transform.r21 * point.y +
	            transform.r22 * point.z + transform.tz};
}

ProjectionLineEndpoints makeCenteredEndpoints(const Vector3D& point1,
                                              const Vector3D& point2,
                                              const ImageParams& params)
{
	return {point1.x - params.off_x, point1.y - params.off_y,
	        point1.z - params.off_z, point2.x - params.off_x,
	        point2.y - params.off_y, point2.z - params.off_z};
}

bool makeBridgeEvent(const ProjectionData& projectionData,
                     const ImageParams& imageParams, bin_t bin,
                     const Line3D& line, bool includeProjectionValues,
                     BridgeEvent& event)
{
	frame_t frame = 0;
	if (projectionData.hasDynamicFraming())
	{
		frame = projectionData.getDynamicFrame(bin);
	}
	if (frame >= imageParams.nt)
	{
		return false;
	}

	event.bin = bin;
	event.line = makeCenteredLine(line, imageParams);
	event.frame = frame;
	if (includeProjectionValues)
	{
		event.projectionValue = projectionData.getProjectionValue(bin);
	}
	return true;
}

bool appendBridgeEvent(const ProjectionData& projectionData,
                       const ImageParams& imageParams, bin_t bin,
                       const Line3D& line, bool includeProjectionValues,
                       std::vector<BridgeEvent>& events)
{
	BridgeEvent event;
	if (!makeBridgeEvent(projectionData, imageParams, bin, line,
	                     includeProjectionValues, event))
	{
		return false;
	}
	events.push_back(event);
	return true;
}

bool tryGatherListModeLUTBridgeEvents(const ProjectionData& projectionData,
                                      const Image& image,
                                      const BinIterator& binIterator,
                                      OperatorProjectorMetalCache* cache,
                                      bool includeProjectionValues,
                                      std::vector<BridgeEvent>& events,
                                      bool& handled)
{
	handled = false;
	if (includeProjectionValues)
	{
		return true;
	}

	const auto* listMode = dynamic_cast<const ListModeLUT*>(&projectionData);
	if (listMode == nullptr)
	{
		return true;
	}

	const Array1DBase<det_id_t>* detector1Array =
	    listMode->getDetector1ArrayPtr();
	const Array1DBase<det_id_t>* detector2Array =
	    listMode->getDetector2ArrayPtr();
	if (detector1Array == nullptr || detector2Array == nullptr ||
	    detector1Array->getRawPointer() == nullptr ||
	    detector2Array->getRawPointer() == nullptr)
	{
		return false;
	}

	const det_id_t* detector1 = detector1Array->getRawPointer();
	const det_id_t* detector2 = detector2Array->getRawPointer();
	const Scanner& scanner = listMode->getScanner();
	const std::size_t numDets = scanner.getNumDets();
	std::vector<Vector3D> localDetectorPositions;
	const std::vector<Vector3D>* detectorPositions = nullptr;
	if (cache != nullptr)
	{
		detectorPositions = &cache->detectorPositions(scanner);
	}
	else
	{
		localDetectorPositions.resize(numDets);
		for (std::size_t det = 0; det < numDets; ++det)
		{
			localDetectorPositions[det] =
			    scanner.getDetectorPos(static_cast<det_id_t>(det));
		}
		detectorPositions = &localDetectorPositions;
	}

	const bool hasMotion = listMode->hasMotion();
	std::vector<transform_t> motionTransforms;
	if (hasMotion)
	{
		const std::size_t numMotionFrames = listMode->getNumMotionFrames();
		motionTransforms.resize(numMotionFrames);
		for (std::size_t frame = 0; frame < numMotionFrames; ++frame)
		{
			motionTransforms[frame] = listMode->getTransformOfMotionFrame(
			    static_cast<frame_t>(frame));
		}
	}

	const bool hasDynamicFraming = listMode->hasDynamicFraming();
	const ImageParams& imageParams = image.getParams();
	std::vector<BridgeEvent> gatheredEvents(binIterator.size());
	std::atomic_bool failed{false};
	util::parallelForChunked(
	    binIterator.size(), globals::getNumThreads(),
	    [&](std::size_t i, std::size_t /*tid*/)
	    {
		    if (failed.load(std::memory_order_relaxed))
		    {
			    return;
		    }

		    const bin_t bin = binIterator.get(static_cast<bin_t>(i));
		    if (bin >= listMode->count())
		    {
			    failed.store(true, std::memory_order_relaxed);
			    return;
		    }

		    const det_id_t det1 = detector1[bin];
		    const det_id_t det2 = detector2[bin];
		    if (det1 >= numDets || det2 >= numDets)
		    {
			    failed.store(true, std::memory_order_relaxed);
			    return;
		    }

		    Vector3D point1 = (*detectorPositions)[det1];
		    Vector3D point2 = (*detectorPositions)[det2];
		    if (hasMotion)
		    {
			    const frame_t motionFrame = listMode->getMotionFrame(bin);
			    if (motionFrame >= 0)
			    {
				    if (static_cast<std::size_t>(motionFrame) >=
				        motionTransforms.size())
				    {
					    failed.store(true, std::memory_order_relaxed);
					    return;
				    }
				    const transform_t& transform =
				        motionTransforms[static_cast<std::size_t>(motionFrame)];
				    point1 = applyTransform(point1, transform);
				    point2 = applyTransform(point2, transform);
			    }
		    }

		    frame_t dynamicFrame = 0;
		    if (hasDynamicFraming)
		    {
			    dynamicFrame = listMode->getDynamicFrame(bin);
		    }
		    if (dynamicFrame >= imageParams.nt)
		    {
			    failed.store(true, std::memory_order_relaxed);
			    return;
		    }

		    BridgeEvent event;
		    event.bin = bin;
		    event.line = makeCenteredEndpoints(point1, point2, imageParams);
		    event.frame = dynamicFrame;
		    gatheredEvents[i] = event;
	    });

	if (failed.load(std::memory_order_relaxed))
	{
		return false;
	}

	events = std::move(gatheredEvents);
	handled = true;
	return true;
}

std::vector<std::vector<BridgeFrameBatch>>
    makeFrameBatchWorkspace(std::size_t numThreads,
                            const ImageParams& imageParams,
                            std::size_t eventCount)
{
	const std::size_t frameCount =
	    imageParams.nt > 0 ? static_cast<std::size_t>(imageParams.nt) : 0;
	std::vector<std::vector<BridgeFrameBatch>> workspace(
	    numThreads, std::vector<BridgeFrameBatch>(frameCount));
	if (frameCount == 0)
	{
		return workspace;
	}

	const std::size_t reservePerFrame =
	    std::max<std::size_t>(1, eventCount / (numThreads * frameCount));
	for (auto& threadBatches : workspace)
	{
		for (std::size_t frame = 0; frame < frameCount; ++frame)
		{
			threadBatches[frame].frame = static_cast<frame_t>(frame);
			threadBatches[frame].lines.reserve(reservePerFrame);
			threadBatches[frame].bins.reserve(reservePerFrame);
		}
	}
	return workspace;
}

bool appendFrameBatchEvent(std::vector<BridgeFrameBatch>& frameBatches,
                           frame_t frame,
                           const ProjectionLineEndpoints& line, bin_t bin)
{
	if (frame < 0)
	{
		return true;
	}
	const auto frameIndex = static_cast<std::size_t>(frame);
	if (frameIndex >= frameBatches.size())
	{
		return false;
	}
	frameBatches[frameIndex].lines.push_back(line);
	frameBatches[frameIndex].bins.push_back(bin);
	return true;
}

void mergeFrameBatchWorkspace(
    std::vector<std::vector<BridgeFrameBatch>>& workspace,
    std::vector<BridgeFrameBatch>& frameBatches)
{
	frameBatches.clear();
	if (workspace.empty())
	{
		return;
	}

	const std::size_t frameCount = workspace.front().size();
	frameBatches.reserve(frameCount);
	for (std::size_t frame = 0; frame < frameCount; ++frame)
	{
		std::size_t eventCount = 0;
		for (const auto& threadBatches : workspace)
		{
			eventCount += threadBatches[frame].lines.size();
		}
		if (eventCount == 0)
		{
			continue;
		}

		BridgeFrameBatch merged;
		merged.frame = static_cast<frame_t>(frame);
		merged.lines.reserve(eventCount);
		merged.bins.reserve(eventCount);
		for (auto& threadBatches : workspace)
		{
			auto& batch = threadBatches[frame];
			merged.lines.insert(merged.lines.end(),
			                    std::make_move_iterator(batch.lines.begin()),
			                    std::make_move_iterator(batch.lines.end()));
			merged.bins.insert(merged.bins.end(), batch.bins.begin(),
			                   batch.bins.end());
		}
		frameBatches.push_back(std::move(merged));
	}
}

bool tryGatherListModeLUTFrameBatches(
    const ProjectionData& projectionData, const Image& image,
    const BinIterator& binIterator, OperatorProjectorMetalCache* cache,
    std::vector<BridgeFrameBatch>& frameBatches, bool& handled)
{
	handled = false;
	const auto* listMode = dynamic_cast<const ListModeLUT*>(&projectionData);
	if (listMode == nullptr)
	{
		return true;
	}

	const Array1DBase<det_id_t>* detector1Array =
	    listMode->getDetector1ArrayPtr();
	const Array1DBase<det_id_t>* detector2Array =
	    listMode->getDetector2ArrayPtr();
	if (detector1Array == nullptr || detector2Array == nullptr ||
	    detector1Array->getRawPointer() == nullptr ||
	    detector2Array->getRawPointer() == nullptr)
	{
		return false;
	}

	const det_id_t* detector1 = detector1Array->getRawPointer();
	const det_id_t* detector2 = detector2Array->getRawPointer();
	const Scanner& scanner = listMode->getScanner();
	const std::size_t numDets = scanner.getNumDets();
	std::vector<Vector3D> localDetectorPositions;
	const std::vector<Vector3D>* detectorPositions = nullptr;
	if (cache != nullptr)
	{
		detectorPositions = &cache->detectorPositions(scanner);
	}
	else
	{
		localDetectorPositions.resize(numDets);
		for (std::size_t det = 0; det < numDets; ++det)
		{
			localDetectorPositions[det] =
			    scanner.getDetectorPos(static_cast<det_id_t>(det));
		}
		detectorPositions = &localDetectorPositions;
	}

	const bool hasMotion = listMode->hasMotion();
	std::vector<transform_t> motionTransforms;
	if (hasMotion)
	{
		const std::size_t numMotionFrames = listMode->getNumMotionFrames();
		motionTransforms.resize(numMotionFrames);
		for (std::size_t frame = 0; frame < numMotionFrames; ++frame)
		{
			motionTransforms[frame] = listMode->getTransformOfMotionFrame(
			    static_cast<frame_t>(frame));
		}
	}

	const bool hasDynamicFraming = listMode->hasDynamicFraming();
	const ImageParams& imageParams = image.getParams();
	const std::size_t numThreads =
	    std::max<std::size_t>(1, globals::getNumThreads());
	auto workspace =
	    makeFrameBatchWorkspace(numThreads, imageParams, binIterator.size());
	if (workspace.empty() || workspace.front().empty())
	{
		return false;
	}

	std::atomic_bool failed{false};
	util::parallelForChunked(
	    binIterator.size(), numThreads,
	    [&](std::size_t i, std::size_t tid)
	    {
		    if (failed.load(std::memory_order_relaxed))
		    {
			    return;
		    }

		    const bin_t bin = binIterator.get(static_cast<bin_t>(i));
		    if (bin >= listMode->count())
		    {
			    failed.store(true, std::memory_order_relaxed);
			    return;
		    }

		    const det_id_t det1 = detector1[bin];
		    const det_id_t det2 = detector2[bin];
		    if (det1 >= numDets || det2 >= numDets)
		    {
			    failed.store(true, std::memory_order_relaxed);
			    return;
		    }

		    Vector3D point1 = (*detectorPositions)[det1];
		    Vector3D point2 = (*detectorPositions)[det2];
		    if (hasMotion)
		    {
			    const frame_t motionFrame = listMode->getMotionFrame(bin);
			    if (motionFrame >= 0)
			    {
				    if (static_cast<std::size_t>(motionFrame) >=
				        motionTransforms.size())
				    {
					    failed.store(true, std::memory_order_relaxed);
					    return;
				    }
				    const transform_t& transform =
				        motionTransforms[static_cast<std::size_t>(motionFrame)];
				    point1 = applyTransform(point1, transform);
				    point2 = applyTransform(point2, transform);
			    }
		    }

		    frame_t dynamicFrame = 0;
		    if (hasDynamicFraming)
		    {
			    dynamicFrame = listMode->getDynamicFrame(bin);
		    }
		    if (!appendFrameBatchEvent(
		            workspace[tid], dynamicFrame,
		            makeCenteredEndpoints(point1, point2, imageParams), bin))
		    {
			    failed.store(true, std::memory_order_relaxed);
		    }
	    });

	if (failed.load(std::memory_order_relaxed))
	{
		return false;
	}

	mergeFrameBatchWorkspace(workspace, frameBatches);
	handled = true;
	return true;
}

bool gatherBridgeFrameBatches(const ProjectionData& projectionData,
                              const Image& image,
                              const BinIterator& binIterator,
                              const BinLoader& binLoader,
                              OperatorProjectorMetalCache* cache,
                              std::vector<BridgeFrameBatch>& frameBatches,
                              bool& handled)
{
	handled = false;
	if (!image.isMemoryValid() ||
	    !binLoader.getPropertyManager().has(ProjectionPropertyType::LOR) ||
	    binLoader.getProjectionPropertiesRawPointer() == nullptr)
	{
		return false;
	}
	if (hasConstraints(binLoader))
	{
		return true;
	}

	bool listModeHandled = false;
	if (!tryGatherListModeLUTFrameBatches(
	        projectionData, image, binIterator, cache, frameBatches,
	        listModeHandled))
	{
		return false;
	}
	if (listModeHandled)
	{
		handled = true;
		return true;
	}

	const ImageParams& imageParams = image.getParams();
	const std::size_t numThreads =
	    std::max<std::size_t>(1, globals::getNumThreads());
	auto workspace =
	    makeFrameBatchWorkspace(numThreads, imageParams, binIterator.size());
	if (workspace.empty() || workspace.front().empty())
	{
		return false;
	}

	std::atomic_bool failed{false};
	util::parallelForChunked(
	    binIterator.size(), numThreads,
	    [&](std::size_t i, std::size_t tid)
	    {
		    if (failed.load(std::memory_order_relaxed))
		    {
			    return;
		    }
		    const bin_t bin = binIterator.get(static_cast<bin_t>(i));
		    BridgeEvent event;
		    if (!makeBridgeEvent(projectionData, imageParams, bin,
		                         projectionData.getLOR(bin), false,
		                         event) ||
		        !appendFrameBatchEvent(workspace[tid], event.frame,
		                               event.line, event.bin))
		    {
			    failed.store(true, std::memory_order_relaxed);
		    }
	    });
	if (failed.load(std::memory_order_relaxed))
	{
		return false;
	}

	mergeFrameBatchWorkspace(workspace, frameBatches);
	handled = true;
	return true;
}

bool gatherBridgeEvents(const ProjectionData& projectionData,
                        const Image& image, const BinIterator& binIterator,
                        const BinLoader& binLoader,
                        OperatorProjectorMetalCache* cache,
                        bool includeProjectionValues,
                        std::vector<BridgeEvent>& events)
{
	if (!image.isMemoryValid() ||
	    !binLoader.getPropertyManager().has(ProjectionPropertyType::LOR) ||
	    binLoader.getProjectionPropertiesRawPointer() == nullptr)
	{
		return false;
	}

	BinFilter::CollectInfoFlags collectInfoFlags(false);
	binLoader.collectFlags(collectInfoFlags);
	const ProjectionPropertyManager& propertyManager =
	    binLoader.getPropertyManager();
	PropertyUnit* properties = binLoader.getProjectionPropertiesRawPointer();
	const ImageParams& imageParams = image.getParams();

	std::vector<BridgeEvent> gatheredEvents;
	gatheredEvents.reserve(binIterator.size());
	if (!hasConstraints(binLoader))
	{
		bool handled = false;
		if (!tryGatherListModeLUTBridgeEvents(
		        projectionData, image, binIterator, cache,
		        includeProjectionValues, events, handled))
		{
			return false;
		}
		if (handled)
		{
			return true;
		}

		gatheredEvents.resize(binIterator.size());
		std::atomic_bool failed{false};
		util::parallelForChunked(
		    binIterator.size(), globals::getNumThreads(),
		    [&](std::size_t i, std::size_t /*tid*/)
		    {
			    if (failed.load(std::memory_order_relaxed))
			    {
				    return;
			    }
			    const bin_t bin = binIterator.get(static_cast<bin_t>(i));
			    BridgeEvent event;
			    if (!makeBridgeEvent(projectionData, imageParams, bin,
			                         projectionData.getLOR(bin),
			                         includeProjectionValues, event))
			    {
				    failed.store(true, std::memory_order_relaxed);
				    return;
			    }
			    gatheredEvents[i] = event;
		    });
		if (failed.load(std::memory_order_relaxed))
		{
			return false;
		}

		events = std::move(gatheredEvents);
		return true;
	}

	for (std::size_t i = 0; i < binIterator.size(); ++i)
	{
		const bin_t bin = binIterator.get(static_cast<bin_t>(i));
		binLoader.collectInfo(bin, 0, 0, projectionData, collectInfoFlags);
		if (!binLoader.verifyConstraints(0))
		{
			continue;
		}

		const Line3D line = propertyManager.getDataValue<Line3D>(
		    properties, 0, ProjectionPropertyType::LOR);
		if (!appendBridgeEvent(projectionData, imageParams, bin, line,
		                       includeProjectionValues, gatheredEvents))
		{
			return false;
		}
	}

	events = std::move(gatheredEvents);
	return true;
}

std::map<frame_t, std::vector<std::size_t>>
    groupNonNegativeFrames(const std::vector<BridgeEvent>& events)
{
	std::map<frame_t, std::vector<std::size_t>> frameGroups;
	for (std::size_t i = 0; i < events.size(); ++i)
	{
		if (events[i].frame >= 0)
		{
			frameGroups[events[i].frame].push_back(i);
		}
	}
	return frameGroups;
}

float getSensitivityCorrection(const ProjectionData& measurements,
                               const Corrector_CPU& corrector,
                               const OperatorProjectorMetalOsemConfig& config,
                               bin_t bin)
{
	return config.usePrecomputedCorrections ?
	           corrector.getPrecomputedSensitivityFactor(bin) :
	           corrector.getSensitivityFactor(measurements, bin);
}

float getAttenuationCorrection(const ProjectionData& measurements,
                               const Corrector_CPU& corrector,
                               const OperatorProjectorMetalOsemConfig& config,
                               bin_t bin)
{
	return config.usePrecomputedCorrections ?
	           corrector.getPrecomputedAttenuationFactor(bin) :
	           corrector.getAttenuationFactorForBin(measurements, bin);
}

float getRandomsCorrection(const ProjectionData& measurements,
                           const Corrector_CPU& corrector,
                           const OperatorProjectorMetalOsemConfig& config,
                           bin_t bin)
{
	return config.usePrecomputedCorrections ?
	           corrector.getPrecomputedRandomsEstimate(bin) :
	           corrector.getRandomsEstimateForBin(measurements, bin);
}

float getScatterCorrection(const ProjectionData& measurements,
                           const Corrector_CPU& corrector,
                           const OperatorProjectorMetalOsemConfig& config,
                           bin_t bin)
{
	return config.usePrecomputedCorrections ?
	           corrector.getPrecomputedScatterEstimate(bin) :
	           corrector.getScatterEstimateForBin(measurements, bin);
}

float getAdditiveCorrection(const ProjectionData& measurements,
                            const Corrector_CPU& corrector,
                            const OperatorProjectorMetalOsemConfig& config,
                            bin_t bin)
{
	if (config.usePrecomputedCorrections)
	{
		float additiveValue = 0.0f;
		if (config.hasRandomsEstimates)
		{
			additiveValue += corrector.getPrecomputedRandomsEstimate(bin);
		}
		if (config.hasScatterEstimates)
		{
			additiveValue += corrector.getPrecomputedScatterEstimate(bin);
		}
		return additiveValue;
	}
	return corrector.getAdditiveCorrectionForBin(measurements, bin);
}

float getInVivoAttenuationCorrection(
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config, bin_t bin)
{
	return config.usePrecomputedCorrections ?
	           corrector.getPrecomputedInVivoAttenuationFactor(bin) :
	           corrector.getInVivoAttenuationFactorForBin(measurements, bin);
}

std::size_t cachedCorrectionsByteCount(const CachedOsemCorrections& corrections)
{
	std::size_t valuesPerEvent = 3;
	if (corrections.hasInVivoAttenuation)
	{
		valuesPerEvent += 1;
	}
	return floatByteCount(corrections.valueCount * valuesPerEvent);
}

bool cachedCorrectionBuffersValid(const CachedOsemCorrections& corrections)
{
	return corrections.valueCount > 0 &&
	       corrections.measurementBuffer.isValid() &&
	       corrections.multiplicativeBuffer.isValid() &&
	       corrections.additiveBuffer.isValid() &&
	       (!corrections.hasInVivoAttenuation ||
	        corrections.inVivoAttenuationBuffer.isValid());
}

bool cachedCorrectionsMatch(const CachedOsemCorrections& corrections,
                            const ProjectionData& measurements,
                            const Corrector_CPU& corrector,
                            const OperatorProjectorMetalOsemConfig& config,
                            std::size_t count)
{
	if (!corrections.valid || corrections.measurements != &measurements ||
	    corrections.corrector != &corrector ||
	    corrections.hasSensitivity != config.hasSensitivity ||
	    corrections.hasAttenuation != config.hasAttenuation ||
	    corrections.hasScatterEstimates != config.hasScatterEstimates ||
	    corrections.hasRandomsEstimates != config.hasRandomsEstimates ||
	    corrections.hasInVivoAttenuation != config.hasInVivoAttenuation ||
	    corrections.valueCount != count)
	{
		return false;
	}

	return cachedCorrectionBuffersValid(corrections);
}

bool buildCachedOsemCorrections(const Context& context,
                                OperatorProjectorMetalProfile* profile,
                                const ProjectionData& measurements,
                                const Corrector_CPU& corrector,
                                const std::vector<bin_t>& bins,
                                const OperatorProjectorMetalOsemConfig& config,
                                CachedOsemCorrections& corrections,
                                bool profileAsAdmission = true)
{
	const std::size_t count = bins.size();
	corrections = CachedOsemCorrections{};
	corrections.measurements = &measurements;
	corrections.corrector = &corrector;
	corrections.hasSensitivity = config.hasSensitivity;
	corrections.hasAttenuation = config.hasAttenuation;
	corrections.hasScatterEstimates = config.hasScatterEstimates;
	corrections.hasRandomsEstimates = config.hasRandomsEstimates;
	corrections.hasInVivoAttenuation = config.hasInVivoAttenuation;
	corrections.valueCount = count;
	corrections.measurementValues.resize(count);
	corrections.multiplicativeValues.resize(count);
	corrections.additiveValues.resize(count);
	if (config.hasInVivoAttenuation)
	{
		corrections.inVivoAttenuationValues.resize(count);
	}

	const auto buildStart = Clock::now();
	const auto fillStart = Clock::now();
	double correctionMeasurementSeconds = 0.0;
	double correctionMultiplicativeSeconds = 0.0;
	double correctionAdditiveSeconds = 0.0;
	double correctionInVivoSeconds = 0.0;
	if (profile == nullptr)
	{
		util::parallelForChunked(
		    count, globals::getNumThreads(),
		    [&](std::size_t i, std::size_t /*tid*/)
		    {
			    const bin_t bin = bins[i];
			    corrections.measurementValues[i] =
			        measurements.getProjectionValue(bin);
			    float multiplicativeValue = config.globalScaleFactor;
			    if (config.hasSensitivity)
			    {
				    multiplicativeValue *= getSensitivityCorrection(
				        measurements, corrector, config, bin);
			    }
			    if (config.hasAttenuation)
			    {
				    multiplicativeValue *= getAttenuationCorrection(
				        measurements, corrector, config, bin);
			    }
			    corrections.multiplicativeValues[i] = multiplicativeValue;

			    corrections.additiveValues[i] =
			        getAdditiveCorrection(measurements, corrector, config, bin);

			    if (config.hasInVivoAttenuation)
			    {
				    corrections.inVivoAttenuationValues[i] =
			        getInVivoAttenuationCorrection(measurements, corrector,
			                                       config, bin);
			    }
		    });
	}
	else
	{
		const auto measurementStart = Clock::now();
		util::parallelForChunked(
		    count, globals::getNumThreads(),
		    [&](std::size_t i, std::size_t /*tid*/)
		    {
			    corrections.measurementValues[i] =
			        measurements.getProjectionValue(bins[i]);
		    });
		correctionMeasurementSeconds =
		    getElapsedSeconds(measurementStart, Clock::now());

		const auto multiplicativeStart = Clock::now();
		util::parallelForChunked(
		    count, globals::getNumThreads(),
		    [&](std::size_t i, std::size_t /*tid*/)
		    {
			    const bin_t bin = bins[i];
			    float multiplicativeValue = config.globalScaleFactor;
			    if (config.hasSensitivity)
			    {
				    multiplicativeValue *= getSensitivityCorrection(
				        measurements, corrector, config, bin);
			    }
			    if (config.hasAttenuation)
			    {
				    multiplicativeValue *= getAttenuationCorrection(
				        measurements, corrector, config, bin);
			    }
			    corrections.multiplicativeValues[i] = multiplicativeValue;
		    });
		correctionMultiplicativeSeconds =
		    getElapsedSeconds(multiplicativeStart, Clock::now());

		const auto additiveStart = Clock::now();
		util::parallelForChunked(
		    count, globals::getNumThreads(),
		    [&](std::size_t i, std::size_t /*tid*/)
		    {
			    corrections.additiveValues[i] = getAdditiveCorrection(
			        measurements, corrector, config, bins[i]);
		    });
		correctionAdditiveSeconds =
		    getElapsedSeconds(additiveStart, Clock::now());

		if (config.hasInVivoAttenuation)
		{
			const auto inVivoStart = Clock::now();
			util::parallelForChunked(
			    count, globals::getNumThreads(),
			    [&](std::size_t i, std::size_t /*tid*/)
			    {
				    corrections.inVivoAttenuationValues[i] =
				        getInVivoAttenuationCorrection(
				            measurements, corrector, config, bins[i]);
			    });
			correctionInVivoSeconds =
			    getElapsedSeconds(inVivoStart, Clock::now());
		}
	}
	const double correctionFillSeconds =
	    getElapsedSeconds(fillStart, Clock::now());
	const auto uploadStart = Clock::now();
	corrections.measurementBuffer =
	    uploadFloatVector(context, corrections.measurementValues);
	corrections.multiplicativeBuffer =
	    uploadFloatVector(context, corrections.multiplicativeValues);
	corrections.additiveBuffer =
	    uploadFloatVector(context, corrections.additiveValues);
	if (config.hasInVivoAttenuation)
	{
		corrections.inVivoAttenuationBuffer =
		    uploadFloatVector(context, corrections.inVivoAttenuationValues);
	}
	const double correctionUploadSeconds =
	    getElapsedSeconds(uploadStart, Clock::now());
	corrections.valid = cachedCorrectionBuffersValid(corrections);
	if (profile != nullptr)
	{
		const double correctionBuildSeconds =
		    getElapsedSeconds(buildStart, Clock::now());
		profile->ratioCorrectionCacheBuildSeconds += correctionBuildSeconds;
		if (profileAsAdmission)
		{
			profile->cacheAdmissionCorrectionBuildSeconds +=
			    correctionBuildSeconds;
			profile->cacheAdmissionCorrectionFillSeconds +=
			    correctionFillSeconds;
			profile->cacheAdmissionCorrectionUploadSeconds +=
			    correctionUploadSeconds;
			profile->cacheAdmissionCorrectionMeasurementSeconds +=
			    correctionMeasurementSeconds;
			profile->cacheAdmissionCorrectionMultiplicativeSeconds +=
			    correctionMultiplicativeSeconds;
			profile->cacheAdmissionCorrectionAdditiveSeconds +=
			    correctionAdditiveSeconds;
			profile->cacheAdmissionCorrectionInVivoSeconds +=
			    correctionInVivoSeconds;
		}
		if (corrections.valid)
		{
			profile->ratioCorrectionCacheBuilds += 1;
			profile->ratioCorrectionCacheBytes +=
			    cachedCorrectionsByteCount(corrections);
		}
	}
	if (!corrections.valid)
	{
		return false;
	}

	releaseFloatVectorStorage(corrections.measurementValues);
	releaseFloatVectorStorage(corrections.multiplicativeValues);
	releaseFloatVectorStorage(corrections.additiveValues);
	releaseFloatVectorStorage(corrections.inVivoAttenuationValues);
	return true;
}

bool uploadOsemRatioBuffers(const Context& context,
                            OperatorProjectorMetalProfile* profile,
                            const ProjectionData& measurements,
                            const Corrector_CPU& corrector,
                            const std::vector<bin_t>& bins,
                            const OperatorProjectorMetalOsemConfig& config,
                            OsemRatioBuffers& buffers)
{
	const std::size_t count = bins.size();
	if (count == 0)
	{
		return false;
	}

	std::vector<float> measurementValues(count);
	std::vector<float> sensitivityValues;
	std::vector<float> attenuationValues;
	std::vector<float> randomsValues;
	std::vector<float> scatterValues;
	std::vector<float> inVivoAttenuationValues;
	if (config.hasSensitivity)
	{
		sensitivityValues.resize(count);
	}
	if (config.hasAttenuation)
	{
		attenuationValues.resize(count);
	}
	if (config.hasRandomsEstimates)
	{
		randomsValues.resize(count);
	}
	if (config.hasScatterEstimates)
	{
		scatterValues.resize(count);
	}
	if (config.hasInVivoAttenuation)
	{
		inVivoAttenuationValues.resize(count);
	}

	const auto packStart = Clock::now();
	util::parallelForChunked(
	    count, globals::getNumThreads(),
	    [&](std::size_t i, std::size_t /*tid*/)
	    {
		    const bin_t bin = bins[i];
		    measurementValues[i] = measurements.getProjectionValue(bin);
		    if (config.hasSensitivity)
		    {
			    sensitivityValues[i] = getSensitivityCorrection(
			        measurements, corrector, config, bin);
		    }
		    if (config.hasAttenuation)
		    {
			    attenuationValues[i] = getAttenuationCorrection(
			        measurements, corrector, config, bin);
		    }
		    if (config.hasRandomsEstimates)
		    {
			    randomsValues[i] =
			        getRandomsCorrection(measurements, corrector, config, bin);
		    }
		    if (config.hasScatterEstimates)
		    {
			    scatterValues[i] =
			        getScatterCorrection(measurements, corrector, config, bin);
		    }
		    if (config.hasInVivoAttenuation)
		    {
			    inVivoAttenuationValues[i] = getInVivoAttenuationCorrection(
			        measurements, corrector, config, bin);
		    }
	    });
	if (profile != nullptr)
	{
		profile->ratioPackSeconds += getElapsedSeconds(packStart, Clock::now());
	}

	const auto uploadStart = Clock::now();
	buffers.measurements = uploadFloatVector(context, measurementValues);
	buffers.sensitivity = uploadOptionalFloatVector(
	    context, sensitivityValues, config.hasSensitivity, 1.0f);
	buffers.attenuation = uploadOptionalFloatVector(
	    context, attenuationValues, config.hasAttenuation, 1.0f);
	buffers.randoms = uploadOptionalFloatVector(
	    context, randomsValues, config.hasRandomsEstimates, 0.0f);
	buffers.scatter = uploadOptionalFloatVector(
	    context, scatterValues, config.hasScatterEstimates, 0.0f);
	buffers.inVivoAttenuation = uploadOptionalFloatVector(
	    context, inVivoAttenuationValues, config.hasInVivoAttenuation, 1.0f);
	if (profile != nullptr)
	{
		profile->ratioBatchUploadSeconds +=
		    getElapsedSeconds(uploadStart, Clock::now());
	}

	return buffers.measurements.isValid() && buffers.sensitivity.isValid() &&
	       buffers.attenuation.isValid() && buffers.randoms.isValid() &&
	       buffers.scatter.isValid() && buffers.inVivoAttenuation.isValid();
}

bool countPostRatioValuesForDiagnostics(OperatorProjectorMetalProfile* profile,
                                        const ProjectionBatchMetal& batch)
{
	if (profile == nullptr || !profileRatioNonzeroDiagnostic())
	{
		return true;
	}

	const auto diagnosticStart = Clock::now();
	std::vector<float> ratioValues;
	if (!batch.copyProjectionValuesToHost(ratioValues))
	{
		return false;
	}

	std::size_t nonzeroCount = 0;
	for (const float value : ratioValues)
	{
		if (std::abs(value) != 0.0f)
		{
			nonzeroCount += 1;
		}
	}

	profile->ratioNonzeroDiagnosticSeconds +=
	    getElapsedSeconds(diagnosticStart, Clock::now());
	profile->ratioValues += ratioValues.size();
	profile->ratioNonzeroValues += nonzeroCount;
	profile->ratioZeroValues += ratioValues.size() - nonzeroCount;
	profile->ratioNonzeroDiagnosticBatches += 1;
	return true;
}

bool applyOsemRatioToBatch(const Context& context,
                           OperatorProjectorMetalProfile* profile,
                           ProjectionBatchMetal& batch,
                           const std::vector<bin_t>& bins,
                           const ProjectionData& measurements,
                           const Corrector_CPU& corrector,
                           const OperatorProjectorMetalOsemConfig& config,
                           const CachedOsemCorrections* corrections = nullptr)
{
	if (!batch.isValid() || batch.size() == 0 || batch.size() != bins.size())
	{
		return false;
	}

	const bool useCachedCorrections =
	    corrections != nullptr &&
	    cachedCorrectionsMatch(*corrections, measurements, corrector, config,
	                           bins.size());
	if (config.cacheCorrectionFactors && corrections != nullptr &&
	    profile != nullptr)
	{
		if (useCachedCorrections)
		{
			profile->ratioCorrectionCacheHits += 1;
		}
		else
		{
			profile->ratioCorrectionCacheMisses += 1;
		}
	}
	if (useCachedCorrections)
	{
		const ProjectionCompactOsemRatioParams params{
		    config.denomThreshold, config.hasInVivoAttenuation ? 1u : 0u};
		const Buffer& inVivoBuffer = config.hasInVivoAttenuation ?
		                                 corrections->inVivoAttenuationBuffer :
		                                 corrections->multiplicativeBuffer;
		const auto kernelStart = Clock::now();
		const bool didRun = launchProjectionCompactOsemRatio(
		    context.device(), context.library(), context.commandQueue(),
		    batch.projectionValuesBuffer(), corrections->measurementBuffer,
		    corrections->multiplicativeBuffer, corrections->additiveBuffer,
		    inVivoBuffer, params, batch.size());
		if (profile != nullptr)
		{
			profile->ratioKernelSeconds +=
			    getElapsedSeconds(kernelStart, Clock::now());
		}
		if (!didRun)
		{
			return false;
		}
		return countPostRatioValuesForDiagnostics(profile, batch);
	}

	OsemRatioBuffers buffers;
	if (!uploadOsemRatioBuffers(context, profile, measurements, corrector, bins,
	                            config, buffers))
	{
		return false;
	}

	const ProjectionOsemRatioParams params{config.globalScaleFactor,
	                                       config.denomThreshold,
	                                       config.hasSensitivity ? 1u : 0u,
	                                       config.hasAttenuation ? 1u : 0u,
	                                       config.hasRandomsEstimates ? 1u : 0u,
	                                       config.hasScatterEstimates ? 1u : 0u,
	                                       config.hasInVivoAttenuation ? 1u :
	                                                                     0u};
	const auto kernelStart = Clock::now();
	const bool didRun = launchProjectionOsemRatio(
	    context.device(), context.library(), context.commandQueue(),
	    batch.projectionValuesBuffer(), buffers.measurements,
	    buffers.sensitivity, buffers.attenuation, buffers.randoms,
	    buffers.scatter, buffers.inVivoAttenuation, params, batch.size());
	if (profile != nullptr)
	{
		profile->ratioKernelSeconds +=
		    getElapsedSeconds(kernelStart, Clock::now());
	}
	if (!didRun)
	{
		return false;
	}
	return countPostRatioValuesForDiagnostics(profile, batch);
}

bool computeHostOsemRatioValue(const ProjectionData& measurements,
                               const Corrector_CPU& corrector,
                               const OperatorProjectorMetalOsemConfig& config,
                               bin_t bin, float estimate, float& ratioValue)
{
	if (config.hasSensitivity)
	{
		estimate *=
		    getSensitivityCorrection(measurements, corrector, config, bin);
	}
	if (config.hasAttenuation)
	{
		estimate *=
		    getAttenuationCorrection(measurements, corrector, config, bin);
	}
	estimate *= config.globalScaleFactor;

	if (config.hasRandomsEstimates)
	{
		estimate += getRandomsCorrection(measurements, corrector, config, bin);
	}
	if (config.hasScatterEstimates)
	{
		estimate += getScatterCorrection(measurements, corrector, config, bin);
	}
	if (config.hasInVivoAttenuation)
	{
		estimate *= getInVivoAttenuationCorrection(measurements, corrector,
		                                           config, bin);
	}

	if (std::abs(estimate) <= config.denomThreshold)
	{
		ratioValue = 0.0f;
		return false;
	}

	ratioValue = measurements.getProjectionValue(bin) / estimate;
	if (config.hasSensitivity)
	{
		ratioValue *=
		    getSensitivityCorrection(measurements, corrector, config, bin);
	}
	if (config.hasAttenuation)
	{
		ratioValue *=
		    getAttenuationCorrection(measurements, corrector, config, bin);
	}
	ratioValue *= config.globalScaleFactor;
	return std::abs(ratioValue) != 0.0f;
}

bool computeHostOsemRatioValues(OperatorProjectorMetalProfile* profile,
                                const std::vector<bin_t>& bins,
                                const std::vector<float>& estimates,
                                const ProjectionData& measurements,
                                const Corrector_CPU& corrector,
                                const OperatorProjectorMetalOsemConfig& config,
                                std::vector<float>& ratioValues,
                                std::size_t& nonzeroCount)
{
	if (bins.size() != estimates.size())
	{
		return false;
	}

	ratioValues.assign(estimates.size(), 0.0f);
	const auto ratioStart = Clock::now();
	const int numThreads = globals::getNumThreads();
	std::vector<std::size_t> nonzeroCounts(numThreads, 0);
	util::parallelForChunked(
	    estimates.size(), numThreads,
	    [&](std::size_t i, int tid)
	    {
		    float ratioValue = 0.0f;
		    if (computeHostOsemRatioValue(measurements, corrector, config,
		                                  bins[i], estimates[i], ratioValue))
		    {
			    nonzeroCounts[tid] += 1;
		    }
		    ratioValues[i] = ratioValue;
	    });
	nonzeroCount = 0;
	for (const std::size_t count : nonzeroCounts)
	{
		nonzeroCount += count;
	}
	if (profile != nullptr)
	{
		profile->ratioPackSeconds +=
		    getElapsedSeconds(ratioStart, Clock::now());
		if (profileRatioNonzeroDiagnostic())
		{
			profile->ratioValues += ratioValues.size();
			profile->ratioNonzeroValues += nonzeroCount;
			profile->ratioZeroValues += ratioValues.size() - nonzeroCount;
			profile->ratioNonzeroDiagnosticBatches += 1;
		}
	}
	return true;
}

bool computeAndUploadHostOsemRatioToBatch(
    OperatorProjectorMetalProfile* profile, ProjectionBatchMetal& batch,
    const std::vector<bin_t>& bins, const ProjectionData& measurements,
    const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    std::size_t& nonzeroCount)
{
	std::vector<float> estimates;
	const auto downloadStart = Clock::now();
	const bool didDownload = batch.copyProjectionValuesToHost(estimates);
	if (profile != nullptr)
	{
		profile->forwardDownloadSeconds +=
		    getElapsedSeconds(downloadStart, Clock::now());
	}
	if (!didDownload || estimates.size() != bins.size())
	{
		return false;
	}

	std::vector<float> ratioValues;
	if (!computeHostOsemRatioValues(profile, bins, estimates, measurements,
	                                corrector, config, ratioValues,
	                                nonzeroCount))
	{
		return false;
	}
	if (nonzeroCount == 0)
	{
		return true;
	}

	const auto uploadStart = Clock::now();
	const bool didUpload = batch.setProjectionValues(ratioValues);
	addBatchUploadProfile(profile, false, false,
	                      getElapsedSeconds(uploadStart, Clock::now()));
	return didUpload;
}

CachedOsemCorrections* findCachedCorrectionsForFrame(
    std::vector<CachedCorrectionBatch>* correctionBatches, frame_t frame,
    const std::vector<bin_t>& bins, const ProjectionData& measurements,
    const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config)
{
	if (correctionBatches == nullptr)
	{
		return nullptr;
	}

	for (CachedCorrectionBatch& correctionBatch : *correctionBatches)
	{
		if (correctionBatch.frame == frame && correctionBatch.bins == bins &&
		    cachedCorrectionsMatch(correctionBatch.osemCorrections,
		                           measurements, corrector, config,
		                           bins.size()))
		{
			return &correctionBatch.osemCorrections;
		}
	}
	return nullptr;
}

std::array<std::vector<std::size_t>, 3> groupIndicesByJosephAxis(
    const std::vector<BridgeEvent>& events,
    const std::vector<std::size_t>& indices,
    const SiddonForwardImageParams& imageParams)
{
	std::array<std::vector<std::size_t>, 3> axisIndices;
	for (const std::size_t index : indices)
	{
		axisIndices[josephMajorAxis(events[index].line, imageParams)]
		    .push_back(index);
	}
	return axisIndices;
}

bool forwardProjectEventsWithImageBuffer(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& image, ForwardImageResources& imageResources,
    ProjectionData& projectionData, const std::vector<BridgeEvent>& events,
    OperatorProjectorMetalKernel projectorKernel)
{
	if (events.empty())
	{
		return true;
	}

	const SiddonProjectorMetal metalProjector(context);
	std::vector<float> results(events.size(), 0.0f);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		const auto packStart = Clock::now();
		std::vector<ProjectionLineEndpoints> lines;
		lines.reserve(indices.size());
		for (const std::size_t index : indices)
		{
			lines.push_back(events[index].line);
		}
		addPackProfile(profile, true, false,
		               getElapsedSeconds(packStart, Clock::now()));

		const auto batchStart = Clock::now();
		auto batch = metalProjector.makeBatch(
		    lines, std::vector<float>(lines.size(), 0.0f));
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
		}
		addBatchUploadProfile(profile, true, false,
		                      getElapsedSeconds(batchStart, Clock::now()));

		auto kernelProfile = makeSiddonKernelProfile(profile);
		std::vector<float> frameResults;
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    batch.isValid() &&
		    makeSiddonForwardImageParams(
		        image, static_cast<std::uint32_t>(frame), imageParams) &&
		    forwardProjectSingleRay(
		        projectorKernel, context, imageResources, batch, imageParams,
		        profile != nullptr ? &kernelProfile : nullptr);
		addForwardKernelProfile(profile, kernelProfile);

		const auto downloadStart = Clock::now();
		const bool didDownload =
		    didForward && batch.copyProjectionValuesToHost(frameResults);
		if (profile != nullptr)
		{
			profile->forwardDownloadSeconds +=
			    getElapsedSeconds(downloadStart, Clock::now());
		}

		if (!didForward || !didDownload ||
		    frameResults.size() != indices.size())
		{
			return false;
		}

		for (std::size_t i = 0; i < indices.size(); ++i)
		{
			results[indices[i]] = frameResults[i];
		}
	}

	const auto hostWriteStart = Clock::now();
	for (std::size_t i = 0; i < events.size(); ++i)
	{
		projectionData.setProjectionValue(events[i].bin, results[i]);
	}
	if (profile != nullptr)
	{
		profile->forwardHostWriteSeconds +=
		    getElapsedSeconds(hostWriteStart, Clock::now());
		profile->forwardEvents += events.size();
	}
	return true;
}

bool backProjectEventsWithImageBuffer(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const ProjectionData& projectionData, Image& workingImage,
    Buffer& imageBuffer, const std::vector<BridgeEvent>& events,
    OperatorProjectorMetalKernel projectorKernel)
{
	if (events.empty())
	{
		return true;
	}

	const SiddonProjectorMetal metalProjector(context);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		SiddonForwardImageParams imageParams{};
		if (!makeSiddonForwardImageParams(
		        workingImage, static_cast<std::uint32_t>(frame), imageParams))
		{
			return false;
		}
		const auto orderedIndices =
		    makeOrderedAdjointIndices(events, indices, &imageParams);
		const auto packStart = Clock::now();
		std::vector<ProjectionLineEndpoints> lines;
		std::vector<float> projectionValues;
		lines.reserve(orderedIndices.size());
		projectionValues.reserve(orderedIndices.size());
		for (const std::size_t index : orderedIndices)
		{
			if (std::abs(events[index].projectionValue) == 0.0f)
			{
				continue;
			}
			lines.push_back(events[index].line);
			projectionValues.push_back(events[index].projectionValue);
		}
		addPackProfile(profile, false, false,
		               getElapsedSeconds(packStart, Clock::now()));
		if (profile != nullptr)
		{
			profile->adjointNonzeroEvents += lines.size();
		}
		if (lines.empty())
		{
			continue;
		}

		const auto batchStart = Clock::now();
		auto batch = metalProjector.makeBatch(lines, projectionValues);
		if (profile != nullptr)
		{
			profile->adjointBatches += 1;
		}
		addBatchUploadProfile(profile, false, false,
		                      getElapsedSeconds(batchStart, Clock::now()));

		auto kernelProfile = makeAdjointSiddonKernelProfile(profile);
		const bool didBackProject =
		    batch.isValid() &&
		    backProjectSingleRay(projectorKernel, context, batch, imageBuffer,
		                         imageParams,
		                         profile != nullptr ? &kernelProfile : nullptr);
		addAdjointKernelProfile(profile, kernelProfile);
		if (!didBackProject)
		{
			return false;
		}
	}

	if (profile != nullptr)
	{
		profile->adjointEvents += events.size();
	}
	(void)projectionData;
	return true;
}

bool applyOsemEventsWithImageBuffers(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, ForwardImageResources& inputImageResources,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    const std::vector<BridgeEvent>& events)
{
	if (events.empty())
	{
		return true;
	}

	const auto kernelOptions = makeProjectorKernelOptions(&config);
	const SiddonProjectorMetal metalProjector(context);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		SiddonForwardImageParams updateImageParamsForOrder{};
		if (!makeSiddonForwardImageParams(workingImage,
		        static_cast<std::uint32_t>(frame), updateImageParamsForOrder))
		{
			return false;
		}
		const auto orderedIndices =
		    makeOrderedAdjointIndices(events, indices,
		        &updateImageParamsForOrder);
		const auto packStart = Clock::now();
		std::vector<ProjectionLineEndpoints> lines;
		std::vector<bin_t> bins;
		lines.reserve(orderedIndices.size());
		bins.reserve(orderedIndices.size());
		for (const std::size_t index : orderedIndices)
		{
			lines.push_back(events[index].line);
			bins.push_back(events[index].bin);
		}
		addPackProfile(profile, true, false,
		               getElapsedSeconds(packStart, Clock::now()));

		const auto batchStart = Clock::now();
		auto batch = metalProjector.makeBatch(
		    lines, std::vector<float>(lines.size(), 0.0f));
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
		}
		addBatchUploadProfile(profile, true, false,
		                      getElapsedSeconds(batchStart, Clock::now()));

		auto kernelProfile = makeSiddonKernelProfile(profile);
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    batch.isValid() &&
		    makeSiddonForwardImageParams(
		        inputImage, static_cast<std::uint32_t>(frame), imageParams) &&
		    forwardProjectSingleRay(
		        config.projectorKernel, context, inputImageResources, batch,
		        imageParams, profile != nullptr ? &kernelProfile : nullptr,
		        kNoJosephAxis, &kernelOptions);
		addForwardKernelProfile(profile, kernelProfile);
		if (!didForward ||
		    !applyOsemRatioToBatch(context, profile, batch, bins, measurements,
		                           corrector, config))
		{
			return false;
		}

		kernelProfile = makeAdjointSiddonKernelProfile(profile);
		SiddonForwardImageParams updateImageParams{};
		const bool didBackProject =
		    makeSiddonForwardImageParams(workingImage,
		                                 static_cast<std::uint32_t>(frame),
		                                 updateImageParams) &&
		    backProjectSingleRay(config.projectorKernel, context, batch,
		                         updateImageBuffer, updateImageParams,
		                         profile != nullptr ? &kernelProfile : nullptr,
		                         kNoJosephAxis, &kernelOptions);
		if (profile != nullptr)
		{
			profile->adjointBatches += 1;
			profile->adjointNonzeroEvents += bins.size();
			addAdjointKernelProfile(profile, kernelProfile);
		}
		if (!didBackProject)
		{
			return false;
		}
	}

	if (profile != nullptr)
	{
		profile->forwardEvents += events.size();
		profile->adjointEvents += events.size();
	}
	return true;
}

bool buildCachedSegment(const Context& context,
                        OperatorProjectorMetalProfile* profile,
                        const Image& image,
                        const std::vector<BridgeEvent>& events,
                        std::size_t offset, std::size_t spanSize,
                        bool profileAsForward,
                        const ProjectionData* measurementsForCorrections,
                        const Corrector_CPU* correctorForCorrections,
                        const OperatorProjectorMetalOsemConfig* osemConfig,
                        CachedIteratorSegment& segment)
{
	segment = CachedIteratorSegment{};
	segment.offset = offset;
	segment.spanSize = spanSize;
	segment.eventCount = events.size();
	segment.byteCount = estimateCachedBytes(events.size(), osemConfig);
	segment.cached = true;
	if (events.empty())
	{
		return true;
	}

	for (const BridgeEvent& event : events)
	{
		if (event.frame < 0)
		{
			segment.zeroProjectionBins.push_back(event.bin);
		}
	}

	const SiddonProjectorMetal metalProjector(context);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		SiddonForwardImageParams imageParams{};
		if (!makeSiddonForwardImageParams(
		        image, static_cast<std::uint32_t>(frame), imageParams))
		{
			return false;
		}
		const auto admissionPackStart = Clock::now();
		const auto orderedIndices =
		    makeOrderedAdjointIndices(events, indices, &imageParams);
		const auto packStart = Clock::now();
		const bool splitByJosephAxis =
		    osemConfig != nullptr &&
		    shouldSplitJosephAxisBatches(osemConfig->projectorKernel);
		std::array<std::vector<std::size_t>, 3> axisIndices;
		if (splitByJosephAxis)
		{
			axisIndices =
			    groupIndicesByJosephAxis(events, orderedIndices, imageParams);
		}
		std::vector<ProjectionLineEndpoints> lines;
		std::vector<bin_t> bins;
		if (!splitByJosephAxis)
		{
			lines.reserve(orderedIndices.size());
			bins.reserve(orderedIndices.size());
			for (const std::size_t index : orderedIndices)
			{
				lines.push_back(events[index].line);
				bins.push_back(events[index].bin);
			}
		}
		if (profile != nullptr)
		{
			profile->cacheAdmissionPackSeconds +=
			    getElapsedSeconds(admissionPackStart, Clock::now());
		}
		addPackProfile(profile, profileAsForward, true,
		               getElapsedSeconds(packStart, Clock::now()));

		auto appendFrameBatch =
		    [&](std::vector<ProjectionLineEndpoints>&& batchLines,
		        std::vector<bin_t>&& batchBins,
		        std::uint32_t josephAxis) -> bool
		{
			const auto batchStart = Clock::now();
			auto batch = metalProjector.makeBatch(
			    batchLines, std::vector<float>(batchLines.size(), 0.0f));
			const double batchUploadSeconds =
			    getElapsedSeconds(batchStart, Clock::now());
			if (profile != nullptr)
			{
				profile->cacheAdmissionBatchUploadSeconds +=
				    batchUploadSeconds;
			}
			addBatchUploadProfile(profile, profileAsForward, true,
			                      batchUploadSeconds);
			if (!batch.isValid())
			{
				return false;
			}

			CachedFrameBatch frameBatch;
			frameBatch.frame = frame;
			frameBatch.bins = std::move(batchBins);
			frameBatch.batch = std::move(batch);
			frameBatch.josephAxis = josephAxis;
			if (osemConfig != nullptr && osemConfig->cacheCorrectionFactors)
			{
				if (measurementsForCorrections == nullptr ||
				    correctorForCorrections == nullptr ||
				    !buildCachedOsemCorrections(
				        context, profile, *measurementsForCorrections,
				        *correctorForCorrections, frameBatch.bins, *osemConfig,
				        frameBatch.osemCorrections))
				{
					return false;
				}
			}
			segment.frameBatches.push_back(std::move(frameBatch));
			return true;
		};

		if (!splitByJosephAxis)
		{
			if (!appendFrameBatch(std::move(lines), std::move(bins),
			                      kNoJosephAxis))
			{
				return false;
			}
			continue;
		}

		for (std::uint32_t axis = 0; axis < axisIndices.size(); ++axis)
		{
			const auto& sourceIndices = axisIndices[axis];
			if (sourceIndices.empty())
			{
				continue;
			}
			std::vector<ProjectionLineEndpoints> axisLines;
			std::vector<bin_t> axisBins;
			axisLines.reserve(sourceIndices.size());
			axisBins.reserve(sourceIndices.size());
			for (const std::size_t index : sourceIndices)
			{
				axisLines.push_back(events[index].line);
				axisBins.push_back(events[index].bin);
			}
			if (!appendFrameBatch(std::move(axisLines), std::move(axisBins),
			                      axis))
			{
				return false;
			}
		}
	}
	return true;
}

CachedIteratorSegment makeLazyCorrectionCachedSegment(
    std::size_t offset, std::size_t spanSize, std::size_t byteCount)
{
	CachedIteratorSegment segment;
	segment.offset = offset;
	segment.spanSize = spanSize;
	segment.eventCount = spanSize;
	segment.byteCount = byteCount;
	segment.cached = false;
	segment.correctionOnlyCached = true;
	segment.correctionOnlyCacheBuilt = false;
	return segment;
}

bool forwardProjectCachedSegmentWithImageBuffer(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& image, ForwardImageResources& imageResources,
    ProjectionData& projectionData, CachedIteratorSegment& segment,
    OperatorProjectorMetalKernel projectorKernel)
{
	if (segment.eventCount == 0)
	{
		return true;
	}

	if (!segment.zeroProjectionBins.empty())
	{
		const auto hostWriteStart = Clock::now();
		for (const bin_t bin : segment.zeroProjectionBins)
		{
			projectionData.setProjectionValue(bin, 0.0f);
		}
		if (profile != nullptr)
		{
			profile->forwardHostWriteSeconds +=
			    getElapsedSeconds(hostWriteStart, Clock::now());
		}
	}

	for (CachedFrameBatch& frameBatch : segment.frameBatches)
	{
		auto kernelProfile = makeSiddonKernelProfile(profile);
		std::vector<float> frameResults;
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    frameBatch.batch.isValid() &&
		    makeSiddonForwardImageParams(
		        image, static_cast<std::uint32_t>(frameBatch.frame),
		        imageParams) &&
		    forwardProjectSingleRay(
		        projectorKernel, context, imageResources, frameBatch.batch,
		        imageParams, profile != nullptr ? &kernelProfile : nullptr,
		        canUseJosephAxisSpecializedForward(projectorKernel) ?
		            frameBatch.josephAxis :
		            kNoJosephAxis);
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
			addForwardKernelProfile(profile, kernelProfile);
		}

		const auto downloadStart = Clock::now();
		const bool didDownload =
		    didForward &&
		    frameBatch.batch.copyProjectionValuesToHost(frameResults);
		if (profile != nullptr)
		{
			profile->forwardDownloadSeconds +=
			    getElapsedSeconds(downloadStart, Clock::now());
		}
		if (!didForward || !didDownload ||
		    frameResults.size() != frameBatch.bins.size())
		{
			return false;
		}

		const auto hostWriteStart = Clock::now();
		for (std::size_t i = 0; i < frameBatch.bins.size(); ++i)
		{
			projectionData.setProjectionValue(frameBatch.bins[i],
			                                  frameResults[i]);
		}
		if (profile != nullptr)
		{
			profile->forwardHostWriteSeconds +=
			    getElapsedSeconds(hostWriteStart, Clock::now());
		}
	}
	if (profile != nullptr)
	{
		profile->forwardEvents += segment.eventCount;
	}
	return true;
}

bool backProjectCachedSegmentWithImageBuffer(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const ProjectionData& projectionData, Image& workingImage,
    Buffer& imageBuffer, CachedIteratorSegment& segment,
    OperatorProjectorMetalKernel projectorKernel)
{
	if (segment.eventCount == 0)
	{
		return true;
	}

	for (CachedFrameBatch& frameBatch : segment.frameBatches)
	{
		const auto packStart = Clock::now();
		std::vector<float> projectionValues;
		projectionValues.reserve(frameBatch.bins.size());
		std::size_t nonzeroCount = 0;
		for (const bin_t bin : frameBatch.bins)
		{
			const float projectionValue =
			    projectionData.getProjectionValue(bin);
			projectionValues.push_back(projectionValue);
			if (std::abs(projectionValue) != 0.0f)
			{
				nonzeroCount += 1;
			}
		}
		addPackProfile(profile, false, false,
		               getElapsedSeconds(packStart, Clock::now()));
		if (profile != nullptr)
		{
			profile->adjointNonzeroEvents += nonzeroCount;
		}
		if (nonzeroCount == 0)
		{
			continue;
		}

		const auto uploadStart = Clock::now();
		const bool didUpload =
		    frameBatch.batch.setProjectionValues(projectionValues);
		if (profile != nullptr)
		{
			profile->adjointBatches += 1;
		}
		addBatchUploadProfile(profile, false, false,
		                      getElapsedSeconds(uploadStart, Clock::now()));
		if (!didUpload)
		{
			return false;
		}

		auto kernelProfile = makeAdjointSiddonKernelProfile(profile);
		SiddonForwardImageParams imageParams{};
		const bool didBackProject =
		    makeSiddonForwardImageParams(
		        workingImage, static_cast<std::uint32_t>(frameBatch.frame),
		        imageParams) &&
		    backProjectSingleRay(projectorKernel, context, frameBatch.batch,
		                         imageBuffer, imageParams,
		                         profile != nullptr ? &kernelProfile : nullptr,
		                         canUseJosephAxisSpecializedAdjoint(projectorKernel) ?
		                             frameBatch.josephAxis :
		                             kNoJosephAxis);
		addAdjointKernelProfile(profile, kernelProfile);
		if (!didBackProject)
		{
			return false;
		}
	}
	if (profile != nullptr)
	{
		profile->adjointEvents += segment.eventCount;
	}
	return true;
}

bool applyOsemCachedSegmentWithImageBuffers(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, ForwardImageResources& inputImageResources,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    CachedIteratorSegment& segment)
{
	if (segment.eventCount == 0)
	{
		return true;
	}

	const auto kernelOptions = makeProjectorKernelOptions(&config);
	for (CachedFrameBatch& frameBatch : segment.frameBatches)
	{
		auto kernelProfile = makeSiddonKernelProfile(profile);
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    frameBatch.batch.isValid() &&
		    makeSiddonForwardImageParams(
		        inputImage, static_cast<std::uint32_t>(frameBatch.frame),
		        imageParams) &&
		    forwardProjectSingleRay(
		        config.projectorKernel, context, inputImageResources,
		        frameBatch.batch, imageParams,
		        profile != nullptr ? &kernelProfile : nullptr,
		        canUseJosephAxisSpecializedForward(config.projectorKernel) ?
		            frameBatch.josephAxis :
		            kNoJosephAxis,
		        &kernelOptions);
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
			addForwardKernelProfile(profile, kernelProfile);
		}
		if (!didForward ||
		    !applyOsemRatioToBatch(context, profile, frameBatch.batch,
		                           frameBatch.bins, measurements, corrector,
		                           config, &frameBatch.osemCorrections))
		{
			return false;
		}

		kernelProfile = makeAdjointSiddonKernelProfile(profile);
		SiddonForwardImageParams updateImageParams{};
		const bool didBackProject =
		    makeSiddonForwardImageParams(
		        workingImage, static_cast<std::uint32_t>(frameBatch.frame),
		        updateImageParams) &&
		    backProjectSingleRay(config.projectorKernel, context,
		                         frameBatch.batch, updateImageBuffer,
		                         updateImageParams,
		                         profile != nullptr ? &kernelProfile : nullptr,
		                         canUseJosephAxisSpecializedAdjoint(config.projectorKernel) ?
		                             frameBatch.josephAxis :
		                             kNoJosephAxis,
		                         &kernelOptions);
		if (profile != nullptr)
		{
			profile->adjointBatches += 1;
			profile->adjointNonzeroEvents += frameBatch.bins.size();
			addAdjointKernelProfile(profile, kernelProfile);
		}
		if (!didBackProject)
		{
			return false;
		}
	}
	if (profile != nullptr)
	{
		profile->forwardEvents += segment.eventCount;
		profile->adjointEvents += segment.eventCount;
	}
	return true;
}

bool applyOsemHostRatioEventsWithImageBuffers(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, ForwardImageResources& inputImageResources,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    const std::vector<BridgeEvent>& events,
    std::vector<CachedCorrectionBatch>* correctionBatches = nullptr,
    bool buildMissingCorrectionBatches = false)
{
	if (events.empty())
	{
		return true;
	}

	const auto kernelOptions = makeProjectorKernelOptions(&config);
	const SiddonProjectorMetal metalProjector(context);
	for (const auto& [frame, indices] : groupNonNegativeFrames(events))
	{
		SiddonForwardImageParams updateImageParamsForOrder{};
		if (!makeSiddonForwardImageParams(workingImage,
		        static_cast<std::uint32_t>(frame), updateImageParamsForOrder))
		{
			return false;
		}
		const auto orderedIndices =
		    makeOrderedAdjointIndices(events, indices,
		        &updateImageParamsForOrder);
		const auto packStart = Clock::now();
		std::vector<ProjectionLineEndpoints> lines;
		std::vector<bin_t> bins;
		lines.reserve(orderedIndices.size());
		bins.reserve(orderedIndices.size());
		for (const std::size_t index : orderedIndices)
		{
			lines.push_back(events[index].line);
			bins.push_back(events[index].bin);
		}
		addPackProfile(profile, true, false,
		               getElapsedSeconds(packStart, Clock::now()));

		const auto batchStart = Clock::now();
		auto batch = metalProjector.makeBatch(
		    lines, std::vector<float>(lines.size(), 0.0f));
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
		}
		addBatchUploadProfile(profile, true, false,
		                      getElapsedSeconds(batchStart, Clock::now()));

		auto kernelProfile = makeSiddonKernelProfile(profile);
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    batch.isValid() &&
		    makeSiddonForwardImageParams(
		        inputImage, static_cast<std::uint32_t>(frame), imageParams) &&
		    forwardProjectSingleRay(
		        config.projectorKernel, context, inputImageResources, batch,
		        imageParams, profile != nullptr ? &kernelProfile : nullptr,
		        kNoJosephAxis, &kernelOptions);
		if (profile != nullptr)
		{
			addForwardKernelProfile(profile, kernelProfile);
		}

		if (!didForward)
		{
			return false;
		}

		CachedOsemCorrections* cachedCorrections =
		    findCachedCorrectionsForFrame(correctionBatches, frame, bins,
		                                  measurements, corrector, config);
		if (cachedCorrections == nullptr && buildMissingCorrectionBatches &&
		    correctionBatches != nullptr && config.cacheCorrectionFactors)
		{
			CachedCorrectionBatch correctionBatch;
			correctionBatch.frame = frame;
			correctionBatch.bins = bins;
			if (!buildCachedOsemCorrections(
			        context, profile, measurements, corrector,
			        correctionBatch.bins, config,
			        correctionBatch.osemCorrections, false))
			{
				return false;
			}
			correctionBatches->push_back(std::move(correctionBatch));
			cachedCorrections =
			    &correctionBatches->back().osemCorrections;
		}
		if (cachedCorrections != nullptr)
		{
			if (!applyOsemRatioToBatch(context, profile, batch, bins,
			                           measurements, corrector, config,
			                           cachedCorrections))
			{
				return false;
			}
			if (profile != nullptr)
			{
				profile->adjointBatches += 1;
				profile->adjointNonzeroEvents += batch.size();
			}
		}
		else
		{
			if (config.cacheCorrectionFactors && correctionBatches != nullptr &&
			    profile != nullptr)
			{
				profile->ratioCorrectionCacheMisses += 1;
			}
			if (config.cacheCorrectionFactors && correctionBatches != nullptr)
			{
				std::size_t nonzeroCount = 0;
				if (!computeAndUploadHostOsemRatioToBatch(
				        profile, batch, bins, measurements, corrector, config,
				        nonzeroCount))
				{
					return false;
				}
				if (nonzeroCount == 0)
				{
					continue;
				}
				if (profile != nullptr)
				{
					profile->adjointBatches += 1;
					profile->adjointNonzeroEvents += nonzeroCount;
				}
			}
			else
			{
				std::size_t nonzeroCount = 0;
				if (!computeAndUploadHostOsemRatioToBatch(
				        profile, batch, bins, measurements, corrector, config,
				        nonzeroCount))
				{
					return false;
				}
				if (nonzeroCount == 0)
				{
					continue;
				}

				if (profile != nullptr)
				{
					profile->adjointBatches += 1;
					profile->adjointNonzeroEvents += nonzeroCount;
				}
			}
		}

		kernelProfile = makeAdjointSiddonKernelProfile(profile);
		SiddonForwardImageParams updateImageParams{};
		const bool didBackProject =
		    makeSiddonForwardImageParams(workingImage,
		                                 static_cast<std::uint32_t>(frame),
		                                 updateImageParams) &&
		    backProjectSingleRay(config.projectorKernel, context, batch,
		                         updateImageBuffer, updateImageParams,
		                         profile != nullptr ? &kernelProfile : nullptr,
		                         kNoJosephAxis, &kernelOptions);
		if (profile != nullptr)
		{
			addAdjointKernelProfile(profile, kernelProfile);
		}
		if (!didBackProject)
		{
			return false;
		}
	}
	if (profile != nullptr)
	{
		profile->forwardEvents += events.size();
		profile->adjointEvents += events.size();
	}
	return true;
}

bool applyOsemHostRatioFrameBatchesWithImageBuffers(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, ForwardImageResources& inputImageResources,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    const std::vector<BridgeFrameBatch>& frameBatches,
    std::size_t eventCount,
    std::vector<CachedCorrectionBatch>* correctionBatches = nullptr,
    bool buildMissingCorrectionBatches = false)
{
	if (frameBatches.empty())
	{
		return true;
	}

	const auto kernelOptions = makeProjectorKernelOptions(&config);
	const SiddonProjectorMetal metalProjector(context);
	for (const BridgeFrameBatch& frameBatch : frameBatches)
	{
		const auto batchStart = Clock::now();
		auto batch = metalProjector.makeBatch(
		    frameBatch.lines,
		    std::vector<float>(frameBatch.lines.size(), 0.0f));
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
		}
		addBatchUploadProfile(profile, true, false,
		                      getElapsedSeconds(batchStart, Clock::now()));

		auto kernelProfile = makeSiddonKernelProfile(profile);
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    batch.isValid() &&
		    makeSiddonForwardImageParams(
		        inputImage, static_cast<std::uint32_t>(frameBatch.frame),
		        imageParams) &&
		    forwardProjectSingleRay(
		        config.projectorKernel, context, inputImageResources, batch,
		        imageParams, profile != nullptr ? &kernelProfile : nullptr,
		        kNoJosephAxis, &kernelOptions);
		if (profile != nullptr)
		{
			addForwardKernelProfile(profile, kernelProfile);
		}

		if (!didForward)
		{
			return false;
		}

		CachedOsemCorrections* cachedCorrections =
		    findCachedCorrectionsForFrame(
		        correctionBatches, frameBatch.frame, frameBatch.bins,
		        measurements, corrector, config);
		if (cachedCorrections == nullptr && buildMissingCorrectionBatches &&
		    correctionBatches != nullptr && config.cacheCorrectionFactors)
		{
			CachedCorrectionBatch correctionBatch;
			correctionBatch.frame = frameBatch.frame;
			correctionBatch.bins = frameBatch.bins;
			if (!buildCachedOsemCorrections(
			        context, profile, measurements, corrector,
			        correctionBatch.bins, config,
			        correctionBatch.osemCorrections, false))
			{
				return false;
			}
			correctionBatches->push_back(std::move(correctionBatch));
			cachedCorrections =
			    &correctionBatches->back().osemCorrections;
		}
		if (cachedCorrections != nullptr)
		{
			if (!applyOsemRatioToBatch(context, profile, batch,
			                           frameBatch.bins, measurements,
			                           corrector, config,
			                           cachedCorrections))
			{
				return false;
			}
			if (profile != nullptr)
			{
				profile->adjointBatches += 1;
				profile->adjointNonzeroEvents += batch.size();
			}
		}
		else
		{
			if (config.cacheCorrectionFactors && correctionBatches != nullptr &&
			    profile != nullptr)
			{
				profile->ratioCorrectionCacheMisses += 1;
			}
			std::size_t nonzeroCount = 0;
			if (!computeAndUploadHostOsemRatioToBatch(
			        profile, batch, frameBatch.bins, measurements, corrector,
			        config, nonzeroCount))
			{
				return false;
			}
			if (nonzeroCount == 0)
			{
				continue;
			}
			if (profile != nullptr)
			{
				profile->adjointBatches += 1;
				profile->adjointNonzeroEvents += nonzeroCount;
			}
		}

		kernelProfile = makeAdjointSiddonKernelProfile(profile);
		SiddonForwardImageParams updateImageParams{};
		const bool didBackProject =
		    makeSiddonForwardImageParams(workingImage,
		                                 static_cast<std::uint32_t>(
		                                     frameBatch.frame),
		                                 updateImageParams) &&
		    backProjectSingleRay(config.projectorKernel, context, batch,
		                         updateImageBuffer, updateImageParams,
		                         profile != nullptr ? &kernelProfile : nullptr,
		                         kNoJosephAxis, &kernelOptions);
		if (profile != nullptr)
		{
			addAdjointKernelProfile(profile, kernelProfile);
		}
		if (!didBackProject)
		{
			return false;
		}
	}
	if (profile != nullptr)
	{
		profile->forwardEvents += eventCount;
		profile->adjointEvents += eventCount;
	}
	return true;
}

bool applyOsemHostRatioBinIteratorWithImageBuffers(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, ForwardImageResources& inputImageResources,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    const BinIterator& batchIterator, const BinLoader& binLoader,
    OperatorProjectorMetalCache* cache,
    std::vector<CachedCorrectionBatch>* correctionBatches = nullptr,
    bool buildMissingCorrectionBatches = false)
{
	if (useDirectHostRatioFrameBatches(config) &&
	    getAdjointEventOrder() == AdjointEventOrder::None)
	{
		const auto gatherStart = Clock::now();
		std::vector<BridgeFrameBatch> frameBatches;
		bool handled = false;
		if (!gatherBridgeFrameBatches(measurements, inputImage, batchIterator,
		                              binLoader, cache, frameBatches,
		                              handled))
		{
			return false;
		}
		if (handled)
		{
			addGatherProfile(profile, binLoader, true, false,
			                 getElapsedSeconds(gatherStart, Clock::now()));
			if (profile != nullptr)
			{
				profile->uncachedBatches += 1;
			}
			return applyOsemHostRatioFrameBatchesWithImageBuffers(
			    context, profile, inputImage, inputImageResources, workingImage,
			    updateImageBuffer, measurements, corrector, config,
			    frameBatches, batchIterator.size(), correctionBatches,
			    buildMissingCorrectionBatches);
		}
	}

	const auto gatherStart = Clock::now();
	std::vector<BridgeEvent> events;
	if (!gatherBridgeEvents(measurements, inputImage, batchIterator, binLoader,
	                        cache, false, events))
	{
		return false;
	}
	addGatherProfile(profile, binLoader, true, false,
	                 getElapsedSeconds(gatherStart, Clock::now()));
	if (profile != nullptr)
	{
		profile->uncachedBatches += 1;
	}
	return applyOsemHostRatioEventsWithImageBuffers(
	    context, profile, inputImage, inputImageResources, workingImage,
	    updateImageBuffer, measurements, corrector, config, events,
	    correctionBatches, buildMissingCorrectionBatches);
}

bool applyOsemHostRatioCachedSegmentWithImageBuffers(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, ForwardImageResources& inputImageResources,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    CachedIteratorSegment& segment)
{
	if (segment.eventCount == 0)
	{
		return true;
	}

	const auto kernelOptions = makeProjectorKernelOptions(&config);
	for (CachedFrameBatch& frameBatch : segment.frameBatches)
	{
		auto kernelProfile = makeSiddonKernelProfile(profile);
		SiddonForwardImageParams imageParams{};
		const bool didForward =
		    frameBatch.batch.isValid() &&
		    makeSiddonForwardImageParams(
		        inputImage, static_cast<std::uint32_t>(frameBatch.frame),
		        imageParams) &&
		    forwardProjectSingleRay(
		        config.projectorKernel, context, inputImageResources,
		        frameBatch.batch, imageParams,
		        profile != nullptr ? &kernelProfile : nullptr,
		        canUseJosephAxisSpecializedForward(config.projectorKernel) ?
		            frameBatch.josephAxis :
		            kNoJosephAxis,
		        &kernelOptions);
		if (profile != nullptr)
		{
			profile->forwardBatches += 1;
			addForwardKernelProfile(profile, kernelProfile);
		}

		if (!didForward)
		{
			return false;
		}

		const bool useCachedCorrections =
		    cachedCorrectionsMatch(frameBatch.osemCorrections, measurements,
		                           corrector, config, frameBatch.bins.size());
		if (useCachedCorrections)
		{
			if (!applyOsemRatioToBatch(context, profile, frameBatch.batch,
			                           frameBatch.bins, measurements, corrector,
			                           config, &frameBatch.osemCorrections))
			{
				return false;
			}
			if (profile != nullptr)
			{
				profile->adjointBatches += 1;
				profile->adjointNonzeroEvents += frameBatch.batch.size();
			}
		}
		else
		{
			if (config.cacheCorrectionFactors && profile != nullptr)
			{
				profile->ratioCorrectionCacheMisses += 1;
			}
			if (config.cacheCorrectionFactors)
			{
				std::size_t nonzeroCount = 0;
				if (!computeAndUploadHostOsemRatioToBatch(
				        profile, frameBatch.batch, frameBatch.bins,
				        measurements, corrector, config, nonzeroCount))
				{
					return false;
				}
				if (nonzeroCount == 0)
				{
					continue;
				}
				if (profile != nullptr)
				{
					profile->adjointBatches += 1;
					profile->adjointNonzeroEvents += nonzeroCount;
				}
			}
			else
			{
				std::size_t nonzeroCount = 0;
				if (!computeAndUploadHostOsemRatioToBatch(
				        profile, frameBatch.batch, frameBatch.bins,
				        measurements, corrector, config, nonzeroCount))
				{
					return false;
				}
				if (nonzeroCount == 0)
				{
					continue;
				}

				if (profile != nullptr)
				{
					profile->adjointBatches += 1;
					profile->adjointNonzeroEvents += nonzeroCount;
				}
			}
		}

		kernelProfile = makeAdjointSiddonKernelProfile(profile);
		SiddonForwardImageParams updateImageParams{};
		const bool didBackProject =
		    makeSiddonForwardImageParams(
		        workingImage, static_cast<std::uint32_t>(frameBatch.frame),
		        updateImageParams) &&
		    backProjectSingleRay(config.projectorKernel, context,
		                         frameBatch.batch, updateImageBuffer,
		                         updateImageParams,
		                         profile != nullptr ? &kernelProfile : nullptr,
		                         canUseJosephAxisSpecializedAdjoint(config.projectorKernel) ?
		                             frameBatch.josephAxis :
		                             kNoJosephAxis,
		                         &kernelOptions);
		addAdjointKernelProfile(profile, kernelProfile);
		if (!didBackProject)
		{
			return false;
		}
	}
	if (profile != nullptr)
	{
		profile->forwardEvents += segment.eventCount;
		profile->adjointEvents += segment.eventCount;
	}
	return true;
}

CachedIteratorSegment makeUncachedSegment(std::size_t offset,
                                          std::size_t spanSize)
{
	CachedIteratorSegment segment;
	segment.offset = offset;
	segment.spanSize = spanSize;
	segment.cached = false;
	return segment;
}

std::size_t getBatchSize(const OperatorProjectorMetalCache* cache,
                         std::size_t totalEvents)
{
	const std::size_t batchLimit =
	    cache != nullptr ? cache->maxBatchEvents() : kDefaultMaxBatchEvents;
	return batchLimit == 0 ?
	           totalEvents :
	           std::max<std::size_t>(1, std::min(batchLimit, totalEvents));
}

bool canReserveCacheBytes(const OperatorProjectorMetalCache& cache,
                          std::size_t pendingBytes, std::size_t byteCount)
{
	const std::size_t maxBytes = cache.maxBytes();
	const std::size_t usedBytes = cache.usedBytes();
	if (byteCount == 0 || byteCount > maxBytes || usedBytes > maxBytes ||
	    pendingBytes > maxBytes - usedBytes)
	{
		return false;
	}
	return byteCount <= maxBytes - usedBytes - pendingBytes;
}

bool canReserveFullCacheBytes(const OperatorProjectorMetalCache& cache,
                              std::size_t pendingBytes, std::size_t byteCount)
{
	if (!canReserveCacheBytes(cache, pendingBytes, byteCount))
	{
		return false;
	}

	const std::size_t reserveBytes =
	    std::min(cache.correctionCacheReserveBytes(), cache.maxBytes());
	if (reserveBytes == 0)
	{
		return true;
	}

	const std::size_t maxFullBytes = cache.maxBytes() - reserveBytes;
	const std::size_t usedBytes = cache.usedBytes();
	if (byteCount == 0 || byteCount > maxFullBytes ||
	    usedBytes > maxFullBytes || pendingBytes > maxFullBytes - usedBytes)
	{
		return false;
	}
	return byteCount <= maxFullBytes - usedBytes - pendingBytes;
}

std::size_t countCachedSegments(const CachedBinIteratorEntry& entry)
{
	std::size_t count = 0;
	for (const CachedIteratorSegment& segment : entry.segments)
	{
		if (segment.cached && segment.eventCount > 0)
		{
			count += 1;
		}
	}
	return count;
}

bool buildCachedEntry(const Context& context,
                      OperatorProjectorMetalProfile* profile,
                      const ProjectionData& projectionData, const Image& image,
                      const BinIterator& binIterator,
                      const BinLoader& binLoader,
                      OperatorProjectorMetalCache& cache, bool profileAsForward,
                      const ProjectionData* measurementsForCorrections,
                      const Corrector_CPU* correctorForCorrections,
                      const OperatorProjectorMetalOsemConfig* osemConfig,
                      CachedBinIteratorEntry& entry, bool& skippedOverBudget)
{
	entry = CachedBinIteratorEntry{};
	entry.eventCount = binIterator.size();
	skippedOverBudget = false;
	if (entry.eventCount == 0)
	{
		return true;
	}

	const bool canCacheCorrectionsOnly =
	    osemConfig != nullptr && osemConfig->cacheCorrectionFactors &&
	    measurementsForCorrections != nullptr &&
	    correctorForCorrections != nullptr;

	const auto gatherAndBuildSegment =
	    [&](const BinIterator& segmentIterator, std::size_t offset,
	        std::size_t spanSize, CachedIteratorSegment& segment) -> bool
	{
		const auto gatherStart = Clock::now();
		std::vector<BridgeEvent> events;
		if (!gatherBridgeEvents(projectionData, image, segmentIterator,
		                        binLoader, &cache, false, events))
		{
			return false;
		}
		const double gatherSeconds = getElapsedSeconds(gatherStart, Clock::now());
		if (profile != nullptr)
		{
			profile->cacheAdmissionGatherSeconds += gatherSeconds;
		}
		addGatherProfile(profile, binLoader, profileAsForward, true,
		                 gatherSeconds);
		return buildCachedSegment(context, profile, image, events, offset,
		    spanSize, profileAsForward, measurementsForCorrections,
		    correctorForCorrections, osemConfig, segment);
	};

	const std::size_t fullEstimate =
	    estimateCachedBytes(entry.eventCount, osemConfig);
	if (canReserveFullCacheBytes(cache, 0, fullEstimate))
	{
		CachedIteratorSegment segment;
		if (!gatherAndBuildSegment(binIterator, 0, entry.eventCount, segment))
		{
			return false;
		}
		entry.cachedEventCount += segment.eventCount;
		entry.byteCount += segment.byteCount;
		entry.segments.push_back(std::move(segment));
		return true;
	}

	const std::size_t batchSize = getBatchSize(&cache, entry.eventCount);
	for (std::size_t offset = 0; offset < entry.eventCount; offset += batchSize)
	{
		const std::size_t currentBatchSize =
		    std::min(batchSize, entry.eventCount - offset);
		const std::size_t estimatedBytes =
		    estimateCachedBytes(currentBatchSize, osemConfig);
		if (!canReserveFullCacheBytes(cache, entry.byteCount, estimatedBytes))
		{
			skippedOverBudget = true;
			const std::size_t correctionOnlyBytes =
			    estimateCorrectionOnlyCachedBytes(currentBatchSize, osemConfig);
			if (canCacheCorrectionsOnly &&
			    canReserveCacheBytes(cache, entry.byteCount,
			                         correctionOnlyBytes))
			{
				CachedIteratorSegment segment =
				    makeLazyCorrectionCachedSegment(offset, currentBatchSize,
				                                    correctionOnlyBytes);
				entry.cachedEventCount += segment.eventCount;
				entry.byteCount += segment.byteCount;
				entry.segments.push_back(std::move(segment));
			}
			else
			{
				entry.segments.push_back(
				    makeUncachedSegment(offset, currentBatchSize));
			}
			continue;
		}

		BinIteratorBatched batchIterator(&binIterator, offset,
		                                 currentBatchSize);
		CachedIteratorSegment segment;
		if (!gatherAndBuildSegment(batchIterator, offset, currentBatchSize,
		                           segment))
		{
			return false;
		}
		entry.cachedEventCount += segment.eventCount;
		entry.byteCount += segment.byteCount;
		entry.segments.push_back(std::move(segment));
	}
	return true;
}

bool forwardProjectCachedEntry(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& image, ProjectionData& projectionData,
    const BinIterator& binIterator, const BinLoader& binLoader,
    OperatorProjectorMetalCache* cache, CachedBinIteratorEntry& entry,
    OperatorProjectorMetalKernel projectorKernel)
{
	if (entry.eventCount == 0)
	{
		return true;
	}

	Buffer imageBuffer;
	SiddonProjectorKernelProfile transferProfile;
	if (!uploadSiddonImageBuffer(context, image, imageBuffer,
	                             profile != nullptr ? &transferProfile :
	                                                  nullptr))
	{
		return false;
	}
	if (profile != nullptr)
	{
		profile->forwardImageUploadSeconds +=
		    transferProfile.imageUploadSeconds;
	}

	ForwardImageResources imageResources{image, imageBuffer};
	for (CachedIteratorSegment& segment : entry.segments)
	{
		if (segment.cached)
		{
			if (!forwardProjectCachedSegmentWithImageBuffer(
			        context, profile, image, imageResources, projectionData,
			        segment, projectorKernel))
			{
				return false;
			}
			continue;
		}

		BinIteratorBatched batchIterator(&binIterator, segment.offset,
		                                 segment.spanSize);
		const auto gatherStart = Clock::now();
		std::vector<BridgeEvent> events;
		if (!gatherBridgeEvents(projectionData, image, batchIterator, binLoader,
		                        cache, false, events))
		{
			return false;
		}
		addGatherProfile(profile, binLoader, true, false,
		                 getElapsedSeconds(gatherStart, Clock::now()));
		if (profile != nullptr)
		{
			profile->uncachedBatches += 1;
		}
		if (!forwardProjectEventsWithImageBuffer(context, profile, image,
		                                         imageResources, projectionData,
		                                         events, projectorKernel))
		{
			return false;
		}
	}
	return true;
}

bool backProjectCachedEntry(const Context& context,
                            OperatorProjectorMetalProfile* profile,
                            const ProjectionData& projectionData,
                            Image& workingImage, const BinIterator& binIterator,
                            const BinLoader& binLoader,
                            OperatorProjectorMetalCache* cache,
                            CachedBinIteratorEntry& entry,
                            OperatorProjectorMetalKernel projectorKernel)
{
	if (entry.eventCount == 0)
	{
		return true;
	}

	Buffer imageBuffer;
	SiddonProjectorKernelProfile transferProfile;
	if (!uploadSiddonImageBuffer(context, workingImage, imageBuffer,
	                             profile != nullptr ? &transferProfile :
	                                                  nullptr))
	{
		return false;
	}
	if (profile != nullptr)
	{
		profile->adjointImageUploadSeconds +=
		    transferProfile.imageUploadSeconds;
	}

	for (CachedIteratorSegment& segment : entry.segments)
	{
		if (segment.cached)
		{
			if (!backProjectCachedSegmentWithImageBuffer(
			        context, profile, projectionData, workingImage, imageBuffer,
			        segment, projectorKernel))
			{
				return false;
			}
			continue;
		}

		BinIteratorBatched batchIterator(&binIterator, segment.offset,
		                                 segment.spanSize);
		const auto gatherStart = Clock::now();
		std::vector<BridgeEvent> events;
		if (!gatherBridgeEvents(projectionData, workingImage, batchIterator,
		                        binLoader, cache, true, events))
		{
			return false;
		}
		addGatherProfile(profile, binLoader, false, false,
		                 getElapsedSeconds(gatherStart, Clock::now()));
		if (profile != nullptr)
		{
			profile->uncachedBatches += 1;
		}
		if (!backProjectEventsWithImageBuffer(context, profile, projectionData,
		                                      workingImage, imageBuffer, events,
		                                      projectorKernel))
		{
			return false;
		}
	}

	transferProfile = {};
	if (!downloadSiddonImageBuffer(context, imageBuffer, workingImage,
	                               profile != nullptr ? &transferProfile :
	                                                    nullptr))
	{
		return false;
	}
	if (profile != nullptr)
	{
		profile->adjointImageDownloadSeconds +=
		    transferProfile.imageDownloadSeconds;
	}
	return true;
}

bool applyOsemCachedEntry(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, const Buffer& inputImageBuffer,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    const BinIterator& binIterator, const BinLoader& binLoader,
    OperatorProjectorMetalCache* cache, CachedBinIteratorEntry& entry)
{
	if (entry.eventCount == 0)
	{
		return true;
	}

	ForwardImageResources inputImageResources{inputImage, inputImageBuffer};
	for (CachedIteratorSegment& segment : entry.segments)
	{
		if (segment.cached)
		{
			if (!applyOsemCachedSegmentWithImageBuffers(
			        context, profile, inputImage, inputImageResources,
			        workingImage, updateImageBuffer, measurements, corrector,
			        config, segment))
			{
				return false;
			}
			continue;
		}

		BinIteratorBatched batchIterator(&binIterator, segment.offset,
		                                 segment.spanSize);
		const auto gatherStart = Clock::now();
		std::vector<BridgeEvent> events;
		if (!gatherBridgeEvents(measurements, inputImage, batchIterator,
		                        binLoader, cache, false, events))
		{
			return false;
		}
		addGatherProfile(profile, binLoader, true, false,
		                 getElapsedSeconds(gatherStart, Clock::now()));
		if (profile != nullptr)
		{
			profile->uncachedBatches += 1;
		}
		if (!applyOsemEventsWithImageBuffers(
		        context, profile, inputImage, inputImageResources, workingImage,
		        updateImageBuffer, measurements, corrector, config, events))
		{
			return false;
		}
	}
	return true;
}

bool applyOsemHostRatioCachedEntry(
    const Context& context, OperatorProjectorMetalProfile* profile,
    const Image& inputImage, const Buffer& inputImageBuffer,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config,
    const BinIterator& binIterator, const BinLoader& binLoader,
    OperatorProjectorMetalCache* cache, CachedBinIteratorEntry& entry)
{
	if (entry.eventCount == 0)
	{
		return true;
	}

	ForwardImageResources inputImageResources{inputImage, inputImageBuffer};
	for (CachedIteratorSegment& segment : entry.segments)
	{
		if (segment.cached)
		{
			if (!applyOsemHostRatioCachedSegmentWithImageBuffers(
			        context, profile, inputImage, inputImageResources,
			        workingImage, updateImageBuffer, measurements, corrector,
			        config, segment))
			{
				return false;
			}
			continue;
		}

		BinIteratorBatched batchIterator(&binIterator, segment.offset,
		                                 segment.spanSize);
		const bool buildMissingCorrectionBatches =
		    segment.correctionOnlyCached && !segment.correctionOnlyCacheBuilt;
		if (!applyOsemHostRatioBinIteratorWithImageBuffers(
		        context, profile, inputImage, inputImageResources, workingImage,
		        updateImageBuffer, measurements, corrector, config,
		        batchIterator, binLoader, cache,
		        &segment.correctionBatches, buildMissingCorrectionBatches))
		{
			return false;
		}
		if (buildMissingCorrectionBatches)
		{
			segment.correctionOnlyCacheBuilt = true;
		}
	}
	return true;
}

}  // namespace

struct OperatorProjectorMetalCache::Impl
{
	std::map<const BinIterator*, CachedBinIteratorEntry> entries;
	std::map<const Scanner*, std::vector<Vector3D>> detectorPositionsByScanner;
	std::size_t maxBytes = kDefaultCacheMaxBytes;
	std::size_t usedBytes = 0;
	std::size_t correctionCacheReserveBytes = 0;
	std::size_t maxBatchEvents = kDefaultMaxBatchEvents;
};

OperatorProjectorMetalCache::OperatorProjectorMetalCache()
    : mp_impl{std::make_unique<Impl>()}
{
}

OperatorProjectorMetalCache::~OperatorProjectorMetalCache() = default;

void OperatorProjectorMetalCache::clear()
{
	mp_impl->entries.clear();
	mp_impl->detectorPositionsByScanner.clear();
	mp_impl->usedBytes = 0;
}

void OperatorProjectorMetalCache::setMaxBytes(std::size_t maxBytes)
{
	mp_impl->maxBytes = maxBytes;
	if (mp_impl->usedBytes > mp_impl->maxBytes)
	{
		clear();
	}
}

std::size_t OperatorProjectorMetalCache::maxBytes() const
{
	return mp_impl->maxBytes;
}

std::size_t OperatorProjectorMetalCache::usedBytes() const
{
	return mp_impl->usedBytes;
}

void OperatorProjectorMetalCache::setCorrectionCacheReserveBytes(
    std::size_t reserveBytes)
{
	mp_impl->correctionCacheReserveBytes = reserveBytes;
}

std::size_t OperatorProjectorMetalCache::correctionCacheReserveBytes() const
{
	return std::min(mp_impl->correctionCacheReserveBytes, mp_impl->maxBytes);
}

void OperatorProjectorMetalCache::setMaxBatchEvents(std::size_t maxBatchEvents)
{
	mp_impl->maxBatchEvents = maxBatchEvents;
}

std::size_t OperatorProjectorMetalCache::maxBatchEvents() const
{
	return mp_impl->maxBatchEvents;
}

void OperatorProjectorMetalCache::setMaxChunkEvents(std::size_t maxChunkEvents)
{
	setMaxBatchEvents(maxChunkEvents);
}

std::size_t OperatorProjectorMetalCache::maxChunkEvents() const
{
	return maxBatchEvents();
}

const std::vector<Vector3D>&
    OperatorProjectorMetalCache::detectorPositions(const Scanner& scanner)
{
	auto [it, didInsert] =
	    mp_impl->detectorPositionsByScanner.try_emplace(&scanner);
	std::vector<Vector3D>& positions = it->second;
	if (didInsert || positions.size() != scanner.getNumDets())
	{
		const std::size_t numDets = scanner.getNumDets();
		positions.resize(numDets);
		for (std::size_t det = 0; det < numDets; ++det)
		{
			positions[det] = scanner.getDetectorPos(static_cast<det_id_t>(det));
		}
	}
	return positions;
}

OperatorProjectorMetalBridge::OperatorProjectorMetalBridge(
    const Context& context, OperatorProjectorMetalProfile* profile,
    OperatorProjectorMetalCache* cache)
    : m_context{context}, mp_profile{profile}, mp_cache{cache}
{
}

bool OperatorProjectorMetalBridge::applyOsemHostRatioWithImageBuffers(
    const Image& inputImage, const Buffer& inputImageBuffer,
    Image& workingImage, Buffer& updateImageBuffer,
    const ProjectionData& measurements, const BinIterator& binIterator,
    const BinLoader& binLoader, const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config) const
{
	const std::size_t totalEvents = binIterator.size();
	if (totalEvents == 0)
	{
		return true;
	}

	auto runUncached = [&]() -> bool
	{
		const std::size_t batchLimit = mp_cache != nullptr ?
		                                   mp_cache->maxBatchEvents() :
		                                   kDefaultMaxBatchEvents;
		const std::size_t batchSize =
		    batchLimit == 0 ?
		        totalEvents :
		        std::max<std::size_t>(1, std::min(batchLimit, totalEvents));

		ForwardImageResources inputImageResources{inputImage, inputImageBuffer};
		for (std::size_t offset = 0; offset < totalEvents; offset += batchSize)
		{
			const std::size_t currentBatchSize =
			    std::min(batchSize, totalEvents - offset);
			BinIteratorBatched batchIterator(&binIterator, offset,
			                                 currentBatchSize);

			if (!applyOsemHostRatioBinIteratorWithImageBuffers(
			        m_context, mp_profile, inputImage, inputImageResources,
			        workingImage, updateImageBuffer, measurements, corrector,
			        config, batchIterator, binLoader, mp_cache))
			{
				return false;
			}
		}
		return true;
	};

	if (mp_cache == nullptr)
	{
		return runUncached();
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookups += 1;
		mp_profile->cacheMaxBytes = mp_cache->maxBytes();
		mp_profile->cacheCorrectionReserveBytes =
		    mp_cache->correctionCacheReserveBytes();
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}

	const auto lookupStart = Clock::now();
	auto entryIt = mp_cache->mp_impl->entries.find(&binIterator);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookupSeconds +=
		    getElapsedSeconds(lookupStart, Clock::now());
	}
	if (entryIt != mp_cache->mp_impl->entries.end())
	{
		if (mp_profile != nullptr)
		{
			mp_profile->cacheHits += 1;
			mp_profile->cacheUsedBytes = mp_cache->usedBytes();
		}
		return applyOsemHostRatioCachedEntry(
		    m_context, mp_profile, inputImage, inputImageBuffer, workingImage,
		    updateImageBuffer, measurements, corrector, config, binIterator,
		    binLoader, mp_cache, entryIt->second);
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheMisses += 1;
	}

	CachedBinIteratorEntry entry;
	bool skippedOverBudget = false;
	const auto admissionStart = Clock::now();
	const bool builtEntry = buildCachedEntry(
	    m_context, mp_profile, measurements, inputImage, binIterator,
	    binLoader, *mp_cache, true,
	    config.cacheCorrectionFactors ? &measurements : nullptr,
	    config.cacheCorrectionFactors ? &corrector : nullptr, &config, entry,
	    skippedOverBudget);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheAdmissionSeconds +=
		    getElapsedSeconds(admissionStart, Clock::now());
	}
	if (!builtEntry)
	{
		return runUncached();
	}
	if (mp_profile != nullptr && skippedOverBudget)
	{
		mp_profile->cacheSkipsOverBudget += 1;
	}
	if (entry.cachedEventCount == 0)
	{
		return runUncached();
	}

	const std::size_t entryBytes = entry.byteCount;
	const auto insertStart = Clock::now();
	auto [insertedIt, didInsert] =
	    mp_cache->mp_impl->entries.emplace(&binIterator, std::move(entry));
	(void)didInsert;
	mp_cache->mp_impl->usedBytes += entryBytes;
	if (mp_profile != nullptr)
	{
		mp_profile->cacheInsertSeconds +=
		    getElapsedSeconds(insertStart, Clock::now());
		mp_profile->cacheBuilds += countCachedSegments(insertedIt->second);
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}
	return applyOsemHostRatioCachedEntry(
	    m_context, mp_profile, inputImage, inputImageBuffer, workingImage,
	    updateImageBuffer, measurements, corrector, config, binIterator,
	    binLoader, mp_cache, insertedIt->second);
}

OperatorProjectorMetalSupport OperatorProjectorMetalBridge::canRunSiddon(
    const OperatorProjector& projector) const
{
	if (!m_context.isValid())
	{
		return unsupported(m_context.errorMessage());
	}
	if (projector.getProjectorType() != ProjectorType::SIDDON)
	{
		return unsupported("Only Siddon projection is supported");
	}
	if (projector.getUpdaterType() != UpdaterType::DEFAULT4D)
	{
		return unsupported("Only DEFAULT4D updater mode is supported");
	}
	if (projector.getTOFHelper() != nullptr)
	{
		return unsupported("TOF projection is not supported");
	}
	if (projector.getProjectionPsfManager() != nullptr)
	{
		return unsupported("Projection-space PSF is not supported");
	}

	const std::set<ProjectionPropertyType> projectionProperties =
	    projector.getProjectionPropertyTypes();
	if (projectionProperties.count(ProjectionPropertyType::DET_ORIENT) != 0)
	{
		return unsupported("Siddon multi-ray projection is not supported");
	}
	if (!hasOnlySupportedProjectionProperties(projectionProperties))
	{
		return unsupported("Projection property set is not supported");
	}
	return supported();
}

bool OperatorProjectorMetalBridge::applyA(
    const OperatorProjector& projector, const Image& image,
    ProjectionData& projectionData, const BinIterator& binIterator,
    const BinLoader& binLoader,
    OperatorProjectorMetalKernel projectorKernel) const
{
	if (!canRunSiddon(projector).supported)
	{
		return false;
	}

	auto runUncached = [&]() -> bool
	{
		const std::size_t totalEvents = binIterator.size();
		if (totalEvents == 0)
		{
			return true;
		}

		Buffer imageBuffer;
		SiddonProjectorKernelProfile transferProfile;
		if (!uploadSiddonImageBuffer(m_context, image, imageBuffer,
		                             mp_profile != nullptr ? &transferProfile :
		                                                     nullptr))
		{
			return false;
		}
		if (mp_profile != nullptr)
		{
			mp_profile->forwardImageUploadSeconds +=
			    transferProfile.imageUploadSeconds;
		}

		const std::size_t batchLimit = mp_cache != nullptr ?
		                                   mp_cache->maxBatchEvents() :
		                                   kDefaultMaxBatchEvents;
		const std::size_t batchSize =
		    batchLimit == 0 ?
		        totalEvents :
		        std::max<std::size_t>(1, std::min(batchLimit, totalEvents));

		ForwardImageResources imageResources{image, imageBuffer};
		for (std::size_t offset = 0; offset < totalEvents; offset += batchSize)
		{
			const std::size_t currentBatchSize =
			    std::min(batchSize, totalEvents - offset);
			BinIteratorBatched batchIterator(&binIterator, offset,
			                                 currentBatchSize);

			const auto gatherStart = Clock::now();
			std::vector<BridgeEvent> events;
			if (!gatherBridgeEvents(projectionData, image, batchIterator,
			                        binLoader, mp_cache, false, events))
			{
				return false;
			}
			addGatherProfile(mp_profile, binLoader, true, false,
			                 getElapsedSeconds(gatherStart, Clock::now()));
			if (mp_profile != nullptr)
			{
				mp_profile->uncachedBatches += 1;
			}

			if (!forwardProjectEventsWithImageBuffer(
			        m_context, mp_profile, image, imageResources,
			        projectionData, events, projectorKernel))
			{
				return false;
			}
		}
		return true;
	};

	if (mp_cache == nullptr)
	{
		return runUncached();
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookups += 1;
		mp_profile->cacheMaxBytes = mp_cache->maxBytes();
		mp_profile->cacheCorrectionReserveBytes =
		    mp_cache->correctionCacheReserveBytes();
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}

	const auto lookupStart = Clock::now();
	auto entryIt = mp_cache->mp_impl->entries.find(&binIterator);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookupSeconds +=
		    getElapsedSeconds(lookupStart, Clock::now());
	}
	if (entryIt != mp_cache->mp_impl->entries.end())
	{
		if (mp_profile != nullptr)
		{
			mp_profile->cacheHits += 1;
			mp_profile->cacheUsedBytes = mp_cache->usedBytes();
		}
		return forwardProjectCachedEntry(
		    m_context, mp_profile, image, projectionData, binIterator,
		    binLoader, mp_cache, entryIt->second, projectorKernel);
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheMisses += 1;
	}

	CachedBinIteratorEntry entry;
	bool skippedOverBudget = false;
	const auto admissionStart = Clock::now();
	const bool builtEntry = buildCachedEntry(
	    m_context, mp_profile, projectionData, image, binIterator, binLoader,
	    *mp_cache, true, nullptr, nullptr, nullptr, entry, skippedOverBudget);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheAdmissionSeconds +=
		    getElapsedSeconds(admissionStart, Clock::now());
	}
	if (!builtEntry)
	{
		return runUncached();
	}
	if (mp_profile != nullptr && skippedOverBudget)
	{
		mp_profile->cacheSkipsOverBudget += 1;
	}
	if (entry.cachedEventCount == 0)
	{
		return runUncached();
	}

	const std::size_t entryBytes = entry.byteCount;
	const auto insertStart = Clock::now();
	auto [insertedIt, didInsert] =
	    mp_cache->mp_impl->entries.emplace(&binIterator, std::move(entry));
	(void)didInsert;
	mp_cache->mp_impl->usedBytes += entryBytes;
	if (mp_profile != nullptr)
	{
		mp_profile->cacheInsertSeconds +=
		    getElapsedSeconds(insertStart, Clock::now());
		mp_profile->cacheBuilds += countCachedSegments(insertedIt->second);
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}
	return forwardProjectCachedEntry(
	    m_context, mp_profile, image, projectionData, binIterator, binLoader,
	    mp_cache, insertedIt->second, projectorKernel);
}

bool OperatorProjectorMetalBridge::applyAH(
    const OperatorProjector& projector, const ProjectionData& projectionData,
    Image& image, const BinIterator& binIterator, const BinLoader& binLoader,
    OperatorProjectorMetalKernel projectorKernel) const
{
	if (!canRunSiddon(projector).supported)
	{
		return false;
	}

	const auto workingImageStart = Clock::now();
	ImageOwned workingImage(image.getParams());
	workingImage.allocate();
	workingImage.copyFromImage(&image);
	if (mp_profile != nullptr)
	{
		mp_profile->adjointHostImageCopySeconds +=
		    getElapsedSeconds(workingImageStart, Clock::now());
	}

	auto finish = [&]() -> bool
	{
		const auto hostCopyStart = Clock::now();
		image.copyFromImage(&workingImage);
		if (mp_profile != nullptr)
		{
			mp_profile->adjointHostImageCopySeconds +=
			    getElapsedSeconds(hostCopyStart, Clock::now());
		}
		return true;
	};

	auto runUncached = [&]() -> bool
	{
		const std::size_t totalEvents = binIterator.size();
		if (totalEvents == 0)
		{
			return finish();
		}

		Buffer imageBuffer;
		SiddonProjectorKernelProfile transferProfile;
		if (!uploadSiddonImageBuffer(m_context, workingImage, imageBuffer,
		                             mp_profile != nullptr ? &transferProfile :
		                                                     nullptr))
		{
			return false;
		}
		if (mp_profile != nullptr)
		{
			mp_profile->adjointImageUploadSeconds +=
			    transferProfile.imageUploadSeconds;
		}

		const std::size_t batchLimit = mp_cache != nullptr ?
		                                   mp_cache->maxBatchEvents() :
		                                   kDefaultMaxBatchEvents;
		const std::size_t batchSize =
		    batchLimit == 0 ?
		        totalEvents :
		        std::max<std::size_t>(1, std::min(batchLimit, totalEvents));

		for (std::size_t offset = 0; offset < totalEvents; offset += batchSize)
		{
			const std::size_t currentBatchSize =
			    std::min(batchSize, totalEvents - offset);
			BinIteratorBatched batchIterator(&binIterator, offset,
			                                 currentBatchSize);

			const auto gatherStart = Clock::now();
			std::vector<BridgeEvent> events;
			if (!gatherBridgeEvents(projectionData, image, batchIterator,
			                        binLoader, mp_cache, true, events))
			{
				return false;
			}
			addGatherProfile(mp_profile, binLoader, false, false,
			                 getElapsedSeconds(gatherStart, Clock::now()));
			if (mp_profile != nullptr)
			{
				mp_profile->uncachedBatches += 1;
			}

			if (!backProjectEventsWithImageBuffer(
			        m_context, mp_profile, projectionData, workingImage,
			        imageBuffer, events, projectorKernel))
			{
				return false;
			}
		}
		transferProfile = {};
		if (!downloadSiddonImageBuffer(
		        m_context, imageBuffer, workingImage,
		        mp_profile != nullptr ? &transferProfile : nullptr))
		{
			return false;
		}
		if (mp_profile != nullptr)
		{
			mp_profile->adjointImageDownloadSeconds +=
			    transferProfile.imageDownloadSeconds;
		}
		return finish();
	};

	if (mp_cache == nullptr)
	{
		return runUncached();
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookups += 1;
		mp_profile->cacheMaxBytes = mp_cache->maxBytes();
		mp_profile->cacheCorrectionReserveBytes =
		    mp_cache->correctionCacheReserveBytes();
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}

	const auto lookupStart = Clock::now();
	auto entryIt = mp_cache->mp_impl->entries.find(&binIterator);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookupSeconds +=
		    getElapsedSeconds(lookupStart, Clock::now());
	}
	if (entryIt != mp_cache->mp_impl->entries.end())
	{
		if (mp_profile != nullptr)
		{
			mp_profile->cacheHits += 1;
			mp_profile->cacheUsedBytes = mp_cache->usedBytes();
		}
		if (!backProjectCachedEntry(m_context, mp_profile, projectionData,
		                            workingImage, binIterator, binLoader,
		                            mp_cache, entryIt->second, projectorKernel))
		{
			return false;
		}
		return finish();
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheMisses += 1;
	}

	CachedBinIteratorEntry entry;
	bool skippedOverBudget = false;
	const auto admissionStart = Clock::now();
	const bool builtEntry = buildCachedEntry(
	    m_context, mp_profile, projectionData, image, binIterator, binLoader,
	    *mp_cache, false, nullptr, nullptr, nullptr, entry, skippedOverBudget);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheAdmissionSeconds +=
		    getElapsedSeconds(admissionStart, Clock::now());
	}
	if (!builtEntry)
	{
		return runUncached();
	}
	if (mp_profile != nullptr && skippedOverBudget)
	{
		mp_profile->cacheSkipsOverBudget += 1;
	}
	if (entry.cachedEventCount == 0)
	{
		return runUncached();
	}

	const std::size_t entryBytes = entry.byteCount;
	const auto insertStart = Clock::now();
	auto [insertedIt, didInsert] =
	    mp_cache->mp_impl->entries.emplace(&binIterator, std::move(entry));
	(void)didInsert;
	mp_cache->mp_impl->usedBytes += entryBytes;
	if (mp_profile != nullptr)
	{
		mp_profile->cacheInsertSeconds +=
		    getElapsedSeconds(insertStart, Clock::now());
		mp_profile->cacheBuilds += countCachedSegments(insertedIt->second);
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}
	if (!backProjectCachedEntry(m_context, mp_profile, projectionData,
	                            workingImage, binIterator, binLoader, mp_cache,
	                            insertedIt->second, projectorKernel))
	{
		return false;
	}
	return finish();
}

bool OperatorProjectorMetalBridge::applyOsemEMUpdateHostRatio(
    const OperatorProjector& projector, const Image& inputImage,
    Image& updateImage, const ProjectionData& measurements,
    const BinIterator& binIterator, const BinLoader& binLoader,
    const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config) const
{
	if (!canRunSiddon(projector).supported)
	{
		return false;
	}

	const auto workingImageStart = Clock::now();
	ImageOwned workingImage(updateImage.getParams());
	workingImage.allocate();
	if (!config.zeroInitializeUpdateImage)
	{
		workingImage.copyFromImage(&updateImage);
	}
	if (mp_profile != nullptr)
	{
		mp_profile->adjointHostImageCopySeconds +=
		    getElapsedSeconds(workingImageStart, Clock::now());
	}

	Buffer inputImageBuffer;
	SiddonProjectorKernelProfile transferProfile;
	if (!uploadSiddonImageBuffer(m_context, inputImage, inputImageBuffer,
	                             mp_profile != nullptr ? &transferProfile :
	                                                     nullptr))
	{
		return false;
	}
	if (mp_profile != nullptr)
	{
		mp_profile->forwardImageUploadSeconds +=
		    transferProfile.imageUploadSeconds;
	}

	Buffer updateImageBuffer;
	transferProfile = {};
	const bool didInitializeUpdateImageBuffer =
	    config.zeroInitializeUpdateImage ?
	        allocateClearedSiddonImageBuffer(
	            m_context, workingImage, updateImageBuffer, 0.0f,
	            mp_profile != nullptr ? &transferProfile : nullptr) :
	        uploadSiddonImageBuffer(m_context, workingImage, updateImageBuffer,
	                                mp_profile != nullptr ? &transferProfile :
	                                                        nullptr);
	if (!didInitializeUpdateImageBuffer)
	{
		return false;
	}
	if (mp_profile != nullptr)
	{
		mp_profile->adjointImageUploadSeconds +=
		    transferProfile.imageUploadSeconds;
	}

	auto finish = [&]() -> bool
	{
		SiddonProjectorKernelProfile downloadProfile;
		if (!downloadSiddonImageBuffer(
		        m_context, updateImageBuffer, workingImage,
		        mp_profile != nullptr ? &downloadProfile : nullptr))
		{
			return false;
		}
		const auto hostCopyStart = Clock::now();
		updateImage.copyFromImage(&workingImage);
		if (mp_profile != nullptr)
		{
			mp_profile->adjointImageDownloadSeconds +=
			    downloadProfile.imageDownloadSeconds;
			mp_profile->adjointHostImageCopySeconds +=
			    getElapsedSeconds(hostCopyStart, Clock::now());
		}
		return true;
	};

	if (!applyOsemHostRatioWithImageBuffers(
	        inputImage, inputImageBuffer, workingImage, updateImageBuffer,
	        measurements, binIterator, binLoader, corrector, config))
	{
		return false;
	}
	return finish();
}

bool OperatorProjectorMetalBridge::applyOsemEMUpdateHostRatioWithBuffers(
    const OperatorProjector& projector, const Image& inputImage,
    const Buffer& inputImageBuffer, Image& updateImage,
    Buffer& updateImageBuffer, const ProjectionData& measurements,
    const BinIterator& binIterator, const BinLoader& binLoader,
    const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config) const
{
	if (!canRunSiddon(projector).supported || !inputImageBuffer.isValid() ||
	    !updateImageBuffer.isValid())
	{
		return false;
	}

	return applyOsemHostRatioWithImageBuffers(
	    inputImage, inputImageBuffer, updateImage, updateImageBuffer,
	    measurements, binIterator, binLoader, corrector, config);
}

bool OperatorProjectorMetalBridge::applyOsemEMUpdate(
    const OperatorProjector& projector, const Image& inputImage,
    Image& updateImage, const ProjectionData& measurements,
    const BinIterator& binIterator, const BinLoader& binLoader,
    const Corrector_CPU& corrector,
    const OperatorProjectorMetalOsemConfig& config) const
{
	if (!canRunSiddon(projector).supported)
	{
		return false;
	}

	const auto workingImageStart = Clock::now();
	ImageOwned workingImage(updateImage.getParams());
	workingImage.allocate();
	if (!config.zeroInitializeUpdateImage)
	{
		workingImage.copyFromImage(&updateImage);
	}
	if (mp_profile != nullptr)
	{
		mp_profile->adjointHostImageCopySeconds +=
		    getElapsedSeconds(workingImageStart, Clock::now());
	}

	Buffer inputImageBuffer;
	SiddonProjectorKernelProfile transferProfile;
	if (!uploadSiddonImageBuffer(m_context, inputImage, inputImageBuffer,
	                             mp_profile != nullptr ? &transferProfile :
	                                                     nullptr))
	{
		return false;
	}
	if (mp_profile != nullptr)
	{
		mp_profile->forwardImageUploadSeconds +=
		    transferProfile.imageUploadSeconds;
	}

	Buffer updateImageBuffer;
	transferProfile = {};
	const bool didInitializeUpdateImageBuffer =
	    config.zeroInitializeUpdateImage ?
	        allocateClearedSiddonImageBuffer(
	            m_context, workingImage, updateImageBuffer, 0.0f,
	            mp_profile != nullptr ? &transferProfile : nullptr) :
	        uploadSiddonImageBuffer(m_context, workingImage, updateImageBuffer,
	                                mp_profile != nullptr ? &transferProfile :
	                                                        nullptr);
	if (!didInitializeUpdateImageBuffer)
	{
		return false;
	}
	if (mp_profile != nullptr)
	{
		mp_profile->adjointImageUploadSeconds +=
		    transferProfile.imageUploadSeconds;
	}

	auto finish = [&]() -> bool
	{
		SiddonProjectorKernelProfile downloadProfile;
		if (!downloadSiddonImageBuffer(
		        m_context, updateImageBuffer, workingImage,
		        mp_profile != nullptr ? &downloadProfile : nullptr))
		{
			return false;
		}
		const auto hostCopyStart = Clock::now();
		updateImage.copyFromImage(&workingImage);
		if (mp_profile != nullptr)
		{
			mp_profile->adjointImageDownloadSeconds +=
			    downloadProfile.imageDownloadSeconds;
			mp_profile->adjointHostImageCopySeconds +=
			    getElapsedSeconds(hostCopyStart, Clock::now());
		}
		return true;
	};

	auto runUncached = [&]() -> bool
	{
		const std::size_t totalEvents = binIterator.size();
		if (totalEvents == 0)
		{
			return finish();
		}

		const std::size_t batchLimit = mp_cache != nullptr ?
		                                   mp_cache->maxBatchEvents() :
		                                   kDefaultMaxBatchEvents;
		const std::size_t batchSize =
		    batchLimit == 0 ?
		        totalEvents :
		        std::max<std::size_t>(1, std::min(batchLimit, totalEvents));

		ForwardImageResources inputImageResources{inputImage, inputImageBuffer};
		for (std::size_t offset = 0; offset < totalEvents; offset += batchSize)
		{
			const std::size_t currentBatchSize =
			    std::min(batchSize, totalEvents - offset);
			BinIteratorBatched batchIterator(&binIterator, offset,
			                                 currentBatchSize);

			const auto gatherStart = Clock::now();
			std::vector<BridgeEvent> events;
			if (!gatherBridgeEvents(measurements, inputImage, batchIterator,
			                        binLoader, mp_cache, false, events))
			{
				return false;
			}
			addGatherProfile(mp_profile, binLoader, true, false,
			                 getElapsedSeconds(gatherStart, Clock::now()));
			if (mp_profile != nullptr)
			{
				mp_profile->uncachedBatches += 1;
			}

			if (!applyOsemEventsWithImageBuffers(
			        m_context, mp_profile, inputImage, inputImageResources,
			        workingImage, updateImageBuffer, measurements, corrector,
			        config, events))
			{
				return false;
			}
		}
		return finish();
	};

	if (mp_cache == nullptr)
	{
		return runUncached();
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookups += 1;
		mp_profile->cacheMaxBytes = mp_cache->maxBytes();
		mp_profile->cacheCorrectionReserveBytes =
		    mp_cache->correctionCacheReserveBytes();
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}

	const auto lookupStart = Clock::now();
	auto entryIt = mp_cache->mp_impl->entries.find(&binIterator);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheLookupSeconds +=
		    getElapsedSeconds(lookupStart, Clock::now());
	}
	if (entryIt != mp_cache->mp_impl->entries.end())
	{
		if (mp_profile != nullptr)
		{
			mp_profile->cacheHits += 1;
			mp_profile->cacheUsedBytes = mp_cache->usedBytes();
		}
		if (!applyOsemCachedEntry(
		        m_context, mp_profile, inputImage, inputImageBuffer,
		        workingImage, updateImageBuffer, measurements, corrector,
		        config, binIterator, binLoader, mp_cache, entryIt->second))
		{
			return false;
		}
		return finish();
	}

	if (mp_profile != nullptr)
	{
		mp_profile->cacheMisses += 1;
	}

	CachedBinIteratorEntry entry;
	bool skippedOverBudget = false;
	const auto admissionStart = Clock::now();
	const bool builtEntry = buildCachedEntry(
	    m_context, mp_profile, measurements, inputImage, binIterator, binLoader,
	    *mp_cache, true, nullptr, nullptr, &config, entry, skippedOverBudget);
	if (mp_profile != nullptr)
	{
		mp_profile->cacheAdmissionSeconds +=
		    getElapsedSeconds(admissionStart, Clock::now());
	}
	if (!builtEntry)
	{
		return runUncached();
	}
	if (mp_profile != nullptr && skippedOverBudget)
	{
		mp_profile->cacheSkipsOverBudget += 1;
	}
	if (entry.cachedEventCount == 0)
	{
		return runUncached();
	}

	const std::size_t entryBytes = entry.byteCount;
	const auto insertStart = Clock::now();
	auto [insertedIt, didInsert] =
	    mp_cache->mp_impl->entries.emplace(&binIterator, std::move(entry));
	(void)didInsert;
	mp_cache->mp_impl->usedBytes += entryBytes;
	if (mp_profile != nullptr)
	{
		mp_profile->cacheInsertSeconds +=
		    getElapsedSeconds(insertStart, Clock::now());
		mp_profile->cacheBuilds += countCachedSegments(insertedIt->second);
		mp_profile->cacheUsedBytes = mp_cache->usedBytes();
	}
	if (!applyOsemCachedEntry(m_context, mp_profile, inputImage,
	                          inputImageBuffer, workingImage, updateImageBuffer,
	                          measurements, corrector, config, binIterator,
	                          binLoader, mp_cache, insertedIt->second))
	{
		return false;
	}
	return finish();
}

}  // namespace yrt::backend::metal
