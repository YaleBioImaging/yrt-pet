/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/backends/metal/MetalContext.hpp"
#include "yrt-pet/backends/metal/ProjectorProfile.hpp"
#include "yrt-pet/geometry/Vector3D.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace yrt
{
class BinIterator;
class BinLoader;
class Corrector_CPU;
class Image;
class OperatorProjector;
class ProjectionData;
class Scanner;
}

namespace yrt::backend::metal
{

struct OperatorProjectorMetalSupport
{
	bool supported = false;
	std::string reason;
};

enum class OperatorProjectorMetalKernel
{
	Siddon,
	Joseph,
	JosephTextureForward
};

struct OperatorProjectorMetalOsemConfig
{
	float globalScaleFactor = 1.0f;
	float denomThreshold = 0.0f;
	bool hasSensitivity = false;
	bool hasAttenuation = false;
	bool hasScatterEstimates = false;
	bool hasRandomsEstimates = false;
	bool hasInVivoAttenuation = false;
	bool zeroInitializeUpdateImage = false;
	bool usePrecomputedCorrections = true;
	bool cacheCorrectionFactors = false;
	OperatorProjectorMetalKernel projectorKernel =
	    OperatorProjectorMetalKernel::Siddon;
	bool directFrameBatchesExplicit = false;
	bool directFrameBatches = false;
	bool nativeFloatAtomicsExplicit = false;
	bool nativeFloatAtomics = false;
	bool josephAdjointAxisSwitchOnceExplicit = false;
	bool josephAdjointAxisSwitchOnce = false;
	bool threadsPerThreadgroupExplicit = false;
	std::size_t threadsPerThreadgroup = 0;
};

struct OperatorProjectorMetalRuntimeOptions
{
	bool nativeFloatAtomicsExplicit = false;
	bool nativeFloatAtomics = false;
	bool josephAdjointAxisSwitchOnceExplicit = false;
	bool josephAdjointAxisSwitchOnce = false;
	bool threadsPerThreadgroupExplicit = false;
	std::size_t threadsPerThreadgroup = 0;
};

class OperatorProjectorMetalCache
{
public:
	OperatorProjectorMetalCache();
	~OperatorProjectorMetalCache();

	void clear();
	void setMaxBytes(std::size_t maxBytes);
	std::size_t maxBytes() const;
	std::size_t usedBytes() const;
	void setCorrectionCacheReserveBytes(std::size_t reserveBytes);
	std::size_t correctionCacheReserveBytes() const;
	void setMaxBatchEvents(std::size_t maxBatchEvents);
	std::size_t maxBatchEvents() const;
	void setMaxChunkEvents(std::size_t maxChunkEvents);
	std::size_t maxChunkEvents() const;
	const std::vector<Vector3D>& detectorPositions(const Scanner& scanner);

private:
	struct Impl;

	std::unique_ptr<Impl> mp_impl;

	friend class OperatorProjectorMetalBridge;
};

class OperatorProjectorMetalBridge
{
public:
	explicit OperatorProjectorMetalBridge(
	    const Context& context,
	    OperatorProjectorMetalProfile* profile = nullptr,
	    OperatorProjectorMetalCache* cache = nullptr);

	OperatorProjectorMetalSupport
	    canRunSiddon(const OperatorProjector& projector) const;

	bool applyA(const OperatorProjector& projector, const Image& image,
	            ProjectionData& projectionData, const BinIterator& binIterator,
	            const BinLoader& binLoader,
	            OperatorProjectorMetalKernel projectorKernel =
	                OperatorProjectorMetalKernel::Siddon,
	            const OperatorProjectorMetalRuntimeOptions* runtimeOptions =
	                nullptr) const;
	bool applyAH(const OperatorProjector& projector,
	             const ProjectionData& projectionData, Image& image,
	             const BinIterator& binIterator,
	             const BinLoader& binLoader,
	             OperatorProjectorMetalKernel projectorKernel =
	                 OperatorProjectorMetalKernel::Siddon,
	             const OperatorProjectorMetalRuntimeOptions* runtimeOptions =
	                 nullptr) const;
	bool applyOsemEMUpdate(const OperatorProjector& projector,
	                       const Image& inputImage, Image& updateImage,
	                       const ProjectionData& measurements,
	                       const BinIterator& binIterator,
	                       const BinLoader& binLoader,
	                       const Corrector_CPU& corrector,
	                       const OperatorProjectorMetalOsemConfig& config)
	    const;
	bool applyOsemEMUpdateHostRatio(
	    const OperatorProjector& projector, const Image& inputImage,
	    Image& updateImage, const ProjectionData& measurements,
	    const BinIterator& binIterator, const BinLoader& binLoader,
	    const Corrector_CPU& corrector,
	    const OperatorProjectorMetalOsemConfig& config) const;
	bool applyOsemEMUpdateHostRatioWithBuffers(
	    const OperatorProjector& projector, const Image& inputImage,
	    const Buffer& inputImageBuffer, Image& updateImage,
	    Buffer& updateImageBuffer, const ProjectionData& measurements,
	    const BinIterator& binIterator, const BinLoader& binLoader,
	    const Corrector_CPU& corrector,
	    const OperatorProjectorMetalOsemConfig& config) const;

private:
	bool applyOsemHostRatioWithImageBuffers(
	    const Image& inputImage, const Buffer& inputImageBuffer,
	    Image& updateImage, Buffer& updateImageBuffer,
	    const ProjectionData& measurements, const BinIterator& binIterator,
	    const BinLoader& binLoader, const Corrector_CPU& corrector,
	    const OperatorProjectorMetalOsemConfig& config) const;

	Context m_context;
	OperatorProjectorMetalProfile* mp_profile = nullptr;
	OperatorProjectorMetalCache* mp_cache = nullptr;
};

}  // namespace yrt::backend::metal
