/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/recon/OSEM.hpp"

#include <memory>

namespace yrt
{

class DetectorMask;
class Histogram3D;
class ListMode;
class ListModeLUTOwned;
class LORMotion;
class ProjectionData;

namespace util
{

void histogram3DToListModeLUT(const Histogram3D* histo, ListModeLUTOwned* lmOut,
                              size_t numEvents = 0);

// Returns the number of mismatched events
size_t compareListModes(const ListMode& lm1, const ListMode& lm2);

std::tuple<timestamp_t, timestamp_t>
    getFullTimeRange(const LORMotion& lorMotion);

template <bool PrintProgress = true>
std::unique_ptr<ImageOwned> timeAverageMoveImage(const LORMotion& lorMotion,
                                                 const Image* unmovedImage);
template <bool PrintProgress = true>
std::unique_ptr<ImageOwned>
    timeAverageMoveImage(const LORMotion& lorMotion, const Image* unmovedImage,
                         timestamp_t timeStart, timestamp_t timeStop);


template <bool RequiresAtomic, bool PrintProgress = true>
void convertToHistogram3D(const ProjectionData& dat, Histogram3D& histoOut,
                          const DetectorMask* detectorMask = nullptr);

template <bool RequiresAtomic, bool PrintProgress = true>
std::unique_ptr<Histogram3DOwned> convertToHistogram3D(const ProjectionData& dat,
						  const DetectorMask* detectorMask = nullptr);

template <bool PrintProgress = true>
std::unique_ptr<ListModeLUTOwned> convertToListModeLUT(const ListMode& lm,
						  const DetectorMask* detectorMask = nullptr);


Line3D getNativeLOR(const Scanner& scanner, const ProjectionData& dat,
                    bin_t binId);

void convertProjectionValuesToACF(ProjectionData& dat, float unitFactor = 0.1f);

std::unique_ptr<OSEM> createOSEM(const Scanner& scanner, bool useGPU = false);

std::tuple<Line3D, Vector3D, Vector3D>
    generateTORRandomDOI(const Scanner& scanner, det_id_t d1, det_id_t d2,
                         int vmax = 256);

// Forward projection
void forwProject(
    const Scanner& scanner, const Image& img, ProjectionData& projData,
    OperatorProjector::ProjectorType projectorType = OperatorProjector::SIDDON,
    bool useGPU = false);
void forwProject(
    const Scanner& scanner, const Image& img, ProjectionData& projData,
    const BinIterator& binIterator,
    OperatorProjector::ProjectorType projectorType = OperatorProjector::SIDDON,
    bool useGPU = false);
void forwProject(
    const Image& img, ProjectionData& projData,
    const OperatorProjectorParams& projParams,
    OperatorProjector::ProjectorType projectorType = OperatorProjector::SIDDON,
    bool useGPU = false);

// Back projection
void backProject(
    const Scanner& scanner, Image& img, const ProjectionData& projData,
    OperatorProjector::ProjectorType projectorType = OperatorProjector::SIDDON,
    bool useGPU = false);
void backProject(
    const Scanner& scanner, Image& img, const ProjectionData& projData,
    const BinIterator& binIterator,
    OperatorProjector::ProjectorType projectorType = OperatorProjector::SIDDON,
    bool useGPU = false);
void backProject(
    Image& img, const ProjectionData& projData,
    const OperatorProjectorParams& projParams,
    OperatorProjector::ProjectorType projectorType = OperatorProjector::SIDDON,
    bool useGPU = false);

}  // namespace util
}  // namespace yrt
