/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/operators/SparseProjection.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/datastruct/projection/SparseHistogram.hpp"
#include "yrt-pet/datastruct/projection/UniformHistogram.hpp"
#include "yrt-pet/operators/OperatorProjector.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include "yrt-pet/utils/ProgressDisplay.hpp"

namespace yrt
{
namespace util
{
template <bool PrintProgress>
void forwProjectToSparseHistogram(const Image& sourceImage,
                                  const Projector& projector,
                                  SparseHistogram& sparseHistogram)
{
	// Create the properties struct
	// Only gather the detector pair because this is what the sparse histogram
	//  can encode
	std::set<ProjectionPropertyType> props = {ProjectionPropertyType::DET_ID};
	props.merge(projector.getProjectionPropertyTypes());
	PropStruct<ProjectionPropertyType> propStruct(props);
	propStruct.allocate(1);

	const auto& projPropManager = *propStruct.getManager();
	PropertyUnit* projectionPropertiesPtr = propStruct.getRawPointer();

	// Iterate over all LORs
	const auto uniformHistogram =
	    std::make_unique<UniformHistogram>(projector.getScanner());
	const size_t numBins = uniformHistogram->count();

	const Image* sourceImage_ptr = &sourceImage;

	ProgressDisplay progress(numBins, 5);

	for (bin_t bin = 0; bin < numBins; ++bin)
	{
		if constexpr (PrintProgress)
		{
			progress.progress(bin);
		}

		uniformHistogram->collectProjectionProperties(
		    projPropManager, projectionPropertiesPtr, 0, bin);

		const float projValue = projector.forwardProjection(
		    sourceImage_ptr, projPropManager, projectionPropertiesPtr, 0);

		if (std::abs(projValue) > SMALL)
		{
			const auto detPair = projPropManager.getDataValue<det_pair_t>(
			    projectionPropertiesPtr, 0, ProjectionPropertyType::DET_ID);
			sparseHistogram.accumulate(detPair, projValue);
		}
	}
}
template void
    forwProjectToSparseHistogram<true>(const Image& sourceImage,
                                       const Projector& projector,
                                       SparseHistogram& sparseHistogram);
template void
    forwProjectToSparseHistogram<false>(const Image& sourceImage,
                                        const Projector& projector,
                                        SparseHistogram& sparseHistogram);
}  // namespace util
}  // namespace yrt
