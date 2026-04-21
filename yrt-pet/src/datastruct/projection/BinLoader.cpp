/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/BinLoader.hpp"

#include "yrt-pet/utils/ProgressDisplay.hpp"

#include <cstring>

#if BUILD_PYBIND11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace yrt
{
void py_setup_binloader(py::module& m)
{
	auto c = py::class_<BinLoader>(m, "BinLoader");
	c.def(py::init<const std::vector<Constraint*>&,
	               const std::set<ProjectionPropertyType>&>(),
	      "contraints"_a, "proj_properties"_a);
	c.def("allocate", &BinLoader::allocate, "num_elements_properties"_a);
	c.def("isAllocated", &BinLoader::isAllocated);
	c.def("clearConstraints", &BinLoader::clearConstraints);
	c.def("collectFromBins", &BinLoader::collectFromBins, "proj_data"_a,
	      "bin_iter"_a, "memset_on_invalid"_a);
	c.def("verifyConstraints", &BinLoader::verifyConstraints, "row"_a);
	c.def("collectInfo", &BinLoader::collectInfo, "bin"_a, "proj_idx"_a,
	      "cons_idx"_a, "proj_data"_a, "collect_flags"_a);
	c.def("getProjectionPropertiesSize",
	      &BinLoader::getProjectionPropertiesSize);
}
}  // namespace yrt

#endif

namespace yrt
{

BinLoader::BinLoader(const std::vector<Constraint*>& constraints,
                     const std::set<ProjectionPropertyType>& projProperties)
    : BinFilter(constraints, projProperties)
{
	setupStructs();
}

void BinLoader::allocate(size_t numElementsProperties)
{
	mp_propStruct->allocate(numElementsProperties);
}

bool BinLoader::isAllocated() const
{
	return mp_propStruct->getRawPointer() != nullptr;
}

void BinLoader::clearConstraints()
{
	BinFilter::clearConstraints();
	mp_constraintStruct = nullptr;
}

void BinLoader::collectFromBins(const ProjectionData& projData,
                                const BinIterator& binIter,
                                bool memsetOnInvalid)
{
	CollectInfoFlags collectInfoFlags(false);
	collectFlags(collectInfoFlags);

	ASSERT_MSG(binIter.size() <= mp_propStruct->size(),
	           "Not enough memory in the BinFilter to collect all bins");

	PropertyUnit* propStructPtr = mp_propStruct->getRawPointer();
	const ProjectionPropertyManager& propManager = *mp_propStruct->getManager();
	size_t propElementSize = propManager.getElementSize();

	util::parallelForChunked(
	    binIter.size(), globals::getNumThreads(),
	    [&collectInfoFlags, &binIter, &projData, propStructPtr, memsetOnInvalid,
	     propElementSize, &propManager, this](bin_t binIdx, int tid)
	    {
		    const bin_t bin = binIter.get(binIdx);

		    // Use projData to gather info, put the constraints-related data in
		    //  the "tid" row of the constraints structure, and the projection
		    //  properties in the "binIdx" row of the property structure.
		    collectInfo(bin, binIdx, tid, projData, collectInfoFlags);

		    if (verifyConstraints(tid))
		    {
			    projData.collectProjectionProperties(propManager, propStructPtr,
			                                         binIdx, bin);
		    }
		    else if (memsetOnInvalid)
		    {
			    PropertyUnit* popStructPtrAtPos =
			        propStructPtr + binIdx * propElementSize;
			    std::memset(popStructPtrAtPos, 0, propElementSize);
		    }
	    });
}

template <bool PrintProgress>
void BinLoader::parallelDoOnBins(
    const ProjectionData& projData, const BinIterator& binIter,
    const std::function<void(const ProjectionPropertyManager& propManager,
                             PropertyUnit* propStruct, size_t pos, bin_t bin)>&
        funcIfValid)
{
	CollectInfoFlags collectInfoFlags(false);
	collectFlags(collectInfoFlags);

	// We use one row for every thread
	const int numThreads = globals::getNumThreads();
	ASSERT_MSG(numThreads <= static_cast<int>(mp_propStruct->size()),
	           "Not enough memory in the BinFilter to use all threads");

	PropertyUnit* propStructPtr = mp_propStruct->getRawPointer();
	const ProjectionPropertyManager& propManager = *mp_propStruct->getManager();
	size_t numBins = binIter.size();

	std::unique_ptr<util::ProgressDisplay> progress;
	if constexpr (PrintProgress)
	{
		progress = std::make_unique<util::ProgressDisplay>(numBins, 10);
	}

	util::parallelForChunked(
	    numBins, numThreads,
	    [&collectInfoFlags, &binIter, &projData, propStructPtr, &funcIfValid,
	     &propManager, &progress, this](bin_t binIdx, int tid)
	    {
		    if constexpr (PrintProgress)
		    {
			    if (tid == 0)
			    {
				    progress->progress(binIdx);
			    }
		    }

		    const bin_t bin = binIter.get(binIdx);

		    // Use projData to gather info, put the constraints-related data in
		    //  the "tid" row of the constraints structure, and the projection
		    //  properties in the "binIdx" row of the property structure.
		    collectInfo(bin, tid, tid, projData, collectInfoFlags);

		    if (verifyConstraints(tid))
		    {
			    projData.collectProjectionProperties(propManager, propStructPtr,
			                                         tid, bin);

			    // Call the lambda function on the current bin
			    funcIfValid(propManager, propStructPtr, tid, bin);
		    }
	    });
}
template void BinLoader::parallelDoOnBins<true>(
    const ProjectionData&, const BinIterator&,
    const std::function<void(const ProjectionPropertyManager&, PropertyUnit*,
                             size_t, bin_t)>&);
template void BinLoader::parallelDoOnBins<false>(
    const ProjectionData&, const BinIterator&,
    const std::function<void(const ProjectionPropertyManager&, PropertyUnit*,
                             size_t, bin_t)>&);

bool BinLoader::verifyConstraints(size_t pos) const
{
	return BinFilter::isValid(*mp_constraintStruct->getManager(),
	                          mp_constraintStruct->getRawPointer(), pos);
}

PropertyUnit* BinLoader::getProjectionPropertiesRawPointer() const
{
	return mp_propStruct->getRawPointer();
}

PropertyUnit* BinLoader::getConstraintVariablesRawPointer() const
{
	return mp_constraintStruct->getRawPointer();
}

void BinLoader::collectInfo(bin_t bin, size_t projIdx, int consIdx,
                            const ProjectionData& projData,
                            const CollectInfoFlags& collectFlags) const
{
	BinFilter::collectInfo(bin, projIdx, consIdx, projData, collectFlags,
	                       getProjectionPropertiesRawPointer(),
	                       getConstraintVariablesRawPointer());
}

size_t BinLoader::getProjectionPropertiesSize() const
{
	return mp_propStruct->size();
}

PropStruct<ConstraintVariableType>* BinLoader::getConstraintPropStruct() const
{
	return mp_constraintStruct.get();
}

PropStruct<ProjectionPropertyType>* BinLoader::getProjectionPropStruct() const
{
	return mp_propStruct.get();
}

void BinLoader::setupStructs()
{
	// The number of elements for the constraint struct is always the number of
	//  threads
	const int numThreads = globals::getNumThreads();
	mp_constraintStruct =
	    std::make_unique<PropStruct<ConstraintVariableType>>(m_consVariables);
	mp_constraintStruct->allocate(numThreads);

	mp_propStruct =
	    std::make_unique<PropStruct<ProjectionPropertyType>>(m_projVariables);

	mp_constraintManagerPtr = mp_constraintStruct->getManager();
	mp_propManagerPtr = mp_propStruct->getManager();
}

}  // namespace yrt
