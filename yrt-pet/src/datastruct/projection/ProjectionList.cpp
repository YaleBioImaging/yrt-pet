/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ProjectionList.hpp"

#include "utils/Assert.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void py_setup_projectionlist(pybind11::module& m)
{
	auto c = pybind11::class_<ProjectionList, ProjectionData>(
	    m, "ProjectionList", py::buffer_protocol());
	c.def_buffer(
	    [](ProjectionList& self) -> py::buffer_info
	    {
		    Array1DBase<float>* d = self.getProjectionsArrayRef();
		    return py::buffer_info(d->getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 1,
		                           d->getDims(), d->getStrides());
	    });

	auto c_owned = pybind11::class_<ProjectionListOwned, ProjectionList>(
	    m, "ProjectionListOwned");
	c_owned.def(pybind11::init<ProjectionData*>(), py::arg("projectionData"));
	c_owned.def("allocate", &ProjectionListOwned::allocate);

	auto c_alias = pybind11::class_<ProjectionListAlias, ProjectionList>(
	    m, "ProjectionListAlias");
	c_alias.def(pybind11::init<ProjectionData*>(), py::arg("projectionData"));
	c_alias.def(
	    "bind",
	    [](ProjectionListAlias& self, pybind11::buffer& projs_in)
	    {
		    pybind11::buffer_info buffer = projs_in.request();
		    if (buffer.ndim != 1)
		    {
			    throw std::runtime_error(
			        "The buffer given has the wrong dimensions");
		    }
		    if (buffer.format != py::format_descriptor<float>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float32 format");
		    }
		    if (buffer.size !=
		        static_cast<ssize_t>(self.getReference()->count()))
		    {
			    throw std::invalid_argument("The buffer shape does not match "
			                                "with the projection data count "
			                                "count()");
		    }
		    reinterpret_cast<Array1DAlias<float>*>(
		        self.getProjectionsArrayRef())
		        ->bind(reinterpret_cast<float*>(buffer.ptr), buffer.size);
	    });
}

#endif  // if BUILD_PYBIND11

ProjectionList::ProjectionList(const ProjectionData* r)
    : ProjectionData(r->getScanner()), mp_reference(r)
{
	ASSERT(mp_reference != nullptr);
}

float ProjectionList::getProjectionValue(bin_t id) const
{
	return (*mp_projs)[id];
}

void ProjectionList::setProjectionValue(bin_t id, float val)
{
	(*mp_projs)[id] = val;
}

void ProjectionList::clearProjections(float value)
{
	mp_projs->fill(value);
}

frame_t ProjectionList::getFrame(bin_t id) const
{
	return mp_reference->getFrame(id);
}

timestamp_t ProjectionList::getTimestamp(bin_t id) const
{
	return mp_reference->getTimestamp(id);
}

size_t ProjectionList::getNumFrames() const
{
	return mp_reference->getNumFrames();
}

bool ProjectionList::isUniform() const
{
	return false;
}

bool ProjectionList::hasRandomsEstimates() const
{
	return mp_reference->hasRandomsEstimates();
}

float ProjectionList::getRandomsEstimate(bin_t id) const
{
	return mp_reference->getRandomsEstimate(id);
}

det_id_t ProjectionList::getDetector1(bin_t evId) const
{
	return mp_reference->getDetector1(evId);
}

det_id_t ProjectionList::getDetector2(bin_t evId) const
{
	return mp_reference->getDetector2(evId);
}

det_pair_t ProjectionList::getDetectorPair(bin_t evId) const
{
	return mp_reference->getDetectorPair(evId);
}

bool ProjectionList::hasTOF() const
{
	return mp_reference->hasTOF();
}

float ProjectionList::getTOFValue(bin_t id) const
{
	return mp_reference->getTOFValue(id);
}

bool ProjectionList::hasMotion() const
{
	return mp_reference->hasMotion();
}

transform_t ProjectionList::getTransformOfFrame(frame_t frame) const
{
	return mp_reference->getTransformOfFrame(frame);
}

bool ProjectionList::hasArbitraryLORs() const
{
	return mp_reference->hasArbitraryLORs();
}

Line3D ProjectionList::getArbitraryLOR(bin_t id) const
{
	return mp_reference->getArbitraryLOR(id);
}

const ProjectionData* ProjectionList::getReference() const
{
	return mp_reference;
}

float* ProjectionList::getRawPointer() const
{
	return mp_projs->getRawPointer();
}

bool ProjectionList::isMemoryValid() const
{
	return mp_projs != nullptr && getRawPointer() != nullptr;
}

Array1DBase<float>* ProjectionList::getProjectionsArrayRef() const
{
	return (mp_projs.get());
}

size_t ProjectionList::count() const
{
	return mp_projs->getSize(0);
}

histo_bin_t ProjectionList::getHistogramBin(bin_t id) const
{
	return mp_reference->getHistogramBin(id);
}

ProjectionListOwned::ProjectionListOwned(const ProjectionData* r)
    : ProjectionList(r)
{
	mp_projs = std::make_unique<Array1D<float>>();
}

void ProjectionListOwned::allocate()
{
	const size_t numBins = mp_reference->count();
	static_cast<Array1D<float>*>(mp_projs.get())->allocate(numBins);
}

ProjectionListAlias::ProjectionListAlias(const ProjectionData* p)
    : ProjectionList(p)
{
	mp_projs = std::make_unique<Array1DAlias<float>>();
}

void ProjectionListAlias::bind(Array1DBase<float>* projs_in)
{
	static_cast<Array1DAlias<float>*>(mp_projs.get())->bind(*projs_in);
	if (mp_projs->getRawPointer() == nullptr)
	{
		throw std::runtime_error(
		    "An error occured during the ProjectionList binding");
	}
}

std::unique_ptr<BinIterator> ProjectionList::getBinIter(int numSubsets,
                                                        int idxSubset) const
{
	return mp_reference->getBinIter(numSubsets, idxSubset);
}
