/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/datastruct/projection/SinogramConverter.hpp"

#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/utils/Assert.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"

#include <atomic>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace yrt
{
void py_setup_sinogram_converter(py::module& m)
{
	using namespace sino;
	using pybind11::arg;

	m.def("fillSinoFromListMode", &fillFromListMode, arg("sino"), arg("lm"),
	      arg("num_threads") = -1,
	      "Accumulate list-mode events into a Sinogram3D in-place.");

	m.def(
	    "sinoFromHistogram3D",
	    [](const Histogram3D& h3d, const SinogramParams& params)
	    { return fromHistogram3D(h3d, params); }, arg("h3d"), arg("params"),
	    "Convert a Histogram3D to a new Sinogram3DOwned.");

	m.def(
	    "sinoToHistogram3D",
	    [](const Sinogram3D& sino) { return toHistogram3D(sino); }, arg("sino"),
	    "Convert a Sinogram3D to a new Histogram3DOwned.");
}
}  // namespace yrt
#endif  // BUILD_PYBIND11

namespace yrt::sino
{

void fillFromListMode(Sinogram3D& sino, const ListMode& lm, int numThreads)
{
	ASSERT_MSG(&sino.getScanner() == &lm.getScanner(),
	           "Sinogram3D and ListMode must share the same Scanner");

	sino.clearProjections(0.0f);
	float* sinoPtr = sino.getData().getRawPointer();
	const size_t numEvents = lm.count();
	const size_t nThreads = static_cast<size_t>(
	    numThreads <= 0 ? globals::getNumThreads() : numThreads);
	const Sinogram3D* sinoConst = &sino;
	const ListMode* lmConst = &lm;

	const bin_t sentinel = sino.sinoSize;
	util::parallelForChunked(
	    numEvents, nThreads,
	    [sinoConst, lmConst, sinoPtr, sentinel](bin_t ev, size_t /*tid*/)
	    {
		    const det_id_t d1 = lmConst->getDetector1(ev);
		    const det_id_t d2 = lmConst->getDetector2(ev);
		    const bin_t b = sinoConst->getBinIdFromDetPair(d1, d2);
		    if (b == sentinel)
			    return;
		    std::atomic_ref<float> slot(sinoPtr[b]);
		    slot += 1.0f;
	    });
}

std::unique_ptr<Sinogram3DOwned> fromHistogram3D(const Histogram3D& h3d,
                                                 const SinogramParams& params)
{
	auto sino = std::make_unique<Sinogram3DOwned>(h3d.getScanner(), params);
	sino->allocate();
	sino->clearProjections(0.0f);

	float* sinoPtr = sino->getData().getRawPointer();
	const size_t numH3DBins = h3d.count();
	const bin_t sentinel = sino->sinoSize;
	const Sinogram3DOwned* sinoConst = sino.get();
	const Histogram3D* h3dConst = &h3d;

	util::parallelForChunked(
	    numH3DBins, static_cast<size_t>(globals::getNumThreads()),
	    [sinoConst, h3dConst, sinoPtr, sentinel](bin_t hBin, size_t /*tid*/)
	    {
		    const float val = h3dConst->getProjectionValue(hBin);
		    if (val == 0.0f)
			    return;
		    const auto [d1, d2] = h3dConst->getDetectorPair(hBin);
		    if (d1 == d2)
			    return;
		    const bin_t b = sinoConst->getBinIdFromDetPair(d1, d2);
		    if (b == sentinel)
			    return;
		    std::atomic_ref<float> slot(sinoPtr[b]);
		    slot += val;
	    });

	return sino;
}

std::unique_ptr<Histogram3DOwned> toHistogram3D(const Sinogram3D& sino)
{
	auto h3d = std::make_unique<Histogram3DOwned>(sino.getScanner());
	h3d->allocate();
	h3d->clearProjections(0.0f);

	float* h3dPtr = h3d->getData().getRawPointer();
	const size_t numSinoBins = sino.count();
	const Sinogram3D* sinoConst = &sino;
	const Histogram3DOwned* h3dConst = h3d.get();

	util::parallelForChunked(
	    numSinoBins, static_cast<size_t>(globals::getNumThreads()),
	    [sinoConst, h3dConst, h3dPtr](bin_t sinoBin, size_t /*tid*/)
	    {
		    const float val = sinoConst->getProjectionValue(sinoBin);
		    if (val == 0.0f)
			    return;
		    const det_id_t d1 = sinoConst->getDetector1(sinoBin);
		    const det_id_t d2 = sinoConst->getDetector2(sinoBin);
		    if (d1 == d2)
			    return;
		    bin_t b;
		    try
		    {
			    b = h3dConst->getBinIdFromDetPair(d1, d2);
		    }
		    catch (...)
		    {
			    return;
		    }
		    std::atomic_ref<float> slot(h3dPtr[b]);
		    slot += val;
	    });

	return h3d;
}

}  // namespace yrt::sino
