/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "yrt-pet/recon/LREM.hpp"

#include "yrt-pet/datastruct/image/Image.hpp"
#include "yrt-pet/geometry/Constants.hpp"
#include "yrt-pet/operators/ProjectorParams.hpp"
#include "yrt-pet/utils/Tools.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace py::literals;

namespace yrt
{
void py_setup_lrem(py::module& m)
{
	auto c = py::class_<LREM>(m, "LREM");
	c.def("isLowRank", &LREM::isLowRank);
	c.def("isDualUpdate", &LREM::isDualUpdate);
	c.def("setUpdateH", &LREM::setUpdateH, "update_h"_a);
	c.def("setHBasis", &LREM::setHBasis, "h_basis"_a);

	c.def(
	    "setHBasisFromNumpy",
	    [](LREM& self, py::buffer& np_data)
	    {
		    const py::buffer_info buffer = np_data.request();

		    if (buffer.ndim != 2)
			    throw std::invalid_argument("HBasis must be 2D (rank x time).");

		    if (buffer.format != py::format_descriptor<float>::format())
			    throw std::invalid_argument("HBasis must be float32.");

		    auto* ptr = static_cast<float*>(buffer.ptr);
		    const size_t rank = static_cast<size_t>(buffer.shape[0]);
		    const size_t T = static_cast<size_t>(buffer.shape[1]);

		    self.getProjectorParams().HBasis.bind(ptr, rank, T);
	    },
	    "h_basis"_a,
	    py::keep_alive<1, 2>()  // keep the buffer owner alive
	);

	c.def("getHBasisNumpy",
	      [](LREM& self)
	      {
		      // Array2DAlias<float>
		      const auto& H = self.getProjectorParams().HBasis;
		      // {rank, T}
		      const std::array<size_t, 2> dims = H.getDims();
		      const int R = static_cast<int>(dims[0]);
		      const int T = static_cast<int>(dims[1]);

		      if (R == 0 || T == 0)
		      {
			      throw std::runtime_error("HBasis not set (empty alias)");
		      }

		      pybind11::array_t<float> arr({R, T});  // C-contiguous
		      std::memcpy(arr.mutable_data(),        // copy all at once
		                  H.getRawPointer(),
		                  static_cast<size_t>(R * T) * sizeof(float));
		      return arr;  // copy
	      });
}
}  // namespace yrt
#endif

namespace yrt
{

bool LREM::isLowRank()
{
	const ProjectorParams& projectorParams = getProjectorParams();
	return (projectorParams.updaterType == UpdaterType::LR ||
	        projectorParams.updaterType == UpdaterType::LRDUALUPDATE);
}

bool LREM::isDualUpdate()
{
	return getProjectorParams().updaterType == UpdaterType::LRDUALUPDATE;
}

void LREM::setUpdateH(bool updateH)
{
	getProjectorParams().updateH = updateH;
}

void LREM::setHBasis(const Array2DBase<float>& HBasis)
{
	// Rebind our alias to the provided alias's storage
	getProjectorParams().HBasis.bind(HBasis);
}

void LREM::saveHBasisBinary(const std::string& base_path, int iter,
                            int numDigitsInFilename)
{
	// Fetch dims and raw pointer from HBasis
	const auto& H = getProjectorParams().HBasis;
	const auto dims = H.getDims();  // works for 2D or 3D HBasis
	if (dims.empty())
	{
		throw std::runtime_error("HBasis is empty, cannot save.");
	}
	const float* H_ptr = H.getRawPointer();
	if (!H_ptr)
		throw std::runtime_error("HBasis raw pointer is null.");

	// Compute size in elements and bytes
	const size_t numel = H.getSizeTotal();
	const size_t nbytes = numel * sizeof(float);

	// Decide iteration string width
	std::ostringstream oss_it;
	oss_it << std::setfill('0') << std::setw(numDigitsInFilename) << (iter);
	const std::string iter_tag = oss_it.str();

	// Compose filenames: base_path + _H_iterationXXXX.bin/json
	const std::string stem = util::addBeforeExtension(
	    base_path, std::string("_iteration") + iter_tag);
	std::string bin_fname = stem;
	std::string json_fname = stem;

	if (bin_fname.size() >= 7 &&
	    bin_fname.substr(bin_fname.size() - 7) == ".nii.gz")
	{
		bin_fname.replace(bin_fname.size() - 7, 7, ".bin");
		json_fname.replace(json_fname.size() - 7, 7, ".json");
	}
	else if (bin_fname.size() >= 4 &&
	         bin_fname.substr(bin_fname.size() - 4) == ".nii")
	{
		bin_fname.replace(bin_fname.size() - 4, 4, ".bin");
		json_fname.replace(json_fname.size() - 4, 4, ".json");
	}
	else
	{
		// fallback if there’s no nii/nii.gz extension
		bin_fname += ".bin";
		json_fname += ".json";
	}

	// Write raw binary (float32, row-major/C order)
	{
		std::ofstream ofs(bin_fname, std::ios::binary);
		if (!ofs)
		{
			throw std::runtime_error("Failed to open " + bin_fname +
			                         " for writing HBasis.");
		}
		ofs.write(reinterpret_cast<const char*>(H_ptr),
		          static_cast<std::streamsize>(nbytes));
		ofs.close();
	}

	// Write JSON sidecar with shape/metadata for easy numpy loading
	{
		std::ofstream jfs(json_fname);
		if (!jfs)
		{
			throw std::runtime_error("Failed to open " + json_fname +
			                         " for writing HBasis metadata.");
		}
		jfs << "{\n";
		jfs << "  \"dtype\": \"float32\",\n";
		jfs << "  \"order\": \"C\",\n";
		jfs << "  \"dims\": [";
		for (size_t i = 0; i < dims.size(); ++i)
		{
			jfs << dims[i];
			if (i + 1 < dims.size())
				jfs << ", ";
		}
		jfs << "],\n";
		// Optional: store your intended semantic layout if useful
		// If you use [rank, z, T] for 3D, or [rank, T] for 2D:
		jfs << "  \"layout_hint\": \""
		    << (dims.size() == 3 ? "H[r,z,t]" : "H[r,t]") << "\"\n";
		jfs << "}\n";
		jfs.close();
	}

	std::cout << "Saved HBasis to " << bin_fname << " and " << json_fname
	          << std::endl;
}

void LREM::resetSensScaling()
{
	std::fill(m_cWUpdate.begin(), m_cWUpdate.end(), 0.0f);
	std::fill(m_cHUpdate.begin(), m_cHUpdate.end(), 0.0f);
}

void LREM::applyHUpdate()
{
	const ProjectorParams& projectorParams = getProjectorParams();

	float* H_old_ptr = projectorParams.HBasis.getRawPointer();  // current H
	const float* Hnum_ptr = mp_HNumerator->getRawPointer();
	// numerator accumulated this subset

	// shapes: rank x T
	std::array<size_t, 2> dims = projectorParams.HBasis.getDims();
	const int rank = dims[0];
	const int T = dims[1];

	// H_new := H_old * (Hnum / c_r)
	util::parallelForChunked(
	    T, globals::getNumThreads(),
	    [&, rank, T, H_old_ptr, Hnum_ptr](int t, int /*tid*/)
	    {
		    for (int r = 0; r < rank; ++r)
		    {
			    const float denom = std::max(m_cHUpdate[r], EPS_FLT);
			    const float inv = 1.0f / denom;
			    float* Hr = H_old_ptr + r * T;
			    const float* Nr = Hnum_ptr + r * T;
			    Hr[t] =
			        Hr[t] * (Nr[t] * inv);  // write the *new H* back over H_old
		    }
	    });
}

void LREM::generateWUpdateSensScaling()
{
	const ProjectorParams& projectorParams = getProjectorParams();

	// HBasis is rank x T
	const auto dims = projectorParams.HBasis.getDims();
	const int rank = static_cast<int>(dims[0]);
	const int T = static_cast<int>(dims[1]);

	float* cWUpdate_r = m_cWUpdate.data();
	for (int r = 0; r < rank; ++r)
	{
		cWUpdate_r[r] = 0.f;
		for (int t = 0; t < T; ++t)
		{
			cWUpdate_r[r] += projectorParams.HBasis[r][t];
		}
	}
}

void LREM::generateHUpdateSensScaling(const Image& Wimage,
                                      const Image& sensImage)
{
	const ImageParams& imageParams = Wimage.getParams();

	const size_t J = imageParams.nx * imageParams.ny * imageParams.nz;
	const float* W_ptr = Wimage.getRawPointer();
	const float* s_ptr = sensImage.getRawPointer();

	const auto dims = getProjectorParams().HBasis.getDims();
	const int rank = static_cast<int>(dims[0]);

	ASSERT_MSG(W_ptr != nullptr, "W image is unallocated");
	ASSERT_MSG(s_ptr != nullptr, "Sensitivity image is unallocated");

	const int numThreads = globals::getNumThreads();
	float* c_HUpdate_r = m_cHUpdate.data();

	for (int r = 0; r < rank; ++r)
	{
		std::vector<float> cr_threadLocal(numThreads, 0.f);
		const auto* Wr = W_ptr + r * J;

		util::parallelForChunked(J, numThreads, [&](size_t j, size_t tid)
		                         { cr_threadLocal[tid] += Wr[j] * s_ptr[j]; });

		float cr = 0.0f;

		for (int t = 0; t < numThreads; ++t)
		{
			cr += cr_threadLocal[t];
		}

		c_HUpdate_r[r] = cr;
	}
}

void LREM::allocateHBasisTmpBuffer()
{
	if (mp_HNumerator == nullptr)
	{
		mp_HNumerator = std::make_unique<Array2DOwned<float>>();
	}
	const std::array<size_t, 2> dims = getProjectorParams().HBasis.getDims();
	const int rank = static_cast<int>(dims[0]);
	const int T = static_cast<int>(dims[1]);
	mp_HNumerator->allocate(rank, T);
}

}  // namespace yrt
