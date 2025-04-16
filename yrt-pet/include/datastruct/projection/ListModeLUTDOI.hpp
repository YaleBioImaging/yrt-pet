/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "utils/Array.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

class Scanner;

class ListModeLUTDOI : public ListModeLUT
{
public:
	~ListModeLUTDOI() override = default;

	bool hasArbitraryLORs() const override;
	Line3D getArbitraryLOR(bin_t id) const override;
	void writeToFile(const std::string& listMode_fname) const override;

protected:
	explicit ListModeLUTDOI(const Scanner& pr_scanner, bool p_flagTOF = false,
	                        int numLayers = 256);
	std::unique_ptr<Array1DBase<unsigned char>> mp_doi1;
	std::unique_ptr<Array1DBase<unsigned char>> mp_doi2;

	int m_numLayers;
};

class ListModeLUTDOIAlias : public ListModeLUTDOI
{
public:
	explicit ListModeLUTDOIAlias(const Scanner& pr_scanner,
	                             bool p_flagTOF = false, int numLayers = 256);
	~ListModeLUTDOIAlias() override = default;
	void bind(const Array1DBase<timestamp_t>* pp_timestamps,
	          const Array1DBase<det_id_t>* pp_detector_ids1,
	          const Array1DBase<det_id_t>* pp_detector_ids2,
	          const Array1DBase<unsigned char>* pp_doi1,
	          const Array1DBase<unsigned char>* pp_doi2,
	          const Array1DBase<float>* pp_tof_ps = nullptr);
#if BUILD_PYBIND11
	void bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	    pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi1,
	    pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi2);
	void bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	    pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi1,
	    pybind11::array_t<unsigned char, pybind11::array::c_style>& p_doi2,
	    pybind11::array_t<float, pybind11::array::c_style>& p_tof_ps);
#endif
};

class ListModeLUTDOIOwned : public ListModeLUTDOI
{
public:
	explicit ListModeLUTDOIOwned(const Scanner& pr_scanner,
	                             bool p_flagTOF = false, int numLayers = 256);
	ListModeLUTDOIOwned(const Scanner& pr_scanner,
	                    const std::string& listMode_fname,
	                    bool p_flagTOF = false, int numLayers = 256);
	~ListModeLUTDOIOwned() override = default;

	void readFromFile(const std::string& listMode_fname);
	void allocate(size_t num_events);

	// For registering the plugin
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const IO::OptionsResult& options);
	static Plugin::OptionsListPerPlugin getOptions();
};
