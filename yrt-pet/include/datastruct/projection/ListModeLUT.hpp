/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/LORMotion.hpp"
#include "datastruct/projection/ListMode.hpp"
#include "utils/Array.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

class Scanner;

class ListModeLUT : public ListMode
{
public:
	// Methods
	~ListModeLUT() override = default;

	timestamp_t getTimestamp(bin_t eventId) const override;
	det_id_t getDetector1(bin_t eventId) const override;
	det_id_t getDetector2(bin_t eventId) const override;
	Line3D getNativeLORFromId(bin_t id) const;
	bool hasTOF() const override;
	float getTOFValue(bin_t id) const override;
	size_t count() const override;
	bool isUniform() const override;
	bool hasMotion() const override;
	frame_t getFrame(bin_t id) const override;
	size_t getNumFrames() const override;
	transform_t getTransformOfFrame(frame_t frame) const override;
	float getDurationOfFrame(frame_t frame) const override;

	void setTimestampOfEvent(bin_t eventId, timestamp_t ts);
	void setDetectorId1OfEvent(bin_t eventId, det_id_t d1);
	void setDetectorId2OfEvent(bin_t eventId, det_id_t d2);
	void setDetectorIdsOfEvent(bin_t eventId, det_id_t d1, det_id_t d2);
	void setTOFValueOfEvent(bin_t eventId, float tofValue);

	Array1DBase<timestamp_t>* getTimestampArrayPtr() const;
	Array1DBase<det_id_t>* getDetector1ArrayPtr() const;
	Array1DBase<det_id_t>* getDetector2ArrayPtr() const;

	virtual void writeToFile(const std::string& listMode_fname) const;

	void addLORMotion(const std::string& lorMotion_fname);

protected:
	explicit ListModeLUT(const Scanner& pr_scanner, bool p_flagTOF = false);

	// Parameters
	// The detector Id of the events.
	std::unique_ptr<Array1DBase<timestamp_t>> mp_timestamps;
	std::unique_ptr<Array1DBase<det_id_t>> mp_detectorId1;
	std::unique_ptr<Array1DBase<det_id_t>> mp_detectorId2;
	bool m_flagTOF;
	std::unique_ptr<Array1DBase<float>> mp_tof_ps;

	std::unique_ptr<LORMotion> mp_lorMotion;
	std::unique_ptr<Array1D<frame_t>> mp_frames;
};


class ListModeLUTAlias : public ListModeLUT
{
public:
	explicit ListModeLUTAlias(const Scanner& pr_scanner,
	                          bool p_flagTOF = false);
	~ListModeLUTAlias() override = default;
	void bind(ListModeLUT* listMode);
	void bind(Array1DBase<timestamp_t>* p_timestamps,
	          Array1DBase<det_id_t>* p_detector_ids1,
	          Array1DBase<det_id_t>* p_detector_ids2,
	          Array1DBase<float>* p_tof_ps = nullptr);
#if BUILD_PYBIND11
	void bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2);
	void bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	    pybind11::array_t<float, pybind11::array::c_style>& p_tof_ps);
#endif
};


class ListModeLUTOwned : public ListModeLUT
{
public:
	explicit ListModeLUTOwned(const Scanner& pr_scanner,
	                          bool p_flagTOF = false);
	ListModeLUTOwned(const Scanner& pr_scanner,
	                 const std::string& listMode_fname, bool p_flagTOF = false);
	~ListModeLUTOwned() override = default;

	void readFromFile(const std::string& listMode_fname);
	void allocate(size_t numEvents);

	// For registering the plugin
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const Plugin::OptionsResult& pluginOptions);
	static Plugin::OptionsListPerPlugin getOptions();
};
