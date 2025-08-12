/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/PluginFramework.hpp"
#include "yrt-pet/datastruct/projection/ListMode.hpp"
#include "yrt-pet/utils/Array.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

namespace yrt
{
class Scanner;

class ListModeLUT : public ListMode
{
public:
	// Methods
	~ListModeLUT() override = default;

	timestamp_t getTimestamp(bin_t eventId) const override;
	det_id_t getDetector1(bin_t eventId) const override;
	det_id_t getDetector2(bin_t eventId) const override;
	Line3D getNativeLORFromId(bin_t eventId) const;
	size_t count() const override;
	bool isUniform() const override;
	bool hasTOF() const override;
	float getTOFValue(bin_t eventId) const override;
	bool hasRandomsEstimates() const override;
	float getRandomsEstimate(bin_t eventId) const override;

	void setTimestampOfEvent(bin_t eventId, timestamp_t ts);
	void setDetectorId1OfEvent(bin_t eventId, det_id_t d1);
	void setDetectorId2OfEvent(bin_t eventId, det_id_t d2);
	void setDetectorIdsOfEvent(bin_t eventId, det_id_t d1, det_id_t d2);
	void setTOFValueOfEvent(bin_t eventId, float tofValue);

	Array1DBase<timestamp_t>* getTimestampArrayPtr() const;
	Array1DBase<det_id_t>* getDetector1ArrayPtr() const;
	Array1DBase<det_id_t>* getDetector2ArrayPtr() const;
	Array1DBase<float>* getTOFArrayPtr() const;
	Array1DBase<float>* getRandomsEstimatesArrayPtr() const;

	virtual void writeToFile(const std::string& listMode_fname) const;

	void addLORMotion(const std::shared_ptr<LORMotion>& pp_lorMotion) override;

	bool isMemoryValid() const;

protected:
	explicit ListModeLUT(const Scanner& pr_scanner);

	// Parameters
	// The detector Id of the events.
	std::unique_ptr<Array1DBase<timestamp_t>> mp_timestamps;
	std::unique_ptr<Array1DBase<det_id_t>> mp_detectorId1;
	std::unique_ptr<Array1DBase<det_id_t>> mp_detectorId2;
	// Time-of-flight: difference of arrival time t2 - t1 in picoseconds
	std::unique_ptr<Array1DBase<float>> mp_tof_ps;
	// Randoms estimate for the given event in counts per second
	std::unique_ptr<Array1DBase<float>> mp_randoms;
};


class ListModeLUTAlias : public ListModeLUT
{
public:
	explicit ListModeLUTAlias(const Scanner& pr_scanner, bool p_flagTOF = false,
	                          bool p_flagRandoms = false);
	~ListModeLUTAlias() override = default;
	void bind(ListModeLUT* listMode);
	void bind(const Array1DBase<timestamp_t>* pp_timestamps,
	          const Array1DBase<det_id_t>* pp_detector_ids1,
	          const Array1DBase<det_id_t>* pp_detector_ids2,
	          const Array1DBase<float>* pp_tof_ps = nullptr,
	          const Array1DBase<float>* pp_randoms = nullptr);
#if BUILD_PYBIND11
	void bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>* pp_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>* pp_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>* pp_detector_ids2,
	    pybind11::array_t<float, pybind11::array::c_style>* pp_tof_ps = nullptr,
	    pybind11::array_t<float, pybind11::array::c_style>* pp_randoms =
	        nullptr);
#endif
};


class ListModeLUTOwned : public ListModeLUT
{
public:
	explicit ListModeLUTOwned(const Scanner& pr_scanner, bool p_flagTOF = false,
	                          bool p_flagRandoms = false);
	ListModeLUTOwned(const Scanner& pr_scanner,
	                 const std::string& listMode_fname, bool p_flagTOF = false,
	                 bool p_flagRandoms = false);
	~ListModeLUTOwned() override = default;

	void readFromFile(const std::string& listMode_fname);
	void allocate(size_t numEvents);

	// For registering the plugin
	static std::unique_ptr<ProjectionData>
	    create(const Scanner& scanner, const std::string& filename,
	           const io::OptionsResult& options);
	static plugin::OptionsListPerPlugin getOptions();
};

}  // namespace yrt
