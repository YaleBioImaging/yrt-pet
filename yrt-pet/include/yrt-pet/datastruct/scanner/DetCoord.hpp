/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/scanner/DetectorMask.hpp"
#include "yrt-pet/datastruct/scanner/DetectorSetup.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <memory>

namespace yrt
{
class DetCoord : public DetectorSetup
{
public:
	~DetCoord() override = default;

	size_t getNumDets() const override;
	float getXpos(det_id_t detID) const override;
	float getYpos(det_id_t detID) const override;
	float getZpos(det_id_t detID) const override;
	float getXorient(det_id_t detID) const override;
	float getYorient(det_id_t detID) const override;
	float getZorient(det_id_t detID) const override;

	void addMask(const std::string& mask_fname);  // Reads mask from file
	void addMask(const DetectorMask& mask);       // Copies the given mask
	bool isDetectorAllowed(det_id_t det) const override;

	void writeToFile(const std::string& detCoord_fname) const override;

	virtual void setXpos(det_id_t detID, float f);
	virtual void setYpos(det_id_t detID, float f);
	virtual void setZpos(det_id_t detID, float f);
	virtual void setXorient(det_id_t detID, float f);
	virtual void setYorient(det_id_t detID, float f);
	virtual void setZorient(det_id_t detID, float f);

	Array1DBase<float>* getXposArrayRef() const;
	Array1DBase<float>* getYposArrayRef() const;
	Array1DBase<float>* getZposArrayRef() const;
	Array1DBase<float>* getXorientArrayRef() const;
	Array1DBase<float>* getYorientArrayRef() const;
	Array1DBase<float>* getZorientArrayRef() const;
	DetectorMask* getMask() const;
	bool hasMask() const override;

protected:
	DetCoord();

protected:
	std::unique_ptr<Array1DBase<float>> mp_Xpos;
	std::unique_ptr<Array1DBase<float>> mp_Ypos;
	std::unique_ptr<Array1DBase<float>> mp_Zpos;
	std::unique_ptr<Array1DBase<float>> mp_Xorient;
	std::unique_ptr<Array1DBase<float>> mp_Yorient;
	std::unique_ptr<Array1DBase<float>> mp_Zorient;
	std::unique_ptr<DetectorMask> mp_mask;
};

class DetCoordAlias : public DetCoord
{
public:
	DetCoordAlias();
	void bind(DetCoord* p_detCoord);
	void bind(Array1DBase<float>* Xpos, Array1DBase<float>* Ypos,
	          Array1DBase<float>* Zpos, Array1DBase<float>* Xorient,
	          Array1DBase<float>* Yorient, Array1DBase<float>* Zorient);
};

class DetCoordOwned : public DetCoord
{
public:
	DetCoordOwned();
	explicit DetCoordOwned(const std::string& filename,
	                       const std::string& mask_fname = "");
	void allocate(size_t numDets);
	void readFromFile(const std::string& filename,
	                  const std::string& mask_fname = "");
};
}  // namespace yrt
