/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

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
	void writeToFile(const std::string& detCoord_fname) const override;
	virtual void setXpos(det_id_t detID, float f);
	virtual void setYpos(det_id_t detID, float f);
	virtual void setZpos(det_id_t detID, float f);
	virtual void setXorient(det_id_t detID, float f);
	virtual void setYorient(det_id_t detID, float f);
	virtual void setZorient(det_id_t detID, float f);

	Array1DBase<float>* getXposArrayRef() const { return (mp_Xpos.get()); }
	Array1DBase<float>* getYposArrayRef() const { return (mp_Ypos.get()); }
	Array1DBase<float>* getZposArrayRef() const { return (mp_Zpos.get()); }
	Array1DBase<float>* getXorientArrayRef() const
	{
		return (mp_Xorient.get());
	}
	Array1DBase<float>* getYorientArrayRef() const
	{
		return (mp_Yorient.get());
	}
	Array1DBase<float>* getZorientArrayRef() const
	{
		return (mp_Zorient.get());
	}

protected:
	DetCoord();

protected:
	std::unique_ptr<Array1DBase<float>> mp_Xpos;
	std::unique_ptr<Array1DBase<float>> mp_Ypos;
	std::unique_ptr<Array1DBase<float>> mp_Zpos;
	std::unique_ptr<Array1DBase<float>> mp_Xorient;
	std::unique_ptr<Array1DBase<float>> mp_Yorient;
	std::unique_ptr<Array1DBase<float>> mp_Zorient;
	// total number of dets in scanner = 258,048 in SAVANT DOI config
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
	DetCoordOwned(const std::string& filename);
	void allocate(size_t numDets);
	void readFromFile(const std::string& filename);
};
}  // namespace yrt
