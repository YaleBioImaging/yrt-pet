/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/scanner/DetectorSetup.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Array.hpp"

#include <memory>

namespace yrt
{
class DetRegular : public DetectorSetup
{
public:
	DetRegular(Scanner* pp_scanner);
	void generateLUT();

	size_t getNumDets() const override;
	float getXpos(det_id_t detID) const override;
	float getYpos(det_id_t detID) const override;
	float getZpos(det_id_t detID) const override;
	float getXorient(det_id_t detID) const override;
	float getYorient(det_id_t detID) const override;
	float getZorient(det_id_t detID) const override;
	void writeToFile(const std::string& detCoord_fname) const override;
	// In case of a small modification after the generation,
	// We add the setters here
	virtual void setXpos(det_id_t detID, float f);
	virtual void setYpos(det_id_t detID, float f);
	virtual void setZpos(det_id_t detID, float f);
	virtual void setXorient(det_id_t detID, float f);
	virtual void setYorient(det_id_t detID, float f);
	virtual void setZorient(det_id_t detID, float f);

	Array1D<float>* getXposArrayRef() const { return (mp_Xpos.get()); }
	Array1D<float>* getYposArrayRef() const { return (mp_Ypos.get()); }
	Array1D<float>* getZposArrayRef() const { return (mp_Zpos.get()); }
	Array1D<float>* getXorientArrayRef() const { return (mp_Xorient.get()); }
	Array1D<float>* getYorientArrayRef() const { return (mp_Yorient.get()); }
	Array1D<float>* getZorientArrayRef() const { return (mp_Zorient.get()); }

	Scanner* getScanner() { return mp_scanner; }
	virtual ~DetRegular() {}

protected:
	void allocate();

protected:
	std::unique_ptr<Array1D<float>> mp_Xpos;
	std::unique_ptr<Array1D<float>> mp_Ypos;
	std::unique_ptr<Array1D<float>> mp_Zpos;
	std::unique_ptr<Array1D<float>> mp_Xorient;
	std::unique_ptr<Array1D<float>> mp_Yorient;
	std::unique_ptr<Array1D<float>> mp_Zorient;
	Scanner* mp_scanner;
};
}  // namespace yrt
