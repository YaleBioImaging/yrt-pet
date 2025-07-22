/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/geometry/Line3D.hpp"

namespace yrt
{
class Cylinder;

class TubeOfResponse
{
public:
	TubeOfResponse(const Vector3D& p1, const Vector3D& p2, const Vector3D& n1,
	               const Vector3D& n2, float p_thickness_trans,
	               float p_thickness_z);

	const Line3D& getLeftLine() const { return leftLine; }
	const Line3D& getRightLine() const { return rightLine; }
	const Line3D& getFrontLine() const { return frontLine; }
	const Line3D& getBackLine() const { return backLine; }

	void setLeftLine(const Line3D& l);
	void setRightLine(const Line3D& l);
	void setFrontLine(const Line3D& l);
	void setBackLine(const Line3D& l);

	const Line3D& getAverageLine() const;
	const Line3D& getAvgLine() const;  // Alias
	bool clip(const Cylinder& cyl);

private:
	void updateAvgLine();

private:
	Line3D leftLine;
	Line3D rightLine;
	Line3D frontLine;
	Line3D backLine;
	Line3D avgLine;
	// The placement of lines is described in the ASV paper
	// Here's a scheme: https://imgur.com/a/VIvuEvq
	// ("rear" has been replaced by "back")
public:
	Vector3D m_n1;
	Vector3D m_n2;
	float thickness_z, thickness_trans;
	bool isMoreHorizontalThanVertical;
};
}  // namespace yrt
