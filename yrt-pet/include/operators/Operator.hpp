/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/Variable.hpp"

class Operator
{
public:
	virtual ~Operator() = default;
	virtual void applyA(const Variable* in, Variable* out) = 0;
	virtual void applyAH(const Variable* in, Variable* out) = 0;
};
