/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <iostream>
#include <string>

namespace yrt
{

namespace version
{


std::string getVersionString();

std::string getGitHash();

bool isDirty();

void printVersion();

};  // namespace version

};  // namespace yrt
