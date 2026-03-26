/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/version.h"

#include <iostream>
#include <string>

namespace yrt
{

namespace version
{

static constexpr const char* versionString = YRT_PET_VERSION_STRING;

std::string getVersionString();

std::string getGitHash();

bool isDirty();

void printVersion();

};  // namespace version

};  // namespace yrt
