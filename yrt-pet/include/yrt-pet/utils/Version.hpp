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

struct VersionStruct
{
	int major;
	int minor;
	int patch;
	std::string hash;
	bool isDirty;

	bool operator==(const VersionStruct& other) const;
	bool operator>(const VersionStruct& other) const;
	bool operator<(const VersionStruct& other) const;
	bool operator>=(const VersionStruct& other) const;
	bool operator<=(const VersionStruct& other) const;
};

struct SimpleVersionStruct
{
	int major;
	int minor;

	bool operator==(const SimpleVersionStruct& other) const;
	bool operator>(const SimpleVersionStruct& other) const;
	bool operator<(const SimpleVersionStruct& other) const;
	bool operator>=(const SimpleVersionStruct& other) const;
	bool operator<=(const SimpleVersionStruct& other) const;
};

std::string getVersionString();
VersionStruct getVersion();

std::string getGitHash();

bool isDirty();

VersionStruct decodeVersion(const std::string& pr_versionString);

std::string encodeVersion(const VersionStruct& vs);

SimpleVersionStruct decodeVersionSimple(const std::string& pr_versionString);

std::string encodeVersionSimple(const SimpleVersionStruct& vs);

void printVersion();

};  // namespace version

};  // namespace yrt
