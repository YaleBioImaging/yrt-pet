/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Array.hpp"

namespace yrt
{
namespace util
{

class FileReader
{
public:
	static constexpr std::streamsize DEFAULT_CACHE_SIZE = 1ull << 30;
	virtual ~FileReader() = default;
	FileReader(std::istream& pr_istream, bool p_useCache = true,
	           size_t p_cacheSize = DEFAULT_CACHE_SIZE);

	virtual std::streamsize read(std::streamoff startPos, char* receivingBuffer,
	                             std::streamsize bytesToRead);

	bool isUsingCache() const;

protected:
	bool readStreamToCache(std::streamoff startPos);
	bool foundEof() const;
	std::streamsize cacheSize() const;
	std::streamoff cacheStart() const;

private:
	Array1D<char> m_cache;
	std::streamoff m_cacheStart;
	std::streamsize m_cacheSize;
	std::istream& mr_istream;
	bool m_foundEof;
	bool m_useCache;
};

class FileReaderContiguous : public FileReader
{
public:
	FileReaderContiguous(std::istream& pr_istream, bool p_useCache = true,
	                     size_t p_cacheSize = DEFAULT_CACHE_SIZE);

	std::streamsize read(char* receivingBuffer, std::streamsize bytesToRead);

	void advanceReadPos(std::streamsize bytesToAdvance);

	bool finishedReading() const;

private:
	std::streamsize read(std::streamoff startPos, char* receivingBuffer,
	                     std::streamsize bytesToRead) override;

private:
	std::streamoff m_readPos;
};

}  // namespace util
}  // namespace yrt
