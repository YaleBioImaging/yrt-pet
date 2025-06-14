/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#include "utils/FileReader.hpp"

#include <algorithm>

namespace Util
{
	FileReader::FileReader(std::istream& pr_istream, bool p_useCache,
	                       size_t p_cacheSize)
	    : m_cacheStart(-1),
	      mr_istream(pr_istream),
	      m_foundEof(false),
	      m_useCache(p_useCache)
	{
		if (isUsingCache())
		{
			m_cache.allocate(p_cacheSize);
			m_cacheSize = static_cast<std::streamsize>(m_cache.getSize(0));
		}
	}

	std::streamsize FileReader::read(std::streamoff startPos,
	                                 char* receivingBuffer,
	                                 std::streamsize bytesToRead)
	{
		if (isUsingCache())
		{
			if (bytesToRead > m_cacheSize)
			{
				throw std::range_error("The requested number of bytes to read "
				                       "exceeds the cache size");
			}

			// Check if hit or miss
			bool cacheHit = m_cacheStart >= 0;     // Before first reading
			cacheHit &= startPos >= m_cacheStart;  // First byte
			cacheHit &= (startPos + bytesToRead) <=
			            (m_cacheStart + m_cacheSize);  // Last byte

			if (!cacheHit)
			{
				if (!m_foundEof)
				{
					m_foundEof = readStreamToCache(startPos);
				}
			}

			const std::streamoff offset = (startPos - m_cacheStart);
			char* initPos = m_cache.getRawPointer() + offset;
			const char* lastPos = std::copy_n(
			    initPos, std::min(m_cacheSize - offset, bytesToRead),
			    receivingBuffer);
			// Return the number of bytes copied
			return lastPos - receivingBuffer;
		}
		// simple read
		const std::streamsize bytesRead =
		    mr_istream.read(receivingBuffer, bytesToRead).gcount();
		m_foundEof = mr_istream.eof();
		return bytesRead;
	}

	bool FileReader::readStreamToCache(std::streamoff startPos)
	{
		m_cacheStart = startPos;
		mr_istream.seekg(startPos, std::ios::beg);
		if (mr_istream.fail())
		{
			throw std::range_error(
			    "Error reading file: Either the seeked position"
			    "is outside the file or the file is corrupt");
		}
		mr_istream.read(m_cache.getRawPointer(), m_cacheSize);
		if (mr_istream.eof())
		{
			m_cacheSize = mr_istream.gcount();
			return true;
		}
		return false;
	}

	bool FileReader::foundEof() const
	{
		return m_foundEof;
	}

	std::streamsize FileReader::cacheSize() const
	{
		return m_cacheSize;
	}

	std::streamoff FileReader::cacheStart() const
	{
		return m_cacheStart;
	}

	bool FileReader::isUsingCache() const
	{
		return m_useCache;
	}

	FileReaderContiguous::FileReaderContiguous(std::istream& pr_istream,
	                                           bool p_useCache,
	                                           size_t p_cacheSize)
	    : FileReader(pr_istream, p_useCache, p_cacheSize), m_readPos(0ull)
	{
	}

	bool FileReaderContiguous::finishedReading() const
	{
		return foundEof() &&
		       (!isUsingCache() || (m_readPos >= cacheStart() + cacheSize()));
	}

	std::streamsize FileReaderContiguous::read(std::streamoff startPos,
	                                           char* receivingBuffer,
	                                           std::streamsize bytesToRead)
	{
		(void)startPos;
		(void)receivingBuffer;
		(void)bytesToRead;
		throw std::runtime_error(
		    "Arbitrary access unsupported on contiguous reading");
	}

	std::streamsize FileReaderContiguous::read(char* receivingBuffer,
	                                           std::streamsize bytesToRead)
	{
		const std::streamsize bytesRead =
		    FileReader::read(m_readPos, receivingBuffer, bytesToRead);
		m_readPos += bytesRead;
		return bytesRead;
	}

	void FileReaderContiguous::advanceReadPos(std::streamsize bytesToAdvance)
	{
		m_readPos += bytesToAdvance;
	}

}  // namespace Util
