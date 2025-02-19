/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <x86intrin.h>  // For SSE/AVX intrinsics (optional)

template <typename TValue>
class PairHashMap
{
public:
	// Hash table entry: (key, index in m_data)
	struct alignas(32) HashEntry
	{
		uint64_t key{};
		uint32_t index{};
		bool occupied = false;
	};
	struct DataEntry
	{
		uint32_t d1, d2;
		TValue value;
	};

	PairHashMap()
	{
		m_hashTable.resize(64);
		m_hashTableSize = m_hashTable.size();
	}

	// Insert/update a pair (d1, d2) with a value
	void insert(uint32_t d1, uint32_t d2, TValue value)
	{
		uint32_t a = std::min(d1, d2);
		uint32_t b = std::max(d1, d2);
		const uint64_t key = makeKey(a, b);

		// Check if key exists
		size_t idx = hash(key);
		while (true)
		{
			if (!m_hashTable[idx].occupied)
				break;
			if (m_hashTable[idx].key == key)
			{
				m_data[m_hashTable[idx].index].value += value;  // Accumulate
				return;
			}
			idx = (idx + 1) % m_hashTableSize;
		}

		// Add new entry
		if (m_size >= m_hashTableSize * LOAD_FACTOR)
		{
			rehash();
		}
		m_data.push_back({a, b, value});
		insertInternal(key, m_data.size() - 1);
	}

	// Random access by insertion order
	DataEntry& operator[](size_t index)
	{
		if (index >= m_data.size())
			throw std::out_of_range("Invalid index");
		return m_data[index];
	}

	const DataEntry& operator[](size_t index) const
	{
		if (index >= m_data.size())
			throw std::out_of_range("Invalid index");
		return m_data[index];
	}

	void reserve(size_t expectedElements)
	{
		// Reserve space for the data vector (stores actual pairs)
		m_data.reserve(expectedElements);

		// Calculate required hash table size to maintain load factor <= 0.7
		const auto requiredTableSize =
		    static_cast<size_t>(std::ceil(expectedElements / LOAD_FACTOR));

		// Find next power-of-two >= required size
		size_t newTableSize = 1;
		while (newTableSize < requiredTableSize)
			newTableSize <<= 1;

		// Only resize if we need a larger table
		if (newTableSize > m_hashTableSize)
		{
			std::vector<HashEntry> newTable(newTableSize);
			std::vector<HashEntry> oldTable;
			std::swap(oldTable, m_hashTable);  // Keep old entries temporarily
			m_hashTable = std::move(newTable);
			m_hashTableSize = m_hashTable.size();
			m_size = 0;  // Reset size counter before reinsertion

			// Reinsert all existing entries into the new table
			for (const auto& entry : oldTable)
			{
				if (entry.occupied)
				{
					insertInternal(entry.key, entry.index);
				}
			}
		}
	}

	// Key-based lookup
	TValue get(uint32_t d1, uint32_t d2) const
	{
		const auto [a, b] = std::minmax(d1, d2);
		const uint64_t key = makeKey(a, b);

		size_t idx = hash(key);
		size_t numChecks = 0ull;  // Anti infinite loop safety
		while (true)
		{
			if (!m_hashTable[idx].occupied)
				return 0.0f;  // Not found
			if (m_hashTable[idx].key == key)
			{
				return m_data[m_hashTable[idx].index].value;
			}
			idx = (idx + 1) % m_hashTableSize;

			if (++numChecks > m_hashTableSize)
			{
				// Not found anywhere
				return 0.0f;
			}
		}
	}

	/// Returns true if the pair (d1, d2) exists in the map
	bool contains(uint32_t d1, uint32_t d2) const
	{
		const auto [a, b] = std::minmax(d1, d2);
		const uint64_t key = makeKey(a, b);

		size_t idx = hash(key);
		while (true)
		{
			if (!m_hashTable[idx].occupied)
			{
				return false;
			}
			if (m_hashTable[idx].key == key)
			{
				return true;
			}
			idx = (idx + 1) % m_hashTableSize;
		}
	}

	/// If (d1, d2) exists: add 'value' to its value.
	/// If not: insert with 'value'.
	/// Returns reference to the updated/inserted value.
	void update_or_insert(uint32_t d1, uint32_t d2, TValue value)
	{
		const uint64_t key = makeKey(d1, d2);
		size_t idx = hash(key);

		// Fast path: assume the key exists in the first probe
		if (m_hashTable[idx].key == key) [[likely]]
		{
			m_data[m_hashTable[idx].index].value += value;
			return;
		}

		// Linear probing with early exit
		while (true)
		{
			if (m_hashTable[idx].key == key)
			{
				m_data[m_hashTable[idx].index].value += value;
				return;
			}
			if (!m_hashTable[idx].occupied)
			{
				if (m_size >= m_hashTableSize * LOAD_FACTOR) [[unlikely]]
				{
					rehash();
					idx = hash(key);  // Recompute after rehash
					continue;
				}
				m_data.push_back({d1, d2, value});
				m_hashTable[idx] = {
				    key, static_cast<uint32_t>(m_data.size() - 1), true};
				m_size++;
				return;
			}
			idx = (idx + 1) & (m_hashTableSize - 1);
		}
	}

	size_t size() const { return m_data.size(); }

private:
	std::vector<DataEntry> m_data;  // Stores pairs in insertion order
	std::vector<HashEntry> m_hashTable;
	size_t m_size = 0;           // Number of occupied entries
	size_t m_hashTableSize = 0;  // Number of occupied entries
	static constexpr double LOAD_FACTOR = 0.5;

	// Combine d1 and d2 into a 64-bit key (d1 <= d2)
	static uint64_t makeKey(uint32_t d1, uint32_t d2)
	{
		const bool swap = d1 > d2;
		const uint32_t a = swap ? d2 : d1;
		const uint32_t b = swap ? d1 : d2;
		return (static_cast<uint64_t>(b) << 32) | a;
	}

	// Hash function using Fibonacci hashing
	size_t hash(uint64_t key) const
	{
		constexpr uint64_t multiplier = 0x9E3779B97F4A7C15;
		return (key * multiplier) & (m_hashTableSize - 1);
	}

	void rehash()
	{
		std::vector<HashEntry> newTable(m_hashTableSize * 2);
		for (auto& entry : m_hashTable)
		{
			if (entry.occupied)
			{
				size_t idx = hash(entry.key) & (newTable.size() - 1);
				while (newTable[idx].occupied)
					idx = (idx + 1) & (newTable.size() - 1);
				newTable[idx] = entry;
			}
		}
		m_hashTable = std::move(newTable);
		m_hashTableSize = m_hashTable.size();
	}

	void insertInternal(uint64_t key, uint32_t index)
	{
		size_t idx = hash(key);
		while (true)
		{
			if (!m_hashTable[idx].occupied)
			{
				m_hashTable[idx] = {key, index, true};
				m_size++;
				return;
			}
			idx = (idx + 1) % m_hashTableSize;
		}
	}
};
