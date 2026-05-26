/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace yrt::util
{
namespace detail
{
#if (!defined(__cpp_lib_atomic_ref) || __cpp_lib_atomic_ref < 201806L) && \
    (defined(__clang__) || defined(__GNUC__))
template <typename T, typename Bits>
inline void atomicAddFloatLike(T& target, T value)
{
	static_assert(sizeof(T) == sizeof(Bits));

	auto* bitsPtr = reinterpret_cast<Bits*>(&target);
	Bits oldBits = __atomic_load_n(bitsPtr, __ATOMIC_RELAXED);
	Bits newBits;
	do
	{
		T oldValue;
		std::memcpy(&oldValue, &oldBits, sizeof(T));
		const T newValue = oldValue + value;
		std::memcpy(&newBits, &newValue, sizeof(T));
	} while (!__atomic_compare_exchange_n(bitsPtr, &oldBits, newBits, false,
	                                      __ATOMIC_RELAXED,
	                                      __ATOMIC_RELAXED));
}
#endif
}  // namespace detail

template <typename T>
inline void atomicAdd(T& target, T value)
{
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
	std::atomic_ref<T> ref(target);
	ref.fetch_add(value, std::memory_order_relaxed);
#elif defined(__clang__) || defined(__GNUC__)
	if constexpr (std::is_integral_v<T>)
	{
		__atomic_fetch_add(&target, value, __ATOMIC_RELAXED);
	}
	else if constexpr (std::is_same_v<T, float>)
	{
		detail::atomicAddFloatLike<T, uint32_t>(target, value);
	}
	else if constexpr (std::is_same_v<T, double>)
	{
		detail::atomicAddFloatLike<T, uint64_t>(target, value);
	}
	else
	{
		static_assert(std::is_arithmetic_v<T>,
		              "Unsupported atomicAdd target type");
	}
#else
	static std::mutex mutex;
	std::lock_guard<std::mutex> lock(mutex);
	target += value;
#endif
}
}  // namespace yrt::util
