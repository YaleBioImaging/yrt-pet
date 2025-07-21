/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace fancyarray
{

	using size_t = unsigned long long;

	// ---------- Selecting a type ---------
	template <size_t Index, typename... Types>
	struct get_nth_type;

	// Base case: end of recursion
	template <typename First, typename... Rest>
	struct get_nth_type<0, First, Rest...>
	{
		using Type = First;
	};

	template <size_t Index, typename First, typename... Rest>
	struct get_nth_type<Index, First, Rest...>
	{
		using Type = typename get_nth_type<Index - 1, Rest...>::Type;
	};

	// Helper alias template
	template <size_t Index, typename... Types>
	using get_nth_type_t = typename get_nth_type<Index, Types...>::Type;


	// ---------- Summing the sizes ---------
	template <size_t Sum, typename... Types>
	struct get_total_sizeof;

	// Base case: end of recursion
	template <size_t Sum, typename Last>
	struct get_total_sizeof<Sum, Last>
	{
		static constexpr size_t FinalSum = Sum + sizeof(Last);
	};

	template <size_t Sum, typename First, typename... Rest>
	struct get_total_sizeof<Sum, First, Rest...>
	{
		static constexpr size_t FinalSum =
		    get_total_sizeof<Sum + sizeof(First), Rest...>::FinalSum;
	};

	// Helper alias
	template <typename... Types>
	using get_total_sizeof_t = get_total_sizeof<0, Types...>;


	// ---------- Summing the sizes ---------
	template <size_t Index, typename... Types>
	struct get_size_until;

	// Base case: end of recursion
	template <typename First, typename... Types>
	struct get_size_until<0, First, Types...>
	{
		static constexpr size_t FinalSum = 0;
	};

	template <size_t Index, typename First, typename... Rest>
	struct get_size_until<Index, First, Rest...>
	{
		static_assert(Index <= sizeof...(Rest) + 1, "Index out of bounds");
		static constexpr size_t FinalSum =
		    sizeof(First) + get_size_until<Index - 1, Rest...>::FinalSum;
	};

	// Helper alias
	template <size_t Index, typename... Types>
	using get_size_until_t = get_size_until<Index, Types...>;
}  // namespace fancyarray
