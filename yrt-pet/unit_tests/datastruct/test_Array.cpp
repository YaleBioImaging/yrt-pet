/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"
#include <stdio.h>

#include "yrt-pet/utils/Array.hpp"

TEST_CASE("array1d", "[array]")
{
	yrt::Array1DBase<int>* arr;
	auto _arr = std::make_unique<yrt::Array1D<int>>();
	_arr->allocate(10);
	arr = _arr.get();

	// Filling the array with some data
	int tmp[10] = {1, 2, 3, 4, 8, 5, 6, 9, 7, 0};
	for (int i = 0; i < 10; i++)
	{
		(*arr)[i] = tmp[i];
	}

	SECTION("array1d-data")
	{
		REQUIRE(arr->getSize(0) == 10);
		REQUIRE((*arr)[4] == 8);
	}

	auto arr_alias = yrt::Array1DAlias<int>();
	arr_alias.bind(*arr);

	SECTION("array1d-binding")
	{
		REQUIRE(arr_alias.getSize(0) == 10);
		REQUIRE((arr_alias)[4] == 8);
	}

	arr->writeToFile("array1d");

	yrt::Array1DBase<int>* arr2;
	auto _arr2 = std::make_unique<yrt::Array1D<int>>();
	arr2 = _arr2.get();
	arr2->readFromFile("array1d");
	std::remove("array1d");


	SECTION("array1d-fileread")
	{
		REQUIRE((*arr)[0] == (*arr2)[0]);
		REQUIRE((*arr)[4] == (*arr2)[4]);
		REQUIRE(arr2->getSize(0) == 10);
	}
}
TEST_CASE("array2d", "[array]")
{

	yrt::Array2DBase<int>* arr;
	auto _arr = std::make_unique<yrt::Array2D<int>>();
	_arr->allocate(2, 10);
	arr = _arr.get();

	// Filling the array with some data
	int tmp[][10] = {{1, 2, 3, 4, 8, 5, 6, 9, 7, 0},
	                 {18, 25, 38, 45, 88, 55, 68, 95, 78, 05}};
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			(*arr)[i][j] = tmp[i][j];
		}
	}

	SECTION("array2d-data")
	{
		REQUIRE(arr->getSize(1) == 10);
		REQUIRE(arr->getSize(0) == 2);
		REQUIRE((*arr)[0][4] == 8);
		REQUIRE((*arr)[1][4] == 88);
	}

	auto arr_alias = yrt::Array2DAlias<int>();
	arr_alias.bind(*arr);

	SECTION("array2d-binding")
	{
		REQUIRE(arr_alias.getSize(1) == 10);
		REQUIRE(arr_alias.getSize(0) == 2);
		REQUIRE((arr_alias)[0][4] == 8);
		REQUIRE((arr_alias)[1][4] == 88);
	}

	arr->writeToFile("array2d");

	yrt::Array2DBase<int>* arr2;
	auto _arr2 = std::make_unique<yrt::Array2D<int>>();
	arr2 = _arr2.get();
	arr2->readFromFile("array2d");
	std::remove("array2d");


	SECTION("array2d-fileread")
	{
		REQUIRE((*arr)[0][0] == (*arr2)[0][0]);
		REQUIRE((*arr)[0][1] == (*arr2)[0][1]);
		REQUIRE((*arr)[1][0] == (*arr2)[1][0]);
		REQUIRE((*arr)[1][4] == (*arr2)[1][4]);
		REQUIRE(arr2->getSize(1) == 10);
		REQUIRE(arr2->getSize(0) == 2);
	}
}
TEST_CASE("array3d", "[array]")
{

	yrt::Array3DBase<int>* arr;
	auto _arr = std::make_unique<yrt::Array3D<int>>();
	_arr->allocate(3, 2, 10);
	arr = _arr.get();

	// Filling the array with some data
	int tmp[][2][10] = {{{1, 2, 3, 4, 8, 5, 6, 9, 7, 0},
	                     {18, 25, 38, 45, 88, 55, 68, 95, 78, 05}},
	                    {{10, 20, 30, 40, 80, 50, 60, 90, 70, 99},
	                     {108, 205, 308, 405, 808, 505, 608, 905, 708, 105}},
	                    {{910, 920, 93, 49, 89, 95, 96, 99, 71, 90},
	                     {918, 925, 938, 945, 988, 955, 968, 995, 978, 905}}};
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 10; k++)
			{
				(*arr)[i][j][k] = tmp[i][j][k];
			}
		}
	}

	SECTION("array3d-data")
	{
		REQUIRE(arr->getSize(2) == 10);
		REQUIRE(arr->getSize(1) == 2);
		REQUIRE(arr->getSize(0) == 3);
		REQUIRE((*arr)[0][0][4] == 8);
		REQUIRE((*arr)[1][0][3] == 40);
		REQUIRE((*arr)[2][0][2] == 93);
		REQUIRE((*arr)[0][1][4] == 88);
		REQUIRE((*arr)[1][1][3] == 405);
		REQUIRE((*arr)[2][1][2] == 938);
	}

	auto arr_alias = yrt::Array3DAlias<int>();
	arr_alias.bind(*arr);

	SECTION("array3d-binding")
	{
		REQUIRE(arr_alias.getSize(2) == 10);
		REQUIRE(arr_alias.getSize(1) == 2);
		REQUIRE(arr_alias.getSize(0) == 3);
		REQUIRE((arr_alias)[0][0][4] == 8);
		REQUIRE((arr_alias)[1][0][3] == 40);
		REQUIRE((arr_alias)[2][0][2] == 93);
		REQUIRE((arr_alias)[0][1][4] == 88);
		REQUIRE((arr_alias)[1][1][3] == 405);
		REQUIRE((arr_alias)[2][1][2] == 938);
	}

	arr->writeToFile("array3d");

	yrt::Array3DBase<int>* arr2;
	auto _arr2 = std::make_unique<yrt::Array3D<int>>();
	arr2 = _arr2.get();
	arr2->readFromFile("array3d");
	std::remove("array3d");


	SECTION("array3d-fileread")
	{
		REQUIRE((*arr)[0][0][0] == (*arr2)[0][0][0]);
		REQUIRE((*arr)[0][0][4] == (*arr2)[0][0][4]);
		REQUIRE((*arr)[1][1][0] == (*arr2)[1][1][0]);
		REQUIRE((*arr)[1][1][4] == (*arr2)[1][1][4]);
		REQUIRE((*arr)[2][1][0] == (*arr2)[2][1][0]);
		REQUIRE((*arr)[2][1][4] == (*arr2)[2][1][4]);
		REQUIRE(arr2->getSize(2) == 10);
		REQUIRE(arr2->getSize(1) == 2);
		REQUIRE(arr2->getSize(0) == 3);
	}

	SECTION("array3d-variadic")
	{
		auto pos = std::array<size_t, 3>({0, 1, 4});
		REQUIRE(arr->get(pos) == arr2->get(pos));
		pos = std::array<size_t, 3>({1, 0, 3});
		REQUIRE(arr->get(pos) == arr2->get(pos));
		pos = std::array<size_t, 3>({2, 0, 2});
		REQUIRE(arr->get(pos) == arr2->get(pos));
	}
}

TEST_CASE("array1d-stl-type", "[array]")
{
	using Pair = std::pair<int, int>;

	yrt::Array1D<Pair> arr;
	arr.allocate(10);
	arr[0] = Pair(12, 13);
	arr[1] = {3, 2};
	SECTION("array1d-stl-assignment")
	{
		REQUIRE(arr[1].first == 3);
	}

	arr.writeToFile("array-stl");
	yrt::Array1D<Pair> arr2;
	arr2.readFromFile("array-stl");
	std::remove("array-stl");

	SECTION("array1d-stl-fileread")
	{
		REQUIRE(arr[0].first == arr2[0].first);
	}

	yrt::Array1DAlias<Pair> arr_alias;
	arr_alias.bind(arr);
	SECTION("array1d-stl-alias")
	{
		REQUIRE(arr[1].second == arr_alias[1].second);
		arr[1].second++;
		REQUIRE(arr[1].second == arr_alias[1].second);
		REQUIRE(arr[1].second != arr2[1].second);
	}
}
