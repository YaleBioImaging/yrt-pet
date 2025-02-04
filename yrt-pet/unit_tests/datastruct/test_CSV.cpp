/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "../../../build-cuda-pybind11/external/Catch/single_include/catch2/catch.hpp"
#include <cstdio>
#include <fstream>

#include "../../include/utils/Tools.hpp"

// #define REQUIRE(x) (std::cout << "test-> " << (x) << std::endl);

template <typename T>
bool compArray(Array2D<T>& x, Array2D<T>& y)
{
	bool pass = true;
	for (size_t row = 0; row < x.getSize(0) && pass; row++)
	{
		for (size_t col = 0; col < x.getSize(1) && pass; col++)
		{
			pass &= x[row][col] == y[row][col];
		}
	}
	return pass;
}

template <typename T>
void writeCSV(Array2D<T>& dat, std::string fname)
{
	std::ofstream myfile;
	myfile.open(fname);
	for (size_t row = 0; row < dat.getSize(0); row++)
	{
		for (size_t col = 0; col < dat.getSize(1); col++)
		{
			myfile << dat[row][col];
			if (col < dat.getSize(1) - 1)
			{
				myfile << ",";
			}
		}
		myfile << std::endl;
	}
	myfile.close();
}

template <typename T>
void get_rand(T& out)
{
	out = rand();
}

template <>
void get_rand(int& out)
{
	out = rand() % 20;
}

template <>
void get_rand(float& out)
{
	out = rand() / (float)RAND_MAX;
}


template <typename T>
void test_helper(int num_rows_max, int num_cols_max)
{
	for (int num_rows = 1; num_rows < num_rows_max; num_rows++)
	{
		for (int num_cols = 1; num_cols < num_cols_max; num_cols++)
		{
			Array2D<int> dat_in;
			dat_in.allocate(num_rows, num_cols);
			for (size_t row = 0; row < dat_in.getSize(0); row++)
			{
				for (size_t col = 0; col < dat_in.getSize(1); col++)
				{
					T val;
					get_rand<T>(val);
					dat_in[row][col] = val;
				}
			}

			// Write CSV file to disk
			std::string fname = "test.csv";
			writeCSV(dat_in, fname);

			// Read CSV file from disk
			Array2D<int> dat_out;
			Util::readCSV(fname, dat_out);
			if (dat_in.getSizeTotal() > 0)
			{
				REQUIRE(dat_in.getSize(0) == dat_out.getSize(0));
				REQUIRE(dat_in.getSize(1) == dat_out.getSize(1));
			}

			REQUIRE(compArray(dat_in, dat_out));

			std::remove(fname.c_str());  // delete CSV
		}
	}
}

TEST_CASE("readCSV", "[CSV]")
{
	test_helper<int>(4, 4);
	test_helper<float>(4, 4);
}
