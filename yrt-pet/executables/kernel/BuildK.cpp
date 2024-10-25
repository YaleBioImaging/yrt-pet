#include "kernel/Kernel.hpp"
#include "utils/Array.hpp"
#include "utils/Assert.hpp"

#include <cxxopts.hpp>
#include <omp.h>

int main(int argc, char** argv)
{
	try
	{
		std::string img_in_fname;
		std::string out_fname;
		std::string out_i_fname;
		std::string out_j_fname;
		int W = 3;
		int P = 1;
		int num_k = 10;
		float sigma2 = 1;
		std::string mode = "neighbors";
		int num_threads = omp_get_max_threads();

		// Parse command line arguments
		cxxopts::Options options(argv[0], "Savant MLEM driver");
		options.positional_help("[optional args]").show_positional_help();

		/* clang-format off */
		options.add_options()
		("i,in", "MRI image", cxxopts::value<std::string>(img_in_fname))
		("o,out", "Image output file", cxxopts::value<std::string>(out_fname))
		("r,row", "Row index output file", cxxopts::value<std::string>(out_i_fname))
		("c,col", "Column index output file", cxxopts::value<std::string>(out_j_fname))
		("W,width", "Neighborhood half-width", cxxopts::value<int>(W))
		("P,patch", "Patch half-width", cxxopts::value<int>(P))
		("k,knn", "Number of neighbors to store", cxxopts::value<int>(num_k))
		("s,sigma2", "Kernel parameter sigma^2", cxxopts::value<float>(sigma2))
		("m,mode", "Mode: 'neighbors', 'knn', 'full'", cxxopts::value<std::string>(mode))
		("t,nthreads", "Number of threads to use", cxxopts::value<int>(num_threads))
		("h,help", "Print help");
		/* clang-format on */

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}

		std::vector<std::string> required_params = {"in", "out"};
		bool missing_args = false;
		for (auto& p : required_params)
		{
			if (result.count(p) == 0)
			{
				std::cerr << "Argument '" << p << "' missing" << std::endl;
				missing_args = true;
			}
		}
		if (missing_args)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}

		// Read input data
		Array3D<float> x_in;
		x_in.readFromFile(img_in_fname);
		size_t shape[3];
		x_in.getDims(shape);
		size_t num_pixels = shape[0] * shape[1] * shape[2];
		size_t num_neighbors = (2 * W + 1) * (2 * W + 1) * (2 * W + 1);
		size_t num_cols = 0;
		if (mode.compare("full") || mode.compare("knn"))
		{
			num_cols = num_k;
		}
		else if (mode.compare("neighbors"))
		{
			num_cols = num_neighbors;
		}

		// Prepare output
		Array2D<float> k_out;
		k_out.allocate(num_pixels, num_cols);
		Array2D<int> k_i_out;
		k_i_out.allocate(num_pixels, num_cols);
		Array2D<int> k_j_out;
		k_j_out.allocate(num_pixels, num_cols);

		// Build K matrix
		if (mode.compare("full") == 0)
		{
			Kernel::build_K_full(x_in.getRawPointer(), k_out.getRawPointer(),
			                     k_i_out.getRawPointer(),
			                     k_j_out.getRawPointer(), shape[0], shape[1],
			                     shape[2], num_k, sigma2, num_threads);
		}
		else if (mode.compare("knn") == 0)
		{
			Kernel::build_K_knn_neighbors(
			    x_in.getRawPointer(), k_out.getRawPointer(),
			    k_i_out.getRawPointer(), k_j_out.getRawPointer(), shape[0],
			    shape[1], shape[2], W, P, num_k, sigma2, num_threads);
		}
		else if (mode.compare("neighbors") == 0)
		{
			Kernel::build_K_neighbors(
			    x_in.getRawPointer(), k_out.getRawPointer(),
			    k_i_out.getRawPointer(), k_j_out.getRawPointer(), shape[0],
			    shape[1], shape[2], W, sigma2, num_threads);
		}

		k_out.writeToFile(out_fname);
		k_i_out.writeToFile(out_i_fname);
		k_j_out.writeToFile(out_j_fname);
		std::cout << "Done." << std::endl;

		return 0;
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e)
	{
		Util::printExceptionMessage(e);
		return -1;
	}
}
