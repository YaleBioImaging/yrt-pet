/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

/*******************************************************************************
 * Def.: Compute the aperture area histogram for LORs in a scanner.
 ******************************************************************************/

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "../ArgumentReader.hpp"
#include "yrt-pet/datastruct/IO.hpp"
#include "yrt-pet/datastruct/projection/Histogram3D.hpp"
#include "yrt-pet/datastruct/scanner/Scanner.hpp"
#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/ProgressDisplayMultiThread.hpp"

using namespace yrt;

struct Point3D
{
	float x, y, z;
};

struct Point2D
{
	float u, v;
	// Cross product of 2D vectors OA and OB
	// Positive if OAB makes a counter-clockwise turn
	static float cross(const Point2D& O, const Point2D& A, const Point2D& B)
	{
		return (A.u - O.u) * (B.v - O.v) - (A.v - O.v) * (B.u - O.u);
	}


	// Sorting comparator
	bool operator<(const Point2D& other) const
	{
		return u < other.u || (u == other.u && v < other.v);
	}
};

// Computes the 2D Convex Hull using Andrew's Monotone Chain Algorithm O(N log
// N)
inline std::vector<Point2D> convexHull2D(std::vector<Point2D>& pts)
{
	size_t n = pts.size(), k = 0;
	if (n <= 3)
		return pts;
	std::vector<Point2D> res(2 * n);

	// Sort points lexicographically
	std::sort(pts.begin(), pts.end());

	// Build lower hull
	for (size_t i = 0; i < n; ++i)
	{
		while (k >= 2 && Point2D::cross(res[k - 2], res[k - 1], pts[i]) <= 0)
			k--;
		res[k++] = pts[i];
	}
	// Build upper hull
	for (size_t i = n - 1, t = k + 1; i > 0; --i)
	{
		while (k >= t &&
		       Point2D::cross(res[k - 2], res[k - 1], pts[i - 1]) <= 0)
			k--;
		res[k++] = pts[i - 1];
	}
	res.resize(k - 1);
	return res;
}

// Generates the 8 corner vertices of a specific crystal based on its LUT
// parameter entries
inline std::vector<Point3D> getCrystalEdges(float cx, float cy, float cz,
                                            float nx, float ny, float nz,
                                            float hsx, float hsy, float hsz)
{
	std::vector<Point3D> edges;
	edges.reserve(8);

	// Base coordinate system matching your scanner layout logic
	float u_x = nx, u_y = ny, u_z = 0.0f;
	float v_x = -ny, v_y = nx, v_z = 0.0f;
	float w_x = 0.0f, w_y = 0.0f, w_z = 1.0f;

	// Evaluate combination of all 8 signs for 3D extent offsets
	for (float sz : {-1.f, 1.f})
	{
		for (float sy : {-1.f, 1.f})
		{
			for (float sx : {-1.f, 1.f})
			{
				float dx = sx * hsx * u_x + sy * hsy * v_x + sz * hsz * w_x;
				float dy = sx * hsx * u_y + sy * hsy * v_y + sz * hsz * w_y;
				float dz = sx * hsx * u_z + sy * hsy * v_z + sz * hsz * w_z;
				edges.push_back({cx + dx, cy + dy, cz + dz});
			}
		}
	}
	return edges;
}

inline float computePolygonArea2D(const std::vector<Point2D>& hull)
{
	size_t n = hull.size();
	if (n < 3)
		return 0.0f;

	float area = 0.0f;
	for (size_t i = 0; i < n; ++i)
	{
		size_t next_i = (i + 1) % n;
		area += (hull[i].u * hull[next_i].v) - (hull[next_i].u * hull[i].v);
	}
	return std::abs(area) * 0.5f;
}

int main(int argc, char* argv[])
{
	try
	{
		io::ArgumentRegistry registry{};

		std::string coreGroup = "0. Core";
		std::string inputGroup = "1. Input";
		std::string outputGroup = "2. Output";

		// Core parameters
		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);

		// Output file
		registry.registerArgument("out", "Output image filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		// Load configuration
		io::ArgumentReader config{registry,
		                          "Compute norm aperture. The"
		                          "output file will be a Histogram3D"};

		if (!config.loadFromCommandLine(argc, argv))
		{
			// "--help" requested. Quit
			return 0;
		}

		if (!config.validate())
		{
			std::cerr
			    << "Invalid configuration. Please check required parameters."
			    << std::endl;
			return -1;
		}

		auto scanner_fname = config.getValue<std::string>("scanner");
		auto out_fname = config.getValue<std::string>("out");
		auto numThreads = config.getValue<int>("num_threads");

		globals::setNumThreads(numThreads);
		numThreads = globals::getNumThreads();

		auto scanner = std::make_unique<Scanner>(scanner_fname);
		float hsx = scanner->crystalDepth / 2.0f;
		float hsy = scanner->crystalSize_trans / 2.0f;
		float hsz = scanner->crystalSize_z / 2.0f;
		Array2DOwned<float> lut;
		scanner->createLUT(lut);
		float* lutPtr = lut.getRawPointer();

		// Create histogram
		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		histo->clearProjections(0.0f);
		size_t totalBins = histo->count();

		util::ProgressDisplayMultiThread progressBar(numThreads, totalBins, 5);

		util::parallelForChunked(
		    totalBins, numThreads,
		    [&progressBar, &histo, lutPtr, hsx, hsy, hsz](size_t binId,
		                                                  size_t threadId)
		    {
			    progressBar.incrementProgress(threadId, 1);
			    // Get the active detector IDs for this specific histogram bin
			    det_pair_t detPair = histo->getDetectorPair(binId);
			    det_id_t det_id_0 = detPair.d1;
			    det_id_t det_id_1 = detPair.d2;

			    // Skip invalid detector configurations or self-coincidences
			    if (det_id_0 == det_id_1)
				    return;

			    // Fetch crystal poses from Look-Up-Table
			    auto& [cx0, cy0, cz0, nx0, ny0, nz0] =
			        reinterpret_cast<float(&)[6]>(lutPtr[6 * det_id_0]);
			    auto& [cx1, cy1, cz1, nx1, ny1, nz1] =
			        reinterpret_cast<float(&)[6]>(lutPtr[6 * det_id_1]);


			    // Compute 16 absolute 3D coordinates for both crystal elements
			    std::vector<Point3D> edges_0 = getCrystalEdges(
			        cx0, cy0, cz0, nx0, ny0, nz0, hsx, hsy, hsz);
			    std::vector<Point3D> edges_1 = getCrystalEdges(
			        cx1, cy1, cz1, nx1, ny1, nz1, hsx, hsy, hsz);

			    // Define the orthogonal LOR coordinates system
			    float nlor_x = cx1 - cx0;
			    float nlor_y = cy1 - cy0;
			    float nlor_z = cz1 - cz0;
			    float len_lor = std::sqrt(nlor_x * nlor_x + nlor_y * nlor_y +
			                              nlor_z * nlor_z);
			    if (len_lor < 1e-5f)
				    return;  // Safeguard for overlapping detectors

			    nlor_x /= len_lor;
			    nlor_y /= len_lor;
			    nlor_z /= len_lor;

			    Point3D clor = {(cx1 + cx0) / 2.0f, (cy1 + cy0) / 2.0f,
			                    (cz1 + cz0) / 2.0f};

			    // ulor = nlor x naz (where naz = 0, 0, 1)
			    float ulor_x = nlor_y;
			    float ulor_y = -nlor_x;
			    float ulor_z = 0.0f;
			    float len_u = std::sqrt(ulor_x * ulor_x + ulor_y * ulor_y);
			    if (len_u < 1e-5f)
				    return;  // Safeguard if parallel to naz
			    ulor_x /= len_u;
			    ulor_y /= len_u;

			    // vlor = nlor x ulor
			    float vlor_x = nlor_y * ulor_z - nlor_z * ulor_y;
			    float vlor_y = nlor_z * ulor_x - nlor_x * ulor_z;
			    float vlor_z = nlor_x * ulor_y - nlor_y * ulor_x;

			    // Flatten the 16 3D space points into the 2D plane subspace via
			    // Dot Product offsets
			    std::vector<Point2D> points_2d;
			    points_2d.reserve(16);

			    auto projectTo2D = [&](const std::vector<Point3D>& edges)
			    {
				    for (const auto& pt : edges)
				    {
					    float dx = pt.x - clor.x;
					    float dy = pt.y - clor.y;
					    float dz = pt.z - clor.z;
					    float u_coord = dx * ulor_x + dy * ulor_y + dz * ulor_z;
					    float v_coord = dx * vlor_x + dy * vlor_y + dz * vlor_z;
					    points_2d.push_back({u_coord, v_coord});
				    }
			    };
			    projectTo2D(edges_0);
			    projectTo2D(edges_1);

			    // Compute the 2D Convex Hull
			    std::vector<Point2D> hull_2d = convexHull2D(points_2d);

			    // Compute the flat area directly using the shoelace equation
			    float intersectionArea = computePolygonArea2D(hull_2d);

			    // Save the value safely back into the active histogram bin ID
			    histo->setProjectionValue(binId, intersectionArea);
		    });

		histo->writeToFile(out_fname);

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
		util::printExceptionMessage(e);
		return -1;
	}
}
