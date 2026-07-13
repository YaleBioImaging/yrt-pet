/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
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
	Point3D operator+(const Point3D& o) const
	{
		return {x + o.x, y + o.y, z + o.z};
	}
	Point3D operator-(const Point3D& o) const
	{
		return {x - o.x, y - o.y, z - o.z};
	}
	Point3D operator*(float s) const { return {x * s, y * s, z * s}; }
};

struct Point2D
{
	float u, v;
	static float cross(const Point2D& O, const Point2D& A, const Point2D& B)
	{
		return (A.u - O.u) * (B.v - O.v) - (A.v - O.v) * (B.u - O.u);
	}
	bool operator<(const Point2D& other) const
	{
		return u < other.u || (u == other.u && v < other.v);
	}
};

struct OBBCrystal
{
	Point3D center;
	Point3D u, v, w;
	Point3D half_dims;
	int layer_id;  // Added to map its specific DOI attenuation factor

	OBBCrystal(float cx, float cy, float cz, float nx, float ny, float nz,
	           float hsx, float hsy, float hsz, int layer)
	{
		center = {cx, cy, cz};
		half_dims = {hsx, hsy, hsz};
		layer_id = layer;

		u = {nx, ny, 0.0f};
		float len_u = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
		if (len_u > 1e-6f)
		{
			u.x /= len_u;
			u.y /= len_u;
			u.z /= len_u;
		}

		w = {0.0f, 0.0f, 1.0f};
		float dot_wu = w.x * u.x + w.y * u.y + w.z * u.z;
		w.x -= dot_wu * u.x;
		w.y -= dot_wu * u.y;
		w.z -= dot_wu * u.z;
		float len_w = std::sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
		if (len_w > 1e-6f)
		{
			w.x /= len_w;
			w.y /= len_w;
			w.z /= len_w;
		}

		v.x = w.y * u.z - w.z * u.y;
		v.y = w.z * u.x - w.x * u.z;
		v.z = w.x * u.y - w.y * u.x;
	}

	inline float intersectRayLength(const Point3D& ray_org,
	                                const Point3D& ray_dir,
	                                float total_len) const
	{
		float t_enter = -std::numeric_limits<float>::infinity();
		float t_exit = std::numeric_limits<float>::infinity();
		Point3D d = ray_org - center;

		std::array<std::pair<float, float>, 3> projections = {
		    {{d.x * u.x + d.y * u.y + d.z * u.z,
		      ray_dir.x * u.x + ray_dir.y * u.y + ray_dir.z * u.z},
		     {d.x * v.x + d.y * v.y + d.z * v.z,
		      ray_dir.x * v.x + ray_dir.y * v.y + ray_dir.z * v.z},
		     {d.x * w.x + d.y * w.y + d.z * w.z,
		      ray_dir.x * w.x + ray_dir.y * w.y + ray_dir.z * w.z}}};

		const float* hdims_ptr = &half_dims.x;
		for (int i = 0; i < 3; ++i)
		{
			float e = projections[i].first;
			float f = projections[i].second;
			float h = hdims_ptr[i];

			if (std::abs(f) > 1e-9f)
			{
				float t1 = (-e - h) / f;
				float t2 = (-e + h) / f;
				if (t1 > t2)
					std::swap(t1, t2);
				if (t1 > t_enter)
					t_enter = t1;
				if (t2 < t_exit)
					t_exit = t2;
				if (t_enter > t_exit)
					return 0.0f;
			}
			else if (-e - h > 0.0f || -e + h < 0.0f)
			{
				return 0.0f;
			}
		}

		float t_enter_clamped = std::max(0.0f, t_enter);
		float t_exit_clamped = std::min(1.0f, t_exit);

		if (t_enter_clamped < t_exit_clamped && t_exit > 0.0f)
		{
			return (t_exit_clamped - t_enter_clamped) * total_len;
		}
		return 0.0f;
	}
};

inline std::vector<Point2D> convexHull2D(std::vector<Point2D>& pts)
{
	size_t n = pts.size(), k = 0;
	if (n <= 3)
		return pts;
	std::vector<Point2D> res(2 * n);
	std::sort(pts.begin(), pts.end());
	for (size_t i = 0; i < n; ++i)
	{
		while (k >= 2 && Point2D::cross(res[k - 2], res[k - 1], pts[i]) <= 0)
			k--;
		res[k++] = pts[i];
	}
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

inline std::vector<Point3D> getCrystalEdges(float cx, float cy, float cz,
                                            float nx, float ny, float nz,
                                            float hsx, float hsy, float hsz)
{
	std::vector<Point3D> edges;
	edges.reserve(8);
	float u_x = nx, u_y = ny, u_z = 0.0f;
	float v_x = -ny, v_y = nx, v_z = 0.0f;
	float w_x = 0.0f, w_y = 0.0f, w_z = 1.0f;

	for (float sz : {-1.f, 1.f})
		for (float sy : {-1.f, 1.f})
			for (float sx : {-1.f, 1.f})
			{
				float dx = sx * hsx * u_x + sy * hsy * v_x + sz * hsz * w_x;
				float dy = sx * hsx * u_y + sy * hsy * v_y + sz * hsz * w_y;
				float dz = sx * hsx * u_z + sy * hsy * v_z + sz * hsz * w_z;
				edges.push_back({cx + dx, cy + dy, cz + dz});
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

inline std::array<Point3D, 27> sampleVolumePoints(const OBBCrystal& crystal)
{
	std::array<Point3D, 27> points;
	int idx = 0;
	std::array<float, 3> steps = {crystal.half_dims.x * 2.0f / 3.0f,
	                              crystal.half_dims.y * 2.0f / 3.0f,
	                              crystal.half_dims.z * 2.0f / 3.0f};
	std::array<float, 3> start_offsets = {
	    -crystal.half_dims.x + steps[0] / 2.0f,
	    -crystal.half_dims.y + steps[1] / 2.0f,
	    -crystal.half_dims.z + steps[2] / 2.0f};

	for (int sz = 0; sz < 3; ++sz)
	{
		float lz = start_offsets[2] + sz * steps[2];
		for (int sy = 0; sy < 3; ++sy)
		{
			float ly = start_offsets[1] + sy * steps[1];
			for (int sx = 0; sx < 3; ++sx)
			{
				float lx = start_offsets[0] + sx * steps[0];
				points[idx++] = crystal.center + crystal.u * lx +
				                crystal.v * ly + crystal.w * lz;
			}
		}
	}
	return points;
}

int main(int argc, char* argv[])
{
	try
	{
		io::ArgumentRegistry registry{};
		std::string coreGroup = "0. Core";
		std::string outputGroup = "2. Output";

		registry.registerArgument("scanner", "Scanner parameters file", true,
		                          io::TypeOfArgument::STRING, "", coreGroup,
		                          "s");
		registry.registerArgument("num_threads", "Number of threads to use",
		                          false, io::TypeOfArgument::INT, -1,
		                          coreGroup);
		registry.registerArgument("mu",
		                          "Comma-separated attenuation values per DOI "
		                          "layer (e.g. 0.086,0.081)",
		                          true, io::TypeOfArgument::STRING, "0.086",
		                          coreGroup, "m");
		registry.registerArgument("out", "Output image filename", true,
		                          io::TypeOfArgument::STRING, "", outputGroup,
		                          "o");

		io::ArgumentReader config{registry,
		                          "Compute comprehensive DOI norm elements."};
		if (!config.loadFromCommandLine(argc, argv))
			return 0;
		if (!config.validate())
			return -1;

		auto scanner_fname = config.getValue<std::string>("scanner");
		auto out_fname = config.getValue<std::string>("out");
		auto numThreads = config.getValue<int>("num_threads");
		auto mu_str = config.getValue<std::string>("mu");

		globals::setNumThreads(numThreads);
		numThreads = globals::getNumThreads();

		auto scanner = std::make_unique<Scanner>(scanner_fname);
		float hsx = scanner->crystalDepth / 2.0f;
		float hsy = scanner->crystalSize_trans / 2.0f;
		float hsz = scanner->crystalSize_z / 2.0f;

		// Determine DOI properties from the scanner layout configuration
		int num_doi_layers =
		    (scanner->numDOI > 0) ? scanner->numDOI : 1;

		// Parse Comma-Separated User Mu Values List
		std::vector<float> mu_layers;
		std::stringstream ss(mu_str);
		std::string item;
		while (std::getline(ss, item, ','))
		{
			mu_layers.push_back(std::stof(item));
		}

		// Apply fallback constraint logic rule if lengths mismatch
		if (mu_layers.size() < static_cast<size_t>(num_doi_layers))
		{
			float fallback_mu = mu_layers.back();
			while (mu_layers.size() < static_cast<size_t>(num_doi_layers))
			{
				mu_layers.push_back(fallback_mu);
			}
		}

		Array2DOwned<float> lut;
		scanner->createLUT(lut);
		float* lutPtr = lut.getRawPointer();
		size_t totalCrystals = lut.getSize(0);

		std::vector<Point3D> lut_mins(totalCrystals);
		std::vector<Point3D> lut_maxs(totalCrystals);
		std::vector<OBBCrystal> global_crystals;
		global_crystals.reserve(totalCrystals);

		for (size_t i = 0; i < totalCrystals; ++i)
		{
			float cx = lutPtr[6 * i + 0];
			float cy = lutPtr[6 * i + 1];
			float cz = lutPtr[6 * i + 2];
			float nx = lutPtr[6 * i + 3];
			float ny = lutPtr[6 * i + 4];
			float nz = lutPtr[6 * i + 5];

			// Resolve modern tracking layer identification safely from scanner
			// object lookup parameters
			int derived_layer = i / (scanner->detsPerRing * scanner->numRings);

			OBBCrystal cr(cx, cy, cz, nx, ny, nz, hsx, hsy, hsz, derived_layer);
			global_crystals.push_back(cr);

			float dx_span = hsx * std::abs(cr.u.x) + hsy * std::abs(cr.v.x) +
			                hsz * std::abs(cr.w.x);
			float dy_span = hsx * std::abs(cr.u.y) + hsy * std::abs(cr.v.y) +
			                hsz * std::abs(cr.w.y);
			float dz_span = hsx * std::abs(cr.u.z) + hsy * std::abs(cr.v.z) +
			                hsz * std::abs(cr.w.z);

			lut_mins[i] = {cx - dx_span, cy - dy_span, cz - dz_span};
			lut_maxs[i] = {cx + dx_span, cy + dy_span, cz + dz_span};
		}

		auto histo = std::make_unique<Histogram3DOwned>(*scanner);
		histo->allocate();
		histo->clearProjections(0.0f);
		size_t totalBins = histo->count();

		util::ProgressDisplayMultiThread progressBar(numThreads, totalBins, 5);

		util::parallelForChunked(
		    totalBins, numThreads,
		    [&progressBar, &histo, &global_crystals, &lut_mins, &lut_maxs,
		     totalCrystals, hsx, hsy, hsz, mu_layers,
		     lutPtr](size_t binId, size_t threadId)
		    {
			    progressBar.incrementProgress(threadId, 1);
			    det_pair_t detPair = histo->getDetectorPair(binId);
			    det_id_t det_id_0 = detPair.d1;
			    det_id_t det_id_1 = detPair.d2;

			    if (det_id_0 == det_id_1)
				    return;

			    // 1. APERTURE AREA CALCULATION
			    auto& [cx0, cy0, cz0, nx0, ny0, nz0] =
			        reinterpret_cast<float(&)[6]>(lutPtr[6 * det_id_0]);
			    auto& [cx1, cy1, cz1, nx1, ny1, nz1] =
			        reinterpret_cast<float(&)[6]>(lutPtr[6 * det_id_1]);

			    std::vector<Point3D> edges_0 = getCrystalEdges(
			        cx0, cy0, cz0, nx0, ny0, nz0, hsx, hsy, hsz);
			    std::vector<Point3D> edges_1 = getCrystalEdges(
			        cx1, cy1, cz1, nx1, ny1, nz1, hsx, hsy, hsz);

			    float nlor_x = cx1 - cx0;
			    float nlor_y = cy1 - cy0;
			    float nlor_z = cz1 - cz0;
			    float len_lor = std::sqrt(nlor_x * nlor_x + nlor_y * nlor_y +
			                              nlor_z * nlor_z);
			    if (len_lor < 1e-5f)
				    return;

			    nlor_x /= len_lor;
			    nlor_y /= len_lor;
			    nlor_z /= len_lor;
			    Point3D clor = {(cx1 + cx0) / 2.0f, (cy1 + cy0) / 2.0f,
			                    (cz1 + cz0) / 2.0f};

			    float ulor_x = nlor_y;
			    float ulor_y = -nlor_x;
			    float ulor_z = 0.0f;
			    float len_u = std::sqrt(ulor_x * ulor_x + ulor_y * ulor_y);
			    if (len_u < 1e-5f)
				    return;
			    ulor_x /= len_u;
			    ulor_y /= len_u;

			    float vlor_x = nlor_y * ulor_z - nlor_z * ulor_y;
			    float vlor_y = nlor_z * ulor_x - nlor_x * ulor_z;
			    float vlor_z = nlor_x * ulor_y - nlor_y * ulor_x;

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

			    std::vector<Point2D> hull_2d = convexHull2D(points_2d);
			    float intersectionArea = computePolygonArea2D(hull_2d);

			    // 2. CANDIDATES SELECTION (AABB BOX INTERSECTION OVERLAP)
			    Point3D lor_min = {
			        std::min(lut_mins[det_id_0].x, lut_mins[det_id_1].x),
			        std::min(lut_mins[det_id_0].y, lut_mins[det_id_1].y),
			        std::min(lut_mins[det_id_0].z, lut_mins[det_id_1].z)};
			    Point3D lor_max = {
			        std::max(lut_maxs[det_id_0].x, lut_maxs[det_id_1].x),
			        std::max(lut_maxs[det_id_0].y, lut_maxs[det_id_1].y),
			        std::max(lut_maxs[det_id_0].z, lut_maxs[det_id_1].z)};

			    std::vector<size_t> candidate_ids;
			    candidate_ids.reserve(512);
			    for (size_t i = 0; i < totalCrystals; ++i)
			    {
				    if (i == det_id_0 || i == det_id_1)
					    continue;
				    if (lut_mins[i].x <= lor_max.x &&
				        lut_maxs[i].x >= lor_min.x &&
				        lut_mins[i].y <= lor_max.y &&
				        lut_maxs[i].y >= lor_min.y &&
				        lut_mins[i].z <= lor_max.z &&
				        lut_maxs[i].z >= lor_min.z)
				    {
					    candidate_ids.push_back(i);
				    }
			    }

			    // 3. PHYSICALLY RECTIFIED VOLUMETRIC COINCIDENCE PROBABILITY
			    // TRACING
			    const OBBCrystal& crystal_0 = global_crystals[det_id_0];
			    const OBBCrystal& crystal_1 = global_crystals[det_id_1];

			    std::array<Point3D, 27> pts_c0 = sampleVolumePoints(crystal_0);
			    std::array<Point3D, 27> pts_c1 = sampleVolumePoints(crystal_1);

			    float accumulated_detection_prob = 0.0f;
			    float mu0 = mu_layers[crystal_0.layer_id];
			    float mu1 = mu_layers[crystal_1.layer_id];

			    for (const auto& p0 : pts_c0)
			    {
				    for (const auto& p1 : pts_c1)
				    {
					    Point3D ray_dir = p1 - p0;
					    float ray_len = std::sqrt(ray_dir.x * ray_dir.x +
					                              ray_dir.y * ray_dir.y +
					                              ray_dir.z * ray_dir.z);
					    if (ray_len < 1e-5f)
						    continue;

					    float l0 =
					        crystal_0.intersectRayLength(p0, ray_dir, ray_len);
					    float l1 =
					        crystal_1.intersectRayLength(p0, ray_dir, ray_len);

					    float structural_shadow_exponent = 0.0f;
					    for (size_t cid : candidate_ids)
					    {
						    float l_interf =
						        global_crystals[cid].intersectRayLength(
						            p0, ray_dir, ray_len);
						    if (l_interf > 0.0f)
						    {
							    structural_shadow_exponent +=
							        mu_layers[global_crystals[cid].layer_id] *
							        l_interf;
						    }
					    }

					    // Mathematically rigorous calculation evaluated
					    // ray-by-ray matching PET imaging physics
					    float efficiency_c0 = 1.0f - std::exp(-mu0 * l0);
					    float efficiency_c1 = 1.0f - std::exp(-mu1 * l1);
					    float transmission_interf =
					        std::exp(-structural_shadow_exponent);

					    accumulated_detection_prob +=
					        (efficiency_c0 * efficiency_c1 *
					         transmission_interf);
				    }
			    }

			    float mean_detection_prob = accumulated_detection_prob / 729.0f;
			    float final_normalization_factor =
			        intersectionArea * mean_detection_prob;
			    histo->setProjectionValue(binId, final_normalization_factor);
		    });

		histo->writeToFile(out_fname);
		std::cout << "Done." << std::endl;
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Execution exception caught: " << e.what() << std::endl;
		return -1;
	}
}
