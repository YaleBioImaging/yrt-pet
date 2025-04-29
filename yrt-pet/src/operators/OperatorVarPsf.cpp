/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

 #include "operators/OperatorVarPsf.hpp"

 #include "utils/Assert.hpp"
 #include "utils/Tools.hpp"
 #include <fstream>
 #include <sstream>
 #include <stdexcept>
 #include <iostream>
 #include <chrono>
 
 #if BUILD_PYBIND11
 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
 namespace py = pybind11;
 
 void py_setup_operatorpsf(py::module& m)
 {
     auto c = py::class_<OperatorPsf, Operator>(m, "OperatorPsf");
     c.def(py::init<>());
     c.def(py::init<const std::string&>());
     c.def("readFromFile", &OperatorPsf::readFromFile);
     c.def("convolve", &OperatorPsf::convolve);
     c.def(
         "applyA", [](OperatorPsf& self, const Image* img_in, Image* img_out)
         { self.applyA(img_in, img_out); }, py::arg("img_in"),
         py::arg("img_out"));
     c.def(
         "applyAH", [](OperatorPsf& self, const Image* img_in, Image* img_out)
         { self.applyAH(img_in, img_out); }, py::arg("img_in"),
         py::arg("img_out"));
 }
 #endif

 OperatorVarPsf::OperatorVarPsf() : Operator{} {}
 
 OperatorVarPsf::OperatorVarPsf(const std::string& imageVarPsf_fname) : OperatorVarPsf{}
 {
     readFromFileInternal(imageVarPsf_fname);
 }
 
 void OperatorVarPsf::readFromFile(const std::string& imageVarPsf_fname)
 {
     readFromFileInternal(imageVarPsf_fname);
 }
 Sigma OperatorVarPsf::find_nearest_sigma(const std::vector<Sigma> &sigma_lookup, float x, float y, float z) const 
 {
    int x_dim = static_cast<int>(std::floor(x_range / x_gap)) + 1;
    int y_dim = static_cast<int>(std::floor(y_range / y_gap)) + 1;
    int z_dim = static_cast<int>(std::floor(z_range / z_gap)) + 1;

    int i = static_cast<int>(std::round(x / x_gap));
    int j = static_cast<int>(std::round(y / y_gap));
    int k = static_cast<int>(std::round(z / z_gap));

    if (i>=x_dim) i = x_dim-1;
    if (j>=y_dim) j = y_dim-1;
    if (k>=z_dim) k = z_dim-1;

    int index = IDX3(i, j, k, x_dim, y_dim);
    Sigma nearest_sigma = sigma_lookup[index];

    return nearest_sigma;
 }
 void OperatorVarPsf::readFromFileInternal(const std::string& imageVarPsf_fname)
 {
     std::cout << "Reading image space Variant PSF sigma lookup table file..." << std::endl;
     Array2D<float> data;
     Util::readCSV<float>(imageVarPsf_fname, data);
     
     size_t dims[2];
     data.getDims(dims);

     for (size_t i = 0; i < dims[0]; ++i)
     {
        if (dims[1] < 6) 
        {
            throw std::runtime_error("CSV file format error: Not enough columns");
        }
        Sigma s;
        s.x = data[i][0];
        s.y = data[i][1];
        s.z = data[i][2];
        s.sigmax = data[i][3];
        s.sigmay = data[i][4];
        s.sigmaz = data[i][5];
        sigma_lookup.push_back(s);
     }
 }
 
 void OperatorVarPsf::applyA(const Variable* in, Variable* out)
 {
     auto start = std::chrono::high_resolution_clock::now();
     const Image* img_in = dynamic_cast<const Image*>(in);
     Image* img_out = dynamic_cast<Image*>(out);
     ASSERT_MSG(img_in != nullptr && img_out != nullptr,
                "Input parameters must be images");
 
     varconvolve<true>(img_in, img_out);
     auto end = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> duration = end - start;
     std::cout << "Var PSF execution time: " << duration.count() << " seconds" << std::endl;

 }
 
 void OperatorVarPsf::applyAH(const Variable* in, Variable* out)
 {
     auto start = std::chrono::high_resolution_clock::now();
     const Image* img_in = dynamic_cast<const Image*>(in);
     Image* img_out = dynamic_cast<Image*>(out);
     ASSERT_MSG(img_in != nullptr && img_out != nullptr,
                "Input parameters must be images");
     varconvolve<false>(img_in, img_out);
     auto end = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> duration = end - start;
     std::cout << "Var Transposed PSF execution time: " << duration.count() << " seconds" << std::endl;
 }
 
 template <bool IS_FWD>
 void OperatorVarPsf::varconvolve(const Image* in, Image* out) const
 {
     const ImageParams& params = in->getParams();
     ASSERT_MSG(params.isSameDimensionsAs(out->getParams()),
                "Dimensions mismatch between the two images");
     const float* inPtr = in->getRawPointer();
     float* outPtr = out->getRawPointer();
     const int nx = params.nx;
     const int ny = params.ny;
     const int nz = params.nz;
     const float vx = params.vx;
     const float vy = params.vy;
     const float vz = params.vz;
     float x_center = nx * vx /2.0f;
     float y_center = ny * vy /2.0f;
     float z_center = nz * vz /2.0f;
 
     const size_t sizeBuffer = std::max(std::max(nx, ny), nz);
     m_buffer_tmp.resize(sizeBuffer);
 
     float xoffset,yoffset,zoffset,temp_x,temp_y,temp_z;
     int ii,jj,kk,kernel_size_x,kernel_size_y,kernel_size_z;
     int i,j,k;
     //padding 0
     #pragma omp parallel for private(temp_x,temp_y,temp_z,i,j,k,ii,jj,kk,kernel_size_x,kernel_size_y,kernel_size_z,xoffset,yoffset,zoffset) shared(outPtr)
     for (int pp =0; pp<nx*ny*nz;pp++)
     {
        i = pp % nx;
        j = (pp / nx) % ny;
        k = pp / (nx * ny);
        temp_x = std::abs((i+0.5) * vx-x_center);
        temp_y = std::abs((j+0.5) * vy-y_center);
        temp_z = std::abs((k+0.5) * vz-z_center);
        Sigma s = find_nearest_sigma(sigma_lookup, temp_x, temp_y, temp_z);
        float N_temp = N/(s.sigmax*s.sigmay*s.sigmaz)*vx*vy*vz;
        kernel_size_x = static_cast<int>(std::floor((s.sigmax*kernel_width_control)/vx))-1;
        kernel_size_y = static_cast<int>(std::floor((s.sigmay*kernel_width_control)/vy))-1;
        kernel_size_z = static_cast<int>(std::floor((s.sigmaz*kernel_width_control)/vz))-1;
        std::vector<std::vector<std::vector<float>>> psf_kernel(kernel_size_x*2+1, std::vector<std::vector<float>>(kernel_size_y*2+1, std::vector<float>(kernel_size_z*2+1, 0.0f))); 
        float inv_2_sigmax2 = 1.0f / (2 * s.sigmax * s.sigmax);
        float inv_2_sigmay2 = 1.0f / (2 * s.sigmay * s.sigmay);
        float inv_2_sigmaz2 = 1.0f / (2 * s.sigmaz * s.sigmaz);
        for (int x_diff = -kernel_size_x;x_diff<=kernel_size_x;x_diff++)
        for (int y_diff = -kernel_size_y;y_diff<=kernel_size_y;y_diff++)
        for (int z_diff = -kernel_size_z;z_diff<=kernel_size_z;z_diff++)
        {
            xoffset = x_diff*vx;
            yoffset = y_diff*vy;
            zoffset = z_diff*vz;
            float temp;
            temp = (-xoffset*xoffset*inv_2_sigmax2)+(-yoffset*yoffset*inv_2_sigmay2)+(-zoffset*zoffset*inv_2_sigmaz2);
            psf_kernel[x_diff + kernel_size_x][y_diff + kernel_size_y][z_diff + kernel_size_z] = exp(temp);
        }
        float temp1 = inPtr[IDX3(i, j, k, nx, ny)];
        for (int x_diff = -kernel_size_x;x_diff<=kernel_size_x;x_diff++)
        for (int y_diff = -kernel_size_y;y_diff<=kernel_size_y;y_diff++)
        for (int z_diff = -kernel_size_z;z_diff<=kernel_size_z;z_diff++)
        {
            ii = i+x_diff;
            jj = j+y_diff;
            kk = k+z_diff;
            if (ii>=0 && ii<nx && jj>=0 && jj<ny && kk>=0 && kk<nz)
            {
		        if constexpr (IS_FWD)
                {
                    #pragma omp atomic
                    outPtr[IDX3(i, j, k, nx, ny)] += inPtr[IDX3(ii, jj, kk, nx, ny)]*N_temp*psf_kernel[x_diff+kernel_size_x][y_diff+kernel_size_y][z_diff+kernel_size_z];
                }
                else
                {
                    #pragma omp atomic
                    outPtr[IDX3(ii, jj, kk, nx, ny)] += temp1*N_temp*psf_kernel[x_diff+kernel_size_x][y_diff+kernel_size_y][z_diff+kernel_size_z];
                }
            }
        }
     }
 }
