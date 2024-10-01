/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/Variable.hpp"

#include "nlohmann/json_fwd.hpp"
#include <string>

#define IMAGEPARAMS_FILE_VERSION 1.1

class ImageParams
{
public:
    int nx;
    int ny;
    int nz;
    double length_x;
    double length_y;
    double length_z;
    double off_x;
    double off_y;
    double off_z;

    // Automatically populated fields
    double vx, vy, vz;
    double fovRadius;

    ImageParams();
    ImageParams(int nxi, int nyi, int nzi, double length_xi, double length_yi,
                double length_zi, double offset_xi = 0.,
                double offset_yi = 0., double offset_zi = 0.);
    ImageParams(const ImageParams& in);
    ImageParams& operator=(const ImageParams& in);
    explicit ImageParams(const std::string& fname);
    bool isSameDimensionsAs(const ImageParams& other) const;
    bool isSameLengthsAs(const ImageParams& other) const;
    bool isSameOffsetsAs(const ImageParams& other) const;
    bool isSameAs(const ImageParams& other) const;

    void copy(const ImageParams& in);
    void setup();
    void serialize(const std::string& fname) const;
    void writeToJSON(nlohmann::json& j) const;
    void deserialize(const std::string& fname);
    void readFromJSON(nlohmann::json& j);
    bool isValid() const;

private:
    static double readLengthFromJSON(nlohmann::json& j, const std::string& length_name, const std::string& v_name, int n);
};

class ImageBase : public Variable
{
public:
    explicit ImageBase(const ImageParams& imgParams);
    ~ImageBase() override = default;

    // Common functions
    double getRadius() const;
    const ImageParams& getParams() const;
    void setParams(const ImageParams& newParams);

    virtual void setValue(double initValue) = 0;
    virtual void addFirstImageToSecond(ImageBase* second) const = 0;
    virtual void applyThreshold(const ImageBase* mask_img, double threshold,
                                double val_le_scale, double val_le_off,
                                double val_gt_scale, double val_gt_off) = 0;
    virtual void writeToFile(const std::string& image_fname) const = 0;
    virtual void updateEMThreshold(ImageBase* update_img,
                                   const ImageBase* norm_img,
                                   double threshold) = 0;

private:
    ImageParams m_params;
};
