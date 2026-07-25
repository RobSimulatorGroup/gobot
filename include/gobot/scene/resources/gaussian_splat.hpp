/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/aabb.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gobot {

// Backend-independent, activated 3D Gaussian data. SH coefficients are stored
// coefficient-major per point: [point][coefficient][rgb].
struct GaussianSplatData {
    std::size_t count = 0;
    int sh_degree = 0;
    std::vector<float> means;
    std::vector<float> rotations_wxyz;
    std::vector<float> scales;
    std::vector<float> opacities;
    std::vector<float> sh_coefficients;
    AABB bounds;

    [[nodiscard]] std::size_t GetCoefficientCount() const {
        const int side = sh_degree + 1;
        return static_cast<std::size_t>(side * side);
    }

    [[nodiscard]] bool IsValid() const;
};

class GOBOT_EXPORT GaussianSplatResource : public Resource {
    GOBCLASS(GaussianSplatResource, Resource)

public:
    void SetSourcePath(const std::string& path);
    [[nodiscard]] const std::string& GetSourcePath() const;

    [[nodiscard]] std::size_t GetGaussianCount() const;
    [[nodiscard]] int GetShDegree() const;
    [[nodiscard]] AABB GetBounds() const;
    [[nodiscard]] std::shared_ptr<const GaussianSplatData> GetData() const;

    bool LoadPly(const std::string& path, std::string* error = nullptr);
    void ResetState() override;

private:
    std::string source_path_;
    std::shared_ptr<const GaussianSplatData> data_;
};

}
