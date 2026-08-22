/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot {

class GOBOT_EXPORT SurfaceMesh : public Resource {
    GOBCLASS(SurfaceMesh, Resource)

public:
    SurfaceMesh() = default;

    void SetVertices(const std::vector<Vector3>& vertices);
    const std::vector<Vector3>& GetVertices() const;

    void SetTriangles(const std::vector<std::uint32_t>& triangles);
    const std::vector<std::uint32_t>& GetTriangles() const;

    std::size_t GetVertexCount() const;
    std::size_t GetTriangleCount() const;

    bool Validate(std::string* error = nullptr) const;

private:
    std::vector<Vector3> vertices_;
    std::vector<std::uint32_t> triangles_;
};

} // namespace gobot
