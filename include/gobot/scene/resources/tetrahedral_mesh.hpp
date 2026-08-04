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

class GOBOT_EXPORT TetrahedralMesh : public Resource {
    GOBCLASS(TetrahedralMesh, Resource)

public:
    TetrahedralMesh() = default;

    void SetVertices(const std::vector<Vector3>& vertices);
    const std::vector<Vector3>& GetVertices() const;

    void SetTetrahedra(const std::vector<std::uint32_t>& tetrahedra);
    const std::vector<std::uint32_t>& GetTetrahedra() const;

    void SetSurfaceTriangles(const std::vector<std::uint32_t>& surface_triangles);
    const std::vector<std::uint32_t>& GetSurfaceTriangles() const;

    std::vector<std::uint32_t> GetResolvedSurfaceTriangles() const;

    std::size_t GetVertexCount() const;
    std::size_t GetTetrahedronCount() const;

    bool Validate(std::string* error = nullptr) const;

private:
    std::vector<Vector3> vertices_;
    std::vector<std::uint32_t> tetrahedra_;
    std::vector<std::uint32_t> surface_triangles_;
};

} // namespace gobot
