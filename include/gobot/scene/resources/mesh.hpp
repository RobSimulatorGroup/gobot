/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/scene/resources/material.hpp"

#include <memory>
#include <vector>

namespace gobot {

struct MeshSurfaceData {
    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Vector3> normals;
    std::vector<Vector4> tangents;
    std::vector<Vector2> uv0;
    std::vector<Color> colors;
    Ref<Material> material;
};

using MeshSurfaceList = std::vector<MeshSurfaceData>;

GOBOT_EXPORT void CompleteMeshSurface(MeshSurfaceData& surface);

class GOBOT_EXPORT Mesh : public Resource {
    GOBCLASS(Mesh, Resource);
public:
    Mesh();

    [[nodiscard]] std::shared_ptr<const MeshSurfaceList> GetSurfaceData() const;

    [[nodiscard]] std::size_t GetSurfaceCount() const;

protected:
    void ReplaceSurfaceData(MeshSurfaceList surfaces);

private:
    std::shared_ptr<const MeshSurfaceList> surface_data_;

};

}
