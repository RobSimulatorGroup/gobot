/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/rid.hpp"

#include <vector>

namespace gobot {

class GOBOT_EXPORT ArrayMesh : public Mesh {
    GOBCLASS(ArrayMesh, Mesh)
public:
    ArrayMesh();

    ~ArrayMesh() override;

    void SetSurface(std::vector<Vector3> vertices,
                    std::vector<uint32_t> indices,
                    std::vector<Vector3> normals = {},
                    std::vector<Color> colors = {});

    const std::vector<Vector3>& GetVertices() const;

    const std::vector<uint32_t>& GetIndices() const;

    const std::vector<Vector3>& GetNormals() const;

    const std::vector<Color>& GetColors() const;

    void SetMaterial(const Ref<Material>& material);

    const Ref<Material>& GetMaterial() const;

    RID GetRid() const override;

private:
    void UploadSurface();

    RID mesh_;
    std::vector<Vector3> vertices_;
    std::vector<uint32_t> indices_;
    std::vector<Vector3> normals_;
    std::vector<Color> colors_;
    Ref<Material> material_{nullptr};
};

} // namespace gobot
