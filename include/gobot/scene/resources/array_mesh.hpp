/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/rid.hpp"

#include <vector>

namespace gobot {

class GOBOT_EXPORT ArrayMesh : public Mesh {
    GOBCLASS(ArrayMesh, Mesh)
public:
    ArrayMesh();

    ~ArrayMesh() override;

    void SetSurface(std::vector<Vector3> vertices, std::vector<uint32_t> indices);

    const std::vector<Vector3>& GetVertices() const;

    const std::vector<uint32_t>& GetIndices() const;

    RID GetRid() const override;

private:
    void UploadSurface();

    RID mesh_;
    std::vector<Vector3> vertices_;
    std::vector<uint32_t> indices_;
};

} // namespace gobot
