/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include "gobot/core/math/math_defs.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_path.hpp"

namespace gobot {

class GOBOT_EXPORT DeformableAttachment3D : public Node {
    GOBCLASS(DeformableAttachment3D, Node)

public:
    void SetEnabled(bool enabled);
    [[nodiscard]] bool IsEnabled() const;

    void SetDeformableBodyPath(const NodePath& path);
    [[nodiscard]] const NodePath& GetDeformableBodyPath() const;

    void SetRigidLinkPath(const NodePath& path);
    [[nodiscard]] const NodePath& GetRigidLinkPath() const;

    void SetVertexIndices(const std::vector<std::uint32_t>& vertex_indices);
    [[nodiscard]] const std::vector<std::uint32_t>& GetVertexIndices() const;

    void SetStrengthRate(RealType strength_rate);
    [[nodiscard]] RealType GetStrengthRate() const;

private:
    bool enabled_{true};
    NodePath deformable_body_path_;
    NodePath rigid_link_path_;
    std::vector<std::uint32_t> vertex_indices_;
    RealType strength_rate_{100.0};
};

} // namespace gobot
