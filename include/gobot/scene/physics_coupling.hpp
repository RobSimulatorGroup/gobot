/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/math/math_defs.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_path.hpp"

namespace gobot {

enum class PhysicsCouplingMode {
    OneWay,
    TwoWay,
};

class GOBOT_EXPORT PhysicsCoupling : public Node {
    GOBCLASS(PhysicsCoupling, Node)

public:
    void SetEnabled(bool enabled);
    [[nodiscard]] bool IsEnabled() const;

    void SetRigidLinkPath(const NodePath& path);
    [[nodiscard]] const NodePath& GetRigidLinkPath() const;

    void SetTargetBodyPath(const NodePath& path);
    [[nodiscard]] const NodePath& GetTargetBodyPath() const;

    void SetMode(PhysicsCouplingMode mode);
    [[nodiscard]] PhysicsCouplingMode GetMode() const;

    void SetForceScale(RealType scale);
    [[nodiscard]] RealType GetForceScale() const;

    void SetTorqueScale(RealType scale);
    [[nodiscard]] RealType GetTorqueScale() const;

private:
    bool enabled_{true};
    NodePath target_body_path_;
    PhysicsCouplingMode mode_{PhysicsCouplingMode::TwoWay};
    RealType force_scale_{1.0};
    RealType torque_scale_{1.0};
};

} // namespace gobot
