/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/physics_coupling.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void PhysicsCoupling::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

bool PhysicsCoupling::IsEnabled() const {
    return enabled_;
}

void PhysicsCoupling::SetRigidLinkPath(const NodePath& path) {
    rigid_link_path_ = path;
}

const NodePath& PhysicsCoupling::GetRigidLinkPath() const {
    return rigid_link_path_;
}

void PhysicsCoupling::SetMode(PhysicsCouplingMode mode) {
    mode_ = mode;
}

PhysicsCouplingMode PhysicsCoupling::GetMode() const {
    return mode_;
}

void PhysicsCoupling::SetForceScale(RealType scale) {
    force_scale_ = scale;
}

RealType PhysicsCoupling::GetForceScale() const {
    return force_scale_;
}

void PhysicsCoupling::SetTorqueScale(RealType scale) {
    torque_scale_ = scale;
}

RealType PhysicsCoupling::GetTorqueScale() const {
    return torque_scale_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    QuickEnumeration_<gobot::PhysicsCouplingMode>("PhysicsCouplingMode");

    Class_<gobot::PhysicsCoupling>("PhysicsCoupling")
            .constructor()(CtorAsRawPtr)
            .property("enabled", &gobot::PhysicsCoupling::IsEnabled,
                      &gobot::PhysicsCoupling::SetEnabled)
            .property("rigid_link_path", &gobot::PhysicsCoupling::GetRigidLinkPath,
                      &gobot::PhysicsCoupling::SetRigidLinkPath)
            .property("mode", &gobot::PhysicsCoupling::GetMode,
                      &gobot::PhysicsCoupling::SetMode)
            .property("force_scale", &gobot::PhysicsCoupling::GetForceScale,
                      &gobot::PhysicsCoupling::SetForceScale)
            .property("torque_scale", &gobot::PhysicsCoupling::GetTorqueScale,
                      &gobot::PhysicsCoupling::SetTorqueScale);
};
