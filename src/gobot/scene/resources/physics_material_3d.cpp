/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/physics_material_3d.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

#define GOBOT_PHYSICS_MATERIAL_PROPERTY(Name, member)          \
    void PhysicsMaterial3D::Set##Name(RealType value) {        \
        if (member != value) {                                 \
            member = value;                                   \
            MarkChanged();                                    \
        }                                                      \
    }                                                          \
    RealType PhysicsMaterial3D::Get##Name() const { return member; }

GOBOT_PHYSICS_MATERIAL_PROPERTY(SlidingFriction, sliding_friction_)
GOBOT_PHYSICS_MATERIAL_PROPERTY(TorsionalFriction, torsional_friction_)
GOBOT_PHYSICS_MATERIAL_PROPERTY(RollingFriction, rolling_friction_)
GOBOT_PHYSICS_MATERIAL_PROPERTY(Restitution, restitution_)
GOBOT_PHYSICS_MATERIAL_PROPERTY(ContactCompliance, contact_compliance_)
GOBOT_PHYSICS_MATERIAL_PROPERTY(ContactDamping, contact_damping_)

#undef GOBOT_PHYSICS_MATERIAL_PROPERTY

void PhysicsMaterial3D::SetContactFrictionMode(const std::string& value) {
    if (value != "auto" && value != "frictionless" && value != "sliding" &&
        value != "sliding_torsional" && value != "sliding_torsional_rolling") {
        LOG_ERROR("Unsupported contact friction mode '{}'.", value);
        return;
    }
    if (contact_friction_mode_ != value) {
        contact_friction_mode_ = value;
        MarkChanged();
    }
}

const std::string& PhysicsMaterial3D::GetContactFrictionMode() const {
    return contact_friction_mode_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::PhysicsMaterial3D>("PhysicsMaterial3D")
            .constructor()(CtorAsRawPtr)
            .property("sliding_friction",
                      &gobot::PhysicsMaterial3D::GetSlidingFriction,
                      &gobot::PhysicsMaterial3D::SetSlidingFriction)
            .property("torsional_friction",
                      &gobot::PhysicsMaterial3D::GetTorsionalFriction,
                      &gobot::PhysicsMaterial3D::SetTorsionalFriction)
            .property("rolling_friction",
                      &gobot::PhysicsMaterial3D::GetRollingFriction,
                      &gobot::PhysicsMaterial3D::SetRollingFriction)
            .property("restitution",
                      &gobot::PhysicsMaterial3D::GetRestitution,
                      &gobot::PhysicsMaterial3D::SetRestitution)
            .property("contact_compliance",
                      &gobot::PhysicsMaterial3D::GetContactCompliance,
                      &gobot::PhysicsMaterial3D::SetContactCompliance)
            .property("contact_damping",
                      &gobot::PhysicsMaterial3D::GetContactDamping,
                      &gobot::PhysicsMaterial3D::SetContactDamping)
            .property("contact_friction_mode",
                      &gobot::PhysicsMaterial3D::GetContactFrictionMode,
                      &gobot::PhysicsMaterial3D::SetContactFrictionMode);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::PhysicsMaterial3D>, gobot::Ref<gobot::Resource>>();
}
