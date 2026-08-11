/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/math_defs.hpp"

namespace gobot {

class GOBOT_EXPORT PhysicsMaterial3D : public Resource {
    GOBCLASS(PhysicsMaterial3D, Resource)

public:
    PhysicsMaterial3D() = default;

    void SetSlidingFriction(RealType value);
    RealType GetSlidingFriction() const;

    void SetTorsionalFriction(RealType value);
    RealType GetTorsionalFriction() const;

    void SetRollingFriction(RealType value);
    RealType GetRollingFriction() const;

    void SetRestitution(RealType value);
    RealType GetRestitution() const;

    void SetContactCompliance(RealType value);
    RealType GetContactCompliance() const;

    void SetContactDamping(RealType value);
    RealType GetContactDamping() const;

private:
    RealType sliding_friction_{1.0};
    RealType torsional_friction_{0.005};
    RealType rolling_friction_{0.0001};
    RealType restitution_{0.0};
    RealType contact_compliance_{0.0};
    RealType contact_damping_{1.0};
};

} // namespace gobot
