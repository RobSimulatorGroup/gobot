/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/math_defs.hpp"
#include <string>
#include <vector>

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

    // Minimum, maximum, width, midpoint, power of the contact impedance curve.
    void SetContactImpedance(const std::vector<RealType>& value);
    const std::vector<RealType>& GetContactImpedance() const;

    // Higher priority selects this material when combining a contact pair.
    void SetContactPriority(int value);
    int GetContactPriority() const;

    void SetContactFrictionMode(const std::string& value);
    const std::string& GetContactFrictionMode() const;

private:
    RealType sliding_friction_{1.0};
    RealType torsional_friction_{0.005};
    RealType rolling_friction_{0.0001};
    RealType restitution_{0.0};
    RealType contact_compliance_{0.0};
    RealType contact_damping_{1.0};
    std::vector<RealType> contact_impedance_{0.9, 0.95, 0.001, 0.5, 2.0};
    int contact_priority_{0};
    std::string contact_friction_mode_{"auto"};
};

} // namespace gobot
