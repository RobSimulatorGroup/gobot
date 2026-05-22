/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gobot/physics/physics_types.hpp"

namespace gobot {

enum class SimulationEntityBaseType {
    FixedBase,
    FloatingBase
};

enum class SimulationEntityArticulationType {
    Rigid,
    Articulated
};

enum class SimulationEntityControlType {
    Passive,
    Actuated,
    Kinematic
};

class GOBOT_EXPORT SimulationEntity {
public:
    SimulationEntity() = default;

    explicit SimulationEntity(const PhysicsRobotSnapshot& robot_snapshot);

    const std::string& GetName() const;

    SimulationEntityBaseType GetBaseType() const;

    SimulationEntityArticulationType GetArticulationType() const;

    SimulationEntityControlType GetControlType() const;

    bool IsFixedBase() const;

    bool IsFloatingBase() const;

    bool IsArticulated() const;

    bool IsActuated() const;

    bool IsKinematic() const;

    const std::vector<std::string>& GetLinkNames() const;

    const std::vector<std::string>& GetJointNames() const;

    const std::vector<std::string>& GetControllableJointNames() const;

    bool HasLink(const std::string& link_name) const;

    bool HasJoint(const std::string& joint_name) const;

    bool HasControllableJoint(const std::string& joint_name) const;

    std::size_t GetJointCount() const;

    std::size_t GetControllableJointCount() const;

private:
    static bool IsControllableJointType(int joint_type);

    std::string name_;
    SimulationEntityBaseType base_type_{SimulationEntityBaseType::FixedBase};
    SimulationEntityArticulationType articulation_type_{SimulationEntityArticulationType::Rigid};
    SimulationEntityControlType control_type_{SimulationEntityControlType::Passive};
    std::vector<std::string> link_names_;
    std::vector<std::string> joint_names_;
    std::vector<std::string> controllable_joint_names_;
};

} // namespace gobot
