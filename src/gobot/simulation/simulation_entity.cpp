/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/simulation_entity.hpp"

#include <algorithm>

#include "gobot/core/robotics_types.hpp"

namespace gobot {

SimulationEntity::SimulationEntity(const PhysicsRobotSnapshot& robot_snapshot)
    : name_(robot_snapshot.name) {
    bool has_floating_joint = false;
    bool has_non_floating_joint = false;
    bool has_controllable_joint = false;
    bool has_physical_link = false;

    link_names_.reserve(robot_snapshot.links.size());
    for (const PhysicsLinkSnapshot& link : robot_snapshot.links) {
        link_names_.push_back(link.name);
        if (link.role == PhysicsLinkRole::Physical) {
            has_physical_link = true;
        }
    }

    joint_names_.reserve(robot_snapshot.joints.size());
    for (const PhysicsJointSnapshot& joint : robot_snapshot.joints) {
        joint_names_.push_back(joint.name);
        const auto joint_type = static_cast<JointType>(joint.joint_type);
        if (joint_type == JointType::Floating) {
            has_floating_joint = true;
            continue;
        }

        has_non_floating_joint = true;
        if (IsControllableJointType(joint.joint_type)) {
            controllable_joint_names_.push_back(joint.name);
            has_controllable_joint = true;
        }
    }

    base_type_ = has_floating_joint ? SimulationEntityBaseType::FloatingBase
                                    : SimulationEntityBaseType::FixedBase;
    articulation_type_ = has_non_floating_joint ? SimulationEntityArticulationType::Articulated
                                                : SimulationEntityArticulationType::Rigid;
    if (!has_physical_link && base_type_ == SimulationEntityBaseType::FixedBase) {
        control_type_ = SimulationEntityControlType::Kinematic;
    } else {
        control_type_ = has_controllable_joint ? SimulationEntityControlType::Actuated
                                               : SimulationEntityControlType::Passive;
    }
}

const std::string& SimulationEntity::GetName() const {
    return name_;
}

SimulationEntityBaseType SimulationEntity::GetBaseType() const {
    return base_type_;
}

SimulationEntityArticulationType SimulationEntity::GetArticulationType() const {
    return articulation_type_;
}

SimulationEntityControlType SimulationEntity::GetControlType() const {
    return control_type_;
}

bool SimulationEntity::IsFixedBase() const {
    return base_type_ == SimulationEntityBaseType::FixedBase;
}

bool SimulationEntity::IsFloatingBase() const {
    return base_type_ == SimulationEntityBaseType::FloatingBase;
}

bool SimulationEntity::IsArticulated() const {
    return articulation_type_ == SimulationEntityArticulationType::Articulated;
}

bool SimulationEntity::IsActuated() const {
    return control_type_ == SimulationEntityControlType::Actuated;
}

bool SimulationEntity::IsKinematic() const {
    return control_type_ == SimulationEntityControlType::Kinematic;
}

const std::vector<std::string>& SimulationEntity::GetLinkNames() const {
    return link_names_;
}

const std::vector<std::string>& SimulationEntity::GetJointNames() const {
    return joint_names_;
}

const std::vector<std::string>& SimulationEntity::GetControllableJointNames() const {
    return controllable_joint_names_;
}

bool SimulationEntity::HasLink(const std::string& link_name) const {
    return std::find(link_names_.begin(), link_names_.end(), link_name) != link_names_.end();
}

bool SimulationEntity::HasJoint(const std::string& joint_name) const {
    return std::find(joint_names_.begin(), joint_names_.end(), joint_name) != joint_names_.end();
}

bool SimulationEntity::HasControllableJoint(const std::string& joint_name) const {
    return std::find(controllable_joint_names_.begin(),
                     controllable_joint_names_.end(),
                     joint_name) != controllable_joint_names_.end();
}

std::size_t SimulationEntity::GetJointCount() const {
    return joint_names_.size();
}

std::size_t SimulationEntity::GetControllableJointCount() const {
    return controllable_joint_names_.size();
}

bool SimulationEntity::IsControllableJointType(int joint_type) {
    const auto type = static_cast<JointType>(joint_type);
    return type == JointType::Revolute ||
           type == JointType::Continuous ||
           type == JointType::Prismatic;
}

} // namespace gobot
