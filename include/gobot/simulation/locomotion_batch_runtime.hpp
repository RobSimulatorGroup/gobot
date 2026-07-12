/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "gobot/physics/physics_types.hpp"
#include "gobot/simulation/locomotion_command_runtime.hpp"
#include "gobot_export.h"

namespace gobot {

class GOBOT_EXPORT LocomotionBatchRuntime {
public:
    LocomotionBatchRuntime(std::size_t environment_count = 0,
                           std::size_t foot_count = 0);

    std::size_t GetEnvironmentCount() const;

    std::size_t GetFootCount() const;

    LocomotionCommandRuntime& CommandRuntime();

    const LocomotionCommandRuntime& CommandRuntime() const;

    void ClearStepContacts(std::size_t environment_index);

    void ClearResetContacts(std::span<const std::size_t> environment_indices);

    void UpdateFootHistory(const PhysicsRobotBatchStepResult& physics_state,
                           std::span<const float> foot_heights,
                           RealType step_dt,
                           RealType physics_dt);

    std::span<float> FootContacts();
    std::span<const float> FootContacts() const;

    std::span<float> FootContactForces();
    std::span<const float> FootContactForces() const;

    std::span<float> IllegalContactCounts();
    std::span<const float> IllegalContactCounts() const;

    std::span<float> SelfCollisionCounts();
    std::span<const float> SelfCollisionCounts() const;

    std::span<float> ShankCollisionCounts();
    std::span<const float> ShankCollisionCounts() const;

    std::span<float> TrunkHeadCollisionCounts();
    std::span<const float> TrunkHeadCollisionCounts() const;

    std::span<float> FootAirTimes();
    std::span<const float> FootAirTimes() const;

    std::span<float> FootPeakHeights();
    std::span<const float> FootPeakHeights() const;

    std::span<float> FirstContacts();
    std::span<const float> FirstContacts() const;

    std::span<float> LandingForces();
    std::span<const float> LandingForces() const;

private:
    void RequireEnvironmentIndex(std::size_t environment_index) const;

    std::size_t environment_count_{0};
    std::size_t foot_count_{0};
    LocomotionCommandRuntime command_runtime_;
    std::vector<float> foot_contacts_;
    std::vector<float> foot_contact_forces_;
    std::vector<float> illegal_contact_counts_;
    std::vector<float> self_collision_counts_;
    std::vector<float> shank_collision_counts_;
    std::vector<float> trunk_head_collision_counts_;
    std::vector<float> foot_air_times_;
    std::vector<float> foot_contact_times_;
    std::vector<float> foot_peak_heights_;
    std::vector<float> first_contacts_;
    std::vector<float> landing_forces_;
    std::vector<std::string> foot_contact_group_names_;
    std::vector<std::size_t> foot_contact_group_indices_;
};

} // namespace gobot
