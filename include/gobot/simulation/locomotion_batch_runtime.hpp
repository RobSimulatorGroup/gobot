/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gobot/physics/physics_types.hpp"
#include "gobot/simulation/locomotion_command_runtime.hpp"
#include "gobot_export.h"

namespace gobot {

struct GOBOT_EXPORT LocomotionBatchStateLayout {
    static constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();

    std::string robot_name;
    std::string base_link;
    std::vector<std::string> joint_names;
    std::vector<std::string> link_names;
    std::vector<std::string> sensor_names;
    std::vector<std::size_t> foot_link_indices;
    std::vector<std::string> foot_shape_names;
    std::vector<std::size_t> foot_height_sensor_indices;
    std::vector<std::size_t> foot_contact_sensor_indices;
    std::size_t height_scan_sensor_index{kInvalidIndex};
    std::size_t imu_sensor_index{kInvalidIndex};
    std::size_t height_scan_count{0};
    bool terminate_on_thigh_contact{true};
};

class GOBOT_EXPORT LocomotionBatchRuntime {
public:
    static constexpr std::string_view kThighContactGroup = "thigh";
    static constexpr std::string_view kShankContactGroup = "shank";
    static constexpr std::string_view kTrunkHeadContactGroup = "trunk_head";

    LocomotionBatchRuntime(std::size_t environment_count = 0,
                           std::size_t foot_count = 0);

    LocomotionBatchRuntime(std::size_t environment_count,
                           LocomotionBatchStateLayout state_layout);

    static std::string FootContactGroupName(std::size_t foot_index);

    std::size_t GetEnvironmentCount() const;

    std::size_t GetJointCount() const;

    std::size_t GetLinkCount() const;

    std::size_t GetFootCount() const;

    std::size_t GetHeightScanCount() const;

    bool HasStateLayout() const;

    LocomotionCommandRuntime& CommandRuntime();

    const LocomotionCommandRuntime& CommandRuntime() const;

    void ClearStepContacts(std::size_t environment_index);

    void ClearResetContacts(std::span<const std::size_t> environment_indices);

    void UpdateState(const PhysicsRobotBatchStepResult& physics_state);

    void UpdateFootHistory(const PhysicsRobotBatchStepResult& physics_state,
                           std::span<const float> foot_heights,
                           RealType step_dt,
                           RealType physics_dt);

    std::span<float> BasePositions();
    std::span<const float> BasePositions() const;

    std::span<float> BaseQuaternions();
    std::span<const float> BaseQuaternions() const;

    std::span<float> BaseLinearVelocities();
    std::span<const float> BaseLinearVelocities() const;

    std::span<float> BaseAngularVelocities();
    std::span<const float> BaseAngularVelocities() const;

    std::span<float> BaseLinearVelocitiesBody();
    std::span<const float> BaseLinearVelocitiesBody() const;

    std::span<float> BaseAngularVelocitiesBody();
    std::span<const float> BaseAngularVelocitiesBody() const;

    std::span<float> ProjectedGravity();
    std::span<const float> ProjectedGravity() const;

    std::span<float> UpVectors();
    std::span<const float> UpVectors() const;

    std::span<float> BaseHeights();
    std::span<const float> BaseHeights() const;

    std::span<float> JointPositions();
    std::span<const float> JointPositions() const;

    std::span<float> JointVelocities();
    std::span<const float> JointVelocities() const;

    std::span<float> JointAccelerations();
    std::span<const float> JointAccelerations() const;

    std::span<float> JointTorques();
    std::span<const float> JointTorques() const;

    std::span<float> JointLowerLimits();
    std::span<const float> JointLowerLimits() const;

    std::span<float> JointUpperLimits();
    std::span<const float> JointUpperLimits() const;

    std::span<float> LinkPositions();
    std::span<const float> LinkPositions() const;

    std::span<float> LinkQuaternions();
    std::span<const float> LinkQuaternions() const;

    std::span<float> LinkLinearVelocities();
    std::span<const float> LinkLinearVelocities() const;

    std::span<float> LinkAngularVelocities();
    std::span<const float> LinkAngularVelocities() const;

    std::span<float> FootPositions();
    std::span<const float> FootPositions() const;

    std::span<float> FootQuaternions();
    std::span<const float> FootQuaternions() const;

    std::span<float> FootVelocities();
    std::span<const float> FootVelocities() const;

    std::span<float> FootHeights();
    std::span<const float> FootHeights() const;

    std::span<float> HeightScan();
    std::span<const float> HeightScan() const;

    std::span<std::uint8_t> HeightScanHits();
    std::span<const std::uint8_t> HeightScanHits() const;

    std::span<float> HeightScanPoints();
    std::span<const float> HeightScanPoints() const;

    std::span<float> HeightScanNormals();
    std::span<const float> HeightScanNormals() const;

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
    void ValidateStateLayout() const;

    void AllocateStateBuffers();

    void ValidatePhysicsState(const PhysicsRobotBatchStepResult& physics_state) const;

    void RequireEnvironmentIndex(std::size_t environment_index) const;

    std::size_t environment_count_{0};
    std::size_t joint_count_{0};
    std::size_t link_count_{0};
    std::size_t foot_count_{0};
    std::size_t height_scan_count_{0};
    bool has_state_layout_{false};
    LocomotionBatchStateLayout state_layout_;
    LocomotionCommandRuntime command_runtime_;
    std::vector<float> base_positions_;
    std::vector<float> base_quaternions_;
    std::vector<float> base_linear_velocities_;
    std::vector<float> base_angular_velocities_;
    std::vector<float> base_linear_velocities_body_;
    std::vector<float> base_angular_velocities_body_;
    std::vector<float> projected_gravity_;
    std::vector<float> up_vectors_;
    std::vector<float> base_heights_;
    std::vector<float> joint_positions_;
    std::vector<float> joint_velocities_;
    std::vector<float> joint_accelerations_;
    std::vector<float> joint_torques_;
    std::vector<float> joint_lower_limits_;
    std::vector<float> joint_upper_limits_;
    std::vector<float> link_positions_;
    std::vector<float> link_quaternions_;
    std::vector<float> link_linear_velocities_;
    std::vector<float> link_angular_velocities_;
    std::vector<float> foot_positions_;
    std::vector<float> foot_quaternions_;
    std::vector<float> foot_velocities_;
    std::vector<float> foot_heights_;
    std::vector<float> height_scan_;
    std::vector<std::uint8_t> height_scan_hits_;
    std::vector<float> height_scan_points_;
    std::vector<float> height_scan_normals_;
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
    std::unordered_map<std::string, std::size_t> foot_index_by_shape_;
};

} // namespace gobot
