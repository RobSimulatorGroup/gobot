/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/locomotion_batch_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <fmt/format.h>

namespace gobot {
namespace {

template <typename T>
void RequireSize(const std::vector<T>& values,
                 std::size_t expected_size,
                 std::string_view name) {
    if (values.size() != expected_size) {
        throw std::invalid_argument(fmt::format(
                "physics_state {} has {} values; expected {}",
                name,
                values.size(),
                expected_size));
    }
}

void RequireSensorIndex(std::size_t index,
                        std::size_t sensor_count,
                        std::string_view name) {
    if (index != LocomotionBatchStateLayout::kInvalidIndex && index >= sensor_count) {
        throw std::invalid_argument(fmt::format(
                "{} index {} is out of range for {} sensors",
                name,
                index,
                sensor_count));
    }
}

} // namespace

LocomotionBatchRuntime::LocomotionBatchRuntime(std::size_t environment_count,
                                               std::size_t foot_count)
    : environment_count_(environment_count),
      foot_count_(foot_count),
      command_runtime_(environment_count),
      foot_contacts_(environment_count * foot_count, 0.0f),
      foot_contact_forces_(environment_count * foot_count * 3, 0.0f),
      illegal_contact_counts_(environment_count, 0.0f),
      self_collision_counts_(environment_count, 0.0f),
      shank_collision_counts_(environment_count, 0.0f),
      trunk_head_collision_counts_(environment_count, 0.0f),
      foot_air_times_(environment_count * foot_count, 0.0f),
      foot_contact_times_(environment_count * foot_count, 0.0f),
      foot_peak_heights_(environment_count * foot_count, 0.0f),
      first_contacts_(environment_count * foot_count, 0.0f),
      landing_forces_(environment_count * foot_count, 0.0f),
      foot_contact_group_indices_(foot_count, 0) {
    foot_contact_group_names_.reserve(foot_count_);
    for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
        foot_contact_group_names_.push_back(FootContactGroupName(foot_index));
    }
}

LocomotionBatchRuntime::LocomotionBatchRuntime(
        std::size_t environment_count,
        LocomotionBatchStateLayout state_layout)
    : LocomotionBatchRuntime(environment_count,
                             state_layout.foot_link_indices.size()) {
    state_layout_ = std::move(state_layout);
    joint_count_ = state_layout_.joint_names.size();
    link_count_ = state_layout_.link_names.size();
    height_scan_count_ = state_layout_.height_scan_count;
    has_state_layout_ = true;
    ValidateStateLayout();
    AllocateStateBuffers();

    for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
        const bool inserted = foot_index_by_shape_.emplace(
                state_layout_.foot_shape_names[foot_index],
                foot_index).second;
        if (!inserted) {
            throw std::invalid_argument(fmt::format(
                    "foot shape '{}' is assigned to more than one foot",
                    state_layout_.foot_shape_names[foot_index]));
        }
    }
}

std::string LocomotionBatchRuntime::FootContactGroupName(std::size_t foot_index) {
    return fmt::format("foot_{}", foot_index);
}

std::size_t LocomotionBatchRuntime::GetEnvironmentCount() const {
    return environment_count_;
}

std::size_t LocomotionBatchRuntime::GetJointCount() const {
    return joint_count_;
}

std::size_t LocomotionBatchRuntime::GetLinkCount() const {
    return link_count_;
}

std::size_t LocomotionBatchRuntime::GetFootCount() const {
    return foot_count_;
}

std::size_t LocomotionBatchRuntime::GetHeightScanCount() const {
    return height_scan_count_;
}

bool LocomotionBatchRuntime::HasStateLayout() const {
    return has_state_layout_;
}

LocomotionCommandRuntime& LocomotionBatchRuntime::CommandRuntime() {
    return command_runtime_;
}

const LocomotionCommandRuntime& LocomotionBatchRuntime::CommandRuntime() const {
    return command_runtime_;
}

void LocomotionBatchRuntime::ValidateStateLayout() const {
    if (state_layout_.robot_name.empty()) {
        throw std::invalid_argument("locomotion state layout robot_name must not be empty");
    }
    if (state_layout_.base_link.empty()) {
        throw std::invalid_argument("locomotion state layout base_link must not be empty");
    }
    if (state_layout_.foot_shape_names.size() != foot_count_) {
        throw std::invalid_argument("foot_shape_names must have one entry per foot");
    }
    if (state_layout_.foot_height_sensor_indices.size() > foot_count_) {
        throw std::invalid_argument("foot_height_sensor_indices cannot contain more entries than feet");
    }
    if (state_layout_.foot_contact_sensor_indices.size() > foot_count_) {
        throw std::invalid_argument("foot_contact_sensor_indices cannot contain more entries than feet");
    }
    for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
        if (state_layout_.foot_link_indices[foot_index] >= link_count_) {
            throw std::invalid_argument(fmt::format(
                    "foot link index {} is out of range for {} links",
                    state_layout_.foot_link_indices[foot_index],
                    link_count_));
        }
        if (state_layout_.foot_shape_names[foot_index].empty()) {
            throw std::invalid_argument("foot_shape_names must not contain empty names");
        }
    }

    const std::size_t sensor_count = state_layout_.sensor_names.size();
    for (const std::size_t index : state_layout_.foot_height_sensor_indices) {
        if (index >= sensor_count) {
            throw std::invalid_argument(fmt::format(
                    "foot height sensor index {} is out of range for {} sensors",
                    index,
                    sensor_count));
        }
    }
    for (const std::size_t index : state_layout_.foot_contact_sensor_indices) {
        if (index >= sensor_count) {
            throw std::invalid_argument(fmt::format(
                    "foot contact sensor index {} is out of range for {} sensors",
                    index,
                    sensor_count));
        }
    }
    RequireSensorIndex(
            state_layout_.height_scan_sensor_index,
            sensor_count,
            "height scan sensor");
    RequireSensorIndex(state_layout_.imu_sensor_index, sensor_count, "IMU sensor");
    if (height_scan_count_ > 0 &&
        state_layout_.height_scan_sensor_index == LocomotionBatchStateLayout::kInvalidIndex) {
        throw std::invalid_argument("height_scan_count requires a height scan sensor index");
    }
}

void LocomotionBatchRuntime::AllocateStateBuffers() {
    base_positions_.assign(environment_count_ * 3, 0.0f);
    base_quaternions_.assign(environment_count_ * 4, 0.0f);
    base_linear_velocities_.assign(environment_count_ * 3, 0.0f);
    base_angular_velocities_.assign(environment_count_ * 3, 0.0f);
    base_linear_velocities_body_.assign(environment_count_ * 3, 0.0f);
    base_angular_velocities_body_.assign(environment_count_ * 3, 0.0f);
    projected_gravity_.assign(environment_count_ * 3, 0.0f);
    up_vectors_.assign(environment_count_ * 3, 0.0f);
    base_heights_.assign(environment_count_, 0.0f);
    joint_positions_.assign(environment_count_ * joint_count_, 0.0f);
    joint_velocities_.assign(environment_count_ * joint_count_, 0.0f);
    joint_accelerations_.assign(environment_count_ * joint_count_, 0.0f);
    joint_torques_.assign(environment_count_ * joint_count_, 0.0f);
    joint_lower_limits_.assign(joint_count_, 0.0f);
    joint_upper_limits_.assign(joint_count_, 0.0f);
    link_positions_.assign(environment_count_ * link_count_ * 3, 0.0f);
    link_quaternions_.assign(environment_count_ * link_count_ * 4, 0.0f);
    link_linear_velocities_.assign(environment_count_ * link_count_ * 3, 0.0f);
    link_angular_velocities_.assign(environment_count_ * link_count_ * 3, 0.0f);
    foot_positions_.assign(environment_count_ * foot_count_ * 3, 0.0f);
    foot_quaternions_.assign(environment_count_ * foot_count_ * 4, 0.0f);
    foot_velocities_.assign(environment_count_ * foot_count_ * 3, 0.0f);
    foot_heights_.assign(environment_count_ * foot_count_, 0.0f);
    height_scan_.assign(environment_count_ * height_scan_count_, 0.0f);
    height_scan_hits_.assign(environment_count_ * height_scan_count_, 0);
    height_scan_points_.assign(environment_count_ * height_scan_count_ * 3, 0.0f);
    height_scan_normals_.assign(environment_count_ * height_scan_count_ * 3, 0.0f);
}

void LocomotionBatchRuntime::ValidatePhysicsState(
        const PhysicsRobotBatchStepResult& physics_state) const {
    if (!has_state_layout_) {
        throw std::logic_error("locomotion runtime has no robot state layout");
    }
    if (physics_state.environment_count != environment_count_) {
        throw std::invalid_argument("physics_state environment count does not match the locomotion runtime");
    }
    if (physics_state.robot_name != state_layout_.robot_name ||
        physics_state.base_link != state_layout_.base_link ||
        physics_state.joint_names != state_layout_.joint_names ||
        physics_state.link_names != state_layout_.link_names ||
        physics_state.sensor_names != state_layout_.sensor_names) {
        throw std::invalid_argument("physics_state names do not match the locomotion state layout");
    }

    const std::size_t sensor_count = state_layout_.sensor_names.size();
    const std::size_t contact_count = environment_count_ * physics_state.max_contact_count;
    const std::size_t group_count = physics_state.contact_shape_group_names.size();
    RequireSize(physics_state.base_position, environment_count_ * 3, "base_position");
    RequireSize(physics_state.base_quaternion, environment_count_ * 4, "base_quaternion");
    RequireSize(physics_state.base_linear_velocity, environment_count_ * 3, "base_linear_velocity");
    RequireSize(physics_state.base_angular_velocity, environment_count_ * 3, "base_angular_velocity");
    RequireSize(physics_state.joint_position, environment_count_ * joint_count_, "joint_position");
    RequireSize(physics_state.joint_velocity, environment_count_ * joint_count_, "joint_velocity");
    RequireSize(physics_state.joint_acceleration, environment_count_ * joint_count_, "joint_acceleration");
    RequireSize(physics_state.joint_effort, environment_count_ * joint_count_, "joint_effort");
    RequireSize(physics_state.joint_lower_limit, joint_count_, "joint_lower_limit");
    RequireSize(physics_state.joint_upper_limit, joint_count_, "joint_upper_limit");
    RequireSize(physics_state.link_position, environment_count_ * link_count_ * 3, "link_position");
    RequireSize(physics_state.link_quaternion, environment_count_ * link_count_ * 4, "link_quaternion");
    RequireSize(
            physics_state.link_linear_velocity,
            environment_count_ * link_count_ * 3,
            "link_linear_velocity");
    RequireSize(
            physics_state.link_angular_velocity,
            environment_count_ * link_count_ * 3,
            "link_angular_velocity");
    RequireSize(physics_state.sensor_value_count, sensor_count, "sensor_value_count");
    RequireSize(physics_state.sensor_hit_count, sensor_count, "sensor_hit_count");
    RequireSize(physics_state.sensor_position, environment_count_ * sensor_count * 3, "sensor_position");
    RequireSize(
            physics_state.sensor_quaternion,
            environment_count_ * sensor_count * 4,
            "sensor_quaternion");
    RequireSize(
            physics_state.sensor_linear_velocity,
            environment_count_ * sensor_count * 3,
            "sensor_linear_velocity");
    RequireSize(
            physics_state.sensor_values,
            environment_count_ * sensor_count * physics_state.max_sensor_values,
            "sensor_values");
    RequireSize(
            physics_state.sensor_hit,
            environment_count_ * sensor_count * physics_state.max_sensor_hits,
            "sensor_hit");
    RequireSize(
            physics_state.sensor_hit_point,
            environment_count_ * sensor_count * physics_state.max_sensor_hits * 3,
            "sensor_hit_point");
    RequireSize(
            physics_state.sensor_hit_normal,
            environment_count_ * sensor_count * physics_state.max_sensor_hits * 3,
            "sensor_hit_normal");
    RequireSize(physics_state.contact_count, environment_count_, "contact_count");
    RequireSize(physics_state.contact_link_index, contact_count * 2, "contact_link_index");
    RequireSize(physics_state.contact_shape_index, contact_count * 2, "contact_shape_index");
    RequireSize(physics_state.contact_force, contact_count * 3, "contact_force");
    RequireSize(
            physics_state.contact_shape_group_tick_count,
            environment_count_ * group_count,
            "contact_shape_group_tick_count");
    RequireSize(
            physics_state.self_contact_tick_count,
            environment_count_,
            "self_contact_tick_count");

    for (const std::size_t sensor_index : state_layout_.foot_height_sensor_indices) {
        if (physics_state.max_sensor_values == 0 ||
            physics_state.sensor_value_count[sensor_index] < 1) {
            throw std::invalid_argument("foot height sensor does not provide a scalar value");
        }
    }
    if (height_scan_count_ > 0) {
        const std::size_t sensor_index = state_layout_.height_scan_sensor_index;
        if (physics_state.max_sensor_values < height_scan_count_ ||
            physics_state.max_sensor_hits < height_scan_count_ ||
            physics_state.sensor_value_count[sensor_index] <
                    static_cast<std::int32_t>(height_scan_count_) ||
            physics_state.sensor_hit_count[sensor_index] <
                    static_cast<std::int32_t>(height_scan_count_)) {
            throw std::invalid_argument("height scan sensor result is smaller than the configured scan layout");
        }
    }
}

void LocomotionBatchRuntime::UpdateState(
        const PhysicsRobotBatchStepResult& physics_state) {
    ValidatePhysicsState(physics_state);

    std::transform(
            physics_state.joint_lower_limit.begin(),
            physics_state.joint_lower_limit.end(),
            joint_lower_limits_.begin(),
            [](RealType value) { return static_cast<float>(value); });
    std::transform(
            physics_state.joint_upper_limit.begin(),
            physics_state.joint_upper_limit.end(),
            joint_upper_limits_.begin(),
            [](RealType value) { return static_cast<float>(value); });

    const auto find_group = [&](std::string_view name) {
        const auto group = std::find(
                physics_state.contact_shape_group_names.begin(),
                physics_state.contact_shape_group_names.end(),
                name);
        return group == physics_state.contact_shape_group_names.end()
                       ? LocomotionBatchStateLayout::kInvalidIndex
                       : static_cast<std::size_t>(std::distance(
                                 physics_state.contact_shape_group_names.begin(),
                                 group));
    };
    const std::size_t thigh_group = find_group(kThighContactGroup);
    const std::size_t shank_group = find_group(kShankContactGroup);
    const std::size_t trunk_head_group = find_group(kTrunkHeadContactGroup);
    const std::size_t sensor_count = state_layout_.sensor_names.size();

    for (std::size_t environment_index = 0;
         environment_index < environment_count_;
         ++environment_index) {
        ClearStepContacts(environment_index);
        const std::size_t environment3 = environment_index * 3;
        const std::size_t environment4 = environment_index * 4;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            base_positions_[environment3 + axis] =
                    static_cast<float>(physics_state.base_position[environment3 + axis]);
            base_linear_velocities_[environment3 + axis] =
                    static_cast<float>(physics_state.base_linear_velocity[environment3 + axis]);
            base_angular_velocities_[environment3 + axis] =
                    static_cast<float>(physics_state.base_angular_velocity[environment3 + axis]);
        }
        for (std::size_t axis = 0; axis < 4; ++axis) {
            base_quaternions_[environment4 + axis] =
                    static_cast<float>(physics_state.base_quaternion[environment4 + axis]);
        }
        base_heights_[environment_index] = base_positions_[environment3 + 2];

        const Quaternion base_orientation(
                physics_state.base_quaternion[environment4 + 0],
                physics_state.base_quaternion[environment4 + 1],
                physics_state.base_quaternion[environment4 + 2],
                physics_state.base_quaternion[environment4 + 3]);
        const Matrix3 base_rotation = base_orientation.normalized().toRotationMatrix();
        const Vector3 linear_world(
                physics_state.base_linear_velocity[environment3 + 0],
                physics_state.base_linear_velocity[environment3 + 1],
                physics_state.base_linear_velocity[environment3 + 2]);
        const Vector3 angular_world(
                physics_state.base_angular_velocity[environment3 + 0],
                physics_state.base_angular_velocity[environment3 + 1],
                physics_state.base_angular_velocity[environment3 + 2]);
        const Vector3 linear_body = base_rotation.transpose() * linear_world;
        const Vector3 angular_body = base_rotation.transpose() * angular_world;
        const Vector3 gravity_body = base_rotation.transpose() * Vector3(0.0, 0.0, -1.0);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            base_linear_velocities_body_[environment3 + axis] =
                    static_cast<float>(linear_body[axis]);
            base_angular_velocities_body_[environment3 + axis] =
                    static_cast<float>(angular_body[axis]);
            projected_gravity_[environment3 + axis] =
                    static_cast<float>(gravity_body[axis]);
            up_vectors_[environment3 + axis] = static_cast<float>(base_rotation(axis, 2));
        }

        if (state_layout_.imu_sensor_index != LocomotionBatchStateLayout::kInvalidIndex) {
            const std::size_t sensor_index = state_layout_.imu_sensor_index;
            const std::size_t sensor4 = (environment_index * sensor_count + sensor_index) * 4;
            const Quaternion imu_orientation(
                    physics_state.sensor_quaternion[sensor4 + 0],
                    physics_state.sensor_quaternion[sensor4 + 1],
                    physics_state.sensor_quaternion[sensor4 + 2],
                    physics_state.sensor_quaternion[sensor4 + 3]);
            const Matrix3 imu_rotation = imu_orientation.normalized().toRotationMatrix();
            for (std::size_t axis = 0; axis < 3; ++axis) {
                up_vectors_[environment3 + axis] = static_cast<float>(imu_rotation(axis, 2));
            }
            if (physics_state.sensor_value_count[sensor_index] >= 10) {
                const std::size_t sensor_values =
                        (environment_index * sensor_count + sensor_index) *
                        physics_state.max_sensor_values;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    base_angular_velocities_body_[environment3 + axis] =
                            static_cast<float>(physics_state.sensor_values[sensor_values + 4 + axis]);
                    base_linear_velocities_body_[environment3 + axis] =
                            static_cast<float>(physics_state.sensor_values[sensor_values + 7 + axis]);
                }
            }
        }

        for (std::size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
            const std::size_t offset = environment_index * joint_count_ + joint_index;
            joint_positions_[offset] = static_cast<float>(physics_state.joint_position[offset]);
            joint_velocities_[offset] = static_cast<float>(physics_state.joint_velocity[offset]);
            joint_accelerations_[offset] = static_cast<float>(physics_state.joint_acceleration[offset]);
            joint_torques_[offset] = static_cast<float>(physics_state.joint_effort[offset]);
        }

        for (std::size_t link_index = 0; link_index < link_count_; ++link_index) {
            const std::size_t link3 = (environment_index * link_count_ + link_index) * 3;
            const std::size_t link4 = (environment_index * link_count_ + link_index) * 4;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                link_positions_[link3 + axis] =
                        static_cast<float>(physics_state.link_position[link3 + axis]);
                link_linear_velocities_[link3 + axis] =
                        static_cast<float>(physics_state.link_linear_velocity[link3 + axis]);
                link_angular_velocities_[link3 + axis] =
                        static_cast<float>(physics_state.link_angular_velocity[link3 + axis]);
            }
            for (std::size_t axis = 0; axis < 4; ++axis) {
                link_quaternions_[link4 + axis] =
                        static_cast<float>(physics_state.link_quaternion[link4 + axis]);
            }
        }

        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot = environment_index * foot_count_ + foot_index;
            const std::size_t foot3 = foot * 3;
            const std::size_t foot4 = foot * 4;
            const std::size_t link_index = state_layout_.foot_link_indices[foot_index];
            const std::size_t link3 = (environment_index * link_count_ + link_index) * 3;
            const std::size_t link4 = (environment_index * link_count_ + link_index) * 4;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                foot_positions_[foot3 + axis] = link_positions_[link3 + axis];
                foot_velocities_[foot3 + axis] = link_linear_velocities_[link3 + axis];
            }
            for (std::size_t axis = 0; axis < 4; ++axis) {
                foot_quaternions_[foot4 + axis] = link_quaternions_[link4 + axis];
            }
            foot_heights_[foot] = 0.0f;

            if (foot_index < state_layout_.foot_contact_sensor_indices.size()) {
                const std::size_t sensor_index =
                        state_layout_.foot_contact_sensor_indices[foot_index];
                const std::size_t sensor3 =
                        (environment_index * sensor_count + sensor_index) * 3;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    foot_positions_[foot3 + axis] =
                            static_cast<float>(physics_state.sensor_position[sensor3 + axis]);
                    foot_velocities_[foot3 + axis] =
                            static_cast<float>(physics_state.sensor_linear_velocity[sensor3 + axis]);
                }
            }
            if (foot_index < state_layout_.foot_height_sensor_indices.size()) {
                const std::size_t sensor_index =
                        state_layout_.foot_height_sensor_indices[foot_index];
                const std::size_t value_offset =
                        (environment_index * sensor_count + sensor_index) *
                        physics_state.max_sensor_values;
                foot_heights_[foot] =
                        static_cast<float>(physics_state.sensor_values[value_offset]);
            }
        }

        if (height_scan_count_ > 0) {
            const std::size_t sensor_index = state_layout_.height_scan_sensor_index;
            const std::size_t value_offset =
                    (environment_index * sensor_count + sensor_index) *
                    physics_state.max_sensor_values;
            const std::size_t hit_offset =
                    (environment_index * sensor_count + sensor_index) *
                    physics_state.max_sensor_hits;
            for (std::size_t sample = 0; sample < height_scan_count_; ++sample) {
                const std::size_t output = environment_index * height_scan_count_ + sample;
                height_scan_[output] =
                        static_cast<float>(physics_state.sensor_values[value_offset + sample]);
                height_scan_hits_[output] = physics_state.sensor_hit[hit_offset + sample];
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    height_scan_points_[output * 3 + axis] = static_cast<float>(
                            physics_state.sensor_hit_point[(hit_offset + sample) * 3 + axis]);
                    height_scan_normals_[output * 3 + axis] = static_cast<float>(
                            physics_state.sensor_hit_normal[(hit_offset + sample) * 3 + axis]);
                }
            }
        }

        const std::size_t environment_contact_count = static_cast<std::size_t>(
                std::max(physics_state.contact_count[environment_index], 0));
        for (std::size_t contact_index = 0;
             contact_index < std::min(environment_contact_count, physics_state.max_contact_count);
             ++contact_index) {
            const std::size_t contact =
                    environment_index * physics_state.max_contact_count + contact_index;
            const int link_a = physics_state.contact_link_index[contact * 2 + 0];
            const int link_b = physics_state.contact_link_index[contact * 2 + 1];
            const int shape_a = physics_state.contact_shape_index[contact * 2 + 0];
            const int shape_b = physics_state.contact_shape_index[contact * 2 + 1];
            if (link_a >= 0 && link_b >= 0) {
                self_collision_counts_[environment_index] += 1.0f;
                continue;
            }
            const int robot_link = link_a >= 0 ? link_a : link_b;
            const int robot_shape = shape_a >= 0 ? shape_a : shape_b;
            if (robot_link < 0 || static_cast<std::size_t>(robot_link) >= link_count_ ||
                robot_shape < 0 ||
                static_cast<std::size_t>(robot_shape) >= physics_state.shape_names.size()) {
                continue;
            }
            const auto foot = foot_index_by_shape_.find(
                    physics_state.shape_names[static_cast<std::size_t>(robot_shape)]);
            if (foot == foot_index_by_shape_.end()) {
                continue;
            }
            const std::size_t foot_offset =
                    environment_index * foot_count_ + foot->second;
            const std::size_t foot3 = foot_offset * 3;
            foot_contacts_[foot_offset] = 1.0f;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                foot_contact_forces_[foot3 + axis] +=
                        static_cast<float>(physics_state.contact_force[contact * 3 + axis]);
            }
        }

        self_collision_counts_[environment_index] =
                static_cast<float>(physics_state.self_contact_tick_count[environment_index]);
        const auto group_tick_count = [&](std::size_t group_index) {
            if (group_index == LocomotionBatchStateLayout::kInvalidIndex) {
                return 0.0f;
            }
            const std::size_t offset =
                    environment_index * physics_state.contact_shape_group_names.size() +
                    group_index;
            return static_cast<float>(physics_state.contact_shape_group_tick_count[offset]);
        };
        illegal_contact_counts_[environment_index] =
                state_layout_.terminate_on_thigh_contact
                        ? group_tick_count(thigh_group)
                        : 0.0f;
        shank_collision_counts_[environment_index] = group_tick_count(shank_group);
        trunk_head_collision_counts_[environment_index] =
                group_tick_count(trunk_head_group);
    }
}

void LocomotionBatchRuntime::ClearStepContacts(std::size_t environment_index) {
    RequireEnvironmentIndex(environment_index);
    illegal_contact_counts_[environment_index] = 0.0f;
    self_collision_counts_[environment_index] = 0.0f;
    shank_collision_counts_[environment_index] = 0.0f;
    trunk_head_collision_counts_[environment_index] = 0.0f;

    const std::size_t foot_begin = environment_index * foot_count_;
    for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
        const std::size_t foot = foot_begin + foot_index;
        const std::size_t force = foot * 3;
        foot_contacts_[foot] = 0.0f;
        if (first_contacts_[foot] > 0.0f) {
            foot_peak_heights_[foot] = 0.0f;
        }
        first_contacts_[foot] = 0.0f;
        landing_forces_[foot] = 0.0f;
        std::fill_n(foot_contact_forces_.begin() + static_cast<std::ptrdiff_t>(force), 3, 0.0f);
    }
}

void LocomotionBatchRuntime::ClearResetContacts(
        std::span<const std::size_t> environment_indices) {
    for (const std::size_t environment_index : environment_indices) {
        ClearStepContacts(environment_index);
        const std::size_t foot_begin = environment_index * foot_count_;
        std::fill_n(
                foot_contact_times_.begin() + static_cast<std::ptrdiff_t>(foot_begin),
                foot_count_,
                0.0f);
        std::fill_n(
                foot_air_times_.begin() + static_cast<std::ptrdiff_t>(foot_begin),
                foot_count_,
                0.0f);
        std::fill_n(
                foot_peak_heights_.begin() + static_cast<std::ptrdiff_t>(foot_begin),
                foot_count_,
                0.0f);
    }
}

void LocomotionBatchRuntime::UpdateFootHistory(
        const PhysicsRobotBatchStepResult& physics_state,
        std::span<const float> foot_heights,
        RealType step_dt,
        RealType physics_dt) {
    if (physics_state.environment_count != environment_count_) {
        throw std::invalid_argument("physics_state environment count does not match the locomotion runtime");
    }
    if (foot_heights.size() != environment_count_ * foot_count_) {
        throw std::invalid_argument("foot_heights must contain environment_count * foot_count values");
    }
    const std::size_t expected_history_size =
            environment_count_ * physics_state.contact_shape_group_names.size() *
            physics_state.contact_history_tick_count;
    if (physics_state.contact_shape_group_history.size() != expected_history_size) {
        throw std::invalid_argument("physics_state contact history does not match its dimensions");
    }

    const float control_dt = static_cast<float>(std::max<RealType>(step_dt, 1.0e-6));
    const float simulation_dt = static_cast<float>(std::max<RealType>(physics_dt, 1.0e-6));
    std::fill(foot_contact_group_indices_.begin(),
              foot_contact_group_indices_.end(),
              physics_state.contact_shape_group_names.size());
    for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
        const auto group = std::find(
                physics_state.contact_shape_group_names.begin(),
                physics_state.contact_shape_group_names.end(),
                foot_contact_group_names_[foot_index]);
        if (group != physics_state.contact_shape_group_names.end()) {
            foot_contact_group_indices_[foot_index] = static_cast<std::size_t>(
                    std::distance(physics_state.contact_shape_group_names.begin(), group));
        }
    }

    for (std::size_t environment_index = 0;
         environment_index < environment_count_;
         ++environment_index) {
        for (std::size_t foot_index = 0; foot_index < foot_count_; ++foot_index) {
            const std::size_t foot = environment_index * foot_count_ + foot_index;
            const std::size_t force = foot * 3;
            bool contact = foot_contacts_[foot] > 0.0f;
            const std::size_t group_index = foot_contact_group_indices_[foot_index];
            if (group_index < physics_state.contact_shape_group_names.size() &&
                physics_state.contact_history_tick_count > 0) {
                for (std::size_t tick = 0; tick < physics_state.contact_history_tick_count; ++tick) {
                    const std::size_t history_offset =
                            (environment_index * physics_state.contact_shape_group_names.size() + group_index) *
                                    physics_state.contact_history_tick_count +
                            tick;
                    const bool tick_contact = physics_state.contact_shape_group_history[history_offset] != 0;
                    if (tick_contact) {
                        foot_contact_times_[foot] += simulation_dt;
                        foot_air_times_[foot] = 0.0f;
                    } else {
                        foot_contact_times_[foot] = 0.0f;
                        foot_air_times_[foot] += simulation_dt;
                    }
                    contact = tick_contact;
                }
                foot_contacts_[foot] = contact ? 1.0f : 0.0f;
            } else if (contact) {
                foot_contact_times_[foot] += control_dt;
                foot_air_times_[foot] = 0.0f;
            } else {
                foot_contact_times_[foot] = 0.0f;
                foot_air_times_[foot] += control_dt;
            }

            first_contacts_[foot] =
                    contact && foot_contact_times_[foot] > 0.0f &&
                            foot_contact_times_[foot] < control_dt + 1.0e-6f
                    ? 1.0f
                    : 0.0f;
            const float force_x = foot_contact_forces_[force + 0];
            const float force_y = foot_contact_forces_[force + 1];
            const float force_z = foot_contact_forces_[force + 2];
            landing_forces_[foot] =
                    std::sqrt(force_x * force_x + force_y * force_y + force_z * force_z) *
                    first_contacts_[foot];
            if (!contact) {
                foot_peak_heights_[foot] = std::max(foot_peak_heights_[foot], foot_heights[foot]);
            }
        }
    }
}

void LocomotionBatchRuntime::RequireEnvironmentIndex(std::size_t environment_index) const {
    if (environment_index >= environment_count_) {
        throw std::out_of_range(fmt::format(
                "environment index {} is out of range for {} environments",
                environment_index,
                environment_count_));
    }
}

std::span<float> LocomotionBatchRuntime::BasePositions() { return base_positions_; }
std::span<const float> LocomotionBatchRuntime::BasePositions() const { return base_positions_; }
std::span<float> LocomotionBatchRuntime::BaseQuaternions() { return base_quaternions_; }
std::span<const float> LocomotionBatchRuntime::BaseQuaternions() const { return base_quaternions_; }
std::span<float> LocomotionBatchRuntime::BaseLinearVelocities() { return base_linear_velocities_; }
std::span<const float> LocomotionBatchRuntime::BaseLinearVelocities() const { return base_linear_velocities_; }
std::span<float> LocomotionBatchRuntime::BaseAngularVelocities() { return base_angular_velocities_; }
std::span<const float> LocomotionBatchRuntime::BaseAngularVelocities() const { return base_angular_velocities_; }
std::span<float> LocomotionBatchRuntime::BaseLinearVelocitiesBody() {
    return base_linear_velocities_body_;
}
std::span<const float> LocomotionBatchRuntime::BaseLinearVelocitiesBody() const {
    return base_linear_velocities_body_;
}
std::span<float> LocomotionBatchRuntime::BaseAngularVelocitiesBody() {
    return base_angular_velocities_body_;
}
std::span<const float> LocomotionBatchRuntime::BaseAngularVelocitiesBody() const {
    return base_angular_velocities_body_;
}
std::span<float> LocomotionBatchRuntime::ProjectedGravity() { return projected_gravity_; }
std::span<const float> LocomotionBatchRuntime::ProjectedGravity() const { return projected_gravity_; }
std::span<float> LocomotionBatchRuntime::UpVectors() { return up_vectors_; }
std::span<const float> LocomotionBatchRuntime::UpVectors() const { return up_vectors_; }
std::span<float> LocomotionBatchRuntime::BaseHeights() { return base_heights_; }
std::span<const float> LocomotionBatchRuntime::BaseHeights() const { return base_heights_; }
std::span<float> LocomotionBatchRuntime::JointPositions() { return joint_positions_; }
std::span<const float> LocomotionBatchRuntime::JointPositions() const { return joint_positions_; }
std::span<float> LocomotionBatchRuntime::JointVelocities() { return joint_velocities_; }
std::span<const float> LocomotionBatchRuntime::JointVelocities() const { return joint_velocities_; }
std::span<float> LocomotionBatchRuntime::JointAccelerations() { return joint_accelerations_; }
std::span<const float> LocomotionBatchRuntime::JointAccelerations() const { return joint_accelerations_; }
std::span<float> LocomotionBatchRuntime::JointTorques() { return joint_torques_; }
std::span<const float> LocomotionBatchRuntime::JointTorques() const { return joint_torques_; }
std::span<float> LocomotionBatchRuntime::JointLowerLimits() { return joint_lower_limits_; }
std::span<const float> LocomotionBatchRuntime::JointLowerLimits() const { return joint_lower_limits_; }
std::span<float> LocomotionBatchRuntime::JointUpperLimits() { return joint_upper_limits_; }
std::span<const float> LocomotionBatchRuntime::JointUpperLimits() const { return joint_upper_limits_; }
std::span<float> LocomotionBatchRuntime::LinkPositions() { return link_positions_; }
std::span<const float> LocomotionBatchRuntime::LinkPositions() const { return link_positions_; }
std::span<float> LocomotionBatchRuntime::LinkQuaternions() { return link_quaternions_; }
std::span<const float> LocomotionBatchRuntime::LinkQuaternions() const { return link_quaternions_; }
std::span<float> LocomotionBatchRuntime::LinkLinearVelocities() { return link_linear_velocities_; }
std::span<const float> LocomotionBatchRuntime::LinkLinearVelocities() const {
    return link_linear_velocities_;
}
std::span<float> LocomotionBatchRuntime::LinkAngularVelocities() { return link_angular_velocities_; }
std::span<const float> LocomotionBatchRuntime::LinkAngularVelocities() const {
    return link_angular_velocities_;
}
std::span<float> LocomotionBatchRuntime::FootPositions() { return foot_positions_; }
std::span<const float> LocomotionBatchRuntime::FootPositions() const { return foot_positions_; }
std::span<float> LocomotionBatchRuntime::FootQuaternions() { return foot_quaternions_; }
std::span<const float> LocomotionBatchRuntime::FootQuaternions() const { return foot_quaternions_; }
std::span<float> LocomotionBatchRuntime::FootVelocities() { return foot_velocities_; }
std::span<const float> LocomotionBatchRuntime::FootVelocities() const { return foot_velocities_; }
std::span<float> LocomotionBatchRuntime::FootHeights() { return foot_heights_; }
std::span<const float> LocomotionBatchRuntime::FootHeights() const { return foot_heights_; }
std::span<float> LocomotionBatchRuntime::HeightScan() { return height_scan_; }
std::span<const float> LocomotionBatchRuntime::HeightScan() const { return height_scan_; }
std::span<std::uint8_t> LocomotionBatchRuntime::HeightScanHits() { return height_scan_hits_; }
std::span<const std::uint8_t> LocomotionBatchRuntime::HeightScanHits() const {
    return height_scan_hits_;
}
std::span<float> LocomotionBatchRuntime::HeightScanPoints() { return height_scan_points_; }
std::span<const float> LocomotionBatchRuntime::HeightScanPoints() const { return height_scan_points_; }
std::span<float> LocomotionBatchRuntime::HeightScanNormals() { return height_scan_normals_; }
std::span<const float> LocomotionBatchRuntime::HeightScanNormals() const { return height_scan_normals_; }
std::span<float> LocomotionBatchRuntime::FootContacts() { return foot_contacts_; }
std::span<const float> LocomotionBatchRuntime::FootContacts() const { return foot_contacts_; }
std::span<float> LocomotionBatchRuntime::FootContactForces() { return foot_contact_forces_; }
std::span<const float> LocomotionBatchRuntime::FootContactForces() const { return foot_contact_forces_; }
std::span<float> LocomotionBatchRuntime::IllegalContactCounts() { return illegal_contact_counts_; }
std::span<const float> LocomotionBatchRuntime::IllegalContactCounts() const { return illegal_contact_counts_; }
std::span<float> LocomotionBatchRuntime::SelfCollisionCounts() { return self_collision_counts_; }
std::span<const float> LocomotionBatchRuntime::SelfCollisionCounts() const { return self_collision_counts_; }
std::span<float> LocomotionBatchRuntime::ShankCollisionCounts() { return shank_collision_counts_; }
std::span<const float> LocomotionBatchRuntime::ShankCollisionCounts() const { return shank_collision_counts_; }
std::span<float> LocomotionBatchRuntime::TrunkHeadCollisionCounts() { return trunk_head_collision_counts_; }
std::span<const float> LocomotionBatchRuntime::TrunkHeadCollisionCounts() const {
    return trunk_head_collision_counts_;
}
std::span<float> LocomotionBatchRuntime::FootAirTimes() { return foot_air_times_; }
std::span<const float> LocomotionBatchRuntime::FootAirTimes() const { return foot_air_times_; }
std::span<float> LocomotionBatchRuntime::FootPeakHeights() { return foot_peak_heights_; }
std::span<const float> LocomotionBatchRuntime::FootPeakHeights() const { return foot_peak_heights_; }
std::span<float> LocomotionBatchRuntime::FirstContacts() { return first_contacts_; }
std::span<const float> LocomotionBatchRuntime::FirstContacts() const { return first_contacts_; }
std::span<float> LocomotionBatchRuntime::LandingForces() { return landing_forces_; }
std::span<const float> LocomotionBatchRuntime::LandingForces() const { return landing_forces_; }

} // namespace gobot
