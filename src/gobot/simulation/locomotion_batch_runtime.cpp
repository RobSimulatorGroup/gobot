/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/locomotion_batch_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

namespace gobot {
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
        foot_contact_group_names_.push_back(fmt::format("foot_{}", foot_index));
    }
}

std::size_t LocomotionBatchRuntime::GetEnvironmentCount() const {
    return environment_count_;
}

std::size_t LocomotionBatchRuntime::GetFootCount() const {
    return foot_count_;
}

LocomotionCommandRuntime& LocomotionBatchRuntime::CommandRuntime() {
    return command_runtime_;
}

const LocomotionCommandRuntime& LocomotionBatchRuntime::CommandRuntime() const {
    return command_runtime_;
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
