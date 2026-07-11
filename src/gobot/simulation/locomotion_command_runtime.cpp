/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/locomotion_command_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace gobot {
namespace {

float ClampRatio(RealType value) {
    return static_cast<float>(std::clamp<RealType>(value, 0.0, 1.0));
}

std::array<float, 2> OrderedRange(const std::array<RealType, 2>& range) {
    const float first = static_cast<float>(range[0]);
    const float second = static_cast<float>(range[1]);
    return first <= second ? std::array<float, 2>{first, second}
                           : std::array<float, 2>{second, first};
}

} // namespace

LocomotionCommandRuntime::LocomotionCommandRuntime(std::size_t environment_count) {
    Allocate(environment_count);
}

void LocomotionCommandRuntime::Allocate(std::size_t environment_count) {
    environment_count_ = environment_count;
    commands_.assign(environment_count_ * kCommandDimension, 0.0f);
    world_commands_.assign(environment_count_ * kCommandDimension, 0.0f);
    heading_targets_.assign(environment_count_, 0.0f);
    heading_errors_.assign(environment_count_, 0.0f);
    time_left_.assign(environment_count_, 0.0f);
    steps_.assign(environment_count_, 0);
    heading_environment_mask_.assign(environment_count_, 0);
    standing_environment_mask_.assign(environment_count_, 0);
    world_environment_mask_.assign(environment_count_, 0);
    forward_environment_mask_.assign(environment_count_, 0);
    ranges_.assign(kRangeCount, 0.0f);
    SeedEnvironmentGenerators(config_.seed);
}

std::size_t LocomotionCommandRuntime::GetEnvironmentCount() const {
    return environment_count_;
}

void LocomotionCommandRuntime::Configure(const LocomotionCommandConfig& config) {
    config_ = config;
    config_.step_dt = std::max<RealType>(config.step_dt, 1.0e-9);
    const auto resampling_time = OrderedRange(config.resampling_time);
    config_.resampling_time = {
            std::max<RealType>(0.0, resampling_time[0]),
            std::max<RealType>(0.0, resampling_time[1])};
    config_.standing_environment_ratio = ClampRatio(config.standing_environment_ratio);
    config_.heading_environment_ratio = ClampRatio(config.heading_environment_ratio);
    config_.world_environment_ratio = ClampRatio(config.world_environment_ratio);
    config_.forward_environment_ratio = ClampRatio(config.forward_environment_ratio);
    config_.zero_small_xy_threshold = std::max<RealType>(0.0, config.zero_small_xy_threshold);

    SetVelocityRanges(config.linear_velocity_x,
                      config.linear_velocity_y,
                      config.angular_velocity_z);
    const auto heading = OrderedRange(config.heading);
    config_.heading = {heading[0], heading[1]};
    ranges_[HeadingMin] = heading[0];
    ranges_[HeadingMax] = heading[1];
    SeedEnvironmentGenerators(config.seed);
    configured_ = true;
}

bool LocomotionCommandRuntime::IsConfigured() const {
    return configured_;
}

RealType LocomotionCommandRuntime::GetStepDt() const {
    return config_.step_dt;
}

void LocomotionCommandRuntime::SetVelocityRanges(
        const std::array<RealType, 2>& linear_velocity_x,
        const std::array<RealType, 2>& linear_velocity_y,
        const std::array<RealType, 2>& angular_velocity_z) {
    const auto x = OrderedRange(linear_velocity_x);
    const auto y = OrderedRange(linear_velocity_y);
    const auto yaw = OrderedRange(angular_velocity_z);
    config_.linear_velocity_x = {x[0], x[1]};
    config_.linear_velocity_y = {y[0], y[1]};
    config_.angular_velocity_z = {yaw[0], yaw[1]};
    if (ranges_.size() != kRangeCount) {
        ranges_.assign(kRangeCount, 0.0f);
    }
    ranges_[LinearVelocityXMin] = x[0];
    ranges_[LinearVelocityXMax] = x[1];
    ranges_[LinearVelocityYMin] = y[0];
    ranges_[LinearVelocityYMax] = y[1];
    ranges_[AngularVelocityZMin] = yaw[0];
    ranges_[AngularVelocityZMax] = yaw[1];
}

void LocomotionCommandRuntime::Reset(std::span<const std::size_t> environment_indices) {
    if (!configured_) {
        return;
    }
    for (const std::size_t environment_index : environment_indices) {
        RequireEnvironmentIndex(environment_index);
        Resample(environment_index);
        time_left_[environment_index] = Uniform(
                environment_index,
                static_cast<float>(config_.resampling_time[0]),
                static_cast<float>(config_.resampling_time[1]));
    }
}

void LocomotionCommandRuntime::SetCommands(
        std::span<const std::size_t> environment_indices,
        std::span<const float> commands,
        std::span<const float> heading_targets,
        std::span<const float> time_left) {
    const std::size_t count = environment_indices.size();
    if (commands.size() != count * kCommandDimension) {
        throw std::invalid_argument("commands must contain len(environment_indices) * 3 values");
    }
    if (heading_targets.size() != count) {
        throw std::invalid_argument("heading_targets must contain len(environment_indices) values");
    }
    if (time_left.size() != count) {
        throw std::invalid_argument("time_left must contain len(environment_indices) values");
    }

    for (std::size_t row = 0; row < count; ++row) {
        const std::size_t environment_index = environment_indices[row];
        RequireEnvironmentIndex(environment_index);
        const std::size_t destination = environment_index * kCommandDimension;
        const std::size_t source = row * kCommandDimension;
        for (std::size_t axis = 0; axis < kCommandDimension; ++axis) {
            commands_[destination + axis] = commands[source + axis];
            world_commands_[destination + axis] = commands[source + axis];
        }
        heading_targets_[environment_index] = heading_targets[row];
        heading_errors_[environment_index] = 0.0f;
        time_left_[environment_index] = time_left[row];
        heading_environment_mask_[environment_index] = config_.heading_command ? 1 : 0;
        standing_environment_mask_[environment_index] =
                std::abs(commands[source + 0]) <= 1.0e-6f &&
                std::abs(commands[source + 1]) <= 1.0e-6f &&
                std::abs(commands[source + 2]) <= 1.0e-6f
                ? 1
                : 0;
        world_environment_mask_[environment_index] = 0;
        forward_environment_mask_[environment_index] = 0;
    }
}

void LocomotionCommandRuntime::SetSteps(
        std::span<const std::size_t> environment_indices,
        std::span<const std::uint32_t> steps) {
    if (steps.size() != environment_indices.size()) {
        throw std::invalid_argument("steps must contain len(environment_indices) values");
    }
    for (std::size_t row = 0; row < environment_indices.size(); ++row) {
        const std::size_t environment_index = environment_indices[row];
        RequireEnvironmentIndex(environment_index);
        steps_[environment_index] = steps[row];
    }
}

void LocomotionCommandRuntime::SetStepResamplingEnabled(bool enabled) {
    step_resampling_enabled_ = enabled;
}

void LocomotionCommandRuntime::Advance(std::span<const float> base_quaternions) {
    if (!configured_) {
        return;
    }
    if (base_quaternions.size() != environment_count_ * kQuaternionDimension) {
        throw std::invalid_argument("base_quaternions must contain environment_count * 4 values");
    }
    if (step_resampling_enabled_) {
        for (std::size_t environment_index = 0; environment_index < environment_count_; ++environment_index) {
            AdvanceStepAndResample(environment_index);
        }
    } else {
        AdvanceTimersAndResample();
    }
    UpdateFrames(base_quaternions);
}

void LocomotionCommandRuntime::UpdateFrames(std::span<const float> base_quaternions) {
    if (!configured_) {
        return;
    }
    if (base_quaternions.size() != environment_count_ * kQuaternionDimension) {
        throw std::invalid_argument("base_quaternions must contain environment_count * 4 values");
    }
    for (std::size_t environment_index = 0; environment_index < environment_count_; ++environment_index) {
        UpdateEnvironmentFrame(environment_index, base_quaternions);
    }
}

void LocomotionCommandRuntime::RequireEnvironmentIndex(std::size_t environment_index) const {
    if (environment_index >= environment_count_) {
        throw std::out_of_range(fmt::format(
                "environment index {} is out of range for {} environments",
                environment_index,
                environment_count_));
    }
}

void LocomotionCommandRuntime::SeedEnvironmentGenerators(std::uint64_t seed) {
    environment_rngs_.resize(environment_count_);
    for (std::size_t environment_index = 0; environment_index < environment_count_; ++environment_index) {
        const std::uint64_t index = static_cast<std::uint64_t>(environment_index);
        std::seed_seq sequence{
                static_cast<std::uint32_t>(seed),
                static_cast<std::uint32_t>(seed >> 32U),
                static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(index >> 32U),
                0x6c6f636fU};
        environment_rngs_[environment_index].seed(sequence);
    }
}

float LocomotionCommandRuntime::Uniform(std::size_t environment_index, float lower, float upper) {
    RequireEnvironmentIndex(environment_index);
    if (upper < lower) {
        std::swap(lower, upper);
    }
    if (std::abs(upper - lower) <= 1.0e-9f) {
        return lower;
    }
    std::uniform_real_distribution<float> distribution(lower, upper);
    return distribution(environment_rngs_[environment_index]);
}

void LocomotionCommandRuntime::Resample(std::size_t environment_index) {
    const std::size_t offset = environment_index * kCommandDimension;
    commands_[offset + 0] = Uniform(
            environment_index, ranges_[LinearVelocityXMin], ranges_[LinearVelocityXMax]);
    commands_[offset + 1] = Uniform(
            environment_index, ranges_[LinearVelocityYMin], ranges_[LinearVelocityYMax]);
    commands_[offset + 2] = Uniform(
            environment_index, ranges_[AngularVelocityZMin], ranges_[AngularVelocityZMax]);

    const float zero_threshold = static_cast<float>(config_.zero_small_xy_threshold);
    if (zero_threshold > 0.0f &&
        std::hypot(commands_[offset + 0], commands_[offset + 1]) <= zero_threshold) {
        commands_[offset + 0] = 0.0f;
        commands_[offset + 1] = 0.0f;
    }

    heading_targets_[environment_index] = Uniform(
            environment_index, ranges_[HeadingMin], ranges_[HeadingMax]);
    heading_environment_mask_[environment_index] =
            config_.heading_command &&
                    Uniform(environment_index, 0.0f, 1.0f) <=
                            static_cast<float>(config_.heading_environment_ratio)
            ? 1
            : 0;
    standing_environment_mask_[environment_index] =
            Uniform(environment_index, 0.0f, 1.0f) <=
                    static_cast<float>(config_.standing_environment_ratio)
            ? 1
            : 0;
    world_environment_mask_[environment_index] =
            Uniform(environment_index, 0.0f, 1.0f) <=
                    static_cast<float>(config_.world_environment_ratio)
            ? 1
            : 0;
    for (std::size_t axis = 0; axis < kCommandDimension; ++axis) {
        world_commands_[offset + axis] = commands_[offset + axis];
    }
    forward_environment_mask_[environment_index] =
            Uniform(environment_index, 0.0f, 1.0f) <=
                    static_cast<float>(config_.forward_environment_ratio)
            ? 1
            : 0;
    if (forward_environment_mask_[environment_index] != 0) {
        commands_[offset + 0] = std::max(std::abs(commands_[offset + 0]), 0.3f);
        commands_[offset + 1] = 0.0f;
        commands_[offset + 2] = 0.0f;
    }
    if (standing_environment_mask_[environment_index] != 0) {
        commands_[offset + 0] = 0.0f;
        commands_[offset + 1] = 0.0f;
        commands_[offset + 2] = 0.0f;
    }
    heading_errors_[environment_index] = 0.0f;
}

float LocomotionCommandRuntime::WrapToPi(float value) {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float two_pi = 2.0f * pi;
    value = std::fmod(value + pi, two_pi);
    if (value < 0.0f) {
        value += two_pi;
    }
    return value - pi;
}

void LocomotionCommandRuntime::AdvanceTimersAndResample() {
    for (std::size_t environment_index = 0; environment_index < environment_count_; ++environment_index) {
        time_left_[environment_index] -= static_cast<float>(config_.step_dt);
        if (time_left_[environment_index] <= 0.0f) {
            Resample(environment_index);
            time_left_[environment_index] = Uniform(
                    environment_index,
                    static_cast<float>(config_.resampling_time[0]),
                    static_cast<float>(config_.resampling_time[1]));
        }
    }
}

void LocomotionCommandRuntime::AdvanceStepAndResample(std::size_t environment_index) {
    const auto interval_steps = std::max<std::uint32_t>(
            static_cast<std::uint32_t>(std::max<RealType>(
                    1.0,
                    std::round(config_.resampling_time[0] /
                               std::max<RealType>(config_.step_dt, 1.0e-9)))),
            1);
    const std::uint32_t step = steps_[environment_index];
    if (step > 0 && step % interval_steps == 0) {
        Resample(environment_index);
    }
    steps_[environment_index] = step + 1;
}

void LocomotionCommandRuntime::UpdateEnvironmentFrame(
        std::size_t environment_index,
        std::span<const float> base_quaternions) {
    const std::size_t command_offset = environment_index * kCommandDimension;
    const std::size_t quaternion_offset = environment_index * kQuaternionDimension;
    const float w = base_quaternions[quaternion_offset + 0];
    const float x = base_quaternions[quaternion_offset + 1];
    const float y = base_quaternions[quaternion_offset + 2];
    const float z = base_quaternions[quaternion_offset + 3];
    const float heading = std::atan2(2.0f * (w * z + x * y),
                                     1.0f - 2.0f * (y * y + z * z));

    if (config_.heading_command && heading_environment_mask_[environment_index] != 0) {
        const float error = WrapToPi(heading_targets_[environment_index] - heading);
        heading_errors_[environment_index] = error;
        commands_[command_offset + 2] = std::clamp(
                static_cast<float>(config_.heading_control_stiffness) * error,
                ranges_[AngularVelocityZMin],
                ranges_[AngularVelocityZMax]);
    }

    if (world_environment_mask_[environment_index] != 0) {
        const float velocity_x_world = world_commands_[command_offset + 0];
        const float velocity_y_world = world_commands_[command_offset + 1];
        const float cosine = std::cos(heading);
        const float sine = std::sin(heading);
        commands_[command_offset + 0] = cosine * velocity_x_world + sine * velocity_y_world;
        commands_[command_offset + 1] = -sine * velocity_x_world + cosine * velocity_y_world;
    }

    if (standing_environment_mask_[environment_index] != 0) {
        for (std::size_t axis = 0; axis < kCommandDimension; ++axis) {
            commands_[command_offset + axis] = 0.0f;
            world_commands_[command_offset + axis] = 0.0f;
        }
    }
}

std::span<float> LocomotionCommandRuntime::Commands() { return commands_; }
std::span<const float> LocomotionCommandRuntime::Commands() const { return commands_; }
std::span<float> LocomotionCommandRuntime::WorldCommands() { return world_commands_; }
std::span<const float> LocomotionCommandRuntime::WorldCommands() const { return world_commands_; }
std::span<float> LocomotionCommandRuntime::HeadingTargets() { return heading_targets_; }
std::span<const float> LocomotionCommandRuntime::HeadingTargets() const { return heading_targets_; }
std::span<float> LocomotionCommandRuntime::HeadingErrors() { return heading_errors_; }
std::span<const float> LocomotionCommandRuntime::HeadingErrors() const { return heading_errors_; }
std::span<float> LocomotionCommandRuntime::TimeLeft() { return time_left_; }
std::span<const float> LocomotionCommandRuntime::TimeLeft() const { return time_left_; }
std::span<std::uint32_t> LocomotionCommandRuntime::Steps() { return steps_; }
std::span<const std::uint32_t> LocomotionCommandRuntime::Steps() const { return steps_; }
std::span<std::uint8_t> LocomotionCommandRuntime::HeadingEnvironmentMask() {
    return heading_environment_mask_;
}
std::span<const std::uint8_t> LocomotionCommandRuntime::HeadingEnvironmentMask() const {
    return heading_environment_mask_;
}
std::span<std::uint8_t> LocomotionCommandRuntime::StandingEnvironmentMask() {
    return standing_environment_mask_;
}
std::span<const std::uint8_t> LocomotionCommandRuntime::StandingEnvironmentMask() const {
    return standing_environment_mask_;
}
std::span<std::uint8_t> LocomotionCommandRuntime::WorldEnvironmentMask() {
    return world_environment_mask_;
}
std::span<const std::uint8_t> LocomotionCommandRuntime::WorldEnvironmentMask() const {
    return world_environment_mask_;
}
std::span<std::uint8_t> LocomotionCommandRuntime::ForwardEnvironmentMask() {
    return forward_environment_mask_;
}
std::span<const std::uint8_t> LocomotionCommandRuntime::ForwardEnvironmentMask() const {
    return forward_environment_mask_;
}
std::span<float> LocomotionCommandRuntime::Ranges() { return ranges_; }
std::span<const float> LocomotionCommandRuntime::Ranges() const { return ranges_; }

} // namespace gobot
