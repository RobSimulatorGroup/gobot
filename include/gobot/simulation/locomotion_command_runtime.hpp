/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "gobot/core/math/math_defs.hpp"
#include "gobot_export.h"

namespace gobot {

struct LocomotionCommandConfig {
    RealType step_dt{0.02};
    std::array<RealType, 2> resampling_time{3.0, 8.0};
    std::array<RealType, 2> linear_velocity_x{-0.6, 1.0};
    std::array<RealType, 2> linear_velocity_y{-0.4, 0.4};
    std::array<RealType, 2> angular_velocity_z{-0.8, 0.8};
    std::array<RealType, 2> heading{-3.14, 3.14};
    RealType standing_environment_ratio{0.1};
    RealType heading_environment_ratio{0.3};
    RealType world_environment_ratio{0.0};
    RealType forward_environment_ratio{0.2};
    bool heading_command{true};
    RealType heading_control_stiffness{0.5};
    RealType zero_small_xy_threshold{0.0};
    std::uint64_t seed{1};
};

class GOBOT_EXPORT LocomotionCommandRuntime {
public:
    static constexpr std::size_t kCommandDimension = 3;
    static constexpr std::size_t kQuaternionDimension = 4;
    static constexpr std::size_t kRangeCount = 8;

    explicit LocomotionCommandRuntime(std::size_t environment_count = 0);

    std::size_t GetEnvironmentCount() const;

    void Configure(const LocomotionCommandConfig& config);

    bool IsConfigured() const;

    RealType GetStepDt() const;

    void SetVelocityRanges(const std::array<RealType, 2>& linear_velocity_x,
                           const std::array<RealType, 2>& linear_velocity_y,
                           const std::array<RealType, 2>& angular_velocity_z);

    void Reset(std::span<const std::size_t> environment_indices);

    void SetCommands(std::span<const std::size_t> environment_indices,
                     std::span<const float> commands,
                     std::span<const float> heading_targets,
                     std::span<const float> time_left);

    void SetSteps(std::span<const std::size_t> environment_indices,
                  std::span<const std::uint32_t> steps);

    void SetStepResamplingEnabled(bool enabled);

    void Advance(std::span<const float> base_quaternions);

    void UpdateFrames(std::span<const float> base_quaternions);

    std::span<float> Commands();
    std::span<const float> Commands() const;

    std::span<float> WorldCommands();
    std::span<const float> WorldCommands() const;

    std::span<float> HeadingTargets();
    std::span<const float> HeadingTargets() const;

    std::span<float> HeadingErrors();
    std::span<const float> HeadingErrors() const;

    std::span<float> TimeLeft();
    std::span<const float> TimeLeft() const;

    std::span<std::uint32_t> Steps();
    std::span<const std::uint32_t> Steps() const;

    std::span<std::uint8_t> HeadingEnvironmentMask();
    std::span<const std::uint8_t> HeadingEnvironmentMask() const;

    std::span<std::uint8_t> StandingEnvironmentMask();
    std::span<const std::uint8_t> StandingEnvironmentMask() const;

    std::span<std::uint8_t> WorldEnvironmentMask();
    std::span<const std::uint8_t> WorldEnvironmentMask() const;

    std::span<std::uint8_t> ForwardEnvironmentMask();
    std::span<const std::uint8_t> ForwardEnvironmentMask() const;

    std::span<float> Ranges();
    std::span<const float> Ranges() const;

private:
    enum RangeIndex : std::size_t {
        LinearVelocityXMin = 0,
        LinearVelocityXMax,
        LinearVelocityYMin,
        LinearVelocityYMax,
        AngularVelocityZMin,
        AngularVelocityZMax,
        HeadingMin,
        HeadingMax,
    };

    void Allocate(std::size_t environment_count);

    void RequireEnvironmentIndex(std::size_t environment_index) const;

    void SeedEnvironmentGenerators(std::uint64_t seed);

    float Uniform(std::size_t environment_index, float lower, float upper);

    void Resample(std::size_t environment_index);

    void AdvanceTimersAndResample();

    void AdvanceStepAndResample(std::size_t environment_index);

    void UpdateEnvironmentFrame(std::size_t environment_index,
                                std::span<const float> base_quaternions);

    static float WrapToPi(float value);

    std::size_t environment_count_{0};
    LocomotionCommandConfig config_;
    bool configured_{false};
    bool step_resampling_enabled_{false};
    std::vector<std::mt19937> environment_rngs_;
    std::vector<float> commands_;
    std::vector<float> world_commands_;
    std::vector<float> heading_targets_;
    std::vector<float> heading_errors_;
    std::vector<float> time_left_;
    std::vector<std::uint32_t> steps_;
    std::vector<std::uint8_t> heading_environment_mask_;
    std::vector<std::uint8_t> standing_environment_mask_;
    std::vector<std::uint8_t> world_environment_mask_;
    std::vector<std::uint8_t> forward_environment_mask_;
    std::vector<float> ranges_;
};

} // namespace gobot
