#include <array>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "gobot/simulation/locomotion_command_runtime.hpp"

namespace gobot {
namespace {

LocomotionCommandConfig DeterministicCommandConfig(std::uint64_t seed) {
    LocomotionCommandConfig config;
    config.seed = seed;
    config.resampling_time = {1.0, 2.0};
    config.linear_velocity_x = {-1.0, 1.0};
    config.linear_velocity_y = {-0.5, 0.5};
    config.angular_velocity_z = {-0.75, 0.75};
    config.heading = {-3.0, 3.0};
    config.standing_environment_ratio = 0.0;
    config.heading_environment_ratio = 0.0;
    config.world_environment_ratio = 0.0;
    config.forward_environment_ratio = 0.0;
    config.heading_command = false;
    return config;
}

std::array<float, 3> CommandForEnvironment(const LocomotionCommandRuntime& runtime,
                                           std::size_t environment_index) {
    const std::span<const float> commands = runtime.Commands();
    const std::size_t offset = environment_index * LocomotionCommandRuntime::kCommandDimension;
    return {commands[offset + 0], commands[offset + 1], commands[offset + 2]};
}

TEST(TestLocomotionCommandRuntime, EnvZeroSequenceDoesNotDependOnBatchSize) {
    LocomotionCommandRuntime single_environment(1);
    LocomotionCommandRuntime batch(8);
    const LocomotionCommandConfig config = DeterministicCommandConfig(1234);
    single_environment.Configure(config);
    batch.Configure(config);

    const std::array<std::size_t, 1> env_zero{0};
    single_environment.Reset(env_zero);
    batch.Reset(env_zero);

    EXPECT_EQ(CommandForEnvironment(single_environment, 0), CommandForEnvironment(batch, 0));
    EXPECT_FLOAT_EQ(single_environment.HeadingTargets()[0], batch.HeadingTargets()[0]);
    EXPECT_FLOAT_EQ(single_environment.TimeLeft()[0], batch.TimeLeft()[0]);
}

TEST(TestLocomotionCommandRuntime, PartialResetsUseIndependentEnvironmentRandomStreams) {
    LocomotionCommandRuntime reference(2);
    LocomotionCommandRuntime interleaved(2);
    const LocomotionCommandConfig config = DeterministicCommandConfig(42);
    reference.Configure(config);
    interleaved.Configure(config);

    const std::array<std::size_t, 1> env_zero{0};
    const std::array<std::size_t, 1> env_one{1};
    reference.Reset(env_zero);
    interleaved.Reset(env_zero);
    for (int reset = 0; reset < 32; ++reset) {
        interleaved.Reset(env_one);
    }

    reference.Reset(env_zero);
    interleaved.Reset(env_zero);
    EXPECT_EQ(CommandForEnvironment(reference, 0), CommandForEnvironment(interleaved, 0));
    EXPECT_FLOAT_EQ(reference.HeadingTargets()[0], interleaved.HeadingTargets()[0]);
    EXPECT_FLOAT_EQ(reference.TimeLeft()[0], interleaved.TimeLeft()[0]);
}

TEST(TestLocomotionCommandRuntime, ConvertsWorldVelocityIntoBodyFrame) {
    LocomotionCommandRuntime runtime(1);
    LocomotionCommandConfig config;
    config.linear_velocity_x = {1.0, 1.0};
    config.linear_velocity_y = {0.0, 0.0};
    config.angular_velocity_z = {0.0, 0.0};
    config.heading_command = false;
    config.standing_environment_ratio = 0.0;
    config.heading_environment_ratio = 0.0;
    config.world_environment_ratio = 1.0;
    config.forward_environment_ratio = 0.0;
    runtime.Configure(config);
    runtime.Reset(std::array<std::size_t, 1>{0});

    constexpr float half_sqrt_two = 0.7071067811865475f;
    const std::array<float, 4> yaw_ninety_degrees{
            half_sqrt_two, 0.0f, 0.0f, half_sqrt_two};
    runtime.UpdateFrames(yaw_ninety_degrees);

    const auto command = CommandForEnvironment(runtime, 0);
    EXPECT_NEAR(command[0], 0.0f, 1.0e-6f);
    EXPECT_NEAR(command[1], -1.0f, 1.0e-6f);
    EXPECT_FLOAT_EQ(command[2], 0.0f);
}

TEST(TestLocomotionCommandRuntime, AppliesHeadingFeedbackWithinYawLimits) {
    LocomotionCommandRuntime runtime(1);
    LocomotionCommandConfig config;
    config.linear_velocity_x = {0.5, 0.5};
    config.linear_velocity_y = {0.0, 0.0};
    config.angular_velocity_z = {-0.6, 0.6};
    config.heading = {1.5707963267948966, 1.5707963267948966};
    config.heading_command = true;
    config.heading_control_stiffness = 1.0;
    config.standing_environment_ratio = 0.0;
    config.heading_environment_ratio = 1.0;
    config.world_environment_ratio = 0.0;
    config.forward_environment_ratio = 0.0;
    runtime.Configure(config);
    runtime.Reset(std::array<std::size_t, 1>{0});

    runtime.UpdateFrames(std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f});

    const auto command = CommandForEnvironment(runtime, 0);
    EXPECT_FLOAT_EQ(command[2], 0.6f);
    EXPECT_NEAR(runtime.HeadingErrors()[0], 1.5707963f, 1.0e-6f);
}

TEST(TestLocomotionCommandRuntime, RunSamplingProducesPureForwardCommands) {
    LocomotionCommandRuntime runtime(64);
    LocomotionCommandConfig config = DeterministicCommandConfig(73);
    config.heading_environment_ratio = 1.0;
    config.world_environment_ratio = 1.0;
    config.forward_environment_ratio = 1.0;
    config.run_environment_ratio = 1.0;
    config.run_velocity_x = {1.8, 2.2};
    runtime.Configure(config);

    std::vector<std::size_t> env_ids(64);
    for (std::size_t index = 0; index < env_ids.size(); ++index) {
        env_ids[index] = index;
    }
    runtime.Reset(env_ids);
    std::vector<float> identity_quaternions(64 * 4, 0.0f);
    for (std::size_t index = 0; index < 64; ++index) {
        identity_quaternions[index * 4] = 1.0f;
    }
    runtime.UpdateFrames(identity_quaternions);

    for (std::size_t index = 0; index < 64; ++index) {
        const auto command = CommandForEnvironment(runtime, index);
        EXPECT_GE(command[0], 1.8f);
        EXPECT_LE(command[0], 2.2f);
        EXPECT_FLOAT_EQ(command[1], 0.0f);
        EXPECT_FLOAT_EQ(command[2], 0.0f);
        EXPECT_EQ(runtime.RunEnvironmentMask()[index], 1);
        EXPECT_EQ(runtime.HeadingEnvironmentMask()[index], 0);
        EXPECT_EQ(runtime.WorldEnvironmentMask()[index], 0);
    }
}

} // namespace
} // namespace gobot
