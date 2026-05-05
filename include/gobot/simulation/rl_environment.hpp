/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_types.hpp"

namespace gobot {

class Node;
class SimulationServer;

struct RLEnvironmentStepResult {
    std::vector<RealType> observation;
    RealType reward{0.0};
    bool terminated{false};
    bool truncated{false};
    std::uint64_t frame_count{0};
    RealType simulation_time{0.0};
    std::string error;
};

struct RLEnvironmentResetResult {
    std::vector<RealType> observation;
    bool ok{false};
    std::uint64_t frame_count{0};
    RealType simulation_time{0.0};
    std::uint32_t seed{0};
    std::string error;
};

struct RLVectorSpec {
    std::string version{"rl_vector_spec_v1"};
    std::vector<std::string> names;
    std::vector<RealType> lower_bounds;
    std::vector<RealType> upper_bounds;
    std::vector<std::string> units;
};

struct RLEnvironmentRewardSettings {
    RealType alive_reward{1.0};
    RealType fallen_reward{0.0};
    RealType minimum_base_height{0.25};
    RealType maximum_base_tilt_radians{1.0};
    bool terminate_on_fall{true};
};

class GOBOT_EXPORT RLEnvironment : public Object {
    GOBCLASS(RLEnvironment, Object)

public:
    explicit RLEnvironment(SimulationServer* simulation = nullptr);

    void SetSimulationServer(SimulationServer* simulation);

    SimulationServer* GetSimulationServer() const;

    void SetSceneRoot(const Node* scene_root);

    const Node* GetSceneRoot() const;

    void SetRobotName(std::string robot_name);

    const std::string& GetRobotName() const;

    void SetMaxEpisodeSteps(std::uint64_t max_episode_steps);

    std::uint64_t GetMaxEpisodeSteps() const;

    void SetRewardSettings(const RLEnvironmentRewardSettings& settings);

    const RLEnvironmentRewardSettings& GetRewardSettings() const;

    RLEnvironmentResetResult Reset(std::uint32_t seed = 0);

    RLEnvironmentStepResult Step(const std::vector<RealType>& action);

    std::vector<RealType> GetObservation() const;

    std::size_t GetActionSize() const;

    std::size_t GetObservationSize() const;

    std::vector<std::string> GetControlledJointNames() const;

    void SetConfiguredControlledJointNames(std::vector<std::string> joint_names);

    const std::vector<std::string>& GetConfiguredControlledJointNames() const;

    void SetDefaultAction(std::vector<RealType> default_action);

    const std::vector<RealType>& GetDefaultAction() const;

    std::vector<std::string> GetContactLinkNames() const;

    RLVectorSpec GetActionSpec() const;

    RLVectorSpec GetObservationSpec() const;

    const std::string& GetLastError() const;

private:
    bool RefreshControlledJointNames();

    bool RefreshContactLinkNames();

    bool RefreshBaseLinkName();

    bool EnsureReady();

    bool IsBaseFallen() const;

    RealType ComputeReward(bool fallen) const;

    void SetLastError(std::string error);

    SimulationServer* simulation_{nullptr};
    const Node* scene_root_{nullptr};
    std::string robot_name_;
    std::uint64_t episode_step_count_{0};
    std::uint64_t max_episode_steps_{0};
    std::uint32_t last_seed_{0};
    std::string base_link_name_;
    std::vector<std::string> configured_controlled_joint_names_;
    std::vector<std::string> controlled_joint_names_;
    std::vector<RealType> default_action_;
    std::vector<std::string> contact_link_names_;
    RLEnvironmentRewardSettings reward_settings_;
    std::string last_error_;
};

} // namespace gobot
