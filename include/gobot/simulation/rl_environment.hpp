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
};

struct RLVectorSpec {
    std::string version{"rl_vector_spec_v1"};
    std::vector<std::string> names;
    std::vector<RealType> lower_bounds;
    std::vector<RealType> upper_bounds;
    std::vector<std::string> units;
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

    bool Reset(std::uint32_t seed = 0);

    RLEnvironmentStepResult Step(const std::vector<RealType>& action);

    std::vector<RealType> GetObservation() const;

    std::size_t GetActionSize() const;

    std::size_t GetObservationSize() const;

    std::vector<std::string> GetControlledJointNames() const;

    RLVectorSpec GetActionSpec() const;

    RLVectorSpec GetObservationSpec() const;

    const std::string& GetLastError() const;

private:
    bool RefreshControlledJointNames();

    bool RefreshBaseLinkName();

    bool EnsureReady();

    void SetLastError(std::string error);

    SimulationServer* simulation_{nullptr};
    const Node* scene_root_{nullptr};
    std::string robot_name_;
    std::uint64_t episode_step_count_{0};
    std::uint64_t max_episode_steps_{0};
    std::uint32_t last_seed_{0};
    std::string base_link_name_;
    std::vector<std::string> controlled_joint_names_;
    std::string last_error_;
};

} // namespace gobot
