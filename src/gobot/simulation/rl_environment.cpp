/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/simulation/rl_environment.hpp"

#include <algorithm>
#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot {
namespace {

const PhysicsRobotState* FindRobotState(const PhysicsSceneState& scene_state,
                                        const std::string& robot_name) {
    for (const PhysicsRobotState& robot_state : scene_state.robots) {
        if (robot_state.name == robot_name) {
            return &robot_state;
        }
    }

    return nullptr;
}

const PhysicsRobotSnapshot* FindRobotSnapshot(const PhysicsSceneSnapshot& scene_snapshot,
                                              const std::string& robot_name) {
    for (const PhysicsRobotSnapshot& robot_snapshot : scene_snapshot.robots) {
        if (robot_snapshot.name == robot_name) {
            return &robot_snapshot;
        }
    }

    return nullptr;
}

bool IsControllableJointType(int joint_type) {
    const auto type = static_cast<JointType>(joint_type);
    return type == JointType::Revolute ||
           type == JointType::Continuous ||
           type == JointType::Prismatic;
}

} // namespace

RLEnvironment::RLEnvironment(SimulationServer* simulation)
    : simulation_(simulation) {
}

void RLEnvironment::SetSimulationServer(SimulationServer* simulation) {
    simulation_ = simulation;
}

SimulationServer* RLEnvironment::GetSimulationServer() const {
    return simulation_;
}

void RLEnvironment::SetSceneRoot(const Node* scene_root) {
    scene_root_ = scene_root;
}

const Node* RLEnvironment::GetSceneRoot() const {
    return scene_root_;
}

void RLEnvironment::SetRobotName(std::string robot_name) {
    robot_name_ = std::move(robot_name);
}

const std::string& RLEnvironment::GetRobotName() const {
    return robot_name_;
}

void RLEnvironment::SetMaxEpisodeSteps(std::uint64_t max_episode_steps) {
    max_episode_steps_ = max_episode_steps;
}

std::uint64_t RLEnvironment::GetMaxEpisodeSteps() const {
    return max_episode_steps_;
}

bool RLEnvironment::Reset(std::uint32_t seed) {
    if (simulation_ == nullptr) {
        SetLastError("RL environment has no SimulationServer.");
        return false;
    }

    last_seed_ = seed;
    episode_step_count_ = 0;

    const bool ok = simulation_->HasWorld()
                            ? simulation_->Reset()
                            : simulation_->BuildWorldFromScene(scene_root_);
    if (!ok) {
        SetLastError(simulation_->GetLastError());
        return false;
    }

    if (!RefreshControlledJointNames()) {
        return false;
    }

    last_error_.clear();
    return true;
}

RLEnvironmentStepResult RLEnvironment::Step(const std::vector<RealType>& action) {
    RLEnvironmentStepResult result;
    if (!EnsureReady()) {
        return result;
    }

    if (action.size() != controlled_joint_names_.size()) {
        SetLastError(fmt::format("RL action size mismatch for robot '{}': expected {}, got {}.",
                                 robot_name_,
                                 controlled_joint_names_.size(),
                                 action.size()));
        return result;
    }

    if (!simulation_->SetRobotJointPositionTargetsFromNormalizedAction(robot_name_, action)) {
        SetLastError(simulation_->GetLastError());
        return result;
    }

    if (!simulation_->StepOnce()) {
        SetLastError(simulation_->GetLastError());
        return result;
    }

    ++episode_step_count_;

    result.observation = GetObservation();
    result.frame_count = simulation_->GetFrameCount();
    result.simulation_time = simulation_->GetSimulationTime();
    result.truncated = max_episode_steps_ > 0 && episode_step_count_ >= max_episode_steps_;
    result.reward = 0.0;
    result.terminated = false;
    last_error_.clear();
    return result;
}

std::vector<RealType> RLEnvironment::GetObservation() const {
    std::vector<RealType> observation;
    if (simulation_ == nullptr || !simulation_->HasWorld()) {
        return observation;
    }

    const PhysicsSceneState& scene_state = simulation_->GetWorld()->GetSceneState();
    const PhysicsRobotState* robot_state = FindRobotState(scene_state, robot_name_);
    if (robot_state == nullptr) {
        return observation;
    }

    observation.reserve(controlled_joint_names_.size() * 2);
    for (const PhysicsJointState& joint_state : robot_state->joints) {
        if (std::find(controlled_joint_names_.begin(),
                      controlled_joint_names_.end(),
                      joint_state.joint_name) == controlled_joint_names_.end()) {
            continue;
        }

        observation.push_back(joint_state.position);
        observation.push_back(joint_state.velocity);
    }

    return observation;
}

std::size_t RLEnvironment::GetActionSize() const {
    return controlled_joint_names_.size();
}

std::size_t RLEnvironment::GetObservationSize() const {
    return controlled_joint_names_.size() * 2;
}

std::vector<std::string> RLEnvironment::GetControlledJointNames() const {
    return controlled_joint_names_;
}

const std::string& RLEnvironment::GetLastError() const {
    return last_error_;
}

bool RLEnvironment::RefreshControlledJointNames() {
    controlled_joint_names_.clear();

    if (simulation_ == nullptr || !simulation_->HasWorld()) {
        SetLastError("RL environment simulation world has not been reset.");
        return false;
    }

    if (robot_name_.empty()) {
        SetLastError("RL environment robot name is empty.");
        return false;
    }

    const PhysicsSceneSnapshot& scene_snapshot = simulation_->GetWorld()->GetSceneSnapshot();
    const PhysicsRobotSnapshot* robot_snapshot = FindRobotSnapshot(scene_snapshot, robot_name_);
    if (robot_snapshot == nullptr) {
        SetLastError(fmt::format("RL environment robot '{}' was not found in the physics scene.", robot_name_));
        return false;
    }

    for (const PhysicsJointSnapshot& joint_snapshot : robot_snapshot->joints) {
        if (!IsControllableJointType(joint_snapshot.joint_type)) {
            continue;
        }

        controlled_joint_names_.push_back(joint_snapshot.name);
    }

    last_error_.clear();
    return true;
}

bool RLEnvironment::EnsureReady() {
    if (simulation_ == nullptr) {
        SetLastError("RL environment has no SimulationServer.");
        return false;
    }

    if (!simulation_->HasWorld()) {
        SetLastError("RL environment simulation world has not been reset.");
        return false;
    }

    if (robot_name_.empty()) {
        SetLastError("RL environment robot name is empty.");
        return false;
    }

    if (controlled_joint_names_.empty() && !RefreshControlledJointNames()) {
        return false;
    }

    return true;
}

void RLEnvironment::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<RLEnvironment>("RLEnvironment")
            .constructor()(CtorAsRawPtr)
            .property("robot_name", &RLEnvironment::GetRobotName, &RLEnvironment::SetRobotName)
            .property("max_episode_steps", &RLEnvironment::GetMaxEpisodeSteps, &RLEnvironment::SetMaxEpisodeSteps)
            .method("reset", &RLEnvironment::Reset)
            .method("step", &RLEnvironment::Step)
            .method("get_observation", &RLEnvironment::GetObservation)
            .method("get_action_size", &RLEnvironment::GetActionSize)
            .method("get_observation_size", &RLEnvironment::GetObservationSize)
            .method("get_controlled_joint_names", &RLEnvironment::GetControlledJointNames)
            .method("get_last_error", &RLEnvironment::GetLastError);

};
