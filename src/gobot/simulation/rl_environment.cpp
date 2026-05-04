/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/simulation/rl_environment.hpp"

#include <algorithm>
#include <limits>
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

const PhysicsLinkState* FindLinkState(const PhysicsRobotState& robot_state,
                                      const std::string& link_name) {
    for (const PhysicsLinkState& link_state : robot_state.links) {
        if (link_state.link_name == link_name) {
            return &link_state;
        }
    }

    return nullptr;
}

const PhysicsJointState* FindJointState(const PhysicsRobotState& robot_state,
                                        const std::string& joint_name) {
    for (const PhysicsJointState& joint_state : robot_state.joints) {
        if (joint_state.joint_name == joint_name) {
            return &joint_state;
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

void PushVector3(std::vector<RealType>* values, const Vector3& vector) {
    values->push_back(vector.x());
    values->push_back(vector.y());
    values->push_back(vector.z());
}

void PushInfiniteBounds(RLVectorSpec* spec, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        spec->lower_bounds.push_back(-std::numeric_limits<RealType>::infinity());
        spec->upper_bounds.push_back(std::numeric_limits<RealType>::infinity());
    }
}

void AddSpecEntry(RLVectorSpec* spec,
                  std::string name,
                  RealType lower_bound,
                  RealType upper_bound,
                  std::string unit) {
    spec->names.emplace_back(std::move(name));
    spec->lower_bounds.push_back(lower_bound);
    spec->upper_bounds.push_back(upper_bound);
    spec->units.emplace_back(std::move(unit));
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

    if (!RefreshBaseLinkName() || !RefreshControlledJointNames()) {
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

    const PhysicsLinkState* base_link_state = FindLinkState(*robot_state, base_link_name_);
    if (base_link_state == nullptr) {
        return observation;
    }

    observation.reserve(GetObservationSize());
    PushVector3(&observation, base_link_state->global_transform.translation());

    const Quaternion base_orientation(base_link_state->global_transform.linear());
    observation.push_back(base_orientation.x());
    observation.push_back(base_orientation.y());
    observation.push_back(base_orientation.z());
    observation.push_back(base_orientation.w());

    PushVector3(&observation, base_link_state->linear_velocity);
    PushVector3(&observation, base_link_state->angular_velocity);

    for (const std::string& joint_name : controlled_joint_names_) {
        const PhysicsJointState* joint_state = FindJointState(*robot_state, joint_name);
        if (joint_state == nullptr) {
            return {};
        }
        observation.push_back(joint_state->position);
        observation.push_back(joint_state->velocity);
    }

    return observation;
}

std::size_t RLEnvironment::GetActionSize() const {
    return controlled_joint_names_.size();
}

std::size_t RLEnvironment::GetObservationSize() const {
    return base_link_name_.empty() ? 0 : 13 + controlled_joint_names_.size() * 2;
}

std::vector<std::string> RLEnvironment::GetControlledJointNames() const {
    return controlled_joint_names_;
}

RLVectorSpec RLEnvironment::GetActionSpec() const {
    RLVectorSpec spec;
    spec.names.reserve(controlled_joint_names_.size());
    spec.lower_bounds.reserve(controlled_joint_names_.size());
    spec.upper_bounds.reserve(controlled_joint_names_.size());
    spec.units.reserve(controlled_joint_names_.size());

    for (const std::string& joint_name : controlled_joint_names_) {
        spec.names.push_back(joint_name + "/target_position_normalized");
        spec.lower_bounds.push_back(-1.0);
        spec.upper_bounds.push_back(1.0);
        spec.units.emplace_back("normalized");
    }

    return spec;
}

RLVectorSpec RLEnvironment::GetObservationSpec() const {
    RLVectorSpec spec;
    spec.names.reserve(GetObservationSize());
    spec.lower_bounds.reserve(GetObservationSize());
    spec.upper_bounds.reserve(GetObservationSize());
    spec.units.reserve(GetObservationSize());

    if (!base_link_name_.empty()) {
        for (const char* axis : {"x", "y", "z"}) {
            AddSpecEntry(&spec,
                         "base/position/" + std::string(axis),
                         -std::numeric_limits<RealType>::infinity(),
                         std::numeric_limits<RealType>::infinity(),
                         "meter");
        }

        for (const char* component : {"x", "y", "z", "w"}) {
            AddSpecEntry(&spec,
                         "base/orientation/" + std::string(component),
                         -1.0,
                         1.0,
                         "quaternion");
        }

        for (const char* axis : {"x", "y", "z"}) {
            AddSpecEntry(&spec,
                         "base/linear_velocity/" + std::string(axis),
                         -std::numeric_limits<RealType>::infinity(),
                         std::numeric_limits<RealType>::infinity(),
                         "meter_per_second");
        }

        for (const char* axis : {"x", "y", "z"}) {
            AddSpecEntry(&spec,
                         "base/angular_velocity/" + std::string(axis),
                         -std::numeric_limits<RealType>::infinity(),
                         std::numeric_limits<RealType>::infinity(),
                         "radian_per_second");
        }
    }

    for (const std::string& joint_name : controlled_joint_names_) {
        AddSpecEntry(&spec,
                     joint_name + "/position",
                     -std::numeric_limits<RealType>::infinity(),
                     std::numeric_limits<RealType>::infinity(),
                     "radian_or_meter");
        AddSpecEntry(&spec,
                     joint_name + "/velocity",
                     -std::numeric_limits<RealType>::infinity(),
                     std::numeric_limits<RealType>::infinity(),
                     "radian_per_second_or_meter_per_second");
    }

    return spec;
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

bool RLEnvironment::RefreshBaseLinkName() {
    base_link_name_.clear();

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
        if (static_cast<JointType>(joint_snapshot.joint_type) == JointType::Floating &&
            !joint_snapshot.child_link.empty()) {
            base_link_name_ = joint_snapshot.child_link;
            last_error_.clear();
            return true;
        }
    }

    for (const PhysicsLinkSnapshot& link_snapshot : robot_snapshot->links) {
        if (link_snapshot.role == PhysicsLinkRole::Physical) {
            base_link_name_ = link_snapshot.name;
            last_error_.clear();
            return true;
        }
    }

    SetLastError(fmt::format("RL environment robot '{}' has no physical base link.", robot_name_));
    return false;
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

    if ((base_link_name_.empty() && !RefreshBaseLinkName()) ||
        (controlled_joint_names_.empty() && !RefreshControlledJointNames())) {
        return false;
    }

    return true;
}

void RLEnvironment::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<RLVectorSpec>("RLVectorSpec")
            .constructor()
            .property("version", &RLVectorSpec::version)
            .property("names", &RLVectorSpec::names)
            .property("lower_bounds", &RLVectorSpec::lower_bounds)
            .property("upper_bounds", &RLVectorSpec::upper_bounds)
            .property("units", &RLVectorSpec::units);

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
            .method("get_action_spec", &RLEnvironment::GetActionSpec)
            .method("get_observation_spec", &RLEnvironment::GetObservationSpec)
            .method("get_last_error", &RLEnvironment::GetLastError);

};
