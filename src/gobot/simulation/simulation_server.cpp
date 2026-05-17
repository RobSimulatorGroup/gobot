/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/simulation_server.hpp"

#include <algorithm>
#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/physics/joint_controller.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/robot_3d.hpp"

namespace gobot {
namespace {

const PhysicsRobotSnapshot* FindRobotSnapshot(const PhysicsSceneSnapshot& snapshot,
                                              std::size_t robot_index,
                                              const std::string& robot_name) {
    if (robot_index < snapshot.robots.size()) {
        return &snapshot.robots[robot_index];
    }

    for (const PhysicsRobotSnapshot& robot_snapshot : snapshot.robots) {
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

const PhysicsLinkSnapshot* FindLinkSnapshot(const PhysicsRobotSnapshot& robot_snapshot,
                                            const std::string& link_name) {
    for (const PhysicsLinkSnapshot& link_snapshot : robot_snapshot.links) {
        if (link_snapshot.name == link_name) {
            return &link_snapshot;
        }
    }

    return nullptr;
}

const char* BackendName(PhysicsBackendType backend_type) {
    switch (backend_type) {
        case PhysicsBackendType::Null:
            return "Null";
        case PhysicsBackendType::MuJoCoCpu:
            return "MuJoCo CPU";
        case PhysicsBackendType::MuJoCoWarp:
            return "MuJoCo Warp";
        case PhysicsBackendType::NewtonGpu:
            return "Newton GPU";
        case PhysicsBackendType::RigidIpcCpu:
            return "Rigid IPC CPU";
    }

    return "Unknown";
}

void ApplyLinkGlobalTransform(Link3D* link, const Affine3& global_transform) {
    if (link == nullptr) {
        return;
    }

    if (link->IsInsideTree()) {
        link->SetGlobalTransform(global_transform);
    } else {
        link->SetTransform(global_transform);
    }
}

void ApplyNodeGlobalTransform(Node3D* node, const Affine3& global_transform) {
    if (node == nullptr) {
        return;
    }

    if (node->IsInsideTree()) {
        node->SetGlobalTransform(global_transform);
    } else {
        node->SetTransform(global_transform);
    }
}

} // namespace

SimulationServer* SimulationServer::s_singleton = nullptr;

SimulationServer::SimulationServer(PhysicsBackendType backend_type)
    : backend_type_(backend_type) {
    s_singleton = this;
}

SimulationServer::~SimulationServer() {
    if (s_singleton == this) {
        s_singleton = nullptr;
    }
}

SimulationServer* SimulationServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initializing SimulationServer.");
    return s_singleton;
}

bool SimulationServer::HasInstance() {
    return s_singleton != nullptr;
}

PhysicsBackendType SimulationServer::GetBackendType() const {
    return backend_type_;
}

void SimulationServer::SetBackendType(PhysicsBackendType backend_type) {
    if (backend_type_ == backend_type) {
        return;
    }

    backend_type_ = backend_type;
    ClearWorld();
}

const PhysicsWorldSettings& SimulationServer::GetPhysicsWorldSettings() const {
    return physics_world_settings_;
}

void SimulationServer::SetPhysicsWorldSettings(const PhysicsWorldSettings& settings) {
    physics_world_settings_ = settings;
    if (world_.IsValid()) {
        world_->SetSettings(physics_world_settings_);
    }
}

const JointControllerGains& SimulationServer::GetDefaultJointGains() const {
    return physics_world_settings_.default_joint_gains;
}

void SimulationServer::SetDefaultJointGains(const JointControllerGains& gains) {
    physics_world_settings_.default_joint_gains = gains;
    if (world_.IsValid()) {
        world_->SetSettings(physics_world_settings_);
    }
}

RealType SimulationServer::GetFixedTimeStep() const {
    return physics_world_settings_.fixed_time_step;
}

void SimulationServer::SetFixedTimeStep(RealType fixed_time_step) {
    if (fixed_time_step <= 0.0) {
        SetLastError("Simulation fixed time step must be greater than zero.");
        return;
    }

    physics_world_settings_.fixed_time_step = fixed_time_step;
    if (world_.IsValid()) {
        world_->SetSettings(physics_world_settings_);
    }
}

RealType SimulationServer::GetTimeScale() const {
    return time_scale_;
}

void SimulationServer::SetTimeScale(RealType time_scale) {
    if (time_scale < 0.0) {
        SetLastError("Simulation time scale cannot be negative.");
        return;
    }

    time_scale_ = time_scale;
}

int SimulationServer::GetMaxSubSteps() const {
    return max_sub_steps_;
}

void SimulationServer::SetMaxSubSteps(int max_sub_steps) {
    if (max_sub_steps <= 0) {
        SetLastError("Simulation max sub-steps must be greater than zero.");
        return;
    }

    max_sub_steps_ = max_sub_steps;
}

int SimulationServer::GetLastStepCount() const {
    return last_step_count_;
}

bool SimulationServer::IsPaused() const {
    return paused_;
}

void SimulationServer::SetPaused(bool paused) {
    paused_ = paused;
}

bool SimulationServer::BuildWorldFromScene(const Node* scene_root) {
    scene_root_ = scene_root;
    world_ = PhysicsServer::CreateWorldForBackend(backend_type_, physics_world_settings_);
    if (!world_.IsValid()) {
        SetLastError("Failed to create physics world.");
        return false;
    }
    if (world_->GetBackendType() != backend_type_) {
        SetLastError(fmt::format("Requested physics backend '{}' is not available or implemented.",
                                 BackendName(backend_type_)));
        world_.Reset();
        return false;
    }

    if (!world_->BuildFromScene(scene_root)) {
        SetLastError(world_->GetLastError());
        world_.Reset();
        return false;
    }

    ResetClock();
    last_error_.clear();
    return true;
}

bool SimulationServer::RebuildWorldFromScene(const Node* scene_root, bool preserve_state) {
    PhysicsSceneState previous_state;
    if (preserve_state && world_.IsValid()) {
        previous_state = world_->GetSceneState();
    }

    scene_root_ = scene_root;
    world_ = PhysicsServer::CreateWorldForBackend(backend_type_, physics_world_settings_);
    if (!world_.IsValid()) {
        SetLastError("Failed to create physics world.");
        return false;
    }
    if (world_->GetBackendType() != backend_type_) {
        SetLastError(fmt::format("Requested physics backend '{}' is not available or implemented.",
                                 BackendName(backend_type_)));
        world_.Reset();
        return false;
    }

    if (!world_->BuildFromScene(scene_root)) {
        SetLastError(world_->GetLastError());
        world_.Reset();
        return false;
    }

    if (preserve_state && !world_->RestoreCompatibleState(previous_state)) {
        SetLastError(world_->GetLastError());
        world_.Reset();
        return false;
    }

    ResetClock();
    ApplyWorldStateToScene();
    last_error_.clear();
    return true;
}

const Node* SimulationServer::GetSceneRoot() const {
    return scene_root_;
}

void SimulationServer::ClearWorld() {
    world_.Reset();
    scene_root_ = nullptr;
    ResetClock();
}

bool SimulationServer::HasWorld() const {
    return world_.IsValid();
}

Ref<PhysicsWorld> SimulationServer::GetWorld() const {
    return world_;
}

bool SimulationServer::Reset() {
    if (!EnsureWorldReady()) {
        return false;
    }

    world_->Reset();
    ResetClock();
    ApplyWorldStateToScene();
    last_error_.clear();
    return true;
}

bool SimulationServer::StepOnce() {
    if (!StepFixed()) {
        last_step_count_ = 0;
        return false;
    }

    accumulator_ = 0.0;
    last_step_count_ = 1;
    last_error_.clear();
    return true;
}

int SimulationServer::Step(RealType delta_time) {
    if (paused_ || delta_time <= 0.0 || time_scale_ <= 0.0) {
        last_step_count_ = 0;
        return 0;
    }

    if (!EnsureWorldReady()) {
        last_step_count_ = 0;
        return 0;
    }

    accumulator_ += delta_time * time_scale_;

    int steps = 0;
    while (accumulator_ + CMP_EPSILON >= physics_world_settings_.fixed_time_step && steps < max_sub_steps_) {
        if (!StepFixed()) {
            last_step_count_ = steps;
            return steps;
        }
        accumulator_ -= physics_world_settings_.fixed_time_step;
        ++steps;
    }

    if (steps == max_sub_steps_ && accumulator_ >= physics_world_settings_.fixed_time_step) {
        accumulator_ = 0.0;
    }

    if (steps > 0) {
        last_error_.clear();
    }

    last_step_count_ = steps;
    return steps;
}

bool SimulationServer::SyncSceneFromWorld() {
    if (!EnsureWorldReady()) {
        return false;
    }

    return ApplyWorldStateToScene();
}

bool SimulationServer::SetJointPositionTarget(const std::string& robot_name,
                                              const std::string& joint_name,
                                              RealType target_position) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->SetJointControl(robot_name, joint_name, PhysicsJointControlMode::Position, target_position)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::SetJointVelocityTarget(const std::string& robot_name,
                                              const std::string& joint_name,
                                              RealType target_velocity) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->SetJointControl(robot_name, joint_name, PhysicsJointControlMode::Velocity, target_velocity)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::SetJointEffortTarget(const std::string& robot_name,
                                            const std::string& joint_name,
                                            RealType target_effort) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->SetJointControl(robot_name, joint_name, PhysicsJointControlMode::Effort, target_effort)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::SetJointPassive(const std::string& robot_name,
                                       const std::string& joint_name) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->SetJointControl(robot_name, joint_name, PhysicsJointControlMode::Passive, 0.0)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::ResetJointState(const std::string& robot_name,
                                       const std::string& joint_name,
                                       RealType position,
                                       RealType velocity) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->ResetJointState(robot_name, joint_name, position, velocity)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    ApplyWorldStateToScene();
    last_error_.clear();
    return true;
}

bool SimulationServer::SetLinkExternalForce(const std::string& robot_name,
                                            const std::string& link_name,
                                            const Vector3& point,
                                            const Vector3& force) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->SetLinkExternalForce(robot_name, link_name, point, force)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::SetLinkSpringForce(const std::string& robot_name,
                                          const std::string& link_name,
                                          const Vector3& local_point,
                                          const Vector3& target_point,
                                          const Vector3& force_hint) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->SetLinkSpringForce(robot_name, link_name, local_point, target_point, force_hint)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

void SimulationServer::ClearExternalForces() {
    if (world_.IsValid()) {
        world_->ClearExternalForces();
    }
}

bool SimulationServer::SetRobotJointPositionTargetsFromNormalizedAction(const std::string& robot_name,
                                                                        const std::vector<RealType>& action) {
    if (!EnsureWorldReady()) {
        return false;
    }

    const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
    const PhysicsSceneState& state = world_->GetSceneState();

    const PhysicsRobotSnapshot* robot_snapshot = nullptr;
    const PhysicsRobotState* robot_state = nullptr;
    for (std::size_t robot_index = 0; robot_index < snapshot.robots.size(); ++robot_index) {
        if (snapshot.robots[robot_index].name != robot_name) {
            continue;
        }

        robot_snapshot = &snapshot.robots[robot_index];
        if (robot_index < state.robots.size()) {
            robot_state = &state.robots[robot_index];
        }
        break;
    }

    if (robot_snapshot == nullptr || robot_state == nullptr) {
        SetLastError(fmt::format("Cannot set normalized action for missing robot '{}'.", robot_name));
        return false;
    }

    std::size_t action_index = 0;
    for (std::size_t joint_index = 0; joint_index < robot_snapshot->joints.size(); ++joint_index) {
        if (joint_index >= robot_state->joints.size()) {
            continue;
        }

        const PhysicsJointSnapshot& joint_snapshot = robot_snapshot->joints[joint_index];
        const PhysicsJointState& joint_state = robot_state->joints[joint_index];
        const auto joint_type = static_cast<JointType>(joint_snapshot.joint_type);
        if (joint_type != JointType::Revolute &&
            joint_type != JointType::Continuous &&
            joint_type != JointType::Prismatic) {
            continue;
        }

        if (action_index >= action.size()) {
            SetLastError(fmt::format("Robot '{}' expected at least {} joint action value(s), got {}.",
                                     robot_name,
                                     action_index + 1,
                                     action.size()));
            return false;
        }

        JointControllerLimits limits = MakeJointControllerLimits(joint_snapshot);
        if (joint_type == JointType::Continuous) {
            limits.has_position_limits = false;
        }

        const RealType target_position =
                JointController::MapNormalizedActionToTargetPosition(action[action_index],
                                                                     limits,
                                                                     joint_state.position,
                                                                     1.0);
        if (!world_->SetJointControl(robot_name,
                                     joint_state.joint_name,
                                     PhysicsJointControlMode::Position,
                                     target_position)) {
            SetLastError(world_->GetLastError());
            return false;
        }

        ++action_index;
    }

    if (action_index < action.size()) {
        SetLastError(fmt::format("Robot '{}' expected {} joint action value(s), got {}.",
                                 robot_name,
                                 action_index,
                                 action.size()));
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::SetRobotJointPositionTargetsFromNormalizedAction(
        const std::string& robot_name,
        const std::vector<std::string>& joint_names,
        const std::vector<RealType>& action) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (joint_names.size() != action.size()) {
        SetLastError(fmt::format("Robot '{}' expected {} named joint action value(s), got {}.",
                                 robot_name,
                                 joint_names.size(),
                                 action.size()));
        return false;
    }

    const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
    const PhysicsSceneState& state = world_->GetSceneState();

    const PhysicsRobotSnapshot* robot_snapshot = nullptr;
    const PhysicsRobotState* robot_state = nullptr;
    for (std::size_t robot_index = 0; robot_index < snapshot.robots.size(); ++robot_index) {
        if (snapshot.robots[robot_index].name != robot_name) {
            continue;
        }

        robot_snapshot = &snapshot.robots[robot_index];
        if (robot_index < state.robots.size()) {
            robot_state = &state.robots[robot_index];
        }
        break;
    }

    if (robot_snapshot == nullptr || robot_state == nullptr) {
        SetLastError(fmt::format("Cannot set normalized action for missing robot '{}'.", robot_name));
        return false;
    }

    for (std::size_t action_index = 0; action_index < joint_names.size(); ++action_index) {
        const std::string& joint_name = joint_names[action_index];
        const auto snapshot_iter = std::find_if(robot_snapshot->joints.begin(),
                                                robot_snapshot->joints.end(),
                                                [&joint_name](const PhysicsJointSnapshot& joint_snapshot) {
                                                    return joint_snapshot.name == joint_name;
                                                });
        const auto state_iter = std::find_if(robot_state->joints.begin(),
                                             robot_state->joints.end(),
                                             [&joint_name](const PhysicsJointState& joint_state) {
                                                 return joint_state.joint_name == joint_name;
                                             });
        if (snapshot_iter == robot_snapshot->joints.end() || state_iter == robot_state->joints.end()) {
            SetLastError(fmt::format("Robot '{}' has no joint named '{}'.", robot_name, joint_name));
            return false;
        }

        const auto joint_type = static_cast<JointType>(snapshot_iter->joint_type);
        if (joint_type != JointType::Revolute &&
            joint_type != JointType::Continuous &&
            joint_type != JointType::Prismatic) {
            SetLastError(fmt::format("Robot '{}' joint '{}' is not controllable by normalized position action.",
                                     robot_name,
                                     joint_name));
            return false;
        }

        JointControllerLimits limits = MakeJointControllerLimits(*snapshot_iter);
        if (joint_type == JointType::Continuous) {
            limits.has_position_limits = false;
        }

        const RealType target_position =
                JointController::MapNormalizedActionToTargetPosition(action[action_index],
                                                                     limits,
                                                                     state_iter->position,
                                                                     1.0);
        if (!world_->SetJointControl(robot_name,
                                     state_iter->joint_name,
                                     PhysicsJointControlMode::Position,
                                     target_position)) {
            SetLastError(world_->GetLastError());
            return false;
        }
    }

    last_error_.clear();
    return true;
}

RealType SimulationServer::GetSimulationTime() const {
    return simulation_time_;
}

std::uint64_t SimulationServer::GetFrameCount() const {
    return frame_count_;
}

RealType SimulationServer::GetAccumulator() const {
    return accumulator_;
}

const std::string& SimulationServer::GetLastError() const {
    return last_error_;
}

bool SimulationServer::EnsureWorldReady() {
    if (!world_.IsValid()) {
        SetLastError("Simulation world has not been built from a scene.");
        return false;
    }

    if (!world_->IsAvailable()) {
        SetLastError(world_->GetLastError());
        return false;
    }

    return true;
}

bool SimulationServer::StepFixed() {
    if (!EnsureWorldReady()) {
        return false;
    }

    world_->Step(physics_world_settings_.fixed_time_step);
    simulation_time_ += physics_world_settings_.fixed_time_step;
    ++frame_count_;
    ApplyWorldStateToScene();
    return true;
}

bool SimulationServer::ApplyWorldStateToScene() {
    if (!world_.IsValid()) {
        return false;
    }

    const PhysicsSceneState& scene_state = world_->GetSceneState();
    const PhysicsSceneSnapshot& scene_snapshot = world_->GetSceneSnapshot();
    for (std::size_t robot_index = 0; robot_index < scene_state.robots.size(); ++robot_index) {
        const PhysicsRobotState& robot_state = scene_state.robots[robot_index];
        auto* robot = const_cast<Robot3D*>(robot_state.node);
        if (!robot || robot->GetMode() != RobotMode::Motion) {
            continue;
        }

        const PhysicsRobotSnapshot* robot_snapshot =
                FindRobotSnapshot(scene_snapshot, robot_index, robot_state.name);
        if (robot_snapshot != nullptr) {
            for (const PhysicsJointSnapshot& joint_snapshot : robot_snapshot->joints) {
                if (static_cast<JointType>(joint_snapshot.joint_type) != JointType::Floating) {
                    continue;
                }

                const PhysicsLinkState* floating_link_state =
                        FindLinkState(robot_state, joint_snapshot.child_link);
                if (floating_link_state == nullptr) {
                    continue;
                }

                auto* joint = const_cast<Joint3D*>(joint_snapshot.node);
                auto* floating_link = const_cast<Link3D*>(floating_link_state->node);
                const PhysicsLinkSnapshot* floating_link_snapshot =
                        FindLinkSnapshot(*robot_snapshot, joint_snapshot.child_link);
                if (joint != nullptr && floating_link != nullptr && floating_link_snapshot != nullptr) {
                    const Affine3 joint_to_child =
                            joint_snapshot.global_transform.inverse() * floating_link_snapshot->global_transform;
                    ApplyNodeGlobalTransform(joint, floating_link_state->global_transform * joint_to_child.inverse());
                    floating_link->SetTransform(joint_to_child);
                } else {
                    ApplyLinkGlobalTransform(floating_link, floating_link_state->global_transform);
                }
            }
        }

        for (const PhysicsJointState& joint_state : robot_state.joints) {
            auto* joint = const_cast<Joint3D*>(joint_state.node);
            if (joint && joint->IsMotionModeEnabled() && joint->GetJointType() != JointType::Floating) {
                joint->SetJointPosition(joint_state.position);
            }
        }
    }

    return true;
}

void SimulationServer::ResetClock() {
    accumulator_ = 0.0;
    simulation_time_ = 0.0;
    frame_count_ = 0;
    last_step_count_ = 0;
}

void SimulationServer::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {
    auto set_robot_normalized_action =
            static_cast<bool (SimulationServer::*)(const std::string&, const std::vector<RealType>&)>(
                    &SimulationServer::SetRobotJointPositionTargetsFromNormalizedAction);

    Class_<SimulationServer>("SimulationServer")
            .constructor()(CtorAsRawPtr)
            .property("backend_type", &SimulationServer::GetBackendType, &SimulationServer::SetBackendType)
            .property("fixed_time_step", &SimulationServer::GetFixedTimeStep, &SimulationServer::SetFixedTimeStep)
            .property("default_joint_gains", &SimulationServer::GetDefaultJointGains, &SimulationServer::SetDefaultJointGains)
            .property("time_scale", &SimulationServer::GetTimeScale, &SimulationServer::SetTimeScale)
            .property("max_sub_steps", &SimulationServer::GetMaxSubSteps, &SimulationServer::SetMaxSubSteps)
            .property_readonly("last_step_count", &SimulationServer::GetLastStepCount)
            .property("paused", &SimulationServer::IsPaused, &SimulationServer::SetPaused)
            .method("build_world_from_scene", &SimulationServer::BuildWorldFromScene)
            .method("rebuild_world_from_scene", &SimulationServer::RebuildWorldFromScene)
            .method("clear_world", &SimulationServer::ClearWorld)
            .method("has_world", &SimulationServer::HasWorld)
            .method("reset", &SimulationServer::Reset)
            .method("step_once", &SimulationServer::StepOnce)
            .method("step", &SimulationServer::Step)
            .method("sync_scene_from_world", &SimulationServer::SyncSceneFromWorld)
            .method("set_joint_position_target", &SimulationServer::SetJointPositionTarget)
            .method("set_joint_velocity_target", &SimulationServer::SetJointVelocityTarget)
            .method("set_joint_effort_target", &SimulationServer::SetJointEffortTarget)
            .method("set_joint_passive", &SimulationServer::SetJointPassive)
            .method("reset_joint_state", &SimulationServer::ResetJointState)
            .method("set_robot_joint_position_targets_from_normalized_action",
                    set_robot_normalized_action)
            .method("get_simulation_time", &SimulationServer::GetSimulationTime)
            .method("get_frame_count", &SimulationServer::GetFrameCount)
            .method("get_last_error", &SimulationServer::GetLastError);

};
