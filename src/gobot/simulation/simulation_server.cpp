/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/simulation/simulation_server.hpp"

#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/error_macros.hpp"
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
        return false;
    }

    accumulator_ = 0.0;
    last_error_.clear();
    return true;
}

int SimulationServer::Step(RealType delta_time) {
    if (paused_ || delta_time <= 0.0 || time_scale_ <= 0.0) {
        return 0;
    }

    if (!EnsureWorldReady()) {
        return 0;
    }

    accumulator_ += delta_time * time_scale_;

    int steps = 0;
    while (accumulator_ + CMP_EPSILON >= physics_world_settings_.fixed_time_step && steps < max_sub_steps_) {
        if (!StepFixed()) {
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

                ApplyLinkGlobalTransform(const_cast<Link3D*>(floating_link_state->node),
                                         floating_link_state->global_transform);
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
}

void SimulationServer::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<SimulationServer>("SimulationServer")
            .constructor()(CtorAsRawPtr)
            .property("backend_type", &SimulationServer::GetBackendType, &SimulationServer::SetBackendType)
            .property("fixed_time_step", &SimulationServer::GetFixedTimeStep, &SimulationServer::SetFixedTimeStep)
            .property("time_scale", &SimulationServer::GetTimeScale, &SimulationServer::SetTimeScale)
            .property("max_sub_steps", &SimulationServer::GetMaxSubSteps, &SimulationServer::SetMaxSubSteps)
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
            .method("get_simulation_time", &SimulationServer::GetSimulationTime)
            .method("get_frame_count", &SimulationServer::GetFrameCount)
            .method("get_last_error", &SimulationServer::GetLastError);

};
