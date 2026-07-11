/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/simulation_server.hpp"

#include <utility>

#include "gobot/core/profile.hpp"
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

const PhysicsLinkSnapshot* FindLinkSnapshot(const PhysicsRobotSnapshot& robot_snapshot,
                                            const std::string& link_name) {
    for (const PhysicsLinkSnapshot& link_snapshot : robot_snapshot.links) {
        if (link_snapshot.name == link_name) {
            return &link_snapshot;
        }
    }

    return nullptr;
}

const PhysicsRobotSceneBinding* FindRobotSceneBinding(const PhysicsSceneBindings& bindings,
                                                      const PhysicsSceneSnapshot& snapshot,
                                                      std::size_t robot_index,
                                                      const std::string& robot_name) {
    if (robot_index < bindings.robots.size() &&
        robot_index < snapshot.robots.size() &&
        snapshot.robots[robot_index].name == robot_name) {
        return &bindings.robots[robot_index];
    }

    for (std::size_t index = 0; index < snapshot.robots.size() && index < bindings.robots.size(); ++index) {
        if (snapshot.robots[index].name == robot_name) {
            return &bindings.robots[index];
        }
    }
    return nullptr;
}

const Joint3D* FindJointSceneNode(const PhysicsRobotSceneBinding& binding,
                                 const PhysicsRobotSnapshot& snapshot,
                                 const std::string& joint_name) {
    for (std::size_t index = 0; index < snapshot.joints.size() && index < binding.joints.size(); ++index) {
        if (snapshot.joints[index].name == joint_name) {
            return binding.joints[index];
        }
    }
    return nullptr;
}

const Link3D* FindLinkSceneNode(const PhysicsRobotSceneBinding& binding,
                               const PhysicsRobotSnapshot& snapshot,
                               const std::string& link_name) {
    for (std::size_t index = 0; index < snapshot.links.size() && index < binding.links.size(); ++index) {
        if (snapshot.links[index].name == link_name) {
            return binding.links[index];
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

SimulationServer::SimulationServer(PhysicsBackendType backend_type, bool register_singleton)
    : backend_type_(backend_type),
      registered_singleton_(register_singleton) {
    if (registered_singleton_) {
        s_singleton = this;
    }
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
    last_error_.clear();
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

bool SimulationServer::ShouldSyncSceneOnFixedStep() const {
    return sync_scene_on_fixed_step_;
}

void SimulationServer::SetSyncSceneOnFixedStep(bool sync_scene_on_fixed_step) {
    sync_scene_on_fixed_step_ = sync_scene_on_fixed_step;
}

bool SimulationServer::BuildWorldFromScene(const Node* scene_root) {
    runtime_scene_.Clear();
    scene_bindings_ = {};
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

    CompiledPhysicsScene compiled_scene;
    std::string compile_error;
    if (!PhysicsSceneCompiler::Compile(scene_root, &compiled_scene, &compile_error)) {
        SetLastError(std::move(compile_error));
        world_.Reset();
        return false;
    }
    if (!world_->Build(std::move(compiled_scene.snapshot))) {
        SetLastError(world_->GetLastError());
        world_.Reset();
        return false;
    }
    scene_bindings_ = std::move(compiled_scene.bindings);

    if (!runtime_scene_.Initialize(world_, scene_root)) {
        SetLastError(runtime_scene_.GetLastError());
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

    runtime_scene_.Clear();
    scene_bindings_ = {};
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

    CompiledPhysicsScene compiled_scene;
    std::string compile_error;
    if (!PhysicsSceneCompiler::Compile(scene_root, &compiled_scene, &compile_error)) {
        SetLastError(std::move(compile_error));
        world_.Reset();
        return false;
    }
    if (!world_->Build(std::move(compiled_scene.snapshot))) {
        SetLastError(world_->GetLastError());
        world_.Reset();
        return false;
    }
    scene_bindings_ = std::move(compiled_scene.bindings);

    if (preserve_state && !world_->RestoreCompatibleState(previous_state)) {
        SetLastError(world_->GetLastError());
        world_.Reset();
        return false;
    }

    if (!runtime_scene_.Initialize(world_, scene_root)) {
        SetLastError(runtime_scene_.GetLastError());
        world_.Reset();
        return false;
    }

    ResetClock();
    ApplyWorldStateToScene();
    last_error_.clear();
    return true;
}

const Node* SimulationServer::GetSceneRoot() const {
    return runtime_scene_.GetSceneRoot();
}

void SimulationServer::ClearWorld() {
    runtime_scene_.Clear();
    scene_bindings_ = {};
    world_.Reset();
    ResetClock();
}

bool SimulationServer::HasWorld() const {
    return world_.IsValid();
}

Ref<PhysicsWorld> SimulationServer::GetWorld() const {
    return world_;
}

SimulationScene* SimulationServer::GetRuntimeScene() {
    return runtime_scene_.IsValid() ? &runtime_scene_ : nullptr;
}

const SimulationScene* SimulationServer::GetRuntimeScene() const {
    return runtime_scene_.IsValid() ? &runtime_scene_ : nullptr;
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
    return StepOnce(FixedStepCallback{});
}

bool SimulationServer::StepOnce(const FixedStepCallback& fixed_step_callback) {
    if (!StepFixed(fixed_step_callback ? &fixed_step_callback : nullptr)) {
        last_step_count_ = 0;
        return false;
    }

    accumulator_ = 0.0;
    last_step_count_ = 1;
    last_error_.clear();
    return true;
}

int SimulationServer::Step(RealType delta_time) {
    return Step(delta_time, FixedStepCallback{});
}

int SimulationServer::Step(RealType delta_time, const FixedStepCallback& fixed_step_callback) {
    GOBOT_PROFILE_ZONE("SimulationServer::Step");
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
        if (!StepFixed(fixed_step_callback ? &fixed_step_callback : nullptr)) {
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
    GOBOT_PROFILE_PLOT("physics_steps_per_frame", steps);
    return steps;
}

bool SimulationServer::ConfigureEnvironmentBatch(std::size_t environment_count) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!world_->ConfigureEnvironmentBatch(environment_count)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

std::size_t SimulationServer::GetEnvironmentCount() const {
    if (!world_.IsValid()) {
        return 0;
    }

    return world_->GetEnvironmentCount();
}

const PhysicsSceneState* SimulationServer::GetEnvironmentState(std::size_t environment_index) const {
    if (!world_.IsValid()) {
        return nullptr;
    }

    return world_->GetEnvironmentState(environment_index);
}

bool SimulationServer::ResetEnvironment(std::size_t environment_index) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!runtime_scene_.ResetEnvironment(environment_index)) {
        SetLastError(runtime_scene_.GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::StepEnvironment(std::size_t environment_index, std::uint64_t ticks) {
    if (!EnsureWorldReady()) {
        return false;
    }

    for (std::uint64_t tick = 0; tick < ticks; ++tick) {
        if (!runtime_scene_.StepEnvironment(environment_index, physics_world_settings_.fixed_time_step)) {
            SetLastError(runtime_scene_.GetLastError());
            return false;
        }
    }

    last_error_.clear();
    return true;
}

bool SimulationServer::StepEnvironmentBatch(std::uint64_t ticks, std::size_t worker_count) {
    if (!EnsureWorldReady()) {
        return false;
    }

    if (!runtime_scene_.StepEnvironmentBatch(physics_world_settings_.fixed_time_step, ticks, worker_count)) {
        SetLastError(runtime_scene_.GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

std::size_t SimulationServer::ResolveEnvironmentBatchWorkerCount(std::size_t worker_count) const {
    if (!world_.IsValid()) {
        return 0;
    }

    return world_->ResolveEnvironmentBatchWorkerCount(worker_count);
}

bool SimulationServer::SyncSceneFromWorld() {
    if (!EnsureWorldReady()) {
        return false;
    }

    return ApplyWorldStateToScene();
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

bool SimulationServer::StepFixed(const FixedStepCallback* fixed_step_callback) {
    GOBOT_PROFILE_ZONE("SimulationServer::StepFixed");
    if (!EnsureWorldReady()) {
        return false;
    }

    if (fixed_step_callback != nullptr) {
        GOBOT_PROFILE_ZONE("SimulationServer::FixedStepCallback");
        (*fixed_step_callback)(physics_world_settings_.fixed_time_step);
    }

    {
        GOBOT_PROFILE_ZONE("SimulationServer::WorldStep");
        world_->Step(physics_world_settings_.fixed_time_step);
    }
    simulation_time_ += physics_world_settings_.fixed_time_step;
    ++frame_count_;
    if (sync_scene_on_fixed_step_) {
        GOBOT_PROFILE_ZONE("SimulationServer::ApplyWorldStateToScene");
        ApplyWorldStateToScene();
    }
    return true;
}

bool SimulationServer::ApplyWorldStateToScene() {
    GOBOT_PROFILE_ZONE("SimulationServer::ApplyWorldStateToScene");
    if (!world_.IsValid()) {
        return false;
    }

    const PhysicsSceneState& scene_state = world_->GetSceneState();
    const PhysicsSceneSnapshot& scene_snapshot = world_->GetSceneSnapshot();
    for (std::size_t robot_index = 0; robot_index < scene_state.robots.size(); ++robot_index) {
        const PhysicsRobotState& robot_state = scene_state.robots[robot_index];
        const PhysicsRobotSnapshot* robot_snapshot =
                FindRobotSnapshot(scene_snapshot, robot_index, robot_state.name);
        const PhysicsRobotSceneBinding* scene_binding =
                FindRobotSceneBinding(scene_bindings_, scene_snapshot, robot_index, robot_state.name);
        auto* robot = scene_binding != nullptr ? const_cast<Robot3D*>(scene_binding->robot) : nullptr;
        if (!robot || robot->GetMode() != RobotMode::Motion) {
            continue;
        }

        std::string floating_base_link;
        if (robot_snapshot != nullptr && scene_binding != nullptr) {
            for (const PhysicsJointSnapshot& joint_snapshot : robot_snapshot->joints) {
                if (static_cast<JointType>(joint_snapshot.joint_type) != JointType::Floating) {
                    continue;
                }
                floating_base_link = joint_snapshot.child_link;

                const PhysicsLinkState* floating_link_state =
                        FindLinkState(robot_state, joint_snapshot.child_link);
                if (floating_link_state == nullptr) {
                    continue;
                }

                auto* joint = const_cast<Joint3D*>(
                        FindJointSceneNode(*scene_binding, *robot_snapshot, joint_snapshot.name));
                auto* floating_link = const_cast<Link3D*>(
                        FindLinkSceneNode(*scene_binding, *robot_snapshot, floating_link_state->link_name));
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
            auto* joint = robot_snapshot != nullptr && scene_binding != nullptr
                                  ? const_cast<Joint3D*>(FindJointSceneNode(
                                            *scene_binding, *robot_snapshot, joint_state.joint_name))
                                  : nullptr;
            if (joint && joint->IsMotionModeEnabled() && joint->GetJointType() != JointType::Floating) {
                joint->SetJointPosition(joint_state.position);
            }
        }

        for (const PhysicsLinkState& link_state : robot_state.links) {
            if (link_state.role == PhysicsLinkRole::VirtualRoot) {
                continue;
            }
            if (!floating_base_link.empty() && link_state.link_name == floating_base_link) {
                continue;
            }

            auto* link = robot_snapshot != nullptr && scene_binding != nullptr
                                 ? const_cast<Link3D*>(FindLinkSceneNode(
                                           *scene_binding, *robot_snapshot, link_state.link_name))
                                 : nullptr;
            if (link != nullptr) {
                ApplyLinkGlobalTransform(link, link_state.global_transform);
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
            .method("step_once", static_cast<bool (SimulationServer::*)()>(&SimulationServer::StepOnce))
            .method("step", static_cast<int (SimulationServer::*)(RealType)>(&SimulationServer::Step))
            .method("configure_environment_batch", &SimulationServer::ConfigureEnvironmentBatch)
            .method("get_environment_count", &SimulationServer::GetEnvironmentCount)
            .method("reset_environment", &SimulationServer::ResetEnvironment)
            .method("step_environment", &SimulationServer::StepEnvironment)
            .method("sync_scene_from_world", &SimulationServer::SyncSceneFromWorld)
            .method("get_simulation_time", &SimulationServer::GetSimulationTime)
            .method("get_frame_count", &SimulationServer::GetFrameCount)
            .method("get_last_error", &SimulationServer::GetLastError);

};
