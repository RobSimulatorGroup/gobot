/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/simulation_server.hpp"

#include <chrono>
#include <cmath>
#include <utility>

#include "gobot/core/profile.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/rigid_body_3d.hpp"
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

struct ResolvedRobotSceneBinding {
    Robot3D* robot{nullptr};
    RigidBody3D* rigid_body{nullptr};
    std::vector<Link3D*> links;
    std::vector<Joint3D*> joints;
};

struct ResolvedSceneBindings {
    Node* scene_root{nullptr};
    std::vector<ResolvedRobotSceneBinding> robots;
};

bool IsOwnedBySceneRoot(const Node* scene_root, const Node* node) {
    return scene_root != nullptr && node != nullptr &&
           (scene_root == node || scene_root->IsAncestorOf(node));
}

template <typename NodeType>
NodeType* ResolveBoundNode(ObjectID object_id, Node* scene_root) {
    auto* node = Object::PointerCastTo<NodeType>(ObjectDB::GetInstance(object_id));
    return IsOwnedBySceneRoot(scene_root, node) ? node : nullptr;
}

bool ResolveSceneBindings(const PhysicsSceneBindings& bindings,
                          const PhysicsSceneSnapshot& snapshot,
                          ResolvedSceneBindings* resolved,
                          std::string* error) {
    *resolved = {};
    resolved->scene_root = Object::PointerCastTo<Node>(
            ObjectDB::GetInstance(bindings.scene_root_id));
    if (resolved->scene_root == nullptr) {
        *error = "Compiled physics scene root is no longer alive.";
        return false;
    }
    if (bindings.robots.size() != snapshot.robots.size()) {
        *error = "Physics scene robot bindings do not match the compiled snapshot.";
        return false;
    }

    resolved->robots.reserve(bindings.robots.size());
    for (std::size_t robot_index = 0; robot_index < bindings.robots.size(); ++robot_index) {
        const PhysicsRobotSceneBinding& binding = bindings.robots[robot_index];
        const PhysicsRobotSnapshot& robot_snapshot = snapshot.robots[robot_index];
        if (binding.link_ids.size() != robot_snapshot.links.size() ||
            binding.joint_ids.size() != robot_snapshot.joints.size()) {
            *error = fmt::format(
                    "Physics scene bindings for robot '{}' do not match the compiled snapshot.",
                    robot_snapshot.name);
            return false;
        }

        ResolvedRobotSceneBinding robot_binding;
        robot_binding.robot = ResolveBoundNode<Robot3D>(binding.robot_id, resolved->scene_root);
        robot_binding.rigid_body =
                ResolveBoundNode<RigidBody3D>(binding.robot_id, resolved->scene_root);
        if (robot_binding.robot == nullptr && robot_binding.rigid_body == nullptr) {
            *error = fmt::format(
                    "Physics scene rigid system '{}' is no longer alive inside the compiled scene root.",
                    robot_snapshot.name);
            return false;
        }

        robot_binding.links.reserve(binding.link_ids.size());
        for (std::size_t link_index = 0; link_index < binding.link_ids.size(); ++link_index) {
            Link3D* link = ResolveBoundNode<Link3D>(binding.link_ids[link_index], resolved->scene_root);
            if (link == nullptr) {
                *error = fmt::format(
                        "Physics scene link '{}.{}' is no longer alive inside the compiled scene root.",
                        robot_snapshot.name,
                        robot_snapshot.links[link_index].name);
                return false;
            }
            robot_binding.links.push_back(link);
        }

        robot_binding.joints.reserve(binding.joint_ids.size());
        for (std::size_t joint_index = 0; joint_index < binding.joint_ids.size(); ++joint_index) {
            Joint3D* joint = ResolveBoundNode<Joint3D>(binding.joint_ids[joint_index], resolved->scene_root);
            if (joint == nullptr) {
                *error = fmt::format(
                        "Physics scene joint '{}.{}' is no longer alive inside the compiled scene root.",
                        robot_snapshot.name,
                        robot_snapshot.joints[joint_index].name);
                return false;
            }
            robot_binding.joints.push_back(joint);
        }
        resolved->robots.push_back(std::move(robot_binding));
    }
    return true;
}

ResolvedRobotSceneBinding* FindRobotSceneBinding(ResolvedSceneBindings& bindings,
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

Joint3D* FindJointSceneNode(const ResolvedRobotSceneBinding& binding,
                           const PhysicsRobotSnapshot& snapshot,
                           const std::string& joint_name) {
    for (std::size_t index = 0; index < snapshot.joints.size() && index < binding.joints.size(); ++index) {
        if (snapshot.joints[index].name == joint_name) {
            return binding.joints[index];
        }
    }
    return nullptr;
}

Link3D* FindLinkSceneNode(const ResolvedRobotSceneBinding& binding,
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
    ClearWorld();
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
    if (!std::isfinite(settings.fixed_time_step) || settings.fixed_time_step <= 0.0) {
        SetLastError("Simulation fixed time step must be finite and greater than zero.");
        return;
    }
    if (external_driver_.IsValid() &&
        settings.fixed_time_step != physics_world_settings_.fixed_time_step) {
        SetLastError("Cannot change the fixed time step while an external simulation session is active.");
        return;
    }
    physics_world_settings_ = settings;
    if (world_.IsValid()) {
        world_->SetSettings(physics_world_settings_);
    }
    if (!faulted_) {
        last_error_.clear();
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
    if (!faulted_) {
        last_error_.clear();
    }
}

RealType SimulationServer::GetFixedTimeStep() const {
    return physics_world_settings_.fixed_time_step;
}

void SimulationServer::SetFixedTimeStep(RealType fixed_time_step) {
    if (!std::isfinite(fixed_time_step) || fixed_time_step <= 0.0) {
        SetLastError("Simulation fixed time step must be finite and greater than zero.");
        return;
    }
    if (external_driver_.IsValid()) {
        if (fixed_time_step == physics_world_settings_.fixed_time_step) {
            if (!faulted_) {
                last_error_.clear();
            }
            return;
        }
        SetLastError("Cannot change the fixed time step while an external simulation session is active.");
        return;
    }

    physics_world_settings_.fixed_time_step = fixed_time_step;
    if (world_.IsValid()) {
        world_->SetSettings(physics_world_settings_);
    }
    if (!faulted_) {
        last_error_.clear();
    }
}

RealType SimulationServer::GetTimeScale() const {
    return time_scale_;
}

void SimulationServer::SetTimeScale(RealType time_scale) {
    if (!std::isfinite(time_scale) || time_scale < 0.0) {
        SetLastError("Simulation time scale must be finite and non-negative.");
        return;
    }

    time_scale_ = time_scale;
    if (!faulted_) {
        last_error_.clear();
    }
}

int SimulationServer::GetMaxSubSteps() const {
    return max_sub_steps_;
}

void SimulationServer::SetMaxSubSteps(int max_sub_steps) {
    if (max_sub_steps <= 0) {
        SetLastError("Simulation max sub-steps must be greater than zero.");
        return;
    }
    if (external_driver_.IsValid()) {
        if (max_sub_steps == max_sub_steps_) {
            if (!faulted_) {
                last_error_.clear();
            }
            return;
        }
        SetLastError("Cannot change max sub-steps while an external simulation session is active.");
        return;
    }

    max_sub_steps_ = max_sub_steps;
    if (!faulted_) {
        last_error_.clear();
    }
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

bool SimulationServer::IsFaulted() const {
    return faulted_;
}

bool SimulationServer::ShouldSyncSceneOnFixedStep() const {
    return sync_scene_on_fixed_step_;
}

void SimulationServer::SetSyncSceneOnFixedStep(bool sync_scene_on_fixed_step) {
    sync_scene_on_fixed_step_ = sync_scene_on_fixed_step;
}

bool SimulationServer::BuildWorldFromScene(const Node* scene_root) {
    const ObjectID requested_scene_root_id =
            scene_root != nullptr ? scene_root->GetInstanceId() : ObjectID{};
    ClearExternalSession();
    scene_root = Object::PointerCastTo<Node>(ObjectDB::GetInstance(requested_scene_root_id));
    if (requested_scene_root_id != ObjectID{} && scene_root == nullptr) {
        SetLastError("Physics scene root was deleted while closing the previous simulation session.");
        return false;
    }
    runtime_scene_.Clear();
    scene_bindings_ = {};
    world_ = PhysicsServer::CreateWorld(backend_type_, physics_world_settings_);
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
    const ObjectID requested_scene_root_id =
            scene_root != nullptr ? scene_root->GetInstanceId() : ObjectID{};
    ClearExternalSession();
    scene_root = Object::PointerCastTo<Node>(ObjectDB::GetInstance(requested_scene_root_id));
    if (requested_scene_root_id != ObjectID{} && scene_root == nullptr) {
        SetLastError("Physics scene root was deleted while closing the previous simulation session.");
        return false;
    }
    PhysicsSceneState previous_state;
    if (preserve_state && world_.IsValid()) {
        previous_state = world_->GetSceneState();
    }

    runtime_scene_.Clear();
    scene_bindings_ = {};
    world_ = PhysicsServer::CreateWorld(backend_type_, physics_world_settings_);
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
    if (!ApplyWorldStateToScene()) {
        return false;
    }
    last_error_.clear();
    return true;
}

const Node* SimulationServer::GetSceneRoot() const {
    if (external_driver_.IsValid()) {
        return Object::PointerCastTo<Node>(ObjectDB::GetInstance(external_scene_root_id_));
    }
    return Object::PointerCastTo<Node>(ObjectDB::GetInstance(scene_bindings_.scene_root_id));
}

void SimulationServer::ClearWorld() {
    ClearExternalSession();
    runtime_scene_.Clear();
    scene_bindings_ = {};
    world_.Reset();
    ResetClock();
}

bool SimulationServer::HasWorld() const {
    return world_.IsValid();
}

bool SimulationServer::HasExternalSession() const {
    return external_driver_.IsValid();
}

bool SimulationServer::HasActiveSession() const {
    return world_.IsValid() || external_driver_.IsValid();
}

std::uint64_t SimulationServer::BeginExternalSession(Ref<ExternalSimulationDriver> driver,
                                                     const Node* scene_root,
                                                     RealType fixed_time_step,
                                                     int max_sub_steps) {
    if (external_session_transitioning_) {
        SetLastError("Cannot begin an external simulation session while another session is closing.");
        return 0;
    }
    if (!driver.IsValid()) {
        SetLastError("External simulation session requires a driver.");
        return 0;
    }
    if (scene_root == nullptr) {
        SetLastError("External simulation session requires a scene root.");
        return 0;
    }
    const ObjectID requested_scene_root_id = scene_root->GetInstanceId();
    if (!std::isfinite(fixed_time_step) || fixed_time_step <= 0.0 || max_sub_steps <= 0) {
        SetLastError(
                "External simulation timing must use a finite positive fixed time step and positive max sub-steps.");
        return 0;
    }

    ClearWorld();
    scene_root = Object::PointerCastTo<Node>(ObjectDB::GetInstance(requested_scene_root_id));
    if (scene_root == nullptr) {
        SetLastError(
                "External simulation scene root was deleted while closing the previous session.");
        return 0;
    }
    saved_fixed_time_step_ = physics_world_settings_.fixed_time_step;
    saved_max_sub_steps_ = max_sub_steps_;
    external_timing_saved_ = true;
    physics_world_settings_.fixed_time_step = fixed_time_step;
    max_sub_steps_ = max_sub_steps;
    external_driver_ = std::move(driver);
    external_scene_root_id_ = requested_scene_root_id;
    external_session_token_ = next_session_token_++;
    if (external_session_token_ == 0) {
        external_session_token_ = next_session_token_++;
    }
    ResetClock();
    last_error_.clear();
    return external_session_token_;
}

bool SimulationServer::EndExternalSession(std::uint64_t session_token) {
    if (!external_driver_.IsValid() || session_token == 0 || session_token != external_session_token_) {
        SetLastError("External simulation session token is stale.");
        return false;
    }
    ClearExternalSession();
    ResetClock();
    last_error_.clear();
    return true;
}

bool SimulationServer::ResetExternalSession(std::uint64_t session_token) {
    if (!external_driver_.IsValid() || session_token == 0 || session_token != external_session_token_) {
        SetLastError("External simulation session token is stale.");
        return false;
    }
    Ref<ExternalSimulationDriver> driver = external_driver_;
    const bool reset_succeeded = driver->Reset();
    if (!IsExternalSessionCurrent(driver, session_token)) {
        SetLastError("Simulation session changed during an external reset callback.");
        return false;
    }
    if (!reset_succeeded) {
        SetLastError(driver->GetLastError());
        LatchFailure("external reset");
        return false;
    }
    ResetClock();
    const bool sync_succeeded = driver->SyncScene();
    if (!IsExternalSessionCurrent(driver, session_token)) {
        SetLastError("Simulation session changed during an external scene sync callback.");
        return false;
    }
    if (!sync_succeeded) {
        SetLastError(driver->GetLastError());
        LatchFailure("external reset synchronization");
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationServer::SyncExternalSession(std::uint64_t session_token) {
    if (!external_driver_.IsValid() || session_token == 0 || session_token != external_session_token_) {
        SetLastError("External simulation session token is stale.");
        return false;
    }
    return SyncSceneFromWorld();
}

bool SimulationServer::SetExternalSessionDiagnostics(
        std::uint64_t session_token,
        ExternalSessionDiagnostics diagnostics) {
    if (!external_driver_.IsValid() || session_token == 0 ||
        session_token != external_session_token_) {
        SetLastError("External simulation session token is stale.");
        return false;
    }
    diagnostics.last_step_latency_ms = external_diagnostics_.last_step_latency_ms;
    diagnostics.average_step_latency_ms = external_diagnostics_.average_step_latency_ms;
    external_diagnostics_ = std::move(diagnostics);
    last_error_.clear();
    return true;
}

const ExternalSessionDiagnostics& SimulationServer::GetExternalSessionDiagnostics() const {
    return external_diagnostics_;
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
    if (!EnsureActiveSessionReady()) {
        return false;
    }

    if (external_driver_.IsValid()) {
        return ResetExternalSession(external_session_token_);
    }

    world_->Reset();
    ResetClock();
    if (!ApplyWorldStateToScene()) {
        LatchFailure("reset synchronization");
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationServer::StepOnce() {
    return StepOnce(FixedStepCallback{});
}

bool SimulationServer::StepOnce(const FixedStepCallback& fixed_step_callback) {
    if (faulted_) {
        last_step_count_ = 0;
        return false;
    }
    const std::uint64_t step_epoch = session_clock_epoch_;
    const FixedStepResult result = StepFixed(
            fixed_step_callback ? &fixed_step_callback : nullptr);
    if (result.session_changed || session_clock_epoch_ != step_epoch) {
        if (session_clock_epoch_ == step_epoch) {
            ResetClock();
        }
        return false;
    }

    if (result.advanced) {
        accumulator_ = 0.0;
        last_step_count_ = 1;
    } else {
        last_step_count_ = 0;
    }
    if (!result.succeeded) {
        LatchFailure("fixed step");
        return false;
    }

    last_error_.clear();
    return true;
}

int SimulationServer::Step(RealType delta_time) {
    return Step(delta_time, FixedStepCallback{});
}

int SimulationServer::Step(RealType delta_time, const FixedStepCallback& fixed_step_callback) {
    GOBOT_PROFILE_ZONE("SimulationServer::Step");
    if (!std::isfinite(delta_time)) {
        SetLastError("Simulation frame delta must be finite.");
        last_step_count_ = 0;
        LatchFailure("frame step");
        return 0;
    }
    if (paused_ || delta_time <= 0.0 || time_scale_ <= 0.0) {
        last_step_count_ = 0;
        return 0;
    }

    if (faulted_) {
        paused_ = true;
        last_step_count_ = 0;
        return 0;
    }

    if (!EnsureActiveSessionReady()) {
        last_step_count_ = 0;
        LatchFailure("fixed step");
        return 0;
    }

    const std::uint64_t step_epoch = session_clock_epoch_;
    accumulator_ += delta_time * time_scale_;

    int steps = 0;
    while (accumulator_ + CMP_EPSILON >= physics_world_settings_.fixed_time_step && steps < max_sub_steps_) {
        const RealType fixed_delta = physics_world_settings_.fixed_time_step;
        const FixedStepResult result = StepFixed(
                fixed_step_callback ? &fixed_step_callback : nullptr);
        if (result.session_changed || session_clock_epoch_ != step_epoch) {
            if (session_clock_epoch_ == step_epoch) {
                ResetClock();
            }
            return 0;
        }
        if (result.advanced) {
            accumulator_ -= fixed_delta;
            ++steps;
        }
        if (!result.succeeded) {
            last_step_count_ = steps;
            LatchFailure("fixed step");
            return steps;
        }
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
    if (!EnsureActiveSessionReady()) {
        return false;
    }

    if (external_driver_.IsValid()) {
        Ref<ExternalSimulationDriver> driver = external_driver_;
        const std::uint64_t session_token = external_session_token_;
        const bool sync_succeeded = driver->SyncScene();
        if (!IsExternalSessionCurrent(driver, session_token)) {
            SetLastError("Simulation session changed during an external scene sync callback.");
            return false;
        }
        if (!sync_succeeded) {
            SetLastError(driver->GetLastError());
            LatchFailure("scene synchronization");
            return false;
        }
        last_error_.clear();
        return true;
    }

    if (!ApplyWorldStateToScene()) {
        LatchFailure("scene synchronization");
        return false;
    }
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

bool SimulationServer::EnsureActiveSessionReady() {
    if (external_driver_.IsValid()) {
        if (ObjectDB::GetInstance(external_scene_root_id_) == nullptr) {
            SetLastError("External simulation scene is no longer alive.");
            return false;
        }
        return true;
    }
    return EnsureWorldReady();
}

bool SimulationServer::IsExternalSessionCurrent(
        const Ref<ExternalSimulationDriver>& driver,
        std::uint64_t session_token) const {
    return driver.IsValid() && external_driver_ == driver && session_token != 0 &&
           external_session_token_ == session_token && !world_.IsValid();
}

void SimulationServer::ClearExternalSession() {
    Ref<ExternalSimulationDriver> driver = std::move(external_driver_);
    external_scene_root_id_ = ObjectID{};
    external_session_token_ = 0;
    external_diagnostics_ = {};
    external_step_latency_sum_ms_ = 0.0;
    external_step_latency_count_ = 0;
    if (external_timing_saved_) {
        physics_world_settings_.fixed_time_step = saved_fixed_time_step_;
        max_sub_steps_ = saved_max_sub_steps_;
        external_timing_saved_ = false;
    }
    if (driver.IsValid()) {
        external_session_transitioning_ = true;
        try {
            driver->Close();
        } catch (const std::exception& error) {
            SetLastError(fmt::format("External simulation close failed: {}", error.what()));
            LOG_ERROR("{}", last_error_);
        } catch (...) {
            SetLastError("External simulation close failed with an unknown error.");
            LOG_ERROR("{}", last_error_);
        }
        external_session_transitioning_ = false;
    }
}

SimulationServer::FixedStepResult SimulationServer::StepFixed(
        const FixedStepCallback* fixed_step_callback) {
    GOBOT_PROFILE_ZONE("SimulationServer::StepFixed");
    if (!EnsureActiveSessionReady()) {
        return {};
    }

    const RealType fixed_delta = physics_world_settings_.fixed_time_step;
    const Ref<ExternalSimulationDriver> active_external_driver = external_driver_;
    const Ref<PhysicsWorld> active_world = world_;
    const std::uint64_t active_external_token = external_session_token_;
    const std::uint64_t active_epoch = session_clock_epoch_;
    const auto session_is_current = [&]() {
        if (session_clock_epoch_ != active_epoch) {
            return false;
        }
        if (active_external_driver.IsValid()) {
            return IsExternalSessionCurrent(active_external_driver, active_external_token);
        }
        return active_world.IsValid() && world_ == active_world && !external_driver_.IsValid();
    };
    const auto fail_if_session_changed = [&]() {
        if (session_is_current()) {
            return false;
        }
        SetLastError("Simulation session changed during a fixed step; the tick was aborted.");
        return true;
    };

    if (fixed_step_callback != nullptr) {
        GOBOT_PROFILE_ZONE("SimulationServer::FixedStepCallback");
        (*fixed_step_callback)(fixed_delta);
        if (fail_if_session_changed()) {
            return {.session_changed = true};
        }
    }

    bool advanced = false;
    if (active_external_driver.IsValid()) {
        GOBOT_PROFILE_ZONE("SimulationServer::ExternalDriverStep");
        const auto step_started_at = std::chrono::steady_clock::now();
        const bool step_succeeded = active_external_driver->Step(fixed_delta);
        const double step_latency_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - step_started_at)
                                               .count();
        external_diagnostics_.last_step_latency_ms = step_latency_ms;
        external_step_latency_sum_ms_ += step_latency_ms;
        ++external_step_latency_count_;
        external_diagnostics_.average_step_latency_ms =
                external_step_latency_sum_ms_ /
                static_cast<double>(external_step_latency_count_);
        advanced = step_succeeded;
        if (fail_if_session_changed()) {
            return {.advanced = advanced, .session_changed = true};
        }
        if (!step_succeeded) {
            SetLastError(active_external_driver->GetLastError());
            return {};
        }
    } else {
        GOBOT_PROFILE_ZONE("SimulationServer::WorldStep");
        active_world->Step(fixed_delta);
        advanced = true;
        if (fail_if_session_changed()) {
            return {.advanced = true, .session_changed = true};
        }
    }
    simulation_time_ += fixed_delta;
    ++frame_count_;
    if (sync_scene_on_fixed_step_) {
        GOBOT_PROFILE_ZONE("SimulationServer::ApplyWorldStateToScene");
        if (active_external_driver.IsValid()) {
            const bool sync_succeeded = active_external_driver->SyncScene();
            if (fail_if_session_changed()) {
                return {.advanced = true, .session_changed = true};
            }
            if (!sync_succeeded) {
                SetLastError(active_external_driver->GetLastError());
                return {.advanced = true};
            }
        } else {
            const bool sync_succeeded = ApplyWorldStateToScene();
            if (fail_if_session_changed()) {
                return {.advanced = true, .session_changed = true};
            }
            if (!sync_succeeded) {
                return {.advanced = true};
            }
        }
    }
    return {.succeeded = true, .advanced = true};
}

bool SimulationServer::ApplyWorldStateToScene() {
    GOBOT_PROFILE_ZONE("SimulationServer::ApplyWorldStateToScene");
    if (!world_.IsValid()) {
        return false;
    }

    const PhysicsSceneState& scene_state = world_->GetSceneState();
    const PhysicsSceneSnapshot& scene_snapshot = world_->GetSceneSnapshot();
    ResolvedSceneBindings resolved_bindings;
    std::string binding_error;
    if (!ResolveSceneBindings(scene_bindings_, scene_snapshot, &resolved_bindings, &binding_error)) {
        SetLastError(std::move(binding_error));
        return false;
    }

    for (std::size_t robot_index = 0; robot_index < scene_state.robots.size(); ++robot_index) {
        const PhysicsRobotState& robot_state = scene_state.robots[robot_index];
        const PhysicsRobotSnapshot* robot_snapshot =
                FindRobotSnapshot(scene_snapshot, robot_index, robot_state.name);
        ResolvedRobotSceneBinding* scene_binding =
                FindRobotSceneBinding(resolved_bindings, scene_snapshot, robot_index, robot_state.name);
        Robot3D* robot = scene_binding != nullptr ? scene_binding->robot : nullptr;
        RigidBody3D* rigid_body =
                scene_binding != nullptr ? scene_binding->rigid_body : nullptr;
        if ((robot == nullptr && rigid_body == nullptr) ||
            (robot != nullptr && robot->GetMode() != RobotMode::Motion)) {
            continue;
        }

        if (rigid_body != nullptr) {
            const PhysicsLinkState* body_state =
                    FindLinkState(robot_state, rigid_body->GetName());
            if (body_state != nullptr) {
                ApplyLinkGlobalTransform(rigid_body, body_state->global_transform);
            }
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

                Joint3D* joint =
                        FindJointSceneNode(*scene_binding, *robot_snapshot, joint_snapshot.name);
                Link3D* floating_link =
                        FindLinkSceneNode(*scene_binding, *robot_snapshot, floating_link_state->link_name);
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
            Joint3D* joint = robot_snapshot != nullptr && scene_binding != nullptr
                                     ? FindJointSceneNode(
                                               *scene_binding, *robot_snapshot, joint_state.joint_name)
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

            Link3D* link = robot_snapshot != nullptr && scene_binding != nullptr
                                   ? FindLinkSceneNode(
                                             *scene_binding, *robot_snapshot, link_state.link_name)
                                   : nullptr;
            if (link != nullptr) {
                ApplyLinkGlobalTransform(link, link_state.global_transform);
            }
        }
    }

    last_error_.clear();
    return true;
}

void SimulationServer::ResetClock() {
    ++session_clock_epoch_;
    if (session_clock_epoch_ == 0) {
        ++session_clock_epoch_;
    }
    accumulator_ = 0.0;
    simulation_time_ = 0.0;
    frame_count_ = 0;
    last_step_count_ = 0;
    faulted_ = false;
}

void SimulationServer::LatchFailure(const char* operation) {
    if (last_error_.empty()) {
        SetLastError("Simulation fixed step failed.");
    }
    if (!faulted_) {
        LOG_ERROR("Simulation paused after {} failure: {}", operation, last_error_);
    }
    faulted_ = true;
    paused_ = true;
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
