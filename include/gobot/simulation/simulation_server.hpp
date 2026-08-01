/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/physics/physics_scene_compiler.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/simulation/external_simulation_driver.hpp"
#include "gobot/simulation/simulation_scene.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT SimulationServer : public Object {
    GOBCLASS(SimulationServer, Object)

public:
    using FixedStepCallback = std::function<void(RealType fixed_delta)>;

    explicit SimulationServer(PhysicsBackendType backend_type = PhysicsBackendType::Null,
                              bool register_singleton = true);

    ~SimulationServer() override;

    static SimulationServer* GetInstance();

    static bool HasInstance();

    PhysicsBackendType GetBackendType() const;

    void SetBackendType(PhysicsBackendType backend_type);

    const PhysicsWorldSettings& GetPhysicsWorldSettings() const;

    void SetPhysicsWorldSettings(const PhysicsWorldSettings& settings);

    const JointControllerGains& GetDefaultJointGains() const;

    void SetDefaultJointGains(const JointControllerGains& gains);

    RealType GetFixedTimeStep() const;

    void SetFixedTimeStep(RealType fixed_time_step);

    RealType GetTimeScale() const;

    void SetTimeScale(RealType time_scale);

    int GetMaxSubSteps() const;

    void SetMaxSubSteps(int max_sub_steps);

    bool IsPaused() const;

    void SetPaused(bool paused);

    bool IsFaulted() const;

    bool ShouldSyncSceneOnFixedStep() const;

    void SetSyncSceneOnFixedStep(bool sync_scene_on_fixed_step);

    bool BuildWorldFromScene(const Node* scene_root);

    bool RebuildWorldFromScene(const Node* scene_root, bool preserve_state = true);

    const Node* GetSceneRoot() const;

    void ClearWorld();

    bool HasWorld() const;

    bool HasExternalSession() const;

    bool HasActiveSession() const;

    std::uint64_t BeginExternalSession(Ref<ExternalSimulationDriver> driver,
                                       const Node* scene_root,
                                       RealType fixed_time_step,
                                       int max_sub_steps = 8);

    bool EndExternalSession(std::uint64_t session_token);

    bool ResetExternalSession(std::uint64_t session_token);

    bool SyncExternalSession(std::uint64_t session_token);

    Ref<PhysicsWorld> GetWorld() const;

    SimulationScene* GetRuntimeScene();

    const SimulationScene* GetRuntimeScene() const;

    bool Reset();

    bool StepOnce();

    bool StepOnce(const FixedStepCallback& fixed_step_callback);

    int Step(RealType delta_time);

    int Step(RealType delta_time, const FixedStepCallback& fixed_step_callback);

    bool ConfigureEnvironmentBatch(std::size_t environment_count);

    std::size_t GetEnvironmentCount() const;

    const PhysicsSceneState* GetEnvironmentState(std::size_t environment_index) const;

    bool ResetEnvironment(std::size_t environment_index);

    bool StepEnvironment(std::size_t environment_index, std::uint64_t ticks);

    bool StepEnvironmentBatch(std::uint64_t ticks, std::size_t worker_count = 0);

    std::size_t ResolveEnvironmentBatchWorkerCount(std::size_t worker_count = 0) const;

    bool SyncSceneFromWorld();

    RealType GetSimulationTime() const;

    std::uint64_t GetFrameCount() const;

    RealType GetAccumulator() const;

    int GetLastStepCount() const;

    const std::string& GetLastError() const;

private:
    struct FixedStepResult {
        bool succeeded{false};
        bool advanced{false};
        bool session_changed{false};
    };

    bool EnsureWorldReady();

    bool EnsureActiveSessionReady();

    bool IsExternalSessionCurrent(const Ref<ExternalSimulationDriver>& driver,
                                  std::uint64_t session_token) const;

    void ClearExternalSession();

    FixedStepResult StepFixed(const FixedStepCallback* fixed_step_callback = nullptr);

    bool ApplyWorldStateToScene();

    void ResetClock();

    void LatchFailure(const char* operation);

    void SetLastError(std::string error);

    static SimulationServer* s_singleton;

    PhysicsBackendType backend_type_{PhysicsBackendType::Null};
    bool registered_singleton_{false};
    PhysicsWorldSettings physics_world_settings_;
    Ref<PhysicsWorld> world_;
    Ref<ExternalSimulationDriver> external_driver_;
    ObjectID external_scene_root_id_{};
    std::uint64_t external_session_token_{0};
    std::uint64_t next_session_token_{1};
    bool external_session_transitioning_{false};
    RealType saved_fixed_time_step_{1.0 / 60.0};
    int saved_max_sub_steps_{8};
    bool external_timing_saved_{false};
    PhysicsSceneBindings scene_bindings_;
    SimulationScene runtime_scene_;
    bool paused_{true};
    bool faulted_{false};
    bool sync_scene_on_fixed_step_{true};
    RealType time_scale_{1.0};
    int max_sub_steps_{8};
    int last_step_count_{0};
    RealType accumulator_{0.0};
    RealType simulation_time_{0.0};
    std::uint64_t frame_count_{0};
    std::uint64_t session_clock_epoch_{1};
    std::string last_error_;
};

} // namespace gobot
