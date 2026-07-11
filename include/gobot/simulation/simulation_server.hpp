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

    bool ShouldSyncSceneOnFixedStep() const;

    void SetSyncSceneOnFixedStep(bool sync_scene_on_fixed_step);

    bool BuildWorldFromScene(const Node* scene_root);

    bool RebuildWorldFromScene(const Node* scene_root, bool preserve_state = true);

    const Node* GetSceneRoot() const;

    void ClearWorld();

    bool HasWorld() const;

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
    bool EnsureWorldReady();

    bool StepFixed(const FixedStepCallback* fixed_step_callback = nullptr);

    bool ApplyWorldStateToScene();

    void ResetClock();

    void SetLastError(std::string error);

    static SimulationServer* s_singleton;

    PhysicsBackendType backend_type_{PhysicsBackendType::Null};
    bool registered_singleton_{false};
    PhysicsWorldSettings physics_world_settings_;
    Ref<PhysicsWorld> world_;
    PhysicsSceneBindings scene_bindings_;
    SimulationScene runtime_scene_;
    bool paused_{true};
    bool sync_scene_on_fixed_step_{true};
    RealType time_scale_{1.0};
    int max_sub_steps_{8};
    int last_step_count_{0};
    RealType accumulator_{0.0};
    RealType simulation_time_{0.0};
    std::uint64_t frame_count_{0};
    std::string last_error_;
};

} // namespace gobot
