/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gobot/physics/physics_world.hpp"
#include "gobot/physics/joint_controller.hpp"

namespace gobot {

class GOBOT_EXPORT MuJoCoPhysicsWorld : public PhysicsWorld {
    GOBCLASS(MuJoCoPhysicsWorld, PhysicsWorld)

public:
    struct Diagnostics {
        RealType timestep{0.0};
        PhysicsSolverType solver{PhysicsSolverType::ProjectedGaussSeidel};
        PhysicsIntegratorType integrator{PhysicsIntegratorType::Euler};
        PhysicsFrictionConeType cone{PhysicsFrictionConeType::Pyramidal};
        PhysicsJacobianType jacobian{PhysicsJacobianType::Dense};
        int iterations{0};
        int line_search_iterations{0};
        int no_slip_iterations{0};
        int convex_collision_iterations{0};
        RealType tolerance{0.0};
        RealType line_search_tolerance{0.0};
        RealType no_slip_tolerance{0.0};
        RealType convex_collision_tolerance{0.0};
        RealType impedance_ratio{0.0};
        int actuator_count{0};
        RealType first_position_actuator_stiffness{0.0};
        int first_position_actuator_control_limited{0};
        Vector2 first_position_actuator_control_range{0.0, 0.0};
        int first_position_actuator_force_limited{0};
        Vector2 first_position_actuator_force_range{0.0, 0.0};
        RealType first_position_actuator_control_value{0.0};
        RealType first_controllable_joint_damping{0.0};
        Vector3 first_collision_friction{0.0, 0.0, 0.0};
        int first_collision_contact_dimension{0};
        Vector2 first_collision_solref{0.0, 0.0};
        std::vector<RealType> first_collision_solimp;
    };

    MuJoCoPhysicsWorld();

    ~MuJoCoPhysicsWorld() override;

    static bool IsBackendAvailable();

    static std::string GetUnavailableReason();

    PhysicsBackendType GetBackendType() const override;

    bool IsAvailable() const override;

    const std::string& GetLastError() const override;

    const PhysicsSceneArtifact* GetSceneArtifact() const override;

    bool Build(PhysicsSceneSnapshot scene_snapshot) override;

    bool RestoreCompatibleState(const PhysicsSceneState& previous_state) override;

    void Reset() override;

    void Step(RealType delta_time) override;

    bool ConfigureEnvironmentBatch(std::size_t environment_count) override;

    std::size_t GetEnvironmentCount() const override;

    const PhysicsSceneState* GetEnvironmentState(std::size_t environment_index) const override;

    bool ResetEnvironment(std::size_t environment_index) override;

    bool StepEnvironment(std::size_t environment_index, RealType delta_time) override;

    bool StepEnvironmentBatch(RealType delta_time,
                              std::uint64_t ticks = 1,
                              std::size_t worker_count = 0) override;

    std::size_t ResolveEnvironmentBatchWorkerCount(std::size_t worker_count) const override;

    bool ResetJointState(const std::string& robot_name,
                         const std::string& joint_name,
                         RealType position,
                         RealType velocity = 0.0) override;

    bool ResetEnvironmentJointState(std::size_t environment_index,
                                    const std::string& robot_name,
                                    const std::string& joint_name,
                                    RealType position,
                                    RealType velocity = 0.0) override;

    bool ResetLinkState(const std::string& robot_name,
                        const std::string& link_name,
                        const Vector3& position,
                        const Quaternion& orientation = Quaternion::Identity(),
                        const Vector3& linear_velocity = Vector3::Zero(),
                        const Vector3& angular_velocity = Vector3::Zero()) override;

    bool ResetEnvironmentLinkState(std::size_t environment_index,
                                   const std::string& robot_name,
                                   const std::string& link_name,
                                   const Vector3& position,
                                   const Quaternion& orientation = Quaternion::Identity(),
                                   const Vector3& linear_velocity = Vector3::Zero(),
                                   const Vector3& angular_velocity = Vector3::Zero()) override;

    bool WriteEnvironmentLinkVelocity(std::size_t environment_index,
                                      const std::string& robot_name,
                                      const std::string& link_name,
                                      const Vector3& linear_velocity,
                                      const Vector3& angular_velocity) override;

    bool ResetEnvironmentRobotStates(const std::vector<PhysicsEnvironmentRobotResetState>& reset_states) override;

    bool SetEnvironmentJointControl(std::size_t environment_index,
                                    const std::string& robot_name,
                                    const std::string& joint_name,
                                    PhysicsJointControlMode control_mode,
                                    RealType target) override;

    bool SetEnvironmentJointControls(const std::string& robot_name,
                                     const std::vector<std::string>& joint_names,
                                     PhysicsJointControlMode control_mode,
                                     const std::vector<RealType>& targets,
                                     std::size_t environment_count) override;

    bool StepRobotBatch(const PhysicsRobotBatchStepRequest& request,
                        PhysicsRobotBatchStepResult& result) override;

    bool SetLinkExternalForce(const std::string& robot_name,
                              const std::string& link_name,
                              const Vector3& point,
                              const Vector3& force) override;

    bool SetLinkSpringForce(const std::string& robot_name,
                            const std::string& link_name,
                            const Vector3& local_point,
                            const Vector3& target_point,
                            const Vector3& force_hint) override;

    void ClearExternalForces() override;

    PhysicsRaycastHit RaycastTerrain(const PhysicsRaycastQuery& query) const override;

    Diagnostics GetDiagnostics() const;

protected:
    PhysicsRaycastHit RaycastTerrainForSensor(const PhysicsRaycastQuery& query,
                                              std::size_t environment_index) const override;

private:
#ifdef GOBOT_HAS_MUJOCO
    struct RobotBatchLayout;

    bool CompileAuthoredModel();

    bool AddAuthoredRobotToSpec(void* parent_spec,
                                const PhysicsRobotSnapshot& robot,
                                std::size_t robot_index,
                                const std::string& prefix);

    void AddLooseSceneGeomsToSpec(void* spec);

    void AddTerrainGeomsToSpec(void* spec);

    void AddFloatingBaseJointsToSpec(void* spec, const PhysicsRobotSnapshot& robot);

    void BuildJointBindings();

    void BuildSensorBindings();

    void BuildLinkBindings();

    std::string GetRobotPrefix(std::size_t robot_index) const;

    bool IsEnvironmentIndexValid(std::size_t environment_index) const;

    PhysicsSceneState& EnvironmentState(std::size_t environment_index);

    const PhysicsSceneState& EnvironmentState(std::size_t environment_index) const;

    void* ModelForEnvironment(std::size_t environment_index) const;

    void* DataForEnvironment(std::size_t environment_index) const;

    void ApplyControlsToMuJoCo(std::size_t environment_index);

    void ApplyExternalForcesToMuJoCo(std::size_t environment_index);

    void FreeModel();

    void SyncStateFromMuJoCo(std::size_t environment_index);

    void SyncStateToMuJoCo(std::size_t environment_index);

    void SyncContactsFromMuJoCo(std::size_t environment_index);

    void SyncSensorsFromMuJoCo(std::size_t environment_index);

    PhysicsRaycastHit RaycastTerrainWithMuJoCo(const PhysicsRaycastQuery& query,
                                               std::size_t environment_index) const;

    struct MuJoCoJointBinding {
        std::size_t robot_index{0};
        std::size_t joint_index{0};
        int mujoco_joint_id{-1};
        int motor_actuator_id{-1};
        int position_actuator_id{-1};
        int velocity_actuator_id{-1};
        RealType position_stiffness{0.0};
        RealType velocity_damping{0.0};
        RealType passive_damping{0.0};
        int qpos_address{-1};
        int dof_address{-1};
        int joint_type{-1};
        std::vector<JointController> controllers;
    };

    struct MuJoCoLinkBinding {
        std::size_t robot_index{0};
        std::size_t link_index{0};
        int body_id{-1};
    };

    struct MuJoCoSensorComponentBinding {
        int sensor_id{-1};
        std::size_t value_offset{0};
    };

    struct MuJoCoSensorBinding {
        std::size_t robot_index{0};
        std::size_t sensor_index{0};
        std::vector<MuJoCoSensorComponentBinding> components;
    };

    struct MuJoCoRobotBinding {
        std::size_t robot_index{0};
        std::string prefix;
    };
#endif

    std::size_t ResolveBatchWorkerCount(std::size_t requested_workers,
                                        std::size_t environment_count) const;

    bool StepEnvironmentBatchInternal(RealType delta_time,
                                      std::uint64_t ticks,
                                      std::size_t worker_count,
                                      bool sync_state,
                                      bool apply_controls = true);

    bool EnsureBatchWorkers(std::size_t worker_count);

    using BatchEnvironmentTask = std::function<void(std::size_t)>;

    bool RunEnvironmentBatchTask(std::size_t environment_count,
                                 std::size_t worker_count,
                                 BatchEnvironmentTask task);

    void StopBatchWorkers();

    void BatchWorkerLoop(std::size_t worker_index);

    bool available_{false};
    PhysicsSceneArtifact scene_artifact_;

#ifdef GOBOT_HAS_MUJOCO
    void* model_{nullptr};
    void* data_{nullptr};
    std::vector<void*> environment_models_;
    std::vector<void*> environment_data_;
    std::vector<PhysicsSceneState> environment_states_;
    std::vector<MuJoCoRobotBinding> robot_bindings_;
    std::vector<MuJoCoJointBinding> joint_bindings_;
    std::vector<MuJoCoLinkBinding> link_bindings_;
    std::vector<MuJoCoSensorBinding> sensor_bindings_;
    std::unique_ptr<RobotBatchLayout> robot_batch_layout_;

    std::vector<std::thread> batch_workers_;
    std::mutex batch_mutex_;
    std::condition_variable batch_cv_;
    std::condition_variable batch_done_cv_;
    std::size_t batch_generation_{0};
    std::atomic<std::size_t> batch_completed_workers_{0};
    std::atomic<std::size_t> batch_next_environment_{0};
    std::size_t batch_active_workers_{0};
    std::size_t batch_environment_count_{0};
    std::size_t batch_work_chunk_{1};
    BatchEnvironmentTask batch_environment_task_;
    std::exception_ptr batch_worker_error_;
    bool batch_stop_{false};
    bool batch_work_pending_{false};
#endif
};

} // namespace gobot
