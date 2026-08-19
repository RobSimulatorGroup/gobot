/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "gobot/physics/ipc_solver_module_api.hpp"

namespace gobot {

inline constexpr std::uint32_t GOBOT_IPC_BATCH_SOLVER_MODULE_ABI_VERSION = 5;

enum IpcBatchSolverOutputFlag : std::uint32_t {
    IpcBatchSolverOutputNone = 0,
    IpcBatchSolverOutputDeformableState = 1U << 0U,
    IpcBatchSolverOutputAffineState = 1U << 1U,
    IpcBatchSolverOutputDeformableContactForces = 1U << 2U,
    IpcBatchSolverOutputAffineContactWrenches = 1U << 3U,
    IpcBatchSolverOutputAll =
            IpcBatchSolverOutputDeformableState |
            IpcBatchSolverOutputAffineState |
            IpcBatchSolverOutputDeformableContactForces |
            IpcBatchSolverOutputAffineContactWrenches,
};

enum class IpcSolverDeviceScalarType : std::uint32_t {
    Float64 = 1,
};

struct IpcSolverDeviceBufferView {
    void* data{nullptr};
    std::uint32_t device_index{0};
    IpcSolverDeviceScalarType scalar_type{IpcSolverDeviceScalarType::Float64};
    std::uint32_t rank{0};
    std::size_t shape[4]{0, 0, 0, 0};
    std::size_t stride[4]{0, 0, 0, 0};
};

struct IpcBatchSolverModuleBuffers {
    IpcSolverDeviceBufferView deformable_positions;
    IpcSolverDeviceBufferView deformable_velocities;
    IpcSolverDeviceBufferView deformable_contact_forces;
    IpcSolverDeviceBufferView affine_targets;
    // World-frame body-origin twists in [linear_xyz, angular_xyz] order.
    IpcSolverDeviceBufferView affine_target_twists;
    IpcSolverDeviceBufferView affine_transforms;
    IpcSolverDeviceBufferView affine_contact_wrenches;
};

struct IpcBatchSolverExecutionContext {
    // CUDA stream owned by the caller. Zero selects the legacy default stream.
    std::uintptr_t cuda_stream{0};
};

enum class IpcSolverPipelineStage : std::uint32_t {
    None = 0,
    RebuildScene,
    PredictMotion,
    ComputeDyTopoEffect,
    SolveGlobalLinearSystem,
    LineSearch,
    UpdateVelocity,
};

enum class IpcSolverFailureKind : std::uint32_t {
    None = 0,
    LineSearchLimit,
    NewtonLimit,
    LinearSystemLimit,
    LinearSystemBreakdown,
    NonFinite,
    Unexpected,
};

struct IpcBatchSolverRuntimeOptions {
    std::uint32_t newton_max_iterations{16};
    std::uint32_t line_search_max_iterations{8};
    double linear_system_tolerance_rate{1.0e-3};
    bool strict_convergence{false};
};

struct IpcBatchSolverModuleConfig {
    IpcSolverModuleConfig solver;
    std::uint32_t environment_count{0};
    std::uint32_t environments_per_shard{0};
    bool external_affine_proxies{true};
    const char* contact_constitution{"ipc"};
    double al_ipc_mu_scale_fem{5.0e7};
    double al_ipc_mu_scale_abd{1.0e5};
    double al_ipc_toi_threshold{0.1};
    double al_ipc_alpha_lower_bound{1.0e-6};
    double al_ipc_decay_factor{0.3};
    std::uint32_t newton_max_iterations{16};
    std::uint32_t line_search_max_iterations{8};
    double linear_system_tolerance_rate{1.0e-3};
    bool strict_convergence{false};
    std::uint32_t output_flags{IpcBatchSolverOutputAll};
};

struct IpcBatchSolverModuleDiagnostics {
    std::uint64_t frame{0};
    std::size_t environment_count{0};
    std::size_t shard_count{0};
    std::size_t deformable_body_count_per_environment{0};
    std::size_t deformable_vertex_count_per_environment{0};
    std::size_t affine_body_count_per_environment{0};
    std::size_t static_collider_count_per_environment{0};
    double last_step_latency_ms{0.0};
    double last_checkpoint_latency_ms{0.0};
    double last_target_staging_latency_ms{0.0};
    double last_ipc_advance_latency_ms{0.0};
    double last_reaction_export_latency_ms{0.0};
    double last_state_sync_latency_ms{0.0};
    std::uint32_t output_flags{IpcBatchSolverOutputAll};
    std::uint64_t deformable_contact_force_frame{0};
    const char* contact_constitution{nullptr};
    bool exact_contact_wrench{false};
    bool checkpoint_active{false};
    bool device_native_coupling{false};
    bool cuda_stream_interop{false};
    std::size_t device_workspace_allocation_count{0};
    IpcSolverPipelineStage solver_stage{IpcSolverPipelineStage::None};
    IpcSolverFailureKind solver_failure{IpcSolverFailureKind::None};
    std::uint32_t newton_iterations{0};
    std::uint32_t line_search_iterations_total{0};
    std::uint32_t line_search_iterations_max{0};
    std::uint32_t pcg_iterations_total{0};
    std::uint32_t pcg_iterations_max{0};
    std::uint32_t pcg_iterations_last{0};
    double pcg_relative_residual{0.0};
    double minimum_step_length{1.0};
    const char* solver_failure_message{nullptr};
    std::size_t failing_shard_index{static_cast<std::size_t>(-1)};
    bool newton_converged{false};
    bool linear_system_converged{true};
    bool strict_convergence{false};
    bool recovered{false};
    bool valid{false};
};

struct IpcBatchSolverModuleApi {
    std::uint32_t abi_version{0};
    const char* provider_name{nullptr};

    void* (*create)(const IpcSolverArtifactView* artifact,
                    const IpcBatchSolverModuleConfig* config,
                    char* error,
                    std::size_t error_size){nullptr};
    void (*destroy)(void* session){nullptr};
    bool (*bind_device_buffers)(void* session,
                                const IpcBatchSolverModuleBuffers* buffers,
                                char* error,
                                std::size_t error_size){nullptr};
    bool (*set_execution_context)(void* session,
                                  const IpcBatchSolverExecutionContext* context,
                                  char* error,
                                  std::size_t error_size){nullptr};
    bool (*step)(void* session,
                 std::uint32_t steps,
                 char* error,
                 std::size_t error_size){nullptr};
    bool (*reset_full)(void* session,
                       char* error,
                       std::size_t error_size){nullptr};
    bool (*capture_checkpoint)(void* session,
                               char* error,
                               std::size_t error_size){nullptr};
    bool (*rewind_checkpoint)(void* session,
                              char* error,
                              std::size_t error_size){nullptr};
    bool (*commit_checkpoint)(void* session,
                              char* error,
                              std::size_t error_size){nullptr};
    bool (*synchronize)(void* session,
                        char* error,
                        std::size_t error_size){nullptr};
    bool (*set_output_flags)(void* session,
                             std::uint32_t output_flags,
                             char* error,
                             std::size_t error_size){nullptr};
    bool (*refresh_outputs)(void* session,
                            std::uint32_t output_flags,
                            char* error,
                            std::size_t error_size){nullptr};
    bool (*set_runtime_options)(
            void* session,
            const IpcBatchSolverRuntimeOptions* options,
            char* error,
            std::size_t error_size){nullptr};

    std::size_t (*deformable_body_count)(void* session){nullptr};
    bool (*deformable_body_info)(void* session,
                                 std::size_t body_index,
                                 IpcSolverModuleBodyInfo* info,
                                 char* error,
                                 std::size_t error_size){nullptr};
    std::size_t (*affine_body_count)(void* session){nullptr};
    bool (*affine_body_info)(void* session,
                             std::size_t body_index,
                             IpcSolverModuleBodyInfo* info,
                             char* error,
                             std::size_t error_size){nullptr};
    bool (*diagnostics)(void* session,
                        IpcBatchSolverModuleDiagnostics* diagnostics,
                        char* error,
                        std::size_t error_size){nullptr};
};

using GetIpcBatchSolverModuleApi = const IpcBatchSolverModuleApi* (*)();

} // namespace gobot

extern "C" GOBOT_IPC_SOLVER_MODULE_EXPORT const gobot::IpcBatchSolverModuleApi*
gobot_ipc_solver_get_batch_api();
