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

inline constexpr std::uint32_t GOBOT_IPC_BATCH_SOLVER_MODULE_ABI_VERSION = 1;

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
    IpcSolverDeviceBufferView affine_transforms;
    IpcSolverDeviceBufferView affine_contact_wrenches;
};

struct IpcBatchSolverModuleConfig {
    IpcSolverModuleConfig solver;
    std::uint32_t environment_count{0};
    std::uint32_t environments_per_shard{0};
    bool external_affine_proxies{true};
};

struct IpcBatchSolverModuleDiagnostics {
    std::uint64_t frame{0};
    std::size_t environment_count{0};
    std::size_t shard_count{0};
    std::size_t deformable_body_count_per_environment{0};
    std::size_t deformable_vertex_count_per_environment{0};
    std::size_t affine_body_count_per_environment{0};
    double last_step_latency_ms{0.0};
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
    bool (*step)(void* session,
                 std::uint32_t steps,
                 char* error,
                 std::size_t error_size){nullptr};
    bool (*reset_full)(void* session,
                       char* error,
                       std::size_t error_size){nullptr};
    bool (*synchronize)(void* session,
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
