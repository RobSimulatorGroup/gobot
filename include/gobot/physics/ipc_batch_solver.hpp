/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gobot/core/macros.hpp"
#include "gobot/physics/ipc_batch_solver_module_api.hpp"
#include "gobot/physics/ipc_solver.hpp"

namespace gobot {

struct IpcBatchSolverConfig {
    IpcSolverConfig solver;
    std::uint32_t environment_count{0};
    std::uint32_t environments_per_shard{0};
    bool external_affine_proxies{true};
    std::string contact_constitution{"ipc"};
    double al_ipc_mu_scale_fem{5.0e7};
    double al_ipc_mu_scale_abd{1.0e5};
    double al_ipc_toi_threshold{0.1};
    double al_ipc_alpha_lower_bound{1.0e-6};
    double al_ipc_decay_factor{0.3};
};

struct IpcBatchSolverDiagnostics {
    std::string provider_name;
    std::uint64_t frame{0};
    std::size_t environment_count{0};
    std::size_t shard_count{0};
    std::size_t deformable_body_count_per_environment{0};
    std::size_t deformable_vertex_count_per_environment{0};
    std::size_t affine_body_count_per_environment{0};
    std::size_t static_collider_count_per_environment{0};
    double last_step_latency_ms{0.0};
    std::string contact_constitution{"ipc"};
    bool exact_contact_wrench{true};
    bool checkpoint_active{false};
    bool valid{false};
};

class GOBOT_EXPORT IpcBatchSolverSession final {
public:
    static std::unique_ptr<IpcBatchSolverSession> Create(
            const IpcSceneArtifact& artifact,
            const IpcBatchSolverConfig& config,
            const std::string& module_path = {},
            std::string* error = nullptr);

    static bool IsModuleAvailable(const std::string& module_path = {},
                                  std::string* error = nullptr);

    ~IpcBatchSolverSession();

    IpcBatchSolverSession(const IpcBatchSolverSession&) = delete;
    IpcBatchSolverSession& operator=(const IpcBatchSolverSession&) = delete;

    bool BindDeviceBuffers(const IpcBatchSolverModuleBuffers& buffers);
    bool Step(std::uint32_t steps = 1);
    bool ResetFull();
    bool CaptureCheckpoint();
    bool RewindCheckpoint();
    bool CommitCheckpoint();
    bool Synchronize();

    const std::vector<IpcSolverBodyInfo>& GetDeformableBodies() const;
    const std::vector<IpcSolverBodyInfo>& GetAffineBodies() const;
    const IpcBatchSolverDiagnostics& GetDiagnostics() const;
    const std::string& GetLastError() const;

private:
    class Impl;
    explicit IpcBatchSolverSession(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace gobot
