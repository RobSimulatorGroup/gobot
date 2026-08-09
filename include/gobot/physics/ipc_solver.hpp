/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gobot/core/macros.hpp"
#include "gobot/physics/ipc_scene_compiler.hpp"

namespace gobot {

struct IpcSolverConfig {
    double fixed_time_step{0.002};
    double gravity[3]{0.0, 0.0, -9.81};
    double friction_coefficient{0.5};
    double contact_activation_distance{0.01};
    double contact_resistance{1.0e9};
    double affine_stiffness{1.0e8};
    double kinematic_strength{100.0};
    std::uint32_t device_index{0};
    std::string workspace;
    std::string backend_module_directory;
};

struct IpcSolverBodyInfo {
    std::string path;
    std::size_t element_offset{0};
    std::size_t element_count{0};
};

struct IpcSolverDiagnostics {
    std::string provider_name;
    std::uint64_t frame{0};
    std::size_t deformable_body_count{0};
    std::size_t deformable_vertex_count{0};
    std::size_t affine_body_count{0};
    double last_step_latency_ms{0.0};
    bool valid{false};
};

class GOBOT_EXPORT IpcSolverSession final {
public:
    static std::unique_ptr<IpcSolverSession> Create(
            const IpcSceneArtifact& artifact,
            const IpcSolverConfig& config,
            const std::string& module_path = {},
            std::string* error = nullptr);

    static bool IsModuleAvailable(const std::string& module_path = {},
                                  std::string* error = nullptr);

    ~IpcSolverSession();

    IpcSolverSession(const IpcSolverSession&) = delete;
    IpcSolverSession& operator=(const IpcSolverSession&) = delete;

    bool Step(std::uint32_t steps = 1);
    bool Reset();
    bool SetAffineTarget(const std::string& path,
                         const double* transform_row_major_4x4);
    bool SetJointTarget(const std::string& path, double position);

    const std::vector<IpcSolverBodyInfo>& GetDeformableBodies() const;
    const std::vector<IpcSolverBodyInfo>& GetAffineBodies() const;
    const std::vector<double>& GetDeformablePositions() const;
    const std::vector<double>& GetDeformableVelocities() const;
    const std::vector<double>& GetDeformableContactForces() const;
    const std::vector<double>& GetAffineTransforms() const;
    const IpcSolverDiagnostics& GetDiagnostics() const;
    const std::string& GetLastError() const;

private:
    class Impl;
    explicit IpcSolverSession(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace gobot
