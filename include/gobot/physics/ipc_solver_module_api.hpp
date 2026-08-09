/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace gobot {

inline constexpr std::uint32_t GOBOT_IPC_SOLVER_MODULE_ABI_VERSION = 4;

struct IpcSolverArtifactBlobView {
    const char* id{nullptr};
    const char* encoding{nullptr};
    const char* sha256{nullptr};
    const std::uint8_t* data{nullptr};
    std::size_t size{0};
};

struct IpcSolverArtifactView {
    std::uint32_t schema_version{0};
    const char* producer{nullptr};
    const char* producer_version{nullptr};
    const char* format{nullptr};
    const char* manifest{nullptr};
    std::size_t manifest_size{0};
    const char* manifest_sha256{nullptr};
    const IpcSolverArtifactBlobView* blobs{nullptr};
    std::size_t blob_count{0};
};

struct IpcSolverModuleConfig {
    double fixed_time_step{0.002};
    double gravity[3]{0.0, 0.0, -9.81};
    double friction_coefficient{0.5};
    double contact_activation_distance{0.01};
    double contact_resistance{1.0e9};
    double affine_stiffness{1.0e8};
    double kinematic_strength{100.0};
    std::uint32_t device_index{0};
    const char* workspace{nullptr};
    const char* backend_module_directory{nullptr};
};

struct IpcSolverModuleBodyInfo {
    const char* path{nullptr};
    std::size_t element_offset{0};
    std::size_t element_count{0};
};

struct IpcSolverModuleDiagnostics {
    std::uint64_t frame{0};
    std::size_t deformable_body_count{0};
    std::size_t deformable_vertex_count{0};
    std::size_t affine_body_count{0};
    double last_step_latency_ms{0.0};
    bool valid{false};
};

struct IpcSolverModuleApi {
    std::uint32_t abi_version{0};
    const char* provider_name{nullptr};

    void* (*create)(const IpcSolverArtifactView* artifact,
                    const IpcSolverModuleConfig* config,
                    char* error,
                    std::size_t error_size){nullptr};
    void (*destroy)(void* session){nullptr};
    bool (*step)(void* session,
                 std::uint32_t steps,
                 char* error,
                 std::size_t error_size){nullptr};
    bool (*reset)(void* session, char* error, std::size_t error_size){nullptr};

    std::size_t (*deformable_body_count)(void* session){nullptr};
    bool (*deformable_body_info)(void* session,
                                 std::size_t body_index,
                                 IpcSolverModuleBodyInfo* info,
                                 char* error,
                                 std::size_t error_size){nullptr};
    bool (*copy_deformable_positions)(void* session,
                                      double* destination_xyz,
                                      std::size_t scalar_count,
                                      char* error,
                                      std::size_t error_size){nullptr};
    bool (*copy_deformable_velocities)(void* session,
                                       double* destination_xyz,
                                       std::size_t scalar_count,
                                       char* error,
                                       std::size_t error_size){nullptr};
    bool (*copy_deformable_contact_forces)(void* session,
                                          double* destination_xyz,
                                          std::size_t scalar_count,
                                          char* error,
                                          std::size_t error_size){nullptr};

    std::size_t (*affine_body_count)(void* session){nullptr};
    bool (*affine_body_info)(void* session,
                             std::size_t body_index,
                             IpcSolverModuleBodyInfo* info,
                             char* error,
                             std::size_t error_size){nullptr};
    bool (*copy_affine_transforms)(void* session,
                                   double* destination_row_major,
                                   std::size_t scalar_count,
                                   char* error,
                                   std::size_t error_size){nullptr};
    bool (*set_affine_target)(void* session,
                              const char* path,
                              const double* transform_row_major,
                              char* error,
                              std::size_t error_size){nullptr};
    bool (*set_joint_target)(void* session,
                             const char* path,
                             double position,
                             char* error,
                             std::size_t error_size){nullptr};

    bool (*diagnostics)(void* session,
                        IpcSolverModuleDiagnostics* diagnostics,
                        char* error,
                        std::size_t error_size){nullptr};
};

using GetIpcSolverModuleApi = const IpcSolverModuleApi* (*)();

} // namespace gobot

#if defined(__GNUC__) || defined(__clang__)
#define GOBOT_IPC_SOLVER_MODULE_EXPORT __attribute__((visibility("default")))
#else
#define GOBOT_IPC_SOLVER_MODULE_EXPORT
#endif

extern "C" GOBOT_IPC_SOLVER_MODULE_EXPORT const gobot::IpcSolverModuleApi*
gobot_ipc_solver_get_api();
