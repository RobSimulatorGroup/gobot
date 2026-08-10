/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_solver_module_api.hpp"
#include "gobot/physics/ipc_batch_solver_module_api.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using gobot::IpcSolverArtifactView;
using gobot::IpcSolverModuleApi;
using gobot::IpcSolverModuleBodyInfo;
using gobot::IpcSolverModuleConfig;
using gobot::IpcSolverModuleDiagnostics;
using gobot::IpcBatchSolverModuleApi;
using gobot::IpcBatchSolverModuleBuffers;
using gobot::IpcBatchSolverModuleConfig;
using gobot::IpcBatchSolverModuleDiagnostics;

constexpr std::array<double, 16> kIdentity{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};

struct Session {
    explicit Session(double activation_distance = 0.01)
        : contact_activation_distance(activation_distance) {
        positions[1] = activation_distance;
    }

    std::vector<double> positions{0.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    std::vector<double> velocities{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> contact_forces{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> affine_transform{kIdentity.begin(), kIdentity.end()};
    double joint_target{0.0};
    double contact_activation_distance{0.01};
    std::uint64_t frame{0};
};

void WriteError(char* destination, std::size_t capacity, std::string_view message) {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    const std::size_t size = std::min(capacity - 1, message.size());
    std::memcpy(destination, message.data(), size);
    destination[size] = '\0';
}

Session* Cast(void* session, char* error, std::size_t error_size) {
    if (session == nullptr) {
        WriteError(error, error_size, "fake IPC session is null");
        return nullptr;
    }
    return static_cast<Session*>(session);
}

bool Copy(const std::vector<double>& source,
          double* destination,
          std::size_t scalar_count,
          char* error,
          std::size_t error_size) {
    if (scalar_count != source.size() ||
        (destination == nullptr && scalar_count != 0)) {
        WriteError(error, error_size, "fake IPC destination has the wrong size");
        return false;
    }
    if (!source.empty()) {
        std::copy(source.begin(), source.end(), destination);
    }
    return true;
}

void* Create(const IpcSolverArtifactView* artifact,
             const IpcSolverModuleConfig* config,
             char* error,
             std::size_t error_size) {
    if (artifact == nullptr || config == nullptr || artifact->manifest == nullptr ||
        artifact->schema_version != 1 || artifact->format == nullptr ||
        std::string_view(artifact->format) != "gobot-ipc") {
        WriteError(error, error_size, "fake IPC module rejected the artifact");
        return nullptr;
    }
    if (!std::isfinite(config->contact_activation_distance) ||
        config->contact_activation_distance <= 0.0) {
        WriteError(error, error_size,
                   "fake IPC contact activation distance must be finite and positive");
        return nullptr;
    }
    return new Session(config->contact_activation_distance);
}

void Destroy(void* session) {
    delete static_cast<Session*>(session);
}

bool Step(void* session,
          std::uint32_t steps,
          char* error,
          std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    if (value == nullptr) {
        return false;
    }
    if (steps == 0) {
        WriteError(error, error_size, "fake IPC step count must be positive");
        return false;
    }
    value->frame += steps;
    for (std::size_t vertex = 0; vertex < 2; ++vertex) {
        value->positions[vertex * 3] += 0.25 * static_cast<double>(steps);
        value->velocities[vertex * 3] = 0.25;
        value->contact_forces[vertex * 3 + 2] =
                static_cast<double>((vertex + 1) * value->frame);
    }
    return true;
}

bool Reset(void* session, char* error, std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    if (value == nullptr) {
        return false;
    }
    *value = Session{value->contact_activation_distance};
    return true;
}

std::size_t DeformableBodyCount(void*) {
    return 1;
}

bool DeformableBodyInfo(void* session,
                        std::size_t index,
                        IpcSolverModuleBodyInfo* info,
                        char* error,
                        std::size_t error_size) {
    if (Cast(session, error, error_size) == nullptr || info == nullptr || index != 0) {
        WriteError(error, error_size, "fake deformable body index is out of range");
        return false;
    }
    *info = IpcSolverModuleBodyInfo{"/World/Soft", 0, 2};
    return true;
}

bool CopyPositions(void* session,
                   double* destination,
                   std::size_t scalar_count,
                   char* error,
                   std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    return value != nullptr && Copy(value->positions, destination, scalar_count, error, error_size);
}

bool CopyVelocities(void* session,
                    double* destination,
                    std::size_t scalar_count,
                    char* error,
                    std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    return value != nullptr && Copy(value->velocities, destination, scalar_count, error, error_size);
}

bool CopyContactForces(void* session,
                       double* destination,
                       std::size_t scalar_count,
                       char* error,
                       std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    return value != nullptr &&
           Copy(value->contact_forces, destination, scalar_count, error, error_size);
}

std::size_t AffineBodyCount(void*) {
    return 1;
}

bool AffineBodyInfo(void* session,
                    std::size_t index,
                    IpcSolverModuleBodyInfo* info,
                    char* error,
                    std::size_t error_size) {
    if (Cast(session, error, error_size) == nullptr || info == nullptr || index != 0) {
        WriteError(error, error_size, "fake affine body index is out of range");
        return false;
    }
    *info = IpcSolverModuleBodyInfo{"/World/Robot/Finger", 0, 1};
    return true;
}

bool CopyAffineTransforms(void* session,
                          double* destination,
                          std::size_t scalar_count,
                          char* error,
                          std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    return value != nullptr &&
           Copy(value->affine_transform, destination, scalar_count, error, error_size);
}

bool SetAffineTarget(void* session,
                     const char* path,
                     const double* transform,
                     char* error,
                     std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    if (value == nullptr || path == nullptr || transform == nullptr ||
        std::string_view(path) != "/World/Robot/Finger") {
        WriteError(error, error_size, "fake affine body path is unknown");
        return false;
    }
    std::copy_n(transform, 16, value->affine_transform.begin());
    return true;
}

bool SetJointTarget(void* session,
                    const char* path,
                    double position,
                    char* error,
                    std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    if (value == nullptr || path == nullptr || !std::isfinite(position) ||
        std::string_view(path) != "/World/Robot/FingerJoint") {
        WriteError(error, error_size, "fake joint path or target is invalid");
        return false;
    }
    value->joint_target = position;
    value->affine_transform[3] = position;
    return true;
}

bool Diagnostics(void* session,
                 IpcSolverModuleDiagnostics* diagnostics,
                 char* error,
                 std::size_t error_size) {
    Session* value = Cast(session, error, error_size);
    if (value == nullptr || diagnostics == nullptr) {
        WriteError(error, error_size, "fake diagnostics output is null");
        return false;
    }
    *diagnostics = IpcSolverModuleDiagnostics{
            value->frame, 1, 2, 1, 0.125, true};
    return true;
}

const IpcSolverModuleApi kApi{
        gobot::GOBOT_IPC_SOLVER_MODULE_ABI_VERSION,
        "fake-ipc",
        &Create,
        &Destroy,
        &Step,
        &Reset,
        &DeformableBodyCount,
        &DeformableBodyInfo,
        &CopyPositions,
        &CopyVelocities,
        &CopyContactForces,
        &AffineBodyCount,
        &AffineBodyInfo,
        &CopyAffineTransforms,
        &SetAffineTarget,
        &SetJointTarget,
        &Diagnostics};

struct BatchSession {
    std::uint32_t environment_count{0};
    std::uint32_t environments_per_shard{0};
    IpcBatchSolverModuleBuffers buffers{};
    std::uint64_t frame{0};
    bool bound{false};
};

double* Data(const gobot::IpcSolverDeviceBufferView& view) {
    return static_cast<double*>(view.data);
}

void InitializeBatchBuffers(BatchSession* session) {
    const std::size_t environments = session->environment_count;
    double* positions = Data(session->buffers.deformable_positions);
    double* velocities = Data(session->buffers.deformable_velocities);
    double* contact_forces = Data(session->buffers.deformable_contact_forces);
    double* transforms = Data(session->buffers.affine_transforms);
    double* targets = Data(session->buffers.affine_targets);
    double* wrenches = Data(session->buffers.affine_contact_wrenches);
    for (std::size_t environment = 0; environment < environments;
         ++environment) {
        const std::size_t vertex_offset = environment * 2 * 3;
        std::fill_n(positions + vertex_offset, 6, 0.0);
        positions[vertex_offset + 2] = 1.0;
        positions[vertex_offset + 5] = 1.0;
        std::fill_n(velocities + vertex_offset, 6, 0.0);
        std::fill_n(contact_forces + vertex_offset, 6, 0.0);
        std::copy_n(targets + environment * 16, 16,
                    transforms + environment * 16);
        std::fill_n(wrenches + environment * 6, 6, 0.0);
    }
}

void* BatchCreate(const IpcSolverArtifactView* artifact,
                  const IpcBatchSolverModuleConfig* config,
                  char* error,
                  std::size_t error_size) {
    if (artifact == nullptr || config == nullptr || artifact->manifest == nullptr ||
        config->environment_count == 0 ||
        config->environments_per_shard == 0 ||
        config->environment_count % config->environments_per_shard != 0) {
        WriteError(error, error_size, "fake IPC batch module rejected its config");
        return nullptr;
    }
    auto session = std::make_unique<BatchSession>();
    session->environment_count = config->environment_count;
    session->environments_per_shard = config->environments_per_shard;
    return session.release();
}

void BatchDestroy(void* session) {
    delete static_cast<BatchSession*>(session);
}

bool BatchBind(void* opaque,
               const IpcBatchSolverModuleBuffers* buffers,
               char* error,
               std::size_t error_size) {
    auto* session = static_cast<BatchSession*>(opaque);
    if (session == nullptr || buffers == nullptr ||
        buffers->deformable_positions.data == nullptr ||
        buffers->affine_targets.data == nullptr) {
        WriteError(error, error_size, "fake IPC batch buffers are invalid");
        return false;
    }
    session->buffers = *buffers;
    session->bound = true;
    InitializeBatchBuffers(session);
    return true;
}

bool BatchStep(void* opaque,
               std::uint32_t steps,
               char* error,
               std::size_t error_size) {
    auto* session = static_cast<BatchSession*>(opaque);
    if (session == nullptr || !session->bound || steps == 0) {
        WriteError(error, error_size, "fake IPC batch step is invalid");
        return false;
    }
    session->frame += steps;
    double* positions = Data(session->buffers.deformable_positions);
    double* velocities = Data(session->buffers.deformable_velocities);
    double* contact_forces = Data(session->buffers.deformable_contact_forces);
    double* transforms = Data(session->buffers.affine_transforms);
    double* targets = Data(session->buffers.affine_targets);
    double* wrenches = Data(session->buffers.affine_contact_wrenches);
    for (std::size_t environment = 0;
         environment < session->environment_count; ++environment) {
        for (std::size_t vertex = 0; vertex < 2; ++vertex) {
            const std::size_t offset = (environment * 2 + vertex) * 3;
            positions[offset] += 0.25 * static_cast<double>(steps);
            velocities[offset] = 0.25;
            contact_forces[offset + 2] =
                    static_cast<double>((vertex + 1) * session->frame);
        }
        std::copy_n(targets + environment * 16, 16,
                    transforms + environment * 16);
        std::fill_n(wrenches + environment * 6, 6, 0.0);
        wrenches[environment * 6 + 2] =
                static_cast<double>(session->frame);
    }
    return true;
}

bool BatchReset(void* opaque, char* error, std::size_t error_size) {
    auto* session = static_cast<BatchSession*>(opaque);
    if (session == nullptr || !session->bound) {
        WriteError(error, error_size, "fake IPC batch reset is invalid");
        return false;
    }
    session->frame = 0;
    InitializeBatchBuffers(session);
    return true;
}

bool BatchSynchronize(void* opaque, char* error, std::size_t error_size) {
    auto* session = static_cast<BatchSession*>(opaque);
    if (session == nullptr || !session->bound) {
        WriteError(error, error_size, "fake IPC batch sync is invalid");
        return false;
    }
    return true;
}

std::size_t BatchDeformableBodyCount(void*) { return 1; }

bool BatchDeformableBodyInfo(void* opaque,
                             std::size_t index,
                             IpcSolverModuleBodyInfo* info,
                             char* error,
                             std::size_t error_size) {
    if (opaque == nullptr || info == nullptr || index != 0) {
        WriteError(error, error_size,
                   "fake IPC batch deformable body index is invalid");
        return false;
    }
    *info = IpcSolverModuleBodyInfo{"/World/Soft", 0, 2};
    return true;
}

std::size_t BatchAffineBodyCount(void*) { return 1; }

bool BatchAffineBodyInfo(void* opaque,
                         std::size_t index,
                         IpcSolverModuleBodyInfo* info,
                         char* error,
                         std::size_t error_size) {
    if (opaque == nullptr || info == nullptr || index != 0) {
        WriteError(error, error_size,
                   "fake IPC batch affine body index is invalid");
        return false;
    }
    *info = IpcSolverModuleBodyInfo{"/World/Robot/Finger", 0, 1};
    return true;
}

bool BatchDiagnostics(void* opaque,
                      IpcBatchSolverModuleDiagnostics* diagnostics,
                      char* error,
                      std::size_t error_size) {
    auto* session = static_cast<BatchSession*>(opaque);
    if (session == nullptr || diagnostics == nullptr) {
        WriteError(error, error_size,
                   "fake IPC batch diagnostics output is invalid");
        return false;
    }
    *diagnostics = IpcBatchSolverModuleDiagnostics{
            session->frame,
            session->environment_count,
            session->environment_count / session->environments_per_shard,
            1,
            2,
            1,
            0.25,
            true};
    return true;
}

const IpcBatchSolverModuleApi kBatchApi{
        gobot::GOBOT_IPC_BATCH_SOLVER_MODULE_ABI_VERSION,
        "fake-ipc-batch",
        &BatchCreate,
        &BatchDestroy,
        &BatchBind,
        &BatchStep,
        &BatchReset,
        &BatchSynchronize,
        &BatchDeformableBodyCount,
        &BatchDeformableBodyInfo,
        &BatchAffineBodyCount,
        &BatchAffineBodyInfo,
        &BatchDiagnostics};

} // namespace

extern "C" __attribute__((visibility("default")))
const gobot::IpcSolverModuleApi* gobot_ipc_solver_get_api() {
    return &kApi;
}

extern "C" __attribute__((visibility("default")))
const gobot::IpcBatchSolverModuleApi* gobot_ipc_solver_get_batch_api() {
    return &kBatchApi;
}
