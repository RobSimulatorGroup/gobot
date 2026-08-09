/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_solver_module_api.hpp"

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

} // namespace

extern "C" __attribute__((visibility("default")))
const gobot::IpcSolverModuleApi* gobot_ipc_solver_get_api() {
    return &kApi;
}
