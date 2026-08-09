/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_solver.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <utility>

#include "gobot/physics/ipc_solver_module_api.hpp"

namespace gobot {
namespace {

constexpr std::size_t kErrorCapacity = 2048;

void IpcSolverModuleAnchor() {}

bool SetError(std::string* destination, std::string message) {
    if (destination != nullptr) {
        *destination = std::move(message);
    }
    return false;
}

std::string DefaultModulePath() {
    Dl_info module_info{};
    if (dladdr(reinterpret_cast<void*>(&IpcSolverModuleAnchor), &module_info) != 0 &&
        module_info.dli_fname != nullptr) {
        const std::filesystem::path directory =
                std::filesystem::path(module_info.dli_fname).parent_path();
        const std::filesystem::path installed =
                directory / "libgobot_libuipc_solver.so";
        if (std::filesystem::exists(installed)) {
            return installed.string();
        }
        const std::filesystem::path build_tree =
                directory / "python" / "gobot" / "libgobot_libuipc_solver.so";
        if (std::filesystem::exists(build_tree)) {
            return build_tree.string();
        }
        return installed.string();
    }
    return "libgobot_libuipc_solver.so";
}

bool ValidateApi(const IpcSolverModuleApi* api, std::string* error) {
    if (api == nullptr) {
        return SetError(error, "IPC solver module returned a null API table");
    }
    if (api->abi_version != GOBOT_IPC_SOLVER_MODULE_ABI_VERSION) {
        return SetError(error, "IPC solver module ABI does not match this Gobot build");
    }
    if (api->provider_name == nullptr || api->create == nullptr || api->destroy == nullptr ||
        api->step == nullptr || api->reset == nullptr ||
        api->deformable_body_count == nullptr || api->deformable_body_info == nullptr ||
        api->copy_deformable_positions == nullptr ||
        api->copy_deformable_velocities == nullptr ||
        api->copy_deformable_contact_forces == nullptr ||
        api->affine_body_count == nullptr || api->affine_body_info == nullptr ||
        api->copy_affine_transforms == nullptr || api->set_affine_target == nullptr ||
        api->set_joint_target == nullptr ||
        api->diagnostics == nullptr) {
        return SetError(error, "IPC solver module API table is incomplete");
    }
    return true;
}

} // namespace

class IpcSolverSession::Impl final {
public:
    ~Impl() {
        if (api_ != nullptr && session_ != nullptr) {
            api_->destroy(session_);
        }
        if (module_ != nullptr) {
            dlclose(module_);
        }
    }

    bool Initialize(const IpcSceneArtifact& artifact,
                    const IpcSolverConfig& config,
                    const std::string& requested_module_path,
                    std::string* error) {
        module_path_ = requested_module_path.empty() ? DefaultModulePath()
                                                     : requested_module_path;
        module_ = dlopen(module_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (module_ == nullptr) {
            const char* loader_error = dlerror();
            return SetError(error,
                            "Cannot load IPC solver module '" + module_path_ + "': " +
                                    (loader_error != nullptr ? loader_error
                                                             : "unknown loader error"));
        }
        dlerror();
        auto get_api = reinterpret_cast<GetIpcSolverModuleApi>(
                dlsym(module_, "gobot_ipc_solver_get_api"));
        const char* symbol_error = dlerror();
        if (symbol_error != nullptr || get_api == nullptr) {
            return SetError(error,
                            "IPC solver module does not export gobot_ipc_solver_get_api: " +
                                    std::string(symbol_error != nullptr ? symbol_error : "unknown error"));
        }
        api_ = get_api();
        if (!ValidateApi(api_, error)) {
            return false;
        }

        blob_views_.reserve(artifact.blobs.size());
        for (const IpcSceneArtifactBlob& blob : artifact.blobs) {
            blob_views_.push_back(IpcSolverArtifactBlobView{
                    blob.id.c_str(), blob.encoding.c_str(), blob.sha256.c_str(),
                    blob.data.data(), blob.data.size()});
        }
        const IpcSolverArtifactView artifact_view{
                artifact.schema_version,
                artifact.producer.c_str(),
                artifact.producer_version.c_str(),
                artifact.format.c_str(),
                artifact.manifest.data(),
                artifact.manifest.size(),
                artifact.manifest_sha256.c_str(),
                blob_views_.data(),
                blob_views_.size()};
        const IpcSolverModuleConfig module_config{
                config.fixed_time_step,
                {config.gravity[0], config.gravity[1], config.gravity[2]},
                config.friction_coefficient,
                config.contact_activation_distance,
                config.contact_resistance,
                config.affine_stiffness,
                config.kinematic_strength,
                config.device_index,
                config.workspace.empty() ? nullptr : config.workspace.c_str(),
                config.backend_module_directory.empty()
                        ? nullptr
                        : config.backend_module_directory.c_str()};

        std::array<char, kErrorCapacity> module_error{};
        session_ = api_->create(
                &artifact_view, &module_config, module_error.data(), module_error.size());
        if (session_ == nullptr) {
            return SetError(error,
                            module_error[0] != '\0'
                                    ? std::string(module_error.data())
                                    : "IPC solver module could not create a session");
        }
        diagnostics_.provider_name = api_->provider_name;
        if (!DiscoverBodies() || !RefreshState()) {
            return SetError(error, last_error_);
        }
        return true;
    }

    bool DiscoverBodies() {
        deformable_bodies_.clear();
        affine_bodies_.clear();
        const std::size_t deformable_count = api_->deformable_body_count(session_);
        const std::size_t affine_count = api_->affine_body_count(session_);
        std::array<char, kErrorCapacity> error{};
        for (std::size_t index = 0; index < deformable_count; ++index) {
            IpcSolverModuleBodyInfo info;
            if (!api_->deformable_body_info(
                        session_, index, &info, error.data(), error.size()) ||
                info.path == nullptr) {
                return Fail(error.data(), "IPC solver returned invalid deformable body metadata");
            }
            deformable_bodies_.push_back(
                    IpcSolverBodyInfo{info.path, info.element_offset, info.element_count});
        }
        for (std::size_t index = 0; index < affine_count; ++index) {
            IpcSolverModuleBodyInfo info;
            error.fill('\0');
            if (!api_->affine_body_info(
                        session_, index, &info, error.data(), error.size()) ||
                info.path == nullptr) {
                return Fail(error.data(), "IPC solver returned invalid affine body metadata");
            }
            affine_bodies_.push_back(
                    IpcSolverBodyInfo{info.path, info.element_offset, info.element_count});
        }

        std::size_t vertex_count = 0;
        for (const IpcSolverBodyInfo& body : deformable_bodies_) {
            if (body.element_count > std::numeric_limits<std::size_t>::max() - vertex_count) {
                return Fail(nullptr, "IPC solver deformable vertex count overflow");
            }
            vertex_count += body.element_count;
        }
        deformable_positions_.assign(vertex_count * 3, 0.0);
        deformable_velocities_.assign(vertex_count * 3, 0.0);
        deformable_contact_forces_.assign(vertex_count * 3, 0.0);
        affine_transforms_.assign(affine_bodies_.size() * 16, 0.0);
        return true;
    }

    bool RefreshState() {
        std::array<char, kErrorCapacity> error{};
        if (!api_->copy_deformable_positions(
                    session_, deformable_positions_.data(), deformable_positions_.size(),
                    error.data(), error.size())) {
            return Fail(error.data(), "IPC solver could not retrieve deformable positions");
        }
        error.fill('\0');
        if (!api_->copy_deformable_velocities(
                    session_, deformable_velocities_.data(), deformable_velocities_.size(),
                    error.data(), error.size())) {
            return Fail(error.data(), "IPC solver could not retrieve deformable velocities");
        }
        error.fill('\0');
        if (!api_->copy_deformable_contact_forces(
                    session_, deformable_contact_forces_.data(),
                    deformable_contact_forces_.size(), error.data(), error.size())) {
            return Fail(error.data(), "IPC solver could not retrieve deformable contact forces");
        }
        error.fill('\0');
        if (!api_->copy_affine_transforms(
                    session_, affine_transforms_.data(), affine_transforms_.size(),
                    error.data(), error.size())) {
            return Fail(error.data(), "IPC solver could not retrieve affine transforms");
        }
        IpcSolverModuleDiagnostics diagnostics;
        error.fill('\0');
        if (!api_->diagnostics(
                    session_, &diagnostics, error.data(), error.size())) {
            return Fail(error.data(), "IPC solver could not retrieve diagnostics");
        }
        diagnostics_.frame = diagnostics.frame;
        diagnostics_.deformable_body_count = diagnostics.deformable_body_count;
        diagnostics_.deformable_vertex_count = diagnostics.deformable_vertex_count;
        diagnostics_.affine_body_count = diagnostics.affine_body_count;
        diagnostics_.last_step_latency_ms = diagnostics.last_step_latency_ms;
        diagnostics_.valid = diagnostics.valid;
        last_error_.clear();
        return true;
    }

    bool Step(std::uint32_t steps) {
        if (steps == 0) {
            return Fail(nullptr, "IPC solver step count must be positive");
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->step(session_, steps, error.data(), error.size())) {
            return Fail(error.data(), "IPC solver step failed");
        }
        return RefreshState();
    }

    bool Reset() {
        std::array<char, kErrorCapacity> error{};
        if (!api_->reset(session_, error.data(), error.size())) {
            return Fail(error.data(), "IPC solver reset failed");
        }
        return RefreshState();
    }

    bool SetAffineTarget(const std::string& path, const double* transform) {
        if (path.empty() || transform == nullptr) {
            return Fail(nullptr, "IPC affine target requires a path and a 4x4 transform");
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->set_affine_target(
                    session_, path.c_str(), transform, error.data(), error.size())) {
            return Fail(error.data(), "IPC solver rejected affine target");
        }
        last_error_.clear();
        return true;
    }

    bool SetJointTarget(const std::string& path, double position) {
        if (path.empty() || !std::isfinite(position)) {
            return Fail(nullptr, "IPC joint target requires a path and a finite position");
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->set_joint_target(
                    session_, path.c_str(), position, error.data(), error.size())) {
            return Fail(error.data(), "IPC solver rejected joint target");
        }
        last_error_.clear();
        return true;
    }

    bool Fail(const char* module_error, const char* fallback) {
        last_error_ = module_error != nullptr && module_error[0] != '\0'
                              ? module_error
                              : fallback;
        diagnostics_.valid = false;
        return false;
    }

    void* module_{nullptr};
    const IpcSolverModuleApi* api_{nullptr};
    void* session_{nullptr};
    std::string module_path_;
    std::vector<IpcSolverArtifactBlobView> blob_views_;
    std::vector<IpcSolverBodyInfo> deformable_bodies_;
    std::vector<IpcSolverBodyInfo> affine_bodies_;
    std::vector<double> deformable_positions_;
    std::vector<double> deformable_velocities_;
    std::vector<double> deformable_contact_forces_;
    std::vector<double> affine_transforms_;
    IpcSolverDiagnostics diagnostics_;
    std::string last_error_;
};

IpcSolverSession::IpcSolverSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IpcSolverSession::~IpcSolverSession() = default;

std::unique_ptr<IpcSolverSession> IpcSolverSession::Create(
        const IpcSceneArtifact& artifact,
        const IpcSolverConfig& config,
        const std::string& module_path,
        std::string* error) {
    auto impl = std::make_unique<Impl>();
    if (!impl->Initialize(artifact, config, module_path, error)) {
        return nullptr;
    }
    return std::unique_ptr<IpcSolverSession>(
            new IpcSolverSession(std::move(impl)));
}

bool IpcSolverSession::IsModuleAvailable(const std::string& requested_path,
                                         std::string* error) {
    const std::string module_path = requested_path.empty() ? DefaultModulePath()
                                                            : requested_path;
    void* module = dlopen(module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) {
        const char* loader_error = dlerror();
        return SetError(error,
                        "Cannot load IPC solver module '" + module_path + "': " +
                                (loader_error != nullptr ? loader_error : "unknown loader error"));
    }
    dlerror();
    auto get_api = reinterpret_cast<GetIpcSolverModuleApi>(
            dlsym(module, "gobot_ipc_solver_get_api"));
    const char* symbol_error = dlerror();
    bool available = symbol_error == nullptr && get_api != nullptr &&
                     ValidateApi(get_api(), error);
    if (!available && symbol_error != nullptr) {
        SetError(error, symbol_error);
    }
    dlclose(module);
    return available;
}

bool IpcSolverSession::Step(std::uint32_t steps) {
    return impl_->Step(steps);
}

bool IpcSolverSession::Reset() {
    return impl_->Reset();
}

bool IpcSolverSession::SetAffineTarget(
        const std::string& path, const double* transform_row_major_4x4) {
    return impl_->SetAffineTarget(path, transform_row_major_4x4);
}

bool IpcSolverSession::SetJointTarget(
        const std::string& path, double position) {
    return impl_->SetJointTarget(path, position);
}

const std::vector<IpcSolverBodyInfo>& IpcSolverSession::GetDeformableBodies() const {
    return impl_->deformable_bodies_;
}

const std::vector<IpcSolverBodyInfo>& IpcSolverSession::GetAffineBodies() const {
    return impl_->affine_bodies_;
}

const std::vector<double>& IpcSolverSession::GetDeformablePositions() const {
    return impl_->deformable_positions_;
}

const std::vector<double>& IpcSolverSession::GetDeformableVelocities() const {
    return impl_->deformable_velocities_;
}

const std::vector<double>& IpcSolverSession::GetDeformableContactForces() const {
    return impl_->deformable_contact_forces_;
}

const std::vector<double>& IpcSolverSession::GetAffineTransforms() const {
    return impl_->affine_transforms_;
}

const IpcSolverDiagnostics& IpcSolverSession::GetDiagnostics() const {
    return impl_->diagnostics_;
}

const std::string& IpcSolverSession::GetLastError() const {
    return impl_->last_error_;
}

} // namespace gobot
