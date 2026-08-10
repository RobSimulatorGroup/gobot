/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_batch_solver.hpp"

#include <array>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <span>
#include <utility>

namespace gobot {
namespace {

constexpr std::size_t kErrorCapacity = 2048;

void IpcBatchSolverModuleAnchor() {}

bool SetError(std::string* destination, std::string message) {
    if (destination != nullptr) {
        *destination = std::move(message);
    }
    return false;
}

std::string DefaultModulePath() {
    Dl_info module_info{};
    if (dladdr(reinterpret_cast<void*>(&IpcBatchSolverModuleAnchor),
               &module_info) != 0 &&
        module_info.dli_fname != nullptr) {
        const std::filesystem::path directory =
                std::filesystem::path(module_info.dli_fname).parent_path();
        const std::filesystem::path installed =
                directory / "libgobot_libuipc_solver.so";
        if (std::filesystem::exists(installed)) {
            return installed.string();
        }
        const std::filesystem::path build_tree =
                directory / "python" / "gobot" /
                "libgobot_libuipc_solver.so";
        if (std::filesystem::exists(build_tree)) {
            return build_tree.string();
        }
        return installed.string();
    }
    return "libgobot_libuipc_solver.so";
}

bool ValidateApi(const IpcBatchSolverModuleApi* api, std::string* error) {
    if (api == nullptr) {
        return SetError(error,
                        "IPC batch solver module returned a null API table");
    }
    if (api->abi_version != GOBOT_IPC_BATCH_SOLVER_MODULE_ABI_VERSION) {
        return SetError(
                error,
                "IPC batch solver module ABI does not match this Gobot build");
    }
    if (api->provider_name == nullptr || api->create == nullptr ||
        api->destroy == nullptr || api->bind_device_buffers == nullptr ||
        api->step == nullptr || api->reset_full == nullptr ||
        api->synchronize == nullptr ||
        api->deformable_body_count == nullptr ||
        api->deformable_body_info == nullptr ||
        api->affine_body_count == nullptr ||
        api->affine_body_info == nullptr || api->diagnostics == nullptr) {
        return SetError(error,
                        "IPC batch solver module API table is incomplete");
    }
    return true;
}

bool CheckedAdd(std::size_t value,
                std::size_t* total,
                std::string_view description,
                std::string* error) {
    if (value > std::numeric_limits<std::size_t>::max() - *total) {
        return SetError(error, std::string(description) + " count overflow");
    }
    *total += value;
    return true;
}

bool ValidateBuffer(const IpcSolverDeviceBufferView& view,
                    std::span<const std::size_t> expected_shape,
                    std::uint32_t expected_device,
                    std::string_view name,
                    std::string* error) {
    if (view.scalar_type != IpcSolverDeviceScalarType::Float64) {
        return SetError(error,
                        std::string(name) + " must use float64 storage");
    }
    if (view.device_index != expected_device) {
        return SetError(error,
                        std::string(name) + " is on the wrong CUDA device");
    }
    if (view.rank != expected_shape.size() || view.rank > 4) {
        return SetError(error, std::string(name) + " has the wrong rank");
    }
    std::size_t element_count = 1;
    for (std::size_t axis = 0; axis < expected_shape.size(); ++axis) {
        if (view.shape[axis] != expected_shape[axis]) {
            return SetError(error, std::string(name) + " has the wrong shape");
        }
        if (view.shape[axis] != 0 &&
            element_count >
                    std::numeric_limits<std::size_t>::max() /
                            view.shape[axis]) {
            return SetError(error,
                            std::string(name) + " element count overflow");
        }
        element_count *= view.shape[axis];
    }
    // Empty CUDA tensors may use implementation-defined strides. No storage is
    // accessed for them, so shape, dtype, and device validation is sufficient.
    if (element_count == 0) {
        return true;
    }
    std::size_t expected_stride = 1;
    for (std::size_t axis = expected_shape.size(); axis-- > 0;) {
        if (view.stride[axis] != expected_stride) {
            return SetError(error,
                            std::string(name) + " must be contiguous");
        }
        expected_stride *= expected_shape[axis];
    }
    if (view.data == nullptr) {
        return SetError(error,
                        std::string(name) + " has a null device pointer");
    }
    return true;
}

} // namespace

class IpcBatchSolverSession::Impl final {
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
                    const IpcBatchSolverConfig& config,
                    const std::string& requested_module_path,
                    std::string* error) {
        if (config.environment_count == 0 ||
            config.environments_per_shard == 0 ||
            config.environment_count % config.environments_per_shard != 0) {
            return SetError(
                    error,
                    "IPC batch environment count must be a positive multiple "
                    "of environments_per_shard");
        }
        config_ = config;
        module_path_ = requested_module_path.empty() ? DefaultModulePath()
                                                     : requested_module_path;
        module_ = dlopen(module_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (module_ == nullptr) {
            const char* loader_error = dlerror();
            return SetError(
                    error,
                    "Cannot load IPC batch solver module '" + module_path_ +
                            "': " +
                            (loader_error != nullptr ? loader_error
                                                     : "unknown loader error"));
        }
        dlerror();
        auto get_api = reinterpret_cast<GetIpcBatchSolverModuleApi>(
                dlsym(module_, "gobot_ipc_solver_get_batch_api"));
        const char* symbol_error = dlerror();
        if (symbol_error != nullptr || get_api == nullptr) {
            return SetError(
                    error,
                    "IPC solver module does not export "
                    "gobot_ipc_solver_get_batch_api: " +
                            std::string(symbol_error != nullptr
                                                ? symbol_error
                                                : "unknown error"));
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
        const IpcSolverModuleConfig solver_config{
                config.solver.fixed_time_step,
                {config.solver.gravity[0], config.solver.gravity[1],
                 config.solver.gravity[2]},
                config.solver.friction_coefficient,
                config.solver.contact_activation_distance,
                config.solver.contact_resistance,
                config.solver.affine_stiffness,
                config.solver.kinematic_strength,
                config.solver.device_index,
                config.solver.workspace.empty()
                        ? nullptr
                        : config.solver.workspace.c_str(),
                config.solver.backend_module_directory.empty()
                        ? nullptr
                        : config.solver.backend_module_directory.c_str()};
        const IpcBatchSolverModuleConfig module_config{
                solver_config,
                config.environment_count,
                config.environments_per_shard,
                config.external_affine_proxies};

        std::array<char, kErrorCapacity> module_error{};
        session_ = api_->create(&artifact_view, &module_config,
                                module_error.data(), module_error.size());
        if (session_ == nullptr) {
            return SetError(
                    error,
                    module_error[0] != '\0'
                            ? std::string(module_error.data())
                            : "IPC batch solver module could not create a session");
        }
        diagnostics_.provider_name = api_->provider_name;
        if (!DiscoverBodies() || !RefreshDiagnostics()) {
            return SetError(error, last_error_);
        }
        return true;
    }

    bool DiscoverBodies() {
        deformable_bodies_.clear();
        affine_bodies_.clear();
        std::array<char, kErrorCapacity> error{};
        const std::size_t deformable_count =
                api_->deformable_body_count(session_);
        const std::size_t affine_count = api_->affine_body_count(session_);
        for (std::size_t index = 0; index < deformable_count; ++index) {
            IpcSolverModuleBodyInfo info;
            if (!api_->deformable_body_info(session_, index, &info,
                                            error.data(), error.size()) ||
                info.path == nullptr) {
                return Fail(error.data(),
                            "IPC batch solver returned invalid deformable body "
                            "metadata");
            }
            deformable_bodies_.push_back(
                    IpcSolverBodyInfo{info.path, info.element_offset,
                                      info.element_count});
        }
        for (std::size_t index = 0; index < affine_count; ++index) {
            IpcSolverModuleBodyInfo info;
            error.fill('\0');
            if (!api_->affine_body_info(session_, index, &info, error.data(),
                                        error.size()) ||
                info.path == nullptr) {
                return Fail(error.data(),
                            "IPC batch solver returned invalid affine body "
                            "metadata");
            }
            affine_bodies_.push_back(
                    IpcSolverBodyInfo{info.path, info.element_offset,
                                      info.element_count});
        }
        return true;
    }

    bool ValidateBuffers(const IpcBatchSolverModuleBuffers& buffers) {
        std::size_t vertex_count = 0;
        for (const IpcSolverBodyInfo& body : deformable_bodies_) {
            if (!CheckedAdd(body.element_count, &vertex_count,
                            "IPC deformable vertex", &last_error_)) {
                return false;
            }
        }
        const std::size_t environments = config_.environment_count;
        const std::size_t affine_count = affine_bodies_.size();
        const std::array<std::size_t, 3> deformable_shape{
                environments, vertex_count, 3};
        const std::array<std::size_t, 4> affine_shape{
                environments, affine_count, 4, 4};
        const std::array<std::size_t, 3> wrench_shape{
                environments, affine_count, 6};
        const std::uint32_t device = config_.solver.device_index;
        return ValidateBuffer(buffers.deformable_positions,
                              deformable_shape, device,
                              "deformable_positions", &last_error_) &&
               ValidateBuffer(buffers.deformable_velocities,
                              deformable_shape, device,
                              "deformable_velocities", &last_error_) &&
               ValidateBuffer(buffers.deformable_contact_forces,
                              deformable_shape, device,
                              "deformable_contact_forces", &last_error_) &&
               ValidateBuffer(buffers.affine_targets, affine_shape, device,
                              "affine_targets", &last_error_) &&
               ValidateBuffer(buffers.affine_transforms, affine_shape, device,
                              "affine_transforms", &last_error_) &&
               ValidateBuffer(buffers.affine_contact_wrenches, wrench_shape,
                              device, "affine_contact_wrenches", &last_error_);
    }

    bool BindDeviceBuffers(const IpcBatchSolverModuleBuffers& buffers) {
        if (!ValidateBuffers(buffers)) {
            diagnostics_.valid = false;
            return false;
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->bind_device_buffers(session_, &buffers, error.data(),
                                       error.size())) {
            return Fail(error.data(),
                        "IPC batch solver rejected device buffers");
        }
        buffers_bound_ = true;
        return RefreshDiagnostics();
    }

    bool Step(std::uint32_t steps) {
        if (!buffers_bound_) {
            return Fail(nullptr,
                        "IPC batch solver device buffers are not bound");
        }
        if (steps == 0) {
            return Fail(nullptr,
                        "IPC batch solver step count must be positive");
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->step(session_, steps, error.data(), error.size())) {
            return Fail(error.data(), "IPC batch solver step failed");
        }
        return RefreshDiagnostics();
    }

    bool ResetFull() {
        if (!buffers_bound_) {
            return Fail(nullptr,
                        "IPC batch solver device buffers are not bound");
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->reset_full(session_, error.data(), error.size())) {
            return Fail(error.data(), "IPC batch solver full reset failed");
        }
        return RefreshDiagnostics();
    }

    bool Synchronize() {
        if (!buffers_bound_) {
            return Fail(nullptr,
                        "IPC batch solver device buffers are not bound");
        }
        std::array<char, kErrorCapacity> error{};
        if (!api_->synchronize(session_, error.data(), error.size())) {
            return Fail(error.data(),
                        "IPC batch solver synchronization failed");
        }
        last_error_.clear();
        return true;
    }

    bool RefreshDiagnostics() {
        IpcBatchSolverModuleDiagnostics diagnostics;
        std::array<char, kErrorCapacity> error{};
        if (!api_->diagnostics(session_, &diagnostics, error.data(),
                               error.size())) {
            return Fail(error.data(),
                        "IPC batch solver could not retrieve diagnostics");
        }
        diagnostics_.frame = diagnostics.frame;
        diagnostics_.environment_count = diagnostics.environment_count;
        diagnostics_.shard_count = diagnostics.shard_count;
        diagnostics_.deformable_body_count_per_environment =
                diagnostics.deformable_body_count_per_environment;
        diagnostics_.deformable_vertex_count_per_environment =
                diagnostics.deformable_vertex_count_per_environment;
        diagnostics_.affine_body_count_per_environment =
                diagnostics.affine_body_count_per_environment;
        diagnostics_.last_step_latency_ms = diagnostics.last_step_latency_ms;
        diagnostics_.valid = diagnostics.valid;
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
    const IpcBatchSolverModuleApi* api_{nullptr};
    void* session_{nullptr};
    std::string module_path_;
    IpcBatchSolverConfig config_;
    std::vector<IpcSolverArtifactBlobView> blob_views_;
    std::vector<IpcSolverBodyInfo> deformable_bodies_;
    std::vector<IpcSolverBodyInfo> affine_bodies_;
    IpcBatchSolverDiagnostics diagnostics_;
    std::string last_error_;
    bool buffers_bound_{false};
};

IpcBatchSolverSession::IpcBatchSolverSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IpcBatchSolverSession::~IpcBatchSolverSession() = default;

std::unique_ptr<IpcBatchSolverSession> IpcBatchSolverSession::Create(
        const IpcSceneArtifact& artifact,
        const IpcBatchSolverConfig& config,
        const std::string& module_path,
        std::string* error) {
    auto impl = std::make_unique<Impl>();
    if (!impl->Initialize(artifact, config, module_path, error)) {
        return nullptr;
    }
    return std::unique_ptr<IpcBatchSolverSession>(
            new IpcBatchSolverSession(std::move(impl)));
}

bool IpcBatchSolverSession::IsModuleAvailable(
        const std::string& requested_path, std::string* error) {
    const std::string module_path = requested_path.empty()
                                            ? DefaultModulePath()
                                            : requested_path;
    void* module = dlopen(module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) {
        const char* loader_error = dlerror();
        return SetError(
                error,
                "Cannot load IPC batch solver module '" + module_path +
                        "': " +
                        (loader_error != nullptr ? loader_error
                                                 : "unknown loader error"));
    }
    dlerror();
    auto get_api = reinterpret_cast<GetIpcBatchSolverModuleApi>(
            dlsym(module, "gobot_ipc_solver_get_batch_api"));
    const char* symbol_error = dlerror();
    const bool available = symbol_error == nullptr && get_api != nullptr &&
                           ValidateApi(get_api(), error);
    if (!available && symbol_error != nullptr) {
        SetError(error, symbol_error);
    }
    dlclose(module);
    return available;
}

bool IpcBatchSolverSession::BindDeviceBuffers(
        const IpcBatchSolverModuleBuffers& buffers) {
    return impl_->BindDeviceBuffers(buffers);
}

bool IpcBatchSolverSession::Step(std::uint32_t steps) {
    return impl_->Step(steps);
}

bool IpcBatchSolverSession::ResetFull() {
    return impl_->ResetFull();
}

bool IpcBatchSolverSession::Synchronize() {
    return impl_->Synchronize();
}

const std::vector<IpcSolverBodyInfo>&
IpcBatchSolverSession::GetDeformableBodies() const {
    return impl_->deformable_bodies_;
}

const std::vector<IpcSolverBodyInfo>&
IpcBatchSolverSession::GetAffineBodies() const {
    return impl_->affine_bodies_;
}

const IpcBatchSolverDiagnostics&
IpcBatchSolverSession::GetDiagnostics() const {
    return impl_->diagnostics_;
}

const std::string& IpcBatchSolverSession::GetLastError() const {
    return impl_->last_error_;
}

} // namespace gobot
