#include "manual_bindings_internal.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <span>

#include "gobot/physics/ipc_batch_solver.hpp"
#include "gobot/physics/ipc_solver.hpp"

namespace gobot::python {
namespace {

template <typename T>
T Required(const py::dict& value, const char* key) {
    if (!value.contains(key)) {
        throw py::value_error(std::string("IPC artifact is missing '") + key + "'");
    }
    return value[key].cast<T>();
}

IpcSceneArtifact ArtifactFromPython(const py::dict& value) {
    IpcSceneArtifact artifact;
    artifact.schema_version = Required<std::uint32_t>(value, "schema_version");
    artifact.producer = Required<std::string>(value, "producer");
    artifact.producer_version = Required<std::string>(value, "producer_version");
    artifact.format = Required<std::string>(value, "format");
    artifact.manifest = Required<std::string>(value, "manifest");
    artifact.manifest_sha256 = Required<std::string>(value, "manifest_sha256");

    const py::sequence blobs = Required<py::sequence>(value, "blobs");
    artifact.blobs.reserve(blobs.size());
    for (const py::handle item : blobs) {
        const py::dict blob = py::reinterpret_borrow<py::dict>(item);
        IpcSceneArtifactBlob converted;
        converted.id = Required<std::string>(blob, "id");
        converted.encoding = Required<std::string>(blob, "encoding");
        converted.sha256 = Required<std::string>(blob, "sha256");
        const std::string bytes = Required<py::bytes>(blob, "data");
        converted.data.assign(
                reinterpret_cast<const std::uint8_t*>(bytes.data()),
                reinterpret_cast<const std::uint8_t*>(bytes.data() + bytes.size()));
        artifact.blobs.push_back(std::move(converted));
    }
    return artifact;
}

template <typename T>
T ConfigValue(const py::dict& value, const char* key, T fallback) {
    return value.contains(key) ? value[key].cast<T>() : std::move(fallback);
}

IpcSolverConfig ConfigFromPython(const py::dict& value) {
    IpcSolverConfig config;
    config.fixed_time_step = ConfigValue(value, "fixed_time_step", config.fixed_time_step);
    config.friction_coefficient = ConfigValue(
            value, "friction_coefficient", config.friction_coefficient);
    config.contact_activation_distance = ConfigValue(
            value, "contact_activation_distance", config.contact_activation_distance);
    config.contact_resistance = ConfigValue(
            value, "contact_resistance", config.contact_resistance);
    config.affine_stiffness = ConfigValue(
            value, "affine_stiffness", config.affine_stiffness);
    config.kinematic_strength = ConfigValue(
            value, "kinematic_strength", config.kinematic_strength);
    config.device_index = ConfigValue(value, "device_index", config.device_index);
    config.workspace = ConfigValue(value, "workspace", config.workspace);
    config.backend_module_directory = ConfigValue(
            value, "backend_module_directory", config.backend_module_directory);
    if (value.contains("gravity")) {
        const auto gravity = value["gravity"].cast<std::array<double, 3>>();
        std::copy(gravity.begin(), gravity.end(), config.gravity);
    }
    return config;
}

IpcBatchSolverConfig BatchConfigFromPython(const py::dict& value) {
    IpcBatchSolverConfig config;
    config.solver = ConfigFromPython(value);
    config.environment_count = Required<std::uint32_t>(
            value, "environment_count");
    config.environments_per_shard = Required<std::uint32_t>(
            value, "environments_per_shard");
    config.external_affine_proxies = ConfigValue(
            value, "external_affine_proxies", true);
    config.contact_constitution = ConfigValue(
            value, "contact_constitution", config.contact_constitution);
    config.al_ipc_mu_scale_fem = ConfigValue(
            value, "al_ipc_mu_scale_fem", config.al_ipc_mu_scale_fem);
    config.al_ipc_mu_scale_abd = ConfigValue(
            value, "al_ipc_mu_scale_abd", config.al_ipc_mu_scale_abd);
    config.al_ipc_toi_threshold = ConfigValue(
            value, "al_ipc_toi_threshold", config.al_ipc_toi_threshold);
    config.al_ipc_alpha_lower_bound = ConfigValue(
            value, "al_ipc_alpha_lower_bound",
            config.al_ipc_alpha_lower_bound);
    config.al_ipc_decay_factor = ConfigValue(
            value, "al_ipc_decay_factor", config.al_ipc_decay_factor);
    return config;
}

py::list BodyInfoToPython(const std::vector<IpcSolverBodyInfo>& bodies) {
    py::list result;
    for (const IpcSolverBodyInfo& body : bodies) {
        py::dict value;
        value["path"] = body.path;
        value["element_offset"] = body.element_offset;
        value["element_count"] = body.element_count;
        result.append(std::move(value));
    }
    return result;
}

py::dict DiagnosticsToPython(const IpcSolverDiagnostics& diagnostics) {
    py::dict result;
    result["provider_name"] = diagnostics.provider_name;
    result["frame"] = diagnostics.frame;
    result["deformable_body_count"] = diagnostics.deformable_body_count;
    result["deformable_vertex_count"] = diagnostics.deformable_vertex_count;
    result["affine_body_count"] = diagnostics.affine_body_count;
    result["last_step_latency_ms"] = diagnostics.last_step_latency_ms;
    result["valid"] = diagnostics.valid;
    return result;
}

py::dict DiagnosticsToPython(const IpcBatchSolverDiagnostics& diagnostics) {
    py::dict result;
    result["provider_name"] = diagnostics.provider_name;
    result["frame"] = diagnostics.frame;
    result["environment_count"] = diagnostics.environment_count;
    result["shard_count"] = diagnostics.shard_count;
    result["deformable_body_count_per_environment"] =
            diagnostics.deformable_body_count_per_environment;
    result["deformable_vertex_count_per_environment"] =
            diagnostics.deformable_vertex_count_per_environment;
    result["affine_body_count_per_environment"] =
            diagnostics.affine_body_count_per_environment;
    result["static_collider_count_per_environment"] =
            diagnostics.static_collider_count_per_environment;
    result["last_step_latency_ms"] = diagnostics.last_step_latency_ms;
    result["contact_constitution"] = diagnostics.contact_constitution;
    result["exact_contact_wrench"] = diagnostics.exact_contact_wrench;
    result["checkpoint_active"] = diagnostics.checkpoint_active;
    result["valid"] = diagnostics.valid;
    return result;
}

IpcSolverDeviceBufferView DeviceBufferFromPython(
        const py::handle& tensor, const char* name) {
    if (!py::hasattr(tensor, "data_ptr") ||
        !py::hasattr(tensor, "shape") ||
        !py::hasattr(tensor, "stride") ||
        !py::hasattr(tensor, "dtype") ||
        !py::hasattr(tensor, "device")) {
        throw py::type_error(std::string(name) + " must be a Torch tensor");
    }
    const std::string dtype = py::str(tensor.attr("dtype"));
    if (dtype != "torch.float64") {
        throw py::value_error(std::string(name) + " must use torch.float64");
    }
    if (!tensor.attr("is_contiguous")().cast<bool>()) {
        throw py::value_error(std::string(name) + " must be contiguous");
    }
    const py::object device = tensor.attr("device");
    const std::string device_type = py::str(device.attr("type"));
    if (device_type != "cuda") {
        throw py::value_error(std::string(name) + " must be a CUDA tensor");
    }
    const py::tuple shape = py::tuple(tensor.attr("shape"));
    const py::tuple stride = tensor.attr("stride")().cast<py::tuple>();
    if (shape.size() > 4 || stride.size() != shape.size()) {
        throw py::value_error(std::string(name) + " has unsupported dimensions");
    }

    IpcSolverDeviceBufferView result;
    const std::uintptr_t address =
            tensor.attr("data_ptr")().cast<std::uintptr_t>();
    result.data = reinterpret_cast<void*>(address);
    const py::object device_index = device.attr("index");
    if (device_index.is_none()) {
        throw py::value_error(std::string(name) + " has no CUDA device index");
    }
    result.device_index = device_index.cast<std::uint32_t>();
    result.scalar_type = IpcSolverDeviceScalarType::Float64;
    result.rank = static_cast<std::uint32_t>(shape.size());
    for (py::ssize_t axis = 0; axis < shape.size(); ++axis) {
        result.shape[static_cast<std::size_t>(axis)] =
                shape[axis].cast<std::size_t>();
        result.stride[static_cast<std::size_t>(axis)] =
                stride[axis].cast<std::size_t>();
    }
    return result;
}

py::array_t<double> ReadOnlyArray(const std::vector<double>& values,
                                  std::vector<py::ssize_t> shape,
                                  const py::object& owner) {
    std::vector<py::ssize_t> strides(shape.size(), sizeof(double));
    for (std::ptrdiff_t index = static_cast<std::ptrdiff_t>(shape.size()) - 2;
         index >= 0;
         --index) {
        strides[static_cast<std::size_t>(index)] =
                strides[static_cast<std::size_t>(index) + 1] *
                shape[static_cast<std::size_t>(index) + 1];
    }
    py::array_t<double> result(
            std::move(shape), std::move(strides), values.data(), owner);
    result.attr("setflags")(false);
    return result;
}

class PyIpcSolverSession final {
public:
    PyIpcSolverSession(const py::dict& artifact,
                       const py::dict& config,
                       const std::string& module_path) {
        std::string error;
        session_ = IpcSolverSession::Create(
                ArtifactFromPython(artifact), ConfigFromPython(config),
                module_path, &error);
        if (session_ == nullptr) {
            throw std::runtime_error(error.empty()
                                             ? "IPC solver session creation failed"
                                             : error);
        }
    }

    static bool IsModuleAvailable(const std::string& module_path) {
        return IpcSolverSession::IsModuleAvailable(module_path);
    }

    void Step(std::uint32_t steps) {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().Step(steps);
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void Reset() {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().Reset();
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void SetAffineTarget(const std::string& path,
                         py::array_t<double, py::array::c_style | py::array::forcecast> value) {
        if (value.size() != 16 ||
            !((value.ndim() == 2 && value.shape(0) == 4 && value.shape(1) == 4) ||
              value.ndim() == 1)) {
            throw py::value_error("IPC affine target must have shape [4,4] or [16]");
        }
        if (!RequireSession().SetAffineTarget(path, value.data())) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void SetJointTarget(const std::string& path, double position) {
        if (!std::isfinite(position)) {
            throw py::value_error("IPC joint target must be finite");
        }
        if (!RequireSession().SetJointTarget(path, position)) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    py::list DeformableBodies() const {
        return BodyInfoToPython(RequireSession().GetDeformableBodies());
    }

    py::list AffineBodies() const {
        return BodyInfoToPython(RequireSession().GetAffineBodies());
    }

    py::array_t<double> Positions(const py::object& owner) const {
        const auto& values = RequireSession().GetDeformablePositions();
        return ReadOnlyArray(values,
                             {static_cast<py::ssize_t>(values.size() / 3), 3},
                             owner);
    }

    py::array_t<double> Velocities(const py::object& owner) const {
        const auto& values = RequireSession().GetDeformableVelocities();
        return ReadOnlyArray(values,
                             {static_cast<py::ssize_t>(values.size() / 3), 3},
                             owner);
    }

    py::array_t<double> DeformableContactForces(const py::object& owner) const {
        const auto& values = RequireSession().GetDeformableContactForces();
        return ReadOnlyArray(values,
                             {static_cast<py::ssize_t>(values.size() / 3), 3},
                             owner);
    }

    py::array_t<double> AffineTransforms(const py::object& owner) const {
        const auto& values = RequireSession().GetAffineTransforms();
        return ReadOnlyArray(values,
                             {static_cast<py::ssize_t>(values.size() / 16), 4, 4},
                             owner);
    }

    py::dict Diagnostics() const {
        return DiagnosticsToPython(RequireSession().GetDiagnostics());
    }

    void Close() {
        closed_ = true;
    }

private:
    IpcSolverSession& RequireSession() const {
        if (closed_ || session_ == nullptr) {
            throw std::runtime_error("IPC solver session is closed");
        }
        return *session_;
    }

    std::unique_ptr<IpcSolverSession> session_;
    bool closed_{false};
};

class PyIpcBatchSolverSession final {
public:
    PyIpcBatchSolverSession(const py::dict& artifact,
                            const py::dict& config,
                            const std::string& module_path) {
        std::string error;
        session_ = IpcBatchSolverSession::Create(
                ArtifactFromPython(artifact), BatchConfigFromPython(config),
                module_path, &error);
        if (session_ == nullptr) {
            throw std::runtime_error(
                    error.empty() ? "IPC batch solver session creation failed"
                                  : error);
        }
    }

    static bool IsModuleAvailable(const std::string& module_path) {
        return IpcBatchSolverSession::IsModuleAvailable(module_path);
    }

    void BindDeviceBuffers(const py::dict& values) {
        IpcBatchSolverModuleBuffers buffers;
        buffers.deformable_positions = DeviceBufferFromPython(
                values["positions"], "positions");
        buffers.deformable_velocities = DeviceBufferFromPython(
                values["velocities"], "velocities");
        buffers.deformable_contact_forces = DeviceBufferFromPython(
                values["contact_forces"], "contact_forces");
        buffers.affine_targets = DeviceBufferFromPython(
                values["affine_targets"], "affine_targets");
        buffers.affine_target_twists = DeviceBufferFromPython(
                values["affine_target_twists"], "affine_target_twists");
        buffers.affine_transforms = DeviceBufferFromPython(
                values["affine_transforms"], "affine_transforms");
        buffers.affine_contact_wrenches = DeviceBufferFromPython(
                values["affine_contact_wrenches"],
                "affine_contact_wrenches");
        if (!RequireSession().BindDeviceBuffers(buffers)) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void Step(std::uint32_t steps) {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().Step(steps);
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void Reset() {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().ResetFull();
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void CaptureCheckpoint() {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().CaptureCheckpoint();
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void RewindCheckpoint() {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().RewindCheckpoint();
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void CommitCheckpoint() {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().CommitCheckpoint();
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    void Synchronize() {
        bool success = false;
        {
            py::gil_scoped_release release;
            success = RequireSession().Synchronize();
        }
        if (!success) {
            throw std::runtime_error(RequireSession().GetLastError());
        }
    }

    py::list DeformableBodies() const {
        return BodyInfoToPython(RequireSession().GetDeformableBodies());
    }

    py::list AffineBodies() const {
        return BodyInfoToPython(RequireSession().GetAffineBodies());
    }

    py::dict Diagnostics() const {
        return DiagnosticsToPython(RequireSession().GetDiagnostics());
    }

    void Close() { closed_ = true; }

private:
    IpcBatchSolverSession& RequireSession() const {
        if (closed_ || session_ == nullptr) {
            throw std::runtime_error("IPC batch solver session is closed");
        }
        return *session_;
    }

    std::unique_ptr<IpcBatchSolverSession> session_;
    bool closed_{false};
};

} // namespace

void RegisterManualIpcSolverBindings(py::module_& module) {
    py::class_<PyIpcSolverSession>(module, "_IpcSolverSession")
            .def(py::init<const py::dict&, const py::dict&, const std::string&>(),
                 py::arg("artifact"), py::arg("config") = py::dict{},
                 py::arg("module_path") = std::string{})
            .def_static("is_module_available", &PyIpcSolverSession::IsModuleAvailable,
                        py::arg("module_path") = std::string{})
            .def("step", &PyIpcSolverSession::Step, py::arg("steps") = 1)
            .def("reset", &PyIpcSolverSession::Reset)
            .def("set_affine_target", &PyIpcSolverSession::SetAffineTarget,
                 py::arg("path"), py::arg("transform"))
            .def("set_joint_target", &PyIpcSolverSession::SetJointTarget,
                 py::arg("path"), py::arg("position"))
            .def_property_readonly("deformable_bodies",
                                   &PyIpcSolverSession::DeformableBodies)
            .def_property_readonly("affine_bodies", &PyIpcSolverSession::AffineBodies)
            .def_property_readonly(
                    "positions", [](PyIpcSolverSession& session) {
                        py::object owner = py::cast(
                                &session, py::return_value_policy::reference);
                        return session.Positions(owner);
                    })
            .def_property_readonly(
                    "velocities", [](PyIpcSolverSession& session) {
                        py::object owner = py::cast(
                                &session, py::return_value_policy::reference);
                        return session.Velocities(owner);
                    })
            .def_property_readonly(
                    "deformable_contact_forces", [](PyIpcSolverSession& session) {
                        py::object owner = py::cast(
                                &session, py::return_value_policy::reference);
                        return session.DeformableContactForces(owner);
                    })
            .def_property_readonly(
                    "affine_transforms", [](PyIpcSolverSession& session) {
                        py::object owner = py::cast(
                                &session, py::return_value_policy::reference);
                        return session.AffineTransforms(owner);
                    })
            .def_property_readonly("diagnostics", &PyIpcSolverSession::Diagnostics)
            .def("close", &PyIpcSolverSession::Close);

    py::class_<PyIpcBatchSolverSession>(module, "_IpcBatchSolverSession")
            .def(py::init<const py::dict&, const py::dict&,
                          const std::string&>(),
                 py::arg("artifact"), py::arg("config") = py::dict{},
                 py::arg("module_path") = std::string{})
            .def_static("is_module_available",
                        &PyIpcBatchSolverSession::IsModuleAvailable,
                        py::arg("module_path") = std::string{})
            .def("bind_device_buffers",
                 &PyIpcBatchSolverSession::BindDeviceBuffers,
                 py::arg("buffers"))
            .def("step", &PyIpcBatchSolverSession::Step,
                 py::arg("steps") = 1)
            .def("reset", &PyIpcBatchSolverSession::Reset)
            .def("capture_checkpoint",
                 &PyIpcBatchSolverSession::CaptureCheckpoint)
            .def("rewind_checkpoint",
                 &PyIpcBatchSolverSession::RewindCheckpoint)
            .def("commit_checkpoint",
                 &PyIpcBatchSolverSession::CommitCheckpoint)
            .def("synchronize", &PyIpcBatchSolverSession::Synchronize)
            .def_property_readonly("deformable_bodies",
                                   &PyIpcBatchSolverSession::DeformableBodies)
            .def_property_readonly("affine_bodies",
                                   &PyIpcBatchSolverSession::AffineBodies)
            .def_property_readonly("diagnostics",
                                   &PyIpcBatchSolverSession::Diagnostics)
            .def("close", &PyIpcBatchSolverSession::Close);
}

} // namespace gobot::python
