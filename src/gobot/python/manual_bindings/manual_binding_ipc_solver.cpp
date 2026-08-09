#include "manual_bindings_internal.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <span>

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
}

} // namespace gobot::python
