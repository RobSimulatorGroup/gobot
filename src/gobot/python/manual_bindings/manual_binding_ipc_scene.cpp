#include "manual_bindings_internal.hpp"

#include <cstring>
#include <limits>

namespace gobot::python {
namespace {

std::vector<std::uint32_t> PythonToIndexTable(
        const py::handle& value, py::ssize_t width, const char* description) {
    auto array = py::array_t<std::int64_t,
                             py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string("expected ") + description + " integer indices");
    }
    const py::buffer_info info = array.request();
    if (info.ndim != 1 && info.ndim != 2) {
        throw std::invalid_argument(std::string(description) + " indices must be one- or two-dimensional");
    }
    if ((info.ndim == 1 && info.shape[0] % width != 0) ||
        (info.ndim == 2 && info.shape[1] != width)) {
        throw std::invalid_argument(
                std::string(description) + " indices must contain groups of " +
                std::to_string(width));
    }
    const auto* source = static_cast<const std::int64_t*>(info.ptr);
    const std::size_t count = static_cast<std::size_t>(array.size());
    std::vector<std::uint32_t> indices;
    indices.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (source[index] < 0 ||
            static_cast<std::uint64_t>(source[index]) >
                    std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                    std::string(description) + " indices must fit unsigned 32-bit integers");
        }
        indices.push_back(static_cast<std::uint32_t>(source[index]));
    }
    return indices;
}

py::array_t<std::uint32_t> IndexTableToPython(
        const std::vector<std::uint32_t>& indices, py::ssize_t width) {
    py::array_t<std::uint32_t> result(
            {static_cast<py::ssize_t>(indices.size()) / width, width});
    if (!indices.empty()) {
        std::memcpy(result.mutable_data(), indices.data(), indices.size() * sizeof(std::uint32_t));
    }
    return result;
}

std::vector<Vector2> PythonToVector2List(const py::handle& value) {
    if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value)) {
        throw std::invalid_argument("expected a sequence of marker pixel coordinates");
    }
    std::vector<Vector2> result;
    for (const py::handle& item : py::reinterpret_borrow<py::sequence>(value)) {
        result.push_back(PythonToVector2(item));
    }
    return result;
}

py::list Vector2ListToPython(const std::vector<Vector2>& values) {
    py::list result;
    for (const Vector2& value : values) {
        result.append(py::make_tuple(value.x(), value.y()));
    }
    return result;
}

std::vector<Vector4> PythonToVector4List(const py::handle& value) {
    if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value)) {
        throw std::invalid_argument("expected a sequence of marker barycentric weights");
    }
    std::vector<Vector4> result;
    for (const py::handle& item : py::reinterpret_borrow<py::sequence>(value)) {
        const std::vector<double> weights =
                PythonToFixedDoubleArray(item, 4, "four-component vector");
        result.emplace_back(weights[0], weights[1], weights[2], weights[3]);
    }
    return result;
}

py::list Vector4ListToPython(const std::vector<Vector4>& values) {
    py::list result;
    for (const Vector4& value : values) {
        result.append(py::make_tuple(value.x(), value.y(), value.z(), value.w()));
    }
    return result;
}

template <typename T>
void SetNodeValue(PyNodeHandle& handle, const char* property, T value) {
    ExecuteSetNodeProperty(handle.Resolve(), property, Variant(std::move(value)));
}

} // namespace

void RegisterManualIpcSceneBindings(
        PyTetrahedralMeshClass& tetrahedral_mesh_class,
        PyTactileSensorConfigClass& tactile_config_class,
        PyPhysicsCouplingClass& physics_coupling_class,
        PyDeformableAttachment3DClass& deformable_attachment_class,
        PyDeformableBody3DClass& deformable_body_class,
        PyTactileSensor3DClass& tactile_sensor_class) {
    physics_coupling_class
            .def_property(
                    "enabled",
                    [](const PyPhysicsCouplingHandle& handle) {
                        return handle.ResolveAs<PhysicsCoupling>()->IsEnabled();
                    },
                    [](PyPhysicsCouplingHandle& handle, bool value) {
                        SetNodeValue(handle, "enabled", value);
                    })
            .def_property(
                    "rigid_link_path",
                    [](const PyPhysicsCouplingHandle& handle) {
                        return static_cast<std::string>(
                                handle.ResolveAs<PhysicsCoupling>()->GetRigidLinkPath());
                    },
                    [](PyPhysicsCouplingHandle& handle, const std::string& value) {
                        SetNodeValue(handle, "rigid_link_path", NodePath(value));
                    })
            .def_property(
                    "mode",
                    [](const PyPhysicsCouplingHandle& handle) {
                        return handle.ResolveAs<PhysicsCoupling>()->GetMode();
                    },
                    [](PyPhysicsCouplingHandle& handle, PhysicsCouplingMode value) {
                        SetNodeValue(handle, "mode", value);
                    })
            .def_property(
                    "force_scale",
                    [](const PyPhysicsCouplingHandle& handle) {
                        return handle.ResolveAs<PhysicsCoupling>()->GetForceScale();
                    },
                    [](PyPhysicsCouplingHandle& handle, RealType value) {
                        SetNodeValue(handle, "force_scale", value);
                    })
            .def_property(
                    "torque_scale",
                    [](const PyPhysicsCouplingHandle& handle) {
                        return handle.ResolveAs<PhysicsCoupling>()->GetTorqueScale();
                    },
                    [](PyPhysicsCouplingHandle& handle, RealType value) {
                        SetNodeValue(handle, "torque_scale", value);
                    });

    deformable_attachment_class
            .def_property(
                    "enabled",
                    [](const PyDeformableAttachment3DHandle& handle) {
                        return handle.ResolveAs<DeformableAttachment3D>()->IsEnabled();
                    },
                    [](PyDeformableAttachment3DHandle& handle, bool value) {
                        SetNodeValue(handle, "enabled", value);
                    })
            .def_property(
                    "deformable_body_path",
                    [](const PyDeformableAttachment3DHandle& handle) {
                        return static_cast<std::string>(
                                handle.ResolveAs<DeformableAttachment3D>()
                                        ->GetDeformableBodyPath());
                    },
                    [](PyDeformableAttachment3DHandle& handle,
                       const std::string& value) {
                        SetNodeValue(handle, "deformable_body_path", NodePath(value));
                    })
            .def_property(
                    "rigid_link_path",
                    [](const PyDeformableAttachment3DHandle& handle) {
                        return static_cast<std::string>(
                                handle.ResolveAs<DeformableAttachment3D>()
                                        ->GetRigidLinkPath());
                    },
                    [](PyDeformableAttachment3DHandle& handle,
                       const std::string& value) {
                        SetNodeValue(handle, "rigid_link_path", NodePath(value));
                    })
            .def_property(
                    "vertex_indices",
                    [](const PyDeformableAttachment3DHandle& handle) {
                        return handle.ResolveAs<DeformableAttachment3D>()
                                ->GetVertexIndices();
                    },
                    [](PyDeformableAttachment3DHandle& handle,
                       const std::vector<std::uint32_t>& value) {
                        SetNodeValue(handle, "vertex_indices", value);
                    })
            .def_property(
                    "strength_rate",
                    [](const PyDeformableAttachment3DHandle& handle) {
                        return handle.ResolveAs<DeformableAttachment3D>()
                                ->GetStrengthRate();
                    },
                    [](PyDeformableAttachment3DHandle& handle, RealType value) {
                        SetNodeValue(handle, "strength_rate", value);
                    });

    tetrahedral_mesh_class
            .def(py::init<>())
            .def_property(
                    "vertices",
                    [](const PyTetrahedralMesh& mesh) {
                        return Vector3ListToPython(mesh.resource->GetVertices());
                    },
                    [](PyTetrahedralMesh& mesh, const py::handle& value) {
                        mesh.resource->SetVertices(PythonToVector3List(value));
                    })
            .def_property(
                    "tetrahedra",
                    [](const PyTetrahedralMesh& mesh) {
                        return IndexTableToPython(mesh.resource->GetTetrahedra(), 4);
                    },
                    [](PyTetrahedralMesh& mesh, const py::handle& value) {
                        mesh.resource->SetTetrahedra(
                                PythonToIndexTable(value, 4, "tetrahedron"));
                    })
            .def_property(
                    "surface_triangles",
                    [](const PyTetrahedralMesh& mesh) {
                        return IndexTableToPython(mesh.resource->GetSurfaceTriangles(), 3);
                    },
                    [](PyTetrahedralMesh& mesh, const py::handle& value) {
                        mesh.resource->SetSurfaceTriangles(
                                PythonToIndexTable(value, 3, "surface triangle"));
                    })
            .def_property_readonly("vertex_count", [](const PyTetrahedralMesh& mesh) {
                return mesh.resource->GetVertexCount();
            })
            .def_property_readonly("tetrahedron_count", [](const PyTetrahedralMesh& mesh) {
                return mesh.resource->GetTetrahedronCount();
            })
            .def("validate", [](const PyTetrahedralMesh& mesh) {
                std::string error;
                if (!mesh.resource->Validate(&error)) {
                    throw py::value_error(error);
                }
            });

    tactile_config_class
            .def(py::init<>())
            .def_property(
                    "image_width",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetImageWidth();
                    },
                    [](PyTactileSensorConfig& config, std::uint32_t value) {
                        config.resource->SetImageWidth(value);
                    })
            .def_property(
                    "image_height",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetImageHeight();
                    },
                    [](PyTactileSensorConfig& config, std::uint32_t value) {
                        config.resource->SetImageHeight(value);
                    })
            .def_property(
                    "near_plane",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetNearPlane();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetNearPlane(value);
                    })
            .def_property(
                    "far_plane",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetFarPlane();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetFarPlane(value);
                    })
            .def_property(
                    "pixel_size",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetPixelSize();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetPixelSize(value);
                    })
            .def_property(
                    "density",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetDensity();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetDensity(value);
                    })
            .def_property(
                    "young_modulus",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetYoungModulus();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetYoungModulus(value);
                    })
            .def_property(
                    "poisson_ratio",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetPoissonRatio();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetPoissonRatio(value);
                    })
            .def_property(
                    "damping",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetDamping();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetDamping(value);
                    })
            .def_property(
                    "friction_coefficient",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetFrictionCoefficient();
                    },
                    [](PyTactileSensorConfig& config, RealType value) {
                        config.resource->SetFrictionCoefficient(value);
                    })
            .def_property(
                    "coat_vertex_indices",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetCoatVertexIndices();
                    },
                    [](PyTactileSensorConfig& config,
                       const std::vector<std::uint32_t>& value) {
                        config.resource->SetCoatVertexIndices(value);
                    })
            .def_property(
                    "stick_vertex_indices",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetStickVertexIndices();
                    },
                    [](PyTactileSensorConfig& config,
                       const std::vector<std::uint32_t>& value) {
                        config.resource->SetStickVertexIndices(value);
                    })
            .def_property(
                    "marker_positions",
                    [](const PyTactileSensorConfig& config) {
                        return Vector2ListToPython(config.resource->GetMarkerPositions());
                    },
                    [](PyTactileSensorConfig& config, const py::handle& value) {
                        config.resource->SetMarkerPositions(PythonToVector2List(value));
                    })
            .def_property(
                    "marker_tetrahedra",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetMarkerTetrahedra();
                    },
                    [](PyTactileSensorConfig& config,
                       const std::vector<std::uint32_t>& value) {
                        config.resource->SetMarkerTetrahedra(value);
                    })
            .def_property(
                    "marker_barycentric",
                    [](const PyTactileSensorConfig& config) {
                        return Vector4ListToPython(config.resource->GetMarkerBarycentric());
                    },
                    [](PyTactileSensorConfig& config, const py::handle& value) {
                        config.resource->SetMarkerBarycentric(PythonToVector4List(value));
                    })
            .def_property(
                    "rgb_model",
                    [](const PyTactileSensorConfig& config) {
                        return config.resource->GetRgbModel();
                    },
                    [](PyTactileSensorConfig& config, const std::string& value) {
                        config.resource->SetRgbModel(value);
                    });

    deformable_body_class
            .def_property(
                    "mesh",
                    [](const PyDeformableBody3DHandle& handle) -> py::object {
                        const Ref<TetrahedralMesh>& mesh =
                                handle.ResolveAs<DeformableBody3D>()->GetMesh();
                        return mesh.IsValid() ? py::cast(PyTetrahedralMesh(mesh)) : py::none();
                    },
                    [](PyDeformableBody3DHandle& handle, const PyTetrahedralMesh& mesh) {
                        SetNodeValue(handle, "mesh", mesh.resource);
                    })
            .def_property(
                    "density",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->GetDensity();
                    },
                    [](PyDeformableBody3DHandle& handle, RealType value) {
                        SetNodeValue(handle, "density", value);
                    })
            .def_property(
                    "young_modulus",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->GetYoungModulus();
                    },
                    [](PyDeformableBody3DHandle& handle, RealType value) {
                        SetNodeValue(handle, "young_modulus", value);
                    })
            .def_property(
                    "poisson_ratio",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->GetPoissonRatio();
                    },
                    [](PyDeformableBody3DHandle& handle, RealType value) {
                        SetNodeValue(handle, "poisson_ratio", value);
                    })
            .def_property(
                    "damping",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->GetDamping();
                    },
                    [](PyDeformableBody3DHandle& handle, RealType value) {
                        SetNodeValue(handle, "damping", value);
                    })
            .def_property(
                    "kinematic",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->IsKinematic();
                    },
                    [](PyDeformableBody3DHandle& handle, bool value) {
                        SetNodeValue(handle, "kinematic", value);
                    })
            .def_property(
                    "collision_layer",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->GetCollisionLayer();
                    },
                    [](PyDeformableBody3DHandle& handle, std::uint32_t value) {
                        SetNodeValue(handle, "collision_layer", value);
                    })
            .def_property(
                    "collision_mask",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->GetCollisionMask();
                    },
                    [](PyDeformableBody3DHandle& handle, std::uint32_t value) {
                        SetNodeValue(handle, "collision_mask", value);
                    })
            .def_property(
                    "self_collision_enabled",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->IsSelfCollisionEnabled();
                    },
                    [](PyDeformableBody3DHandle& handle, bool value) {
                        SetNodeValue(handle, "self_collision_enabled", value);
                    })
            .def_property(
                    "debug_surface_color",
                    [](const PyDeformableBody3DHandle& handle) {
                        return ColorToPython(
                                handle.ResolveAs<DeformableBody3D>()->GetDebugSurfaceColor());
                    },
                    [](PyDeformableBody3DHandle& handle, const py::handle& value) {
                        SetNodeValue(handle, "debug_surface_color", PythonToColor4(value));
                    })
            .def_property(
                    "debug_wireframe_visible",
                    [](const PyDeformableBody3DHandle& handle) {
                        return handle.ResolveAs<DeformableBody3D>()->IsDebugWireframeVisible();
                    },
                    [](PyDeformableBody3DHandle& handle, bool value) {
                        SetNodeValue(handle, "debug_wireframe_visible", value);
                    });

    tactile_sensor_class
            .def_property(
                    "config",
                    [](const PyTactileSensor3DHandle& handle) -> py::object {
                        const Ref<TactileSensorConfig>& config =
                                handle.ResolveAs<TactileSensor3D>()->GetConfig();
                        return config.IsValid() ? py::cast(PyTactileSensorConfig(config)) : py::none();
                    },
                    [](PyTactileSensor3DHandle& handle, const PyTactileSensorConfig& config) {
                        SetNodeValue(handle, "config", config.resource);
                    })
            .def_property(
                    "gel_mesh",
                    [](const PyTactileSensor3DHandle& handle) -> py::object {
                        const Ref<TetrahedralMesh>& mesh =
                                handle.ResolveAs<TactileSensor3D>()->GetGelMesh();
                        return mesh.IsValid() ? py::cast(PyTetrahedralMesh(mesh)) : py::none();
                    },
                    [](PyTactileSensor3DHandle& handle, const PyTetrahedralMesh& mesh) {
                        SetNodeValue(handle, "gel_mesh", mesh.resource);
                    })
            .def_property(
                    "collision_layer",
                    [](const PyTactileSensor3DHandle& handle) {
                        return handle.ResolveAs<TactileSensor3D>()->GetCollisionLayer();
                    },
                    [](PyTactileSensor3DHandle& handle, std::uint32_t value) {
                        SetNodeValue(handle, "collision_layer", value);
                    })
            .def_property(
                    "collision_mask",
                    [](const PyTactileSensor3DHandle& handle) {
                        return handle.ResolveAs<TactileSensor3D>()->GetCollisionMask();
                    },
                    [](PyTactileSensor3DHandle& handle, std::uint32_t value) {
                        SetNodeValue(handle, "collision_mask", value);
                    });
}

} // namespace gobot::python
