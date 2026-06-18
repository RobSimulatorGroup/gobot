#include "manual_bindings_internal.hpp"

namespace gobot::python {
namespace {

const PhysicsSensorState& RequiredSensorStateForHandle(const PySensor3DHandle& handle) {
    const Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
    const PhysicsRobotState& robot = RequiredRobotStateForNodeHandle(handle);
    const PhysicsSensorState* sensor_state = FindSensorState(robot, sensor->GetName());
    if (sensor_state == nullptr) {
        throw std::runtime_error("Gobot runtime state robot '" + robot.name +
                                 "' has no sensor '" + sensor->GetName() + "'");
    }
    return *sensor_state;
}

} // namespace

void RegisterManualTerrainSensorBindings(PyTerrain3DClass& terrain3d_class,
                                         PySensor3DClass& sensor3d_class,
                                         PyIMUSensor3DClass& imu_sensor3d_class,
                                         PyAngularMomentumSensor3DClass& angular_momentum_sensor3d_class,
                                         PyContactSensor3DClass& contact_sensor3d_class,
                                         PyRayCastSensor3DClass& raycast_sensor3d_class,
                                         PyTerrainHeightSensor3DClass& terrain_height_sensor3d_class,
                                         PyHeightScanner3DClass& height_scanner3d_class,
                                         PyMeshInstance3DClass& mesh_instance_class) {
    terrain3d_class
            .def("clear_terrain",
                 [](PyTerrain3DHandle& handle) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     ExecuteSetNodeProperty(terrain, "boxes", Variant(std::vector<TerrainBox>{}));
                     ExecuteSetNodeProperty(terrain, "heightfields", Variant(std::vector<TerrainHeightField>{}));
                     ExecuteSetNodeProperty(terrain, "mesh_patches", Variant(std::vector<TerrainMeshPatch>{}));
                     ExecuteSetNodeProperty(terrain, "spawn_origins", Variant(std::vector<Vector3>{}));
                 })
            .def("add_box",
                 [](PyTerrain3DHandle& handle,
                    const py::handle& center,
                    const py::handle& size,
                    const py::handle& rotation_degrees,
                    const py::handle& color) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     std::vector<TerrainBox> boxes = terrain->GetBoxes();
                     TerrainBox box;
                     box.center = PythonToVector3(center);
                     box.size = PythonToVector3(size);
                     box.rotation_degrees = PythonToVector3(rotation_degrees);
                     box.color = PythonToColor4(color);
                     boxes.push_back(box);
                     ExecuteSetNodeProperty(terrain, "boxes", Variant(boxes));
                 },
                 py::arg("center"),
                 py::arg("size"),
                 py::arg("rotation_degrees") = py::make_tuple(0.0, 0.0, 0.0),
                 py::arg("color") = py::make_tuple(1.0, 1.0, 1.0, 1.0))
            .def("add_heightfield",
                 [](PyTerrain3DHandle& handle,
                    const py::handle& center,
                    const py::handle& size,
                    int rows,
                    int cols,
                    const std::vector<RealType>& heights,
                    RealType base_thickness,
                    const std::vector<RealType>& normalized_elevation,
                    RealType z_offset) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     std::vector<TerrainHeightField> heightfields = terrain->GetHeightFields();
                     TerrainHeightField heightfield;
                     heightfield.center = PythonToVector3(center);
                     heightfield.size = PythonToVector2(size);
                     heightfield.rows = rows;
                     heightfield.cols = cols;
                     heightfield.heights = heights;
                     heightfield.normalized_elevation = normalized_elevation;
                     heightfield.base_thickness = base_thickness;
                     heightfield.z_offset = z_offset;
                     heightfields.push_back(std::move(heightfield));
                     ExecuteSetNodeProperty(terrain, "heightfields", Variant(heightfields));
                 },
                 py::arg("center"),
                 py::arg("size"),
                 py::arg("rows"),
                 py::arg("cols"),
                 py::arg("heights"),
                 py::arg("base_thickness") = 0.1,
                 py::arg("normalized_elevation") = std::vector<RealType>{},
                 py::arg("z_offset") = 0.0)
            .def("add_mesh_patch",
                 [](PyTerrain3DHandle& handle,
                    const py::handle& center,
                    const py::handle& vertices,
                    const std::vector<std::uint32_t>& indices,
                    const py::handle& rotation_degrees,
                    const py::handle& color) {
                     Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                     std::vector<TerrainMeshPatch> mesh_patches = terrain->GetMeshPatches();
                     TerrainMeshPatch mesh_patch;
                     mesh_patch.center = PythonToVector3(center);
                     mesh_patch.vertices = PythonToVector3List(vertices);
                     mesh_patch.indices = indices;
                     mesh_patch.rotation_degrees = PythonToVector3(rotation_degrees);
                     mesh_patch.color = PythonToColor4(color);
                     mesh_patches.push_back(std::move(mesh_patch));
                     ExecuteSetNodeProperty(terrain, "mesh_patches", Variant(mesh_patches));
                 },
                 py::arg("center"),
                 py::arg("vertices"),
                 py::arg("indices"),
                 py::arg("rotation_degrees") = py::make_tuple(0.0, 0.0, 0.0),
                 py::arg("color") = py::make_tuple(1.0, 1.0, 1.0, 1.0))
            .def_property_readonly("box_count",
                                   [](const PyTerrain3DHandle& handle) {
                                       return handle.ResolveAs<Terrain3D>()->GetBoxes().size();
                                   })
            .def_property_readonly("heightfield_count",
                                   [](const PyTerrain3DHandle& handle) {
                                       return handle.ResolveAs<Terrain3D>()->GetHeightFields().size();
                                   })
            .def_property_readonly("mesh_patch_count",
                                   [](const PyTerrain3DHandle& handle) {
                                       return handle.ResolveAs<Terrain3D>()->GetMeshPatches().size();
                                   })
            .def("get_heightfield_heights",
                 [](const PyTerrain3DHandle& handle, std::size_t index) {
                     const auto& heightfields = handle.ResolveAs<Terrain3D>()->GetHeightFields();
                     if (index >= heightfields.size()) {
                         throw py::index_error("Terrain3D heightfield index out of range");
                     }
                     return heightfields[index].heights;
                 },
                 py::arg("index"))
            .def_property("spawn_origins",
                          [](const PyTerrain3DHandle& handle) {
                              return Vector3ListToPython(handle.ResolveAs<Terrain3D>()->GetSpawnOrigins());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "spawn_origins", Variant(PythonToVector3List(value)));
                          })
            .def_property("surface_color",
                          [](const PyTerrain3DHandle& handle) {
                              return ColorToPython(handle.ResolveAs<Terrain3D>()->GetSurfaceColor());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "surface_color", Variant(PythonToColor4(value)));
                          })
            .def_property("color_mode",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetColorMode();
                          },
                          [](PyTerrain3DHandle& handle, TerrainColorMode value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "color_mode", Variant(value));
                          })
            .def_property("height_low_color",
                          [](const PyTerrain3DHandle& handle) {
                              return ColorToPython(handle.ResolveAs<Terrain3D>()->GetHeightLowColor());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_low_color", Variant(PythonToColor4(value)));
                          })
            .def_property("height_high_color",
                          [](const PyTerrain3DHandle& handle) {
                              return ColorToPython(handle.ResolveAs<Terrain3D>()->GetHeightHighColor());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_high_color", Variant(PythonToColor4(value)));
                          })
            .def_property("height_range_min",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetHeightRangeMin();
                          },
                          [](PyTerrain3DHandle& handle, RealType value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_range_min", Variant(value));
                          })
            .def_property("height_range_max",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetHeightRangeMax();
                          },
                          [](PyTerrain3DHandle& handle, RealType value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "height_range_max", Variant(value));
                          })
            .def_property("friction",
                          [](const PyTerrain3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Terrain3D>()->GetFriction());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "friction", Variant(PythonToVector3(value)));
                          })
            .def_property("solref",
                          [](const PyTerrain3DHandle& handle) {
                              return Vector2ToPython(handle.ResolveAs<Terrain3D>()->GetSolref());
                          },
                          [](PyTerrain3DHandle& handle, const py::handle& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "solref", Variant(PythonToVector2(value)));
                          })
            .def_property("solimp",
                          [](const PyTerrain3DHandle& handle) {
                              return handle.ResolveAs<Terrain3D>()->GetSolimp();
                          },
                          [](PyTerrain3DHandle& handle, const std::vector<RealType>& value) {
                              Terrain3D* terrain = handle.ResolveAs<Terrain3D>();
                              ExecuteSetNodeProperty(terrain, "solimp", Variant(value));
                          });

    sensor3d_class
            .def_property("enabled",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->IsEnabled();
                          },
                          [](PySensor3DHandle& handle, bool enabled) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "enabled", Variant(enabled));
                          })
            .def_property("sensor_period",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->GetSensorPeriod();
                          },
                          [](PySensor3DHandle& handle, RealType sensor_period) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "sensor_period", Variant(sensor_period));
                          })
            .def_property("noise_stddev",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->GetNoiseStddev();
                          },
                          [](PySensor3DHandle& handle, RealType noise_stddev) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "noise_stddev", Variant(noise_stddev));
                          })
            .def_property("visualize_debug",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->ShouldVisualizeDebug();
                          },
                          [](PySensor3DHandle& handle, bool visualize_debug) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "visualize_debug", Variant(visualize_debug));
                          })
            .def_property("debug_marker_radius",
                          [](const PySensor3DHandle& handle) {
                              return handle.ResolveAs<Sensor3D>()->GetDebugMarkerRadius();
                          },
                          [](PySensor3DHandle& handle, RealType debug_marker_radius) {
                              Sensor3D* sensor = handle.ResolveAs<Sensor3D>();
                              ExecuteSetNodeProperty(sensor, "debug_marker_radius", Variant(debug_marker_radius));
                          })
            .def("get_runtime_state",
                 [](const PySensor3DHandle& handle) {
                     RuntimeSceneForNodeHandle(handle);
                     return SensorStateToPythonDict(RequiredSensorStateForHandle(handle));
                 });

    contact_sensor3d_class
            .def_property("radius",
                          [](const PyContactSensor3DHandle& handle) {
                              return handle.ResolveAs<ContactSensor3D>()->GetRadius();
                          },
                          [](PyContactSensor3DHandle& handle, RealType radius) {
                              ContactSensor3D* sensor = handle.ResolveAs<ContactSensor3D>();
                              ExecuteSetNodeProperty(sensor, "radius", Variant(radius));
                          })
            .def_property("min_threshold",
                          [](const PyContactSensor3DHandle& handle) {
                              return handle.ResolveAs<ContactSensor3D>()->GetMinThreshold();
                          },
                          [](PyContactSensor3DHandle& handle, RealType min_threshold) {
                              ContactSensor3D* sensor = handle.ResolveAs<ContactSensor3D>();
                              ExecuteSetNodeProperty(sensor, "min_threshold", Variant(min_threshold));
                          })
            .def_property("max_threshold",
                          [](const PyContactSensor3DHandle& handle) {
                              return handle.ResolveAs<ContactSensor3D>()->GetMaxThreshold();
                          },
                          [](PyContactSensor3DHandle& handle, RealType max_threshold) {
                              ContactSensor3D* sensor = handle.ResolveAs<ContactSensor3D>();
                              ExecuteSetNodeProperty(sensor, "max_threshold", Variant(max_threshold));
                          });

    raycast_sensor3d_class
            .def_property("sample_offsets",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return Vector3ListToPython(
                                      handle.ResolveAs<RayCastSensor3D>()->GetSampleOffsets());
                          },
                          [](PyRayCastSensor3DHandle& handle, const py::handle& value) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "sample_offsets", Variant(PythonToVector3List(value)));
                          })
            .def_property("ray_direction",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<RayCastSensor3D>()->GetRayDirection());
                          },
                          [](PyRayCastSensor3DHandle& handle, const py::handle& value) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "ray_direction", Variant(PythonToVector3(value)));
                          })
            .def_property("ray_direction_world_space",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return handle.ResolveAs<RayCastSensor3D>()->IsRayDirectionWorldSpace();
                          },
                          [](PyRayCastSensor3DHandle& handle, bool value) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "ray_direction_world_space", Variant(value));
                          })
            .def_property("max_distance",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return handle.ResolveAs<RayCastSensor3D>()->GetMaxDistance();
                          },
                          [](PyRayCastSensor3DHandle& handle, RealType max_distance) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "max_distance", Variant(max_distance));
                          })
            .def_property("pattern_mode",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return handle.ResolveAs<RayCastSensor3D>()->GetPatternMode();
                          },
                          [](PyRayCastSensor3DHandle& handle, RayPatternMode pattern_mode) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "pattern_mode", Variant(pattern_mode));
                          })
            .def_property("grid_size",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return Vector2ToPython(handle.ResolveAs<RayCastSensor3D>()->GetGridSize());
                          },
                          [](PyRayCastSensor3DHandle& handle, const py::handle& value) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "grid_size", Variant(PythonToVector2(value)));
                          })
            .def_property("grid_resolution",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return handle.ResolveAs<RayCastSensor3D>()->GetGridResolution();
                          },
                          [](PyRayCastSensor3DHandle& handle, RealType grid_resolution) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "grid_resolution", Variant(grid_resolution));
                          })
            .def_property("ray_alignment",
                          [](const PyRayCastSensor3DHandle& handle) {
                              return handle.ResolveAs<RayCastSensor3D>()->GetRayAlignment();
                          },
                          [](PyRayCastSensor3DHandle& handle, RayAlignmentMode ray_alignment) {
                              RayCastSensor3D* sensor = handle.ResolveAs<RayCastSensor3D>();
                              ExecuteSetNodeProperty(sensor, "ray_alignment", Variant(ray_alignment));
                          });

    terrain_height_sensor3d_class
            .def_property("reduction_mode",
                          [](const PyTerrainHeightSensor3DHandle& handle) {
                              return handle.ResolveAs<TerrainHeightSensor3D>()->GetReductionMode();
                          },
                          [](PyTerrainHeightSensor3DHandle& handle, RayReductionMode reduction_mode) {
                              TerrainHeightSensor3D* sensor = handle.ResolveAs<TerrainHeightSensor3D>();
                              ExecuteSetNodeProperty(sensor, "reduction_mode", Variant(reduction_mode));
                          });

    GOB_UNUSED(imu_sensor3d_class);
    GOB_UNUSED(angular_momentum_sensor3d_class);
    GOB_UNUSED(height_scanner3d_class);

    mesh_instance_class
            .def_property("surface_color",
                          [](const PyMeshInstance3DHandle& handle) {
                              const Color color = handle.ResolveAs<MeshInstance3D>()->GetSurfaceColor();
                              return py::make_tuple(color.red(), color.green(), color.blue(), color.alpha());
                          },
                          [](PyMeshInstance3DHandle& handle, const py::handle& value) {
                              py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
                              if (sequence.size() != 4) {
                                  throw std::invalid_argument("expected a 4-element RGBA color");
                              }
                              MeshInstance3D* mesh_instance = handle.ResolveAs<MeshInstance3D>();
                              ExecuteSetNodeProperty(mesh_instance, "surface_color", Variant(Color{
                                      py::cast<float>(sequence[0]),
                                      py::cast<float>(sequence[1]),
                                      py::cast<float>(sequence[2]),
                                      py::cast<float>(sequence[3])
                              }));
                          });
}

} // namespace gobot::python
