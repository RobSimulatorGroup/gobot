#include "manual_bindings_internal.hpp"

namespace gobot::python {

void RegisterManualApis(py::module_& module) {
    RegisterManualCommonBindings(module);

    auto node_class = py::class_<PyNodeHandle>(module, "Node");
    auto node3d_class = py::class_<PyNode3DHandle, PyNodeHandle>(module, "Node3D");
    auto robot3d_class = py::class_<PyRobot3DHandle, PyNode3DHandle>(module, "Robot3D");
    auto link3d_class = py::class_<PyLink3DHandle, PyNode3DHandle>(module, "Link3D");
    auto joint3d_class = py::class_<PyJoint3DHandle, PyNode3DHandle>(module, "Joint3D");
    auto collision_shape_class =
            py::class_<PyCollisionShape3DHandle, PyNode3DHandle>(module, "CollisionShape3D");
    auto mesh_instance_class =
            py::class_<PyMeshInstance3DHandle, PyNode3DHandle>(module, "MeshInstance3D");
    auto terrain3d_class = py::class_<PyTerrain3DHandle, PyNode3DHandle>(module, "Terrain3D");
    auto sensor3d_class = py::class_<PySensor3DHandle, PyNode3DHandle>(module, "Sensor3D");
    auto imu_sensor3d_class = py::class_<PyIMUSensor3DHandle, PySensor3DHandle>(module, "IMUSensor3D");
    auto angular_momentum_sensor3d_class =
            py::class_<PyAngularMomentumSensor3DHandle, PySensor3DHandle>(module, "AngularMomentumSensor3D");
    auto contact_sensor3d_class =
            py::class_<PyContactSensor3DHandle, PySensor3DHandle>(module, "ContactSensor3D");
    auto raycast_sensor3d_class =
            py::class_<PyRayCastSensor3DHandle, PySensor3DHandle>(module, "RayCastSensor3D");
    auto terrain_height_sensor3d_class =
            py::class_<PyTerrainHeightSensor3DHandle, PyRayCastSensor3DHandle>(module, "TerrainHeightSensor3D");
    auto height_scanner3d_class =
            py::class_<PyHeightScanner3DHandle, PyTerrainHeightSensor3DHandle>(module, "HeightScanner3D");
    auto velocity_command_debug3d_class =
            py::class_<PyVelocityCommandDebug3DHandle, PyNode3DHandle>(module, "VelocityCommandDebug3D");

    py::implicitly_convertible<PyRobot3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyLink3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyJoint3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyCollisionShape3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyMeshInstance3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PySensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyIMUSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyAngularMomentumSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyContactSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyRayCastSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyTerrainHeightSensor3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyHeightScanner3DHandle, PyNodeHandle>();
    py::implicitly_convertible<PyVelocityCommandDebug3DHandle, PyNodeHandle>();

    GOB_UNUSED(velocity_command_debug3d_class);

    RegisterManualAppContextBindings(module);
    RegisterManualNodeBindings(node_class, node3d_class);
    RegisterManualRobotBindings(robot3d_class, link3d_class, joint3d_class, collision_shape_class);
    RegisterManualTerrainSensorBindings(terrain3d_class,
                                        sensor3d_class,
                                        imu_sensor3d_class,
                                        angular_momentum_sensor3d_class,
                                        contact_sensor3d_class,
                                        raycast_sensor3d_class,
                                        terrain_height_sensor3d_class,
                                        height_scanner3d_class,
                                        mesh_instance_class);
    RegisterManualModuleFunctions(module);
}

} // namespace gobot::python
