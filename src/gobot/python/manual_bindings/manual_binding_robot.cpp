#include "manual_bindings_internal.hpp"

namespace gobot::python {
namespace {

const std::string& RuntimeRobotNameForHandle(const PyNodeHandle& handle) {
    return RuntimeRobotForNodeHandle(handle)->GetName();
}

const PhysicsLinkState& RequiredLinkStateForHandle(const PyLink3DHandle& handle) {
    const Link3D* link = handle.ResolveAs<Link3D>();
    const PhysicsRobotState& robot = RequiredRobotStateForNodeHandle(handle);
    const PhysicsLinkState* link_state = FindLinkState(robot, link->GetName());
    if (link_state == nullptr) {
        throw std::runtime_error("Gobot runtime state robot '" + robot.name +
                                 "' has no link '" + link->GetName() + "'");
    }
    return *link_state;
}

const PhysicsJointState& RequiredJointStateForHandle(const PyJoint3DHandle& handle) {
    const Joint3D* joint = handle.ResolveAs<Joint3D>();
    const PhysicsRobotState& robot = RequiredRobotStateForNodeHandle(handle);
    const PhysicsJointState* joint_state = FindJointState(robot, joint->GetName());
    if (joint_state == nullptr) {
        throw std::runtime_error("Gobot runtime state robot '" + robot.name +
                                 "' has no joint '" + joint->GetName() + "'");
    }
    return *joint_state;
}

} // namespace

void RegisterManualRobotBindings(PyRobot3DClass& robot3d_class,
                                 PyLink3DClass& link3d_class,
                                 PyJoint3DClass& joint3d_class,
                                 PyCollisionShape3DClass& collision_shape_class) {
    robot3d_class
            .def_property("source_path",
                          [](const PyRobot3DHandle& handle) {
                              return handle.ResolveAs<Robot3D>()->GetSourcePath();
                          },
                          [](PyRobot3DHandle& handle, const std::string& source_path) {
                              Robot3D* robot = handle.ResolveAs<Robot3D>();
                              ExecuteSetNodeProperty(robot, "source_path", Variant(source_path));
                          })
            .def_property("mode",
                          [](const PyRobot3DHandle& handle) {
                              return handle.ResolveAs<Robot3D>()->GetMode();
                          },
                          [](PyRobot3DHandle& handle, RobotMode mode) {
                              Robot3D* robot = handle.ResolveAs<Robot3D>();
                              ExecuteSetNodeProperty(robot, "mode", Variant(mode));
                          });

    link3d_class
            .def_property("has_inertial",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->HasInertial();
                          },
                          [](PyLink3DHandle& handle, bool has_inertial) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "has_inertial", Variant(has_inertial));
                          })
            .def_property("mass",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->GetMass();
                          },
                          [](PyLink3DHandle& handle, RealType mass) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "mass", Variant(mass));
                          })
            .def_property("center_of_mass",
                          [](const PyLink3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Link3D>()->GetCenterOfMass());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "center_of_mass", Variant(PythonToVector3(value)));
                          })
            .def_property("inertia_orientation",
                          [](const PyLink3DHandle& handle) {
                              return QuaternionWxyzToPython(handle.ResolveAs<Link3D>()->GetInertiaOrientation());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "inertia_orientation", Variant(PythonToQuaternionWxyz(value)));
                          })
            .def_property("inertia_diagonal",
                          [](const PyLink3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Link3D>()->GetInertiaDiagonal());
                          },
                          [](PyLink3DHandle& handle, const py::handle& value) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "inertia_diagonal", Variant(PythonToVector3(value)));
                          })
            .def_property("role",
                          [](const PyLink3DHandle& handle) {
                              return handle.ResolveAs<Link3D>()->GetRole();
                          },
                          [](PyLink3DHandle& handle, LinkRole role) {
                              Link3D* link = handle.ResolveAs<Link3D>();
                              ExecuteSetNodeProperty(link, "role", Variant(role));
                          })
            .def("reset_runtime_state",
                 [](PyLink3DHandle& handle,
                    const py::object& position,
                    const py::object& orientation,
                    const py::object& linear_velocity,
                    const py::object& angular_velocity) {
                     Link3D* link = handle.ResolveAs<Link3D>();
                     SimulationScene* runtime_scene = RuntimeSceneForNodeHandle(handle);
                     if (!runtime_scene->ResetLinkState(RuntimeRobotNameForHandle(handle),
                                                        link->GetName(),
                                                        PythonToVector3(position),
                                                        PythonToQuaternionWxyz(orientation),
                                                        PythonToVector3(linear_velocity),
                                                        PythonToVector3(angular_velocity))) {
                         throw std::runtime_error(runtime_scene->GetLastError());
                     }
                 },
                 py::arg("position"),
                 py::arg("orientation") = py::make_tuple(1.0, 0.0, 0.0, 0.0),
                 py::arg("linear_velocity") = py::make_tuple(0.0, 0.0, 0.0),
                 py::arg("angular_velocity") = py::make_tuple(0.0, 0.0, 0.0))
            .def("get_runtime_state",
                 [](const PyLink3DHandle& handle) {
                     RuntimeSceneForNodeHandle(handle);
                     return LinkStateToPythonDict(RequiredLinkStateForHandle(handle));
                 });

    joint3d_class
            .def_property("joint_type",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetJointType();
                          },
                          [](PyJoint3DHandle& handle, JointType joint_type) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "joint_type", Variant(joint_type));
                          })
            .def_property("parent_link",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetParentLink();
                          },
                          [](PyJoint3DHandle& handle, const std::string& parent_link) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "parent_link", Variant(parent_link));
                          })
            .def_property("child_link",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetChildLink();
                          },
                          [](PyJoint3DHandle& handle, const std::string& child_link) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "child_link", Variant(child_link));
                          })
            .def_property("axis",
                          [](const PyJoint3DHandle& handle) {
                              return Vector3ToPython(handle.ResolveAs<Joint3D>()->GetAxis());
                          },
                          [](PyJoint3DHandle& handle, const py::handle& value) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "axis", Variant(PythonToVector3(value)));
                          })
            .def_property("lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "lower_limit", Variant(lower_limit));
                          })
            .def_property("upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "upper_limit", Variant(upper_limit));
                          })
            .def_property("effort_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetEffortLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType effort_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "effort_limit", Variant(effort_limit));
                          })
            .def_property("velocity_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetVelocityLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType velocity_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "velocity_limit", Variant(velocity_limit));
                          })
            .def_property("damping",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDamping();
                          },
                          [](PyJoint3DHandle& handle, RealType damping) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "damping", Variant(damping));
                          })
            .def_property("armature",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetArmature();
                          },
                          [](PyJoint3DHandle& handle, RealType armature) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "armature", Variant(armature));
                          })
            .def_property("friction_loss",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetFrictionLoss();
                          },
                          [](PyJoint3DHandle& handle, RealType friction_loss) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "friction_loss", Variant(friction_loss));
                          })
            .def_property("joint_position",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetJointPosition();
                          },
                          [](PyJoint3DHandle& handle, RealType joint_position) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "joint_position", Variant(joint_position));
                          })
            .def_property("initial_position",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetInitialPosition();
                          },
                          [](PyJoint3DHandle& handle, RealType initial_position) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "initial_position", Variant(initial_position));
                          })
            .def_property("drive_mode",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDriveMode();
                          },
                          [](PyJoint3DHandle& handle, JointDriveMode drive_mode) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "drive_mode", Variant(drive_mode));
                          })
            .def_property("drive_stiffness",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDriveStiffness();
                          },
                          [](PyJoint3DHandle& handle, RealType stiffness) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "drive_stiffness", Variant(stiffness));
                          })
            .def_property("drive_damping",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetDriveDamping();
                          },
                          [](PyJoint3DHandle& handle, RealType damping) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "drive_damping", Variant(damping));
                          })
            .def_property("control_lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetControlLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "control_lower_limit", Variant(lower_limit));
                          })
            .def_property("control_upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetControlUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "control_upper_limit", Variant(upper_limit));
                          })
            .def_property("force_lower_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetForceLowerLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType lower_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "force_lower_limit", Variant(lower_limit));
                          })
            .def_property("force_upper_limit",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetForceUpperLimit();
                          },
                          [](PyJoint3DHandle& handle, RealType upper_limit) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "force_upper_limit", Variant(upper_limit));
                          })
            .def_property("gear",
                          [](const PyJoint3DHandle& handle) {
                              return handle.ResolveAs<Joint3D>()->GetGear();
                          },
                          [](PyJoint3DHandle& handle, const std::vector<RealType>& gear) {
                              Joint3D* joint = handle.ResolveAs<Joint3D>();
                              ExecuteSetNodeProperty(joint, "gear", Variant(gear));
                          })
            .def("set_position_target",
                 [](PyJoint3DHandle& handle, RealType target_position) {
                     Joint3D* joint = handle.ResolveAs<Joint3D>();
                     SimulationScene* runtime_scene = RuntimeSceneForNodeHandle(handle);
                     if (!runtime_scene->SetJointPositionTarget(RuntimeRobotNameForHandle(handle),
                                                                joint->GetName(),
                                                                target_position)) {
                         throw std::runtime_error(runtime_scene->GetLastError());
                     }
                 },
                 py::arg("target"))
            .def("set_velocity_target",
                 [](PyJoint3DHandle& handle, RealType target_velocity) {
                     Joint3D* joint = handle.ResolveAs<Joint3D>();
                     SimulationScene* runtime_scene = RuntimeSceneForNodeHandle(handle);
                     if (!runtime_scene->SetJointVelocityTarget(RuntimeRobotNameForHandle(handle),
                                                                joint->GetName(),
                                                                target_velocity)) {
                         throw std::runtime_error(runtime_scene->GetLastError());
                     }
                 },
                 py::arg("target"))
            .def("set_effort_target",
                 [](PyJoint3DHandle& handle, RealType target_effort) {
                     Joint3D* joint = handle.ResolveAs<Joint3D>();
                     SimulationScene* runtime_scene = RuntimeSceneForNodeHandle(handle);
                     if (!runtime_scene->SetJointEffortTarget(RuntimeRobotNameForHandle(handle),
                                                              joint->GetName(),
                                                              target_effort)) {
                         throw std::runtime_error(runtime_scene->GetLastError());
                     }
                 },
                 py::arg("target"))
            .def("set_passive",
                 [](PyJoint3DHandle& handle) {
                     Joint3D* joint = handle.ResolveAs<Joint3D>();
                     SimulationScene* runtime_scene = RuntimeSceneForNodeHandle(handle);
                     if (!runtime_scene->SetJointPassive(RuntimeRobotNameForHandle(handle), joint->GetName())) {
                         throw std::runtime_error(runtime_scene->GetLastError());
                     }
                 })
            .def("reset_runtime_state",
                 [](PyJoint3DHandle& handle, RealType position, RealType velocity) {
                     Joint3D* joint = handle.ResolveAs<Joint3D>();
                     SimulationScene* runtime_scene = RuntimeSceneForNodeHandle(handle);
                     if (!runtime_scene->ResetJointState(RuntimeRobotNameForHandle(handle),
                                                         joint->GetName(),
                                                         position,
                                                         velocity)) {
                         throw std::runtime_error(runtime_scene->GetLastError());
                     }
                 },
                 py::arg("position"),
                 py::arg("velocity") = 0.0)
            .def("get_runtime_state",
                 [](const PyJoint3DHandle& handle) {
                     RuntimeSceneForNodeHandle(handle);
                     return JointStateToPythonDict(RequiredJointStateForHandle(handle));
                 });

    collision_shape_class
            .def_property("disabled",
                          [](const PyCollisionShape3DHandle& handle) {
                              return handle.ResolveAs<CollisionShape3D>()->IsDisabled();
                          },
                          [](PyCollisionShape3DHandle& handle, bool disabled) {
                              CollisionShape3D* collision_shape = handle.ResolveAs<CollisionShape3D>();
                              ExecuteSetNodeProperty(collision_shape, "disabled", Variant(disabled));
                          });
}

} // namespace gobot::python
