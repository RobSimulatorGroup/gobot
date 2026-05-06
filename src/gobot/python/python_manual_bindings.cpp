#include "gobot/python/python_binding_registry.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/node_creation_registry.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/rl_environment.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot::python {
namespace {

EngineContext* s_active_app_context = nullptr;

struct GobotRuntime {
    ProjectSettings* project_settings{nullptr};
    PhysicsServer* physics_server{nullptr};
    SimulationServer* app_simulation_server{nullptr};
    std::unique_ptr<EngineContext> app_context;
    bool scene_initializer_ready{false};

    GobotRuntime() {
        project_settings = Object::New<ProjectSettings>();
        physics_server = Object::New<PhysicsServer>();
        app_simulation_server = Object::New<SimulationServer>();
        app_context = std::make_unique<EngineContext>(project_settings,
                                                      physics_server,
                                                      app_simulation_server);
        SceneInitializer::Init();
        scene_initializer_ready = true;
    }

    ~GobotRuntime() {
        app_context.reset();
        if (app_simulation_server != nullptr) {
            Object::Delete(app_simulation_server);
            app_simulation_server = nullptr;
        }
        if (scene_initializer_ready) {
            SceneInitializer::Destroy();
            scene_initializer_ready = false;
        }
        if (physics_server != nullptr) {
            Object::Delete(physics_server);
            physics_server = nullptr;
        }
        if (project_settings != nullptr) {
            Object::Delete(project_settings);
            project_settings = nullptr;
        }
    }

    EngineContext& GetAppContext() {
        return *app_context;
    }
};

GobotRuntime& Runtime() {
    static GobotRuntime runtime;
    return runtime;
}

Robot3D* CreateTestRobotScene() {
    auto* robot = Object::New<Robot3D>();
    robot->SetName("robot");

    auto* base_link = Object::New<Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({0.0, 0.0, 1.0});
    base_link->SetHasInertial(true);
    base_link->SetMass(1.0);

    auto* collision_shape = Object::New<CollisionShape3D>();
    collision_shape->SetName("base_collision");
    auto box = MakeRef<BoxShape3D>();
    box->SetSize({0.5, 0.5, 0.5});
    collision_shape->SetShape(box);
    base_link->AddChild(collision_shape);

    auto* joint = Object::New<Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);

    auto* tip_link = Object::New<Link3D>();
    tip_link->SetName("tip");

    robot->AddChild(base_link);
    robot->AddChild(joint);
    joint->AddChild(tip_link);
    return robot;
}

CollisionShape3D* CreateBoxCollision(const std::string& name,
                                      const Vector3& size,
                                      const Vector3& position = Vector3::Zero()) {
    auto* collision_shape = Object::New<CollisionShape3D>();
    collision_shape->SetName(name);
    collision_shape->SetPosition(position);
    auto box = MakeRef<BoxShape3D>();
    box->SetSize(size);
    collision_shape->SetShape(box);
    return collision_shape;
}

MeshInstance3D* CreateBoxVisual(const std::string& name,
                                const Vector3& size,
                                const Vector3& position = Vector3::Zero()) {
    auto* mesh_instance = Object::New<MeshInstance3D>();
    mesh_instance->SetName(name);
    mesh_instance->SetPosition(position);
    auto box = MakeRef<BoxMesh>();
    box->SetSize(size);
    mesh_instance->SetMesh(box);
    return mesh_instance;
}

Node* LoadSceneRoot(const std::string& scene_path) {
    Ref<Resource> resource =
            ResourceLoader::Load(scene_path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        throw std::runtime_error("failed to load PackedScene from '" + scene_path + "'");
    }

    Node* scene_root = packed_scene->Instantiate();
    if (scene_root == nullptr) {
        throw std::runtime_error("failed to instantiate PackedScene from '" + scene_path + "'");
    }

    return scene_root;
}

bool SaveSceneRoot(Node* root, const std::string& path) {
    if (root == nullptr) {
        throw std::invalid_argument("cannot save a null Gobot scene root");
    }
    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    if (!packed_scene->Pack(root)) {
        return false;
    }
    USING_ENUM_BITWISE_OPERATORS;
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    return ResourceSaver::Save(packed_scene,
                               global_path,
                               ResourceSaverFlags::ReplaceSubResourcePaths |
                                       ResourceSaverFlags::ChangePath);
}

PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return PhysicsBackendType::Null;
    }
    if (backend == "mujoco") {
        return PhysicsBackendType::MuJoCoCpu;
    }
    throw std::invalid_argument("unknown Gobot physics backend '" + backend + "'");
}

struct PyRLEnvironment {
    SimulationServer* simulation{nullptr};
    RLEnvironment* environment{nullptr};
    Node* scene_root{nullptr};

    PyRLEnvironment(const std::string& scene_path,
                    const std::string& robot,
                    const std::string& backend) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        simulation = Object::New<SimulationServer>(ParseBackend(backend));
        environment = Object::New<RLEnvironment>(simulation);
        scene_root = scene_path.empty() ? static_cast<Node*>(CreateTestRobotScene()) : LoadSceneRoot(scene_path);
        environment->SetSceneRoot(scene_root);
        environment->SetRobotName(robot);
    }

    ~PyRLEnvironment() {
        if (environment != nullptr) {
            Object::Delete(environment);
            environment = nullptr;
        }
        if (simulation != nullptr) {
            Object::Delete(simulation);
            simulation = nullptr;
        }
        if (scene_root != nullptr) {
            Object::Delete(scene_root);
            scene_root = nullptr;
        }
    }

    RLEnvironmentResetResult Reset(std::uint32_t seed) {
        return environment->Reset(seed);
    }

    RLEnvironmentStepResult Step(const std::vector<RealType>& action) {
        return environment->Step(action);
    }

    void SetDefaultJointGains(const JointControllerGains& gains) {
        simulation->SetDefaultJointGains(gains);
    }

    const JointControllerGains& GetDefaultJointGains() const {
        return simulation->GetDefaultJointGains();
    }
};

struct PyRLControllerConfig {
    std::vector<std::string> controlled_joints;
    std::vector<RealType> default_action;
    JointControllerGains joint_gains{100.0, 10.0, 0.0, 0.0};
};

py::dict ControllerConfigToDict(const PyRLControllerConfig& config) {
    py::dict result;
    result["controlled_joints"] = config.controlled_joints;
    result["default_action"] = config.default_action;
    result["joint_gains"] = ReflectedToPythonDict(config.joint_gains);
    return result;
}

PyRLControllerConfig ControllerConfigFromDict(py::dict dict) {
    PyRLControllerConfig config;
    if (dict.contains("controlled_joints")) {
        config.controlled_joints = py::cast<std::vector<std::string>>(dict["controlled_joints"]);
    }
    if (dict.contains("default_action")) {
        config.default_action = py::cast<std::vector<RealType>>(dict["default_action"]);
    }
    if (dict.contains("joint_gains")) {
        config.joint_gains = DictToReflected<JointControllerGains>(
                py::reinterpret_borrow<py::dict>(dict["joint_gains"]));
    }
    return config;
}

PyRLControllerConfig MakeControllerConfigFromEnv(PyRLEnvironment& env) {
    PyRLControllerConfig config;
    config.controlled_joints = env.environment->GetControlledJointNames();
    config.default_action = env.environment->GetDefaultAction();
    if (config.default_action.empty()) {
        config.default_action.assign(config.controlled_joints.size(), 0.0);
    }
    config.joint_gains = env.GetDefaultJointGains();
    return config;
}

void ApplyControllerConfigToEnv(PyRLEnvironment& env, py::dict dict) {
    const PyRLControllerConfig config = ControllerConfigFromDict(dict);
    env.environment->SetConfiguredControlledJointNames(config.controlled_joints);
    env.environment->SetDefaultAction(config.default_action);
    env.SetDefaultJointGains(config.joint_gains);
    if (!config.default_action.empty() &&
        config.default_action.size() != config.controlled_joints.size()) {
        throw std::invalid_argument("default_action size must match controlled_joints size");
    }
}

struct PyScene {
    Node* root{nullptr};

    explicit PyScene(Node* root_node)
        : root(root_node) {
    }

    ~PyScene() {
        if (root != nullptr) {
            Object::Delete(root);
            root = nullptr;
        }
    }

    PyScene(const PyScene&) = delete;
    PyScene& operator=(const PyScene&) = delete;
};

std::string NodeTypeName(const Node& node) {
    return node.GetClassStringName().data();
}

std::vector<Node*> GetNodeChildren(Node& node) {
    std::vector<Node*> children;
    children.reserve(node.GetChildCount());
    for (std::size_t index = 0; index < node.GetChildCount(); ++index) {
        children.push_back(node.GetChild(static_cast<int>(index)));
    }
    return children;
}

Node* GetNodeChild(Node& node, int index) {
    Node* child = node.GetChild(index);
    if (child == nullptr) {
        throw py::index_error("Gobot node child index is out of range");
    }
    return child;
}

Node* FindNode(Node& node, const std::string& path) {
    return node.GetNodeOrNull(NodePath(path));
}

py::object NodeGetProperty(const Node& node, const std::string& name) {
    Variant value = node.Get(name);
    if (!value.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    return VariantToPython(value);
}

void NodeSetProperty(Node& node, const std::string& name, const py::handle& value) {
    const Property property = node.GetType().get_property(name);
    if (!property.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    if (!node.Set(name, PythonToVariantForType(value, property.get_type()))) {
        throw std::runtime_error("failed to set Gobot property '" + name + "'");
    }
    GetActiveAppContext().NotifySceneChanged();
}

void NodeAddChild(Node& node, Node& child, bool force_readable_name) {
    node.AddChild(&child, force_readable_name);
    GetActiveAppContext().NotifySceneChanged();
}

void NodeRemoveChild(Node& node, Node& child, bool delete_child) {
    node.RemoveChild(&child);
    if (delete_child) {
        Object::Delete(&child);
    }
    GetActiveAppContext().NotifySceneChanged();
}

void NodeReparent(Node& node, Node& parent) {
    node.Reparent(&parent);
    GetActiveAppContext().NotifySceneChanged();
}

Node* NodeGetParent(Node& node) {
    return node.GetParent();
}

py::list NodeGetPropertyNames(const Node& node) {
    py::list names;
    for (const Property& property : node.GetType().get_properties()) {
        names.append(py::str(property.get_name().data()));
    }
    return names;
}

Node* CreateNode(const std::string& type_name, const std::string& name) {
    Node* node = NodeCreationRegistry::CreateNode(type_name);
    if (node == nullptr) {
        throw std::invalid_argument("unknown Gobot node type '" + type_name + "'");
    }
    if (!name.empty()) {
        node->SetName(name);
    }
    return node;
}

py::dict ResourceToPythonDict(const Ref<Resource>& resource) {
    py::dict result;
    if (!resource.IsValid()) {
        return result;
    }

    result["type"] = py::str(resource->GetClassStringName().data());
    result["path"] = resource->GetPath();
    result["name"] = resource->GetName();
    py::dict properties;
    Instance instance(resource.Get());
    const Type type = resource->GetType();
    for (const Property& property : type.get_properties()) {
        Variant value = property.get_value(instance);
        if (!value.is_valid()) {
            continue;
        }
        properties[py::str(property.get_name().data())] = VariantToPython(value);
    }
    result["properties"] = properties;
    for (const auto& item : properties) {
        result[item.first] = item.second;
    }
    return result;
}

py::dict StepInfoFromResult(const py::dict& result) {
    py::dict info;
    info["frame_count"] = result["frame_count"];
    info["simulation_time"] = result["simulation_time"];
    info["error"] = result["error"];
    return info;
}

} // namespace

void SetActiveAppContext(EngineContext* context) {
    s_active_app_context = context;
}

EngineContext& GetActiveAppContext() {
    if (s_active_app_context != nullptr) {
        return *s_active_app_context;
    }
    return Runtime().GetAppContext();
}

EngineContext* GetActiveAppContextOrNull() {
    return s_active_app_context;
}

void RegisterRuntime(py::module_& module) {
    module.doc() = "Gobot robotics scene, simulation, and rendering engine bindings.";
}

void RegisterManualApis(py::module_& module) {
    py::enum_<JointType>(module, "JointType")
            .value("Fixed", JointType::Fixed)
            .value("Revolute", JointType::Revolute)
            .value("Continuous", JointType::Continuous)
            .value("Prismatic", JointType::Prismatic)
            .value("Floating", JointType::Floating)
            .value("Planar", JointType::Planar)
            .export_values();

    py::enum_<RobotMode>(module, "RobotMode")
            .value("Assembly", RobotMode::Assembly)
            .value("Motion", RobotMode::Motion)
            .export_values();

    py::enum_<LinkRole>(module, "LinkRole")
            .value("Physical", LinkRole::Physical)
            .value("VirtualRoot", LinkRole::VirtualRoot)
            .export_values();

    py::class_<EngineContext>(module, "AppContext")
            .def_property_readonly("project_path", &EngineContext::GetProjectPath)
            .def_property_readonly("scene_path", &EngineContext::GetScenePath)
            .def_property_readonly("root", &EngineContext::GetSceneRoot,
                                   py::return_value_policy::reference_internal)
            .def_property("backend_type",
                          &EngineContext::GetBackendType,
                          &EngineContext::SetBackendType)
            .def_property_readonly("has_scene", &EngineContext::HasScene)
            .def_property_readonly("has_world", &EngineContext::HasWorld)
            .def_property_readonly("simulation_time", &EngineContext::GetSimulationTime)
            .def_property_readonly("frame_count", &EngineContext::GetFrameCount)
            .def_property_readonly("gravity", [](const EngineContext& context) {
                return Vector3ToPython(context.GetGravity());
            })
            .def("set_project_path", [](EngineContext& context, const std::string& project_path) {
                if (!context.SetProjectPath(project_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("project_path"))
            .def("load_scene", [](EngineContext& context, const std::string& scene_path) {
                if (!context.LoadScene(scene_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
                return context.GetSceneRoot();
            }, py::arg("scene_path"), py::return_value_policy::reference_internal)
            .def("clear_scene", &EngineContext::ClearScene)
            .def("notify_scene_changed", &EngineContext::NotifySceneChanged)
            .def("build_world", [](EngineContext& context, PhysicsBackendType backend_type) {
                context.SetBackendType(backend_type);
                if (!context.BuildWorld()) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("backend_type") = PhysicsBackendType::Null)
            .def("rebuild_world", [](EngineContext& context, bool preserve_state) {
                if (!context.RebuildWorld(preserve_state)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("preserve_state") = true)
            .def("clear_world", &EngineContext::ClearWorld)
            .def("reset_simulation", [](EngineContext& context) {
                if (!context.ResetSimulation()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step_once", [](EngineContext& context) {
                if (!context.StepOnce()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step", [](EngineContext& context, std::uint64_t ticks) {
                if (!context.StepTicks(ticks)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("ticks") = 1)
            .def("get_last_error", &EngineContext::GetLastError);

    py::class_<PyScene, std::unique_ptr<PyScene>>(module, "Scene")
            .def_property_readonly("root", [](PyScene& scene) {
                return scene.root;
            }, py::return_value_policy::reference_internal);

    py::class_<Node>(module, "Node")
            .def_property("name", &Node::GetName, &Node::SetName)
            .def_property_readonly("type", &NodeTypeName)
            .def_property_readonly("child_count", &Node::GetChildCount)
            .def_property_readonly("children", &GetNodeChildren, py::return_value_policy::reference_internal)
            .def_property_readonly("parent", &NodeGetParent, py::return_value_policy::reference_internal)
            .def("child", &GetNodeChild, py::arg("index"), py::return_value_policy::reference_internal)
            .def("find", &FindNode, py::arg("path"), py::return_value_policy::reference_internal)
            .def("add_child", &NodeAddChild, py::arg("child"), py::arg("force_readable_name") = false)
            .def("remove_child", &NodeRemoveChild, py::arg("child"), py::arg("delete") = false)
            .def("reparent", &NodeReparent, py::arg("parent"))
            .def("get", &NodeGetProperty, py::arg("property"))
            .def("set", &NodeSetProperty, py::arg("property"), py::arg("value"))
            .def("property_names", &NodeGetPropertyNames);

    py::class_<Node3D, Node>(module, "Node3D")
            .def_property("position",
                          [](const Node3D& node) { return Vector3ToPython(node.GetPosition()); },
                          [](Node3D& node, const py::handle& value) {
                              node.SetPosition(PythonToVector3(value));
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("rotation_degrees",
                          [](const Node3D& node) { return Vector3ToPython(node.GetEulerDegree()); },
                          [](Node3D& node, const py::handle& value) {
                              node.SetEulerDegree(PythonToVector3(value));
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("scale",
                          [](const Node3D& node) { return Vector3ToPython(node.GetScale()); },
                          [](Node3D& node, const py::handle& value) {
                              node.SetScale(PythonToVector3(value));
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("visible",
                          &Node3D::IsVisible,
                          [](Node3D& node, bool visible) {
                              node.SetVisible(visible);
                              GetActiveAppContext().NotifySceneChanged();
                          });

    py::class_<Robot3D, Node3D>(module, "Robot3D")
            .def_property("source_path",
                          &Robot3D::GetSourcePath,
                          [](Robot3D& robot, const std::string& source_path) {
                              robot.SetSourcePath(source_path);
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("mode",
                          &Robot3D::GetMode,
                          [](Robot3D& robot, RobotMode mode) {
                              robot.SetMode(mode);
                              GetActiveAppContext().NotifySceneChanged();
                          });

    py::class_<Link3D, Node3D>(module, "Link3D")
            .def_property("has_inertial", &Link3D::HasInertial, &Link3D::SetHasInertial)
            .def_property("mass", &Link3D::GetMass, &Link3D::SetMass)
            .def_property("center_of_mass",
                          [](const Link3D& link) { return Vector3ToPython(link.GetCenterOfMass()); },
                          [](Link3D& link, const py::handle& value) {
                              link.SetCenterOfMass(PythonToVector3(value));
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("inertia_diagonal",
                          [](const Link3D& link) { return Vector3ToPython(link.GetInertiaDiagonal()); },
                          [](Link3D& link, const py::handle& value) {
                              link.SetInertiaDiagonal(PythonToVector3(value));
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("role", &Link3D::GetRole, &Link3D::SetRole);

    py::class_<Joint3D, Node3D>(module, "Joint3D")
            .def_property("joint_type", &Joint3D::GetJointType, &Joint3D::SetJointType)
            .def_property("parent_link", &Joint3D::GetParentLink, &Joint3D::SetParentLink)
            .def_property("child_link", &Joint3D::GetChildLink, &Joint3D::SetChildLink)
            .def_property("axis",
                          [](const Joint3D& joint) { return Vector3ToPython(joint.GetAxis()); },
                          [](Joint3D& joint, const py::handle& value) {
                              joint.SetAxis(PythonToVector3(value));
                              GetActiveAppContext().NotifySceneChanged();
                          })
            .def_property("lower_limit", &Joint3D::GetLowerLimit, &Joint3D::SetLowerLimit)
            .def_property("upper_limit", &Joint3D::GetUpperLimit, &Joint3D::SetUpperLimit)
            .def_property("effort_limit", &Joint3D::GetEffortLimit, &Joint3D::SetEffortLimit)
            .def_property("velocity_limit", &Joint3D::GetVelocityLimit, &Joint3D::SetVelocityLimit)
            .def_property("joint_position", &Joint3D::GetJointPosition, &Joint3D::SetJointPosition);

    py::class_<CollisionShape3D, Node3D>(module, "CollisionShape3D")
            .def_property("disabled", &CollisionShape3D::IsDisabled, &CollisionShape3D::SetDisabled);

    py::class_<MeshInstance3D, Node3D>(module, "MeshInstance3D")
            .def_property("surface_color",
                          [](const MeshInstance3D& mesh_instance) {
                              const Color color = mesh_instance.GetSurfaceColor();
                              return py::make_tuple(color.red(), color.green(), color.blue(), color.alpha());
                          },
                          [](MeshInstance3D& mesh_instance, const py::handle& value) {
                              py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
                              if (sequence.size() != 4) {
                                  throw std::invalid_argument("expected a 4-element RGBA color");
                              }
                              mesh_instance.SetSurfaceColor({
                                      py::cast<float>(sequence[0]),
                                      py::cast<float>(sequence[1]),
                                      py::cast<float>(sequence[2]),
                                      py::cast<float>(sequence[3])
                              });
                              GetActiveAppContext().NotifySceneChanged();
                          });

    py::class_<PyRLControllerConfig>(module, "RLControllerConfig")
            .def(py::init<>())
            .def_readwrite("controlled_joints", &PyRLControllerConfig::controlled_joints)
            .def_readwrite("default_action", &PyRLControllerConfig::default_action)
            .def_property("joint_gains",
                          [](const PyRLControllerConfig& config) {
                              return ReflectedToPythonDict(config.joint_gains);
                          },
                          [](PyRLControllerConfig& config, py::dict gains) {
                              config.joint_gains = DictToReflected<JointControllerGains>(gains);
                          })
            .def("to_dict", &ControllerConfigToDict)
            .def_static("from_dict", &ControllerConfigFromDict, py::arg("config"));

    py::class_<PyRLEnvironment>(module, "RLEnvironment")
            .def(py::init<const std::string&, const std::string&, const std::string&>(),
                 py::arg("scene_path") = "",
                 py::arg("robot") = "robot",
                 py::arg("backend") = "null")
            .def("reset", [](PyRLEnvironment& env, std::uint32_t seed) {
                py::dict result = ReflectedToPythonDict(env.Reset(seed));
                py::dict info = StepInfoFromResult(result);
                info["ok"] = result["ok"];
                info["seed"] = result["seed"];
                return py::make_tuple(result["observation"], info);
            }, py::arg("seed") = 0)
            .def("step", [](PyRLEnvironment& env, const std::vector<RealType>& action) {
                py::dict result = ReflectedToPythonDict(env.Step(action));
                return py::make_tuple(result["observation"],
                                      result["reward"],
                                      result["terminated"],
                                      result["truncated"],
                                      StepInfoFromResult(result));
            }, py::arg("action"))
            .def("reset_result", [](PyRLEnvironment& env, std::uint32_t seed) {
                return ReflectedToPythonDict(env.Reset(seed));
            }, py::arg("seed") = 0)
            .def("step_result", [](PyRLEnvironment& env, const std::vector<RealType>& action) {
                return ReflectedToPythonDict(env.Step(action));
            }, py::arg("action"))
            .def("get_observation", [](const PyRLEnvironment& env) {
                return env.environment->GetObservation();
            })
            .def("get_action_size", [](const PyRLEnvironment& env) {
                return env.environment->GetActionSize();
            })
            .def("get_observation_size", [](const PyRLEnvironment& env) {
                return env.environment->GetObservationSize();
            })
            .def("get_controlled_joint_names", [](const PyRLEnvironment& env) {
                return env.environment->GetControlledJointNames();
            })
            .def("get_contact_link_names", [](const PyRLEnvironment& env) {
                return env.environment->GetContactLinkNames();
            })
            .def("get_action_spec", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetActionSpec());
            })
            .def("get_observation_spec", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetObservationSpec());
            })
            .def("set_reward_settings", [](PyRLEnvironment& env, py::dict settings) {
                env.environment->SetRewardSettings(DictToReflected<RLEnvironmentRewardSettings>(settings));
            })
            .def("get_reward_settings", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetRewardSettings());
            })
            .def("set_default_joint_gains", [](PyRLEnvironment& env, py::dict gains) {
                env.SetDefaultJointGains(DictToReflected<JointControllerGains>(gains));
            })
            .def("get_default_joint_gains", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.GetDefaultJointGains());
            })
            .def("get_controller_config", &MakeControllerConfigFromEnv)
            .def("apply_controller_config", &ApplyControllerConfigToEnv, py::arg("config"))
            .def("get_last_error", [](const PyRLEnvironment& env) {
                return env.environment->GetLastError();
            });

    module.def("set_project_path", [](const std::string& project_path) {
        EngineContext& context = GetActiveAppContext();
        if (!context.SetProjectPath(project_path)) {
            throw std::runtime_error("failed to set Gobot project path '" + project_path + "'");
        }
    });

    py::module_ app_module = module.def_submodule("app", "Gobot application runtime context.");
    app_module.def("context", []() -> EngineContext& {
        return GetActiveAppContext();
    }, py::return_value_policy::reference);

    module.def("load_scene", [](const std::string& scene_path) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        return std::make_unique<PyScene>(LoadSceneRoot(scene_path));
    }, py::arg("scene_path"));

    module.def("create_node", [](const std::string& type_name, const std::string& name) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        return CreateNode(type_name, name);
    }, py::arg("type_name"), py::arg("name") = "", py::return_value_policy::reference);

    module.def("create_box_collision", [](const std::string& name,
                                          const py::handle& size,
                                          const py::handle& position) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        return CreateBoxCollision(name, PythonToVector3(size), PythonToVector3(position));
    }, py::arg("name"), py::arg("size"), py::arg("position") = py::make_tuple(0.0, 0.0, 0.0),
       py::return_value_policy::reference);

    module.def("create_box_visual", [](const std::string& name,
                                       const py::handle& size,
                                       const py::handle& position) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        return CreateBoxVisual(name, PythonToVector3(size), PythonToVector3(position));
    }, py::arg("name"), py::arg("size"), py::arg("position") = py::make_tuple(0.0, 0.0, 0.0),
       py::return_value_policy::reference);

    module.def("save_scene", [](Node& root, const std::string& path) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        if (!SaveSceneRoot(&root, path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + path + "'");
        }
    }, py::arg("root"), py::arg("path"));

    module.def("load_resource", [](const std::string& path, const std::string& type_hint) {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        return ResourceToPythonDict(ResourceLoader::Load(path, type_hint));
    }, py::arg("path"), py::arg("type_hint") = "");

    module.def("create_test_scene", []() {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        return std::make_unique<PyScene>(CreateTestRobotScene());
    });

    module.def("backend_infos", []() {
        if (s_active_app_context == nullptr) {
            Runtime();
        }
        py::list infos;
        for (const PhysicsBackendInfo& info : PhysicsServer::GetBackendInfosForAllBackends()) {
            infos.append(ReflectedToPythonDict(info));
        }
        return infos;
    });
}

void RegisterModule(py::module_& module) {
    RegisterRuntime(module);
    RegisterReflectedTypes(module);
    RegisterManualApis(module);
}

} // namespace gobot::python
