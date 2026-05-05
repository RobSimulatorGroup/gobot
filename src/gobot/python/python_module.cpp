#include <string>
#include <stdexcept>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_format_scene.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/types.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/rl_environment.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace py = pybind11;

namespace {

py::object VariantToPython(const gobot::Variant& variant);

py::tuple Vector3ToPython(const gobot::Vector3& vector) {
    return py::make_tuple(vector.x(), vector.y(), vector.z());
}

gobot::Vector3 PythonToVector3(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() != 3) {
        throw std::invalid_argument("expected a 3-element vector");
    }
    return {
            py::cast<gobot::RealType>(sequence[0]),
            py::cast<gobot::RealType>(sequence[1]),
            py::cast<gobot::RealType>(sequence[2])
    };
}

py::object ArithmeticVariantToPython(const gobot::Variant& variant) {
    const gobot::Type type = variant.get_type();
    if (type == gobot::Type::get<bool>()) {
        return py::bool_(variant.to_bool());
    }
    if (type == gobot::Type::get<std::uint8_t>() ||
        type == gobot::Type::get<std::uint16_t>() ||
        type == gobot::Type::get<std::uint32_t>() ||
        type == gobot::Type::get<std::uint64_t>() ||
        type == gobot::Type::get<std::int8_t>() ||
        type == gobot::Type::get<std::int16_t>() ||
        type == gobot::Type::get<std::int32_t>() ||
        type == gobot::Type::get<std::int64_t>() ||
        type == gobot::Type::get<int>() ||
        type == gobot::Type::get<unsigned int>()) {
        return py::int_(variant.to_int64());
    }
    if (type == gobot::Type::get<float>() ||
        type == gobot::Type::get<double>() ||
        type == gobot::Type::get<gobot::RealType>()) {
        return py::float_(variant.to_double());
    }
    return py::none();
}

py::object SequentialVariantToPython(gobot::Variant variant) {
    gobot::VariantListView view = variant.create_sequential_view();
    py::list list;
    for (const gobot::Variant& value : view) {
        list.append(VariantToPython(value.extract_wrapped_value()));
    }
    return list;
}

py::object ObjectToPythonDict(gobot::Variant variant) {
    if (variant.get_type().is_wrapper()) {
        variant = variant.extract_wrapped_value();
    }

    const gobot::Type type = variant.get_type();
    py::dict dict;
    for (const gobot::Property& property : type.get_properties()) {
        gobot::Variant value = property.get_value(variant);
        if (!value.is_valid()) {
            continue;
        }
        dict[py::str(property.get_name().data())] = VariantToPython(value);
    }
    return dict;
}

py::object VariantToPython(const gobot::Variant& variant) {
    if (!variant.is_valid()) {
        return py::none();
    }

    gobot::Variant value = variant;
    if (value.get_type().is_wrapper()) {
        value = value.extract_wrapped_value();
    }

    const gobot::Type type = value.get_type();
    if (type == gobot::Type::get<std::string>()) {
        return py::str(value.to_string());
    }
    if (type == gobot::Type::get<gobot::Vector3>()) {
        return Vector3ToPython(value.get_value<gobot::Vector3>());
    }
    if (type.is_arithmetic()) {
        return ArithmeticVariantToPython(value);
    }
    if (value.is_sequential_container()) {
        return SequentialVariantToPython(value);
    }
    if (!type.get_properties().empty()) {
        return ObjectToPythonDict(value);
    }

    return py::str(value.to_string());
}

template <typename T>
py::dict ReflectedToPythonDict(const T& value) {
    return py::cast<py::dict>(ObjectToPythonDict(gobot::Variant(value)));
}

template <typename T>
T DictToReflected(py::dict dict) {
    T value;
    const gobot::Type type = gobot::Type::get<T>();
    for (const auto& item : dict) {
        const std::string name = py::cast<std::string>(item.first);
        const gobot::Property property = type.get_property(name);
        if (!property.is_valid()) {
            throw std::invalid_argument("unknown property '" + name + "' for " + type.get_name().to_string());
        }

        const gobot::Type property_type = property.get_type();
        if (property_type == gobot::Type::get<bool>()) {
            property.set_value(value, py::cast<bool>(item.second));
        } else if (property_type == gobot::Type::get<gobot::Vector3>()) {
            property.set_value(value, PythonToVector3(item.second));
        } else if (property_type == gobot::Type::get<float>()) {
            property.set_value(value, py::cast<float>(item.second));
        } else if (property_type == gobot::Type::get<double>() ||
                   property_type == gobot::Type::get<gobot::RealType>()) {
            property.set_value(value, py::cast<gobot::RealType>(item.second));
        } else if (property_type == gobot::Type::get<std::string>()) {
            property.set_value(value, py::cast<std::string>(item.second));
        } else if (property_type == gobot::Type::get<std::uint64_t>()) {
            property.set_value(value, py::cast<std::uint64_t>(item.second));
        } else if (property_type == gobot::Type::get<std::uint32_t>()) {
            property.set_value(value, py::cast<std::uint32_t>(item.second));
        } else {
            throw std::invalid_argument("unsupported reflected property type '" +
                                        property_type.get_name().to_string() + "' for property '" + name + "'");
        }
    }
    return value;
}

gobot::Variant PythonToVariantForType(const py::handle& object, const gobot::Type& type) {
    if (type == gobot::Type::get<bool>()) {
        return gobot::Variant(py::cast<bool>(object));
    }
    if (type == gobot::Type::get<float>()) {
        return gobot::Variant(py::cast<float>(object));
    }
    if (type == gobot::Type::get<double>() ||
        type == gobot::Type::get<gobot::RealType>()) {
        return gobot::Variant(py::cast<gobot::RealType>(object));
    }
    if (type == gobot::Type::get<std::string>()) {
        return gobot::Variant(py::cast<std::string>(object));
    }
    if (type == gobot::Type::get<gobot::Vector3>()) {
        return gobot::Variant(PythonToVector3(object));
    }
    throw std::invalid_argument("unsupported Gobot reflected property type '" +
                                type.get_name().to_string() + "'");
}

struct GobotRuntime {
    gobot::ProjectSettings* project_settings{nullptr};
    gobot::PhysicsServer* physics_server{nullptr};
    bool scene_initializer_ready{false};

    GobotRuntime() {
        project_settings = gobot::Object::New<gobot::ProjectSettings>();
        physics_server = gobot::Object::New<gobot::PhysicsServer>();
        gobot::SceneInitializer::Init();
        scene_initializer_ready = true;
    }

    ~GobotRuntime() {
        if (scene_initializer_ready) {
            gobot::SceneInitializer::Destroy();
            scene_initializer_ready = false;
        }
        if (physics_server != nullptr) {
            gobot::Object::Delete(physics_server);
            physics_server = nullptr;
        }
        if (project_settings != nullptr) {
            gobot::Object::Delete(project_settings);
            project_settings = nullptr;
        }
    }
};

GobotRuntime& Runtime() {
    static GobotRuntime runtime;
    return runtime;
}

gobot::Robot3D* CreateTestRobotScene() {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({0.0, 0.0, 1.0});
    base_link->SetHasInertial(true);
    base_link->SetMass(1.0);

    auto* collision_shape = gobot::Object::New<gobot::CollisionShape3D>();
    collision_shape->SetName("base_collision");
    auto box = gobot::MakeRef<gobot::BoxShape3D>();
    box->SetSize({0.5, 0.5, 0.5});
    collision_shape->SetShape(box);
    base_link->AddChild(collision_shape);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);

    auto* tip_link = gobot::Object::New<gobot::Link3D>();
    tip_link->SetName("tip");

    robot->AddChild(base_link);
    robot->AddChild(joint);
    joint->AddChild(tip_link);
    return robot;
}

gobot::Node* LoadSceneRoot(const std::string& scene_path) {
    gobot::Ref<gobot::Resource> resource =
            gobot::ResourceLoader::Load(scene_path, "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    gobot::Ref<gobot::PackedScene> packed_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        throw std::runtime_error("failed to load PackedScene from '" + scene_path + "'");
    }

    gobot::Node* scene_root = packed_scene->Instantiate();
    if (scene_root == nullptr) {
        throw std::runtime_error("failed to instantiate PackedScene from '" + scene_path + "'");
    }

    return scene_root;
}

gobot::PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return gobot::PhysicsBackendType::Null;
    }
    if (backend == "mujoco") {
        return gobot::PhysicsBackendType::MuJoCoCpu;
    }
    throw std::invalid_argument("unknown Gobot physics backend '" + backend + "'");
}

struct PyRLEnvironment {
    gobot::SimulationServer* simulation{nullptr};
    gobot::RLEnvironment* environment{nullptr};
    gobot::Node* scene_root{nullptr};

    PyRLEnvironment(const std::string& scene_path,
                    const std::string& robot,
                    const std::string& backend) {
        Runtime();
        simulation = gobot::Object::New<gobot::SimulationServer>(ParseBackend(backend));
        environment = gobot::Object::New<gobot::RLEnvironment>(simulation);
        scene_root = scene_path.empty() ? static_cast<gobot::Node*>(CreateTestRobotScene()) : LoadSceneRoot(scene_path);
        environment->SetSceneRoot(scene_root);
        environment->SetRobotName(robot);
    }

    ~PyRLEnvironment() {
        if (environment != nullptr) {
            gobot::Object::Delete(environment);
            environment = nullptr;
        }
        if (simulation != nullptr) {
            gobot::Object::Delete(simulation);
            simulation = nullptr;
        }
        if (scene_root != nullptr) {
            gobot::Object::Delete(scene_root);
            scene_root = nullptr;
        }
    }

    gobot::RLEnvironmentResetResult Reset(std::uint32_t seed) {
        return environment->Reset(seed);
    }

    gobot::RLEnvironmentStepResult Step(const std::vector<gobot::RealType>& action) {
        return environment->Step(action);
    }

    void SetDefaultJointGains(const gobot::JointControllerGains& gains) {
        simulation->SetDefaultJointGains(gains);
    }

    const gobot::JointControllerGains& GetDefaultJointGains() const {
        return simulation->GetDefaultJointGains();
    }
};

struct PyRLControllerConfig {
    std::vector<std::string> controlled_joints;
    std::vector<gobot::RealType> default_action;
    gobot::JointControllerGains joint_gains{100.0, 10.0, 0.0, 0.0};
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
        config.default_action = py::cast<std::vector<gobot::RealType>>(dict["default_action"]);
    }
    if (dict.contains("joint_gains")) {
        config.joint_gains = DictToReflected<gobot::JointControllerGains>(
                py::reinterpret_borrow<py::dict>(dict["joint_gains"]));
    }
    return config;
}

PyRLControllerConfig MakeControllerConfigFromEnv(PyRLEnvironment& env) {
    PyRLControllerConfig config;
    config.controlled_joints = env.environment->GetControlledJointNames();
    config.default_action.assign(config.controlled_joints.size(), 0.0);
    config.joint_gains = env.GetDefaultJointGains();
    return config;
}

void ApplyControllerConfigToEnv(PyRLEnvironment& env, py::dict dict) {
    const PyRLControllerConfig config = ControllerConfigFromDict(dict);
    env.SetDefaultJointGains(config.joint_gains);
    if (!config.default_action.empty() &&
        config.default_action.size() != env.environment->GetActionSize()) {
        throw std::invalid_argument("default_action size must match the current RL environment action size");
    }
}

struct PyScene {
    gobot::Node* root{nullptr};

    explicit PyScene(gobot::Node* root_node)
        : root(root_node) {
    }

    ~PyScene() {
        if (root != nullptr) {
            gobot::Object::Delete(root);
            root = nullptr;
        }
    }

    PyScene(const PyScene&) = delete;
    PyScene& operator=(const PyScene&) = delete;
};

std::string NodeTypeName(const gobot::Node& node) {
    return node.GetClassStringName().data();
}

std::vector<gobot::Node*> GetNodeChildren(gobot::Node& node) {
    std::vector<gobot::Node*> children;
    children.reserve(node.GetChildCount());
    for (std::size_t index = 0; index < node.GetChildCount(); ++index) {
        children.push_back(node.GetChild(static_cast<int>(index)));
    }
    return children;
}

gobot::Node* GetNodeChild(gobot::Node& node, int index) {
    gobot::Node* child = node.GetChild(index);
    if (child == nullptr) {
        throw py::index_error("Gobot node child index is out of range");
    }
    return child;
}

py::object NodeGetProperty(const gobot::Node& node, const std::string& name) {
    gobot::Variant value = node.Get(name);
    if (!value.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    return VariantToPython(value);
}

void NodeSetProperty(gobot::Node& node, const std::string& name, const py::handle& value) {
    const gobot::Property property = node.GetType().get_property(name);
    if (!property.is_valid()) {
        throw py::key_error("unknown Gobot property '" + name + "'");
    }
    if (!node.Set(name, PythonToVariantForType(value, property.get_type()))) {
        throw std::runtime_error("failed to set Gobot property '" + name + "'");
    }
}

py::list NodeGetPropertyNames(const gobot::Node& node) {
    py::list names;
    for (const gobot::Property& property : node.GetType().get_properties()) {
        names.append(py::str(property.get_name().data()));
    }
    return names;
}

py::dict ResourceToPythonDict(const gobot::Ref<gobot::Resource>& resource) {
    py::dict result;
    if (!resource.IsValid()) {
        return result;
    }

    result["type"] = py::str(resource->GetClassStringName().data());
    result["path"] = resource->GetPath();
    result["name"] = resource->GetName();
    py::dict properties;
    gobot::Instance instance(resource.Get());
    const gobot::Type type = resource->GetType();
    for (const gobot::Property& property : type.get_properties()) {
        gobot::Variant value = property.get_value(instance);
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

PYBIND11_MODULE(gobot, module) {
    Runtime();

    py::enum_<gobot::PhysicsBackendType>(module, "PhysicsBackendType")
            .value("Null", gobot::PhysicsBackendType::Null)
            .value("MuJoCoCpu", gobot::PhysicsBackendType::MuJoCoCpu);

    py::class_<PyScene, std::unique_ptr<PyScene>>(module, "Scene")
            .def_property_readonly("root", [](PyScene& scene) {
                return scene.root;
            }, py::return_value_policy::reference_internal);

    py::class_<gobot::Node>(module, "Node")
            .def_property("name", &gobot::Node::GetName, &gobot::Node::SetName)
            .def_property_readonly("type", &NodeTypeName)
            .def_property_readonly("child_count", &gobot::Node::GetChildCount)
            .def_property_readonly("children", &GetNodeChildren, py::return_value_policy::reference_internal)
            .def("child", &GetNodeChild, py::arg("index"), py::return_value_policy::reference_internal)
            .def("get", &NodeGetProperty, py::arg("property"))
            .def("set", &NodeSetProperty, py::arg("property"), py::arg("value"))
            .def("property_names", &NodeGetPropertyNames);

    py::class_<gobot::Node3D, gobot::Node>(module, "Node3D")
            .def_property("position",
                          [](const gobot::Node3D& node) { return Vector3ToPython(node.GetPosition()); },
                          [](gobot::Node3D& node, const py::handle& value) { node.SetPosition(PythonToVector3(value)); })
            .def_property("rotation_degrees",
                          [](const gobot::Node3D& node) { return Vector3ToPython(node.GetEulerDegree()); },
                          [](gobot::Node3D& node, const py::handle& value) { node.SetEulerDegree(PythonToVector3(value)); })
            .def_property("scale",
                          [](const gobot::Node3D& node) { return Vector3ToPython(node.GetScale()); },
                          [](gobot::Node3D& node, const py::handle& value) { node.SetScale(PythonToVector3(value)); })
            .def_property("visible", &gobot::Node3D::IsVisible, &gobot::Node3D::SetVisible);

    py::class_<PyRLControllerConfig>(module, "RLControllerConfig")
            .def(py::init<>())
            .def_readwrite("controlled_joints", &PyRLControllerConfig::controlled_joints)
            .def_readwrite("default_action", &PyRLControllerConfig::default_action)
            .def_property("joint_gains",
                          [](const PyRLControllerConfig& config) {
                              return ReflectedToPythonDict(config.joint_gains);
                          },
                          [](PyRLControllerConfig& config, py::dict gains) {
                              config.joint_gains = DictToReflected<gobot::JointControllerGains>(gains);
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
            .def("step", [](PyRLEnvironment& env, const std::vector<gobot::RealType>& action) {
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
            .def("step_result", [](PyRLEnvironment& env, const std::vector<gobot::RealType>& action) {
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
                env.environment->SetRewardSettings(DictToReflected<gobot::RLEnvironmentRewardSettings>(settings));
            })
            .def("get_reward_settings", [](const PyRLEnvironment& env) {
                return ReflectedToPythonDict(env.environment->GetRewardSettings());
            })
            .def("set_default_joint_gains", [](PyRLEnvironment& env, py::dict gains) {
                env.SetDefaultJointGains(DictToReflected<gobot::JointControllerGains>(gains));
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
        Runtime();
        if (!gobot::ProjectSettings::GetInstance()->SetProjectPath(project_path)) {
            throw std::runtime_error("failed to set Gobot project path '" + project_path + "'");
        }
    });

    module.def("load_scene", [](const std::string& scene_path) {
        Runtime();
        return std::make_unique<PyScene>(LoadSceneRoot(scene_path));
    }, py::arg("scene_path"));

    module.def("load_resource", [](const std::string& path, const std::string& type_hint) {
        Runtime();
        return ResourceToPythonDict(gobot::ResourceLoader::Load(path, type_hint));
    }, py::arg("path"), py::arg("type_hint") = "");

    module.def("create_test_scene", []() {
        Runtime();
        return std::make_unique<PyScene>(CreateTestRobotScene());
    });

    module.def("backend_infos", []() {
        Runtime();
        py::list infos;
        for (const gobot::PhysicsBackendInfo& info : gobot::PhysicsServer::GetBackendInfosForAllBackends()) {
            infos.append(ReflectedToPythonDict(info));
        }
        return infos;
    });
}
