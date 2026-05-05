#include <memory>
#include <string>
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
#include "gobot/simulation/rl_environment.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace py = pybind11;

namespace {

py::object VariantToPython(const gobot::Variant& variant);

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

struct GobotRuntime {
    gobot::ProjectSettings* project_settings{nullptr};
    gobot::PhysicsServer* physics_server{nullptr};
    gobot::Ref<gobot::ResourceFormatLoaderScene> scene_loader;

    GobotRuntime() {
        project_settings = gobot::Object::New<gobot::ProjectSettings>();
        physics_server = gobot::Object::New<gobot::PhysicsServer>();
        scene_loader = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
        gobot::ResourceLoader::AddResourceFormatLoader(scene_loader, true);
    }

    ~GobotRuntime() {
        if (scene_loader.IsValid()) {
            gobot::ResourceLoader::RemoveResourceFormatLoader(scene_loader);
            scene_loader.Reset();
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
            .def("get_last_error", [](const PyRLEnvironment& env) {
                return env.environment->GetLastError();
            });

    module.def("set_project_path", [](const std::string& project_path) {
        Runtime();
        if (!gobot::ProjectSettings::GetInstance()->SetProjectPath(project_path)) {
            throw std::runtime_error("failed to set Gobot project path '" + project_path + "'");
        }
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
