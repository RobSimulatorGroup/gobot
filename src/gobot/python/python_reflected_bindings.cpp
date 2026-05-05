#include "gobot/python/python_binding_registry.hpp"

#include <string>

#include <pybind11/stl.h>

#include "gobot/physics/physics_types.hpp"
#include "gobot/python/python_binding_policy.hpp"
#include "gobot/simulation/rl_environment.hpp"

namespace gobot::python {
namespace {

py::module_ EnsureSubmodule(py::module_& module, const std::string& name) {
    if (name.empty()) {
        return module;
    }
    if (py::hasattr(module, name.c_str())) {
        return py::reinterpret_borrow<py::module_>(module.attr(name.c_str()));
    }
    return module.def_submodule(name.c_str());
}

template <typename Enum>
void RegisterReflectedEnum(py::module_& root_module, const char* module_name, const char* alias = nullptr) {
    const Type type = Type::get<Enum>();
    const Enumeration enumeration = type.get_enumeration();
    py::module_ target_module = EnsureSubmodule(root_module, module_name);
    py::enum_<Enum> py_enum(target_module, alias == nullptr ? type.get_name().data() : alias);
    for (const auto& value : enumeration.get_values()) {
        const Enum enum_value = value.get_value<Enum>();
        py_enum.value(enumeration.value_to_name(value).data(), enum_value);
    }
    py_enum.export_values();

    if (!target_module.is(root_module)) {
        root_module.attr(alias == nullptr ? type.get_name().data() : alias) =
                target_module.attr(alias == nullptr ? type.get_name().data() : alias);
    }
}

template <typename T>
void RegisterReflectedValueType(py::module_& root_module, const char* module_name, const char* alias = nullptr) {
    const Type type = Type::get<T>();
    py::module_ target_module = EnsureSubmodule(root_module, module_name);
    const char* python_name = alias == nullptr ? type.get_name().data() : alias;
    py::class_<T> py_class(target_module, python_name);
    py_class.def(py::init<>());

    for (const Property& property : type.get_properties()) {
        const std::string property_name = property.get_name().to_string();
        py_class.def_property(
                property_name.c_str(),
                [property](const T& instance) {
                    Variant value = property.get_value(instance);
                    return VariantToPython(value);
                },
                [property, property_name](T& instance, const py::handle& value) {
                    if (!SetReflectedPropertyFromPython(instance, property, value)) {
                        throw std::invalid_argument("failed to set reflected property '" + property_name + "'");
                    }
                });
    }

    py_class.def("to_dict", [](const T& value) {
        return ReflectedToPythonDict(value);
    });
    py_class.def_static("from_dict", [](py::dict dict) {
        return DictToReflected<T>(dict);
    }, py::arg("value"));

    if (!target_module.is(root_module)) {
        root_module.attr(python_name) = target_module.attr(python_name);
    }
}

} // namespace

void RegisterReflectedTypes(py::module_& module) {
    EnsureSubmodule(module, "sim");
    EnsureSubmodule(module, "scene");
    EnsureSubmodule(module, "physics");
    EnsureSubmodule(module, "rl");

    RegisterReflectedEnum<PhysicsBackendType>(module, "physics");
    RegisterReflectedValueType<JointControllerGains>(module, "sim");
    RegisterReflectedValueType<RLEnvironmentRewardSettings>(module, "rl");
    RegisterReflectedValueType<RLVectorSpec>(module, "rl");
    RegisterReflectedValueType<PhysicsBackendInfo>(module, "physics");
}

} // namespace gobot::python
