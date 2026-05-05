/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/types.hpp"

namespace gobot::python {

namespace py = pybind11;

void RegisterRuntime(py::module_& module);

void RegisterReflectedTypes(py::module_& module);

void RegisterManualApis(py::module_& module);

py::object VariantToPython(const Variant& variant);

Variant PythonToVariant(const py::handle& object);

Variant PythonToVariantForType(const py::handle& object, const Type& type);

bool SetReflectedPropertyFromPython(Instance instance, const Property& property, const py::handle& value);

Vector3 PythonToVector3(const py::handle& object);

py::tuple Vector3ToPython(const Vector3& vector);

py::dict ReflectedToPythonDict(const Variant& value);

Variant DictToReflected(py::dict dict, const Type& type);

template <typename T>
py::dict ReflectedToPythonDict(const T& value) {
    return ReflectedToPythonDict(Variant(value));
}

template <typename T>
T DictToReflected(py::dict dict) {
    T value{};
    const Type type = Type::get<T>();
    for (const auto& item : dict) {
        const std::string name = py::cast<std::string>(item.first);
        const Property property = type.get_property(name);
        if (!property.is_valid()) {
            throw std::invalid_argument("unknown property '" + name + "' for " + type.get_name().to_string());
        }

        if (!SetReflectedPropertyFromPython(value, property, item.second)) {
            throw std::invalid_argument("failed to set property '" + name + "' for " +
                                        type.get_name().to_string());
        }
    }
    return value;
}

} // namespace gobot::python
