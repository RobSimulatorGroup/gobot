/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

#include "gobot/core/math/geometry.hpp"
#include "gobot/core/types.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot_export.h"

namespace gobot::python {

namespace py = pybind11;

GOBOT_EXPORT void RegisterRuntime(py::module_& module);

GOBOT_EXPORT void RegisterReflectedTypes(py::module_& module);

GOBOT_EXPORT void RegisterManualApis(py::module_& module);

GOBOT_EXPORT void RegisterModule(py::module_& module);

GOBOT_EXPORT py::object VariantToPython(const Variant& variant);

GOBOT_EXPORT Variant PythonToVariant(const py::handle& object);

GOBOT_EXPORT Variant PythonToVariantForType(const py::handle& object, const Type& type);

GOBOT_EXPORT bool SetReflectedPropertyFromPython(Instance instance, const Property& property, const py::handle& value);

GOBOT_EXPORT Vector3 PythonToVector3(const py::handle& object);

GOBOT_EXPORT py::object Vector3ToPython(const Vector3& vector);

GOBOT_EXPORT py::dict ReflectedToPythonDict(const Variant& value);

GOBOT_EXPORT Variant DictToReflected(py::dict dict, const Type& type);

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
