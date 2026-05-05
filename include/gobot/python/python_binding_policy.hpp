/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <string>

#include <rttr/registration>

#include "gobot/core/types.hpp"

namespace gobot::python {

enum class PythonBindingPolicy {
    Exported,
    Internal,
    ManualOnly
};

struct PythonClassInfo {
    std::string module;
    PythonBindingPolicy policy{PythonBindingPolicy::Internal};

    PythonClassInfo& SetModule(std::string module_name) {
        module = std::move(module_name);
        return *this;
    }

    PythonClassInfo& SetPolicy(PythonBindingPolicy binding_policy) {
        policy = binding_policy;
        return *this;
    }

    PythonClassInfo& SetExported(bool exported) {
        policy = exported ? PythonBindingPolicy::Exported : PythonBindingPolicy::Internal;
        return *this;
    }
};

inline constexpr const char* PYTHON_BINDING_POLICY_KEY = "PYTHON_BINDING_POLICY_KEY";
inline constexpr const char* PYTHON_MODULE_KEY = "PYTHON_MODULE_KEY";

inline MetaData AddMetaPythonPolicy(PythonBindingPolicy policy) {
    return rttr::metadata(PYTHON_BINDING_POLICY_KEY, static_cast<int>(policy));
}

inline MetaData AddMetaPythonModule(const char* module) {
    return rttr::metadata(PYTHON_MODULE_KEY, module);
}

} // namespace gobot::python
