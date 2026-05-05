#include <pybind11/pybind11.h>

#include "gobot/python/python_binding_registry.hpp"

PYBIND11_MODULE(gobot, module) {
    gobot::python::RegisterRuntime(module);
    gobot::python::RegisterReflectedTypes(module);
    gobot::python::RegisterManualApis(module);
}
