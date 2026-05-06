#include <pybind11/pybind11.h>

#include "gobot/python/python_binding_registry.hpp"

PYBIND11_MODULE(_core, module) {
    gobot::python::RegisterModule(module);
}
