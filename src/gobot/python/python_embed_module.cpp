#include <pybind11/embed.h>

#include "gobot/python/python_binding_registry.hpp"

PYBIND11_EMBEDDED_MODULE(gobot, module) {
    gobot::python::RegisterModule(module);
}
