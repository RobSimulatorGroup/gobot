#include <pybind11/embed.h>

#include "gobot/python/python_binding_registry.hpp"

namespace {

pybind11::module_::module_def pybind11_module_def_gobot_core;

PyObject* PyInitGobotCore() {
    auto module = pybind11::module_::create_extension_module("gobot._core",
                                                             nullptr,
                                                             &pybind11_module_def_gobot_core);
    try {
        module.attr("__package__") = "gobot";
        gobot::python::RegisterModule(module);
        return module.ptr();
    } catch (const pybind11::error_already_set& error) {
        PyErr_SetString(PyExc_ImportError, error.what());
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ImportError, error.what());
        return nullptr;
    }
}

struct EmbeddedGobotCoreModule {
    EmbeddedGobotCoreModule() {
        if (PyImport_AppendInittab("gobot._core", PyInitGobotCore) == -1) {
            pybind11::pybind11_fail("Insufficient memory to add embedded gobot._core module");
        }
    }
};

EmbeddedGobotCoreModule embedded_gobot_core_module;

} // namespace
