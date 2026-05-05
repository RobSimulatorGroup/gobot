#include "gobot/python/python_script_runner.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <pybind11/embed.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/python/python_binding_registry.hpp"

namespace gobot::python {
namespace {

namespace py = pybind11;

bool& InterpreterStartedByRunner() {
    static bool started = false;
    return started;
}

void EnsureInterpreter() {
    if (Py_IsInitialized()) {
        return;
    }

    setenv("PYTHONNOUSERSITE", "1", 1);
    Py_NoUserSiteDirectory = 1;
    py::initialize_interpreter();
    InterpreterStartedByRunner() = true;
}

void AddProjectPathToSysPath(EngineContext* context) {
    if (context == nullptr || context->GetProjectPath().empty()) {
        return;
    }

    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    const std::string project_path = context->GetProjectPath();
    for (const py::handle item : path) {
        if (py::cast<std::string>(item) == project_path) {
            return;
        }
    }
    path.insert(0, project_path);
}

PythonExecutionResult ExecuteCompiledCode(const std::string& source,
                                          EngineContext* context,
                                          const std::string& filename) {
    PythonExecutionResult result;
    EngineContext* previous_context = nullptr;

    try {
        EnsureInterpreter();

        py::gil_scoped_acquire gil;
        previous_context = GetActiveAppContextOrNull();
        SetActiveAppContext(context);
        AddProjectPathToSysPath(context);
        py::module_ sys = py::module_::import("sys");
        py::dict modules = sys.attr("modules");
        modules.attr("pop")("gobot", py::none());
        py::module_::import("gobot");

        py::module_ io = py::module_::import("io");
        py::module_ contextlib = py::module_::import("contextlib");
        py::object stdout_buffer = io.attr("StringIO")();
        py::object stderr_buffer = io.attr("StringIO")();
        py::dict globals;
        globals["__name__"] = "__main__";
        globals["__file__"] = filename;

        py::object stdout_redirect = contextlib.attr("redirect_stdout")(stdout_buffer);
        py::object stderr_redirect = contextlib.attr("redirect_stderr")(stderr_buffer);
        stdout_redirect.attr("__enter__")();
        stderr_redirect.attr("__enter__")();
        try {
            py::exec(source, globals);
            result.ok = true;
        } catch (...) {
            stderr_redirect.attr("__exit__")(py::none(), py::none(), py::none());
            stdout_redirect.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        stderr_redirect.attr("__exit__")(py::none(), py::none(), py::none());
        stdout_redirect.attr("__exit__")(py::none(), py::none(), py::none());

        result.output = py::cast<std::string>(stdout_buffer.attr("getvalue")());
        result.error = py::cast<std::string>(stderr_buffer.attr("getvalue")());
    } catch (const py::error_already_set& error) {
        result.ok = false;
        result.error = error.what();
    } catch (const std::exception& error) {
        result.ok = false;
        result.error = error.what();
    }

    SetActiveAppContext(previous_context);
    return result;
}

} // namespace

bool PythonScriptRunner::IsAvailable() {
    return true;
}

PythonExecutionResult PythonScriptRunner::ExecuteString(const std::string& source,
                                                        EngineContext* context,
                                                        const std::string& filename) {
    return ExecuteCompiledCode(source, context, filename);
}

PythonExecutionResult PythonScriptRunner::ExecuteFile(const std::string& path,
                                                      EngineContext* context) {
    std::ifstream stream(path);
    if (!stream) {
        return {false, "", "Cannot open Python script: " + path};
    }

    std::stringstream buffer;
    buffer << stream.rdbuf();
    return ExecuteString(buffer.str(), context, path);
}

void PythonScriptRunner::Shutdown() {
    if (InterpreterStartedByRunner() && Py_IsInitialized()) {
        SetActiveAppContext(nullptr);
        py::finalize_interpreter();
        InterpreterStartedByRunner() = false;
    }
}

} // namespace gobot::python
