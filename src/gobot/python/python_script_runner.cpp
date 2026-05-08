#include "gobot/python/python_script_runner.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>

#include <pybind11/embed.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot/python/python_binding_registry.hpp"

namespace gobot::python {
namespace {

namespace py = pybind11;

class SourceLocationWriter {
public:
    explicit SourceLocationWriter(std::string default_filename) :
            default_filename_(std::move(default_filename)) {}

    void Write(const std::string& text) {
        for (char character : text) {
            pending_line_.push_back(character);
            if (character == '\n') {
                FlushPendingLine();
            }
        }
    }

    void Flush() {
        FlushPendingLine();
    }

    std::string GetValue() const {
        return buffer_.str();
    }

private:
    std::pair<std::string, int> CurrentSourceLocation() const {
        PyFrameObject* frame = PyEval_GetFrame();
        if (frame == nullptr) {
            return {default_filename_, 0};
        }

        PyCodeObject* code = PyFrame_GetCode(frame);
        if (code == nullptr) {
            return {default_filename_, PyFrame_GetLineNumber(frame)};
        }

        py::object filename_object = py::reinterpret_borrow<py::object>(code->co_filename);
        std::string filename = py::cast<std::string>(filename_object);
        Py_DECREF(code);
        if (filename.empty()) {
            filename = default_filename_;
        }

        return {filename, PyFrame_GetLineNumber(frame)};
    }

    void FlushPendingLine() {
        if (pending_line_.empty()) {
            return;
        }

        const bool only_newline = pending_line_ == "\n";
        if (!only_newline) {
            auto [filename, line] = CurrentSourceLocation();
            buffer_ << "[" << filename;
            if (line > 0) {
                buffer_ << ":" << line;
            }
            buffer_ << "] ";
        }
        buffer_ << pending_line_;
        pending_line_.clear();
    }

    std::string default_filename_;
    std::string pending_line_;
    std::stringstream buffer_;
};

py::object MakeWriterObject(const std::shared_ptr<SourceLocationWriter>& writer) {
    py::object object = py::module_::import("types").attr("SimpleNamespace")();
    object.attr("write") = py::cpp_function([writer](const std::string& text) {
        writer->Write(text);
    });
    object.attr("flush") = py::cpp_function([writer]() {
        writer->Flush();
    });
    return object;
}

bool& InterpreterStartedByRunner() {
    static bool started = false;
    return started;
}

void EnsureInterpreter() {
    if (Py_IsInitialized()) {
        return;
    }

    setenv("PYTHONNOUSERSITE", "1", 1);
#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;
    config.install_signal_handlers = 1;
    config.user_site_directory = 0;
    py::initialize_interpreter(&config);
#else
    Py_NoUserSiteDirectory = 1;
    py::initialize_interpreter();
#endif
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

        py::module_ contextlib = py::module_::import("contextlib");
        std::shared_ptr<SourceLocationWriter> stdout_writer =
                std::make_shared<SourceLocationWriter>(filename);
        std::shared_ptr<SourceLocationWriter> stderr_writer =
                std::make_shared<SourceLocationWriter>(filename);
        py::object stdout_buffer = MakeWriterObject(stdout_writer);
        py::object stderr_buffer = MakeWriterObject(stderr_writer);
        py::dict globals;
        globals["__name__"] = "__main__";
        globals["__file__"] = filename;

        py::object stdout_redirect = contextlib.attr("redirect_stdout")(stdout_buffer);
        py::object stderr_redirect = contextlib.attr("redirect_stderr")(stderr_buffer);
        stdout_redirect.attr("__enter__")();
        stderr_redirect.attr("__enter__")();
        try {
            py::module_ builtins = py::module_::import("builtins");
            py::object code = builtins.attr("compile")(source, filename, "exec");
            builtins.attr("exec")(code, globals);
            result.ok = true;
        } catch (...) {
            stderr_redirect.attr("__exit__")(py::none(), py::none(), py::none());
            stdout_redirect.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        stderr_redirect.attr("__exit__")(py::none(), py::none(), py::none());
        stdout_redirect.attr("__exit__")(py::none(), py::none(), py::none());
        stdout_writer->Flush();
        stderr_writer->Flush();

        result.output = stdout_writer->GetValue();
        result.error = stderr_writer->GetValue();
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
