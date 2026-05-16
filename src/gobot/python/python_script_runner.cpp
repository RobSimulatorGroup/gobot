#include "gobot/python/python_script_runner.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <pybind11/embed.h>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/python_script.hpp"
#include "gobot/core/object.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/python/python_app_context.hpp"
#include "gobot/python/python_binding_registry.hpp"
#include "gobot/scene/node.hpp"

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

class ScopedPythonOutputCapture {
public:
    explicit ScopedPythonOutputCapture(const std::string& filename) :
            stdout_writer_(std::make_shared<SourceLocationWriter>(filename)),
            stderr_writer_(std::make_shared<SourceLocationWriter>(filename)) {
        py::module_ contextlib = py::module_::import("contextlib");
        stdout_buffer_ = MakeWriterObject(stdout_writer_);
        stderr_buffer_ = MakeWriterObject(stderr_writer_);
        stdout_redirect_ = contextlib.attr("redirect_stdout")(stdout_buffer_);
        stderr_redirect_ = contextlib.attr("redirect_stderr")(stderr_buffer_);
        stdout_redirect_.attr("__enter__")();
        stderr_redirect_.attr("__enter__")();
        active_ = true;
    }

    ~ScopedPythonOutputCapture() {
        try {
            Close();
        } catch (...) {
        }
    }

    void Close() {
        if (!active_) {
            return;
        }
        stderr_redirect_.attr("__exit__")(py::none(), py::none(), py::none());
        stdout_redirect_.attr("__exit__")(py::none(), py::none(), py::none());
        stdout_writer_->Flush();
        stderr_writer_->Flush();
        active_ = false;
    }

    std::string GetOutput() const {
        return stdout_writer_->GetValue();
    }

    std::string GetError() const {
        return stderr_writer_->GetValue();
    }

private:
    std::shared_ptr<SourceLocationWriter> stdout_writer_;
    std::shared_ptr<SourceLocationWriter> stderr_writer_;
    py::object stdout_buffer_;
    py::object stderr_buffer_;
    py::object stdout_redirect_;
    py::object stderr_redirect_;
    bool active_{false};
};

bool& InterpreterStartedByRunner() {
    static bool started = false;
    return started;
}

EngineContext*& SceneScriptContext() {
    static EngineContext* context = nullptr;
    return context;
}

Node*& SceneScriptRoot() {
    static Node* root = nullptr;
    return root;
}

std::uint64_t& SceneScriptEpoch() {
    static std::uint64_t epoch = 0;
    return epoch;
}

bool& ExecutingSceneScript() {
    static thread_local bool executing = false;
    return executing;
}

class ScopedSceneScriptExecution {
public:
    ScopedSceneScriptExecution() {
        previous_ = ExecutingSceneScript();
        ExecutingSceneScript() = true;
    }

    ~ScopedSceneScriptExecution() {
        ExecutingSceneScript() = previous_;
    }

private:
    bool previous_{false};
};

struct SceneScriptInstance {
    py::object instance;
    std::string path;
    bool disabled{false};
};

std::unordered_map<ObjectID, SceneScriptInstance>& SceneScriptInstances() {
    static auto* instances = new std::unordered_map<ObjectID, SceneScriptInstance>();
    return *instances;
}

py::dict& ScriptGlobals() {
    static auto* globals = new py::dict();
    return *globals;
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

    if (const char* python_executable = std::getenv("GOBOT_PYTHON_EXECUTABLE");
        python_executable != nullptr && python_executable[0] != '\0') {
        PyStatus status = PyConfig_SetBytesString(&config, &config.program_name, python_executable);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            throw std::runtime_error("Failed to configure embedded Python program_name");
        }

        status = PyConfig_SetBytesString(&config, &config.executable, python_executable);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            throw std::runtime_error("Failed to configure embedded Python executable");
        }
    }

    if (const char* python_home = std::getenv("GOBOT_PYTHON_HOME");
        python_home != nullptr && python_home[0] != '\0') {
        PyStatus status = PyConfig_SetBytesString(&config, &config.home, python_home);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            throw std::runtime_error("Failed to configure embedded Python home");
        }
    }

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

std::string SanitizeModuleName(std::string value) {
    for (char& character : value) {
        const bool valid =
                (character >= 'a' && character <= 'z') ||
                (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') ||
                character == '_';
        if (!valid) {
            character = '_';
        }
    }
    if (value.empty() || (value[0] >= '0' && value[0] <= '9')) {
        value.insert(value.begin(), '_');
    }
    return value;
}

std::string ModuleNameForScript(const PythonScript& script) {
    const std::string path = script.GetPath();
    if (!path.empty()) {
        return "__gobot_script_" + SanitizeModuleName(path);
    }
    return "__gobot_script_builtin_" +
           std::to_string(static_cast<std::uint64_t>(script.GetInstanceId()));
}

Node* FindRoot(Node* node) {
    if (SceneScriptRoot() != nullptr) {
        return SceneScriptRoot();
    }
    if (SceneScriptContext() != nullptr && SceneScriptContext()->GetSceneRoot() != nullptr) {
        return SceneScriptContext()->GetSceneRoot();
    }
    Node* root = node;
    while (root != nullptr && root->GetParent() != nullptr) {
        root = root->GetParent();
    }
    return root;
}

py::object MakeNodeHandle(Node* node) {
    py::module_ gobot = py::module_::import("gobot");
    return gobot.attr("_node_from_id")(static_cast<std::uint64_t>(node->GetInstanceId()));
}

py::object MakeContextHandle() {
    py::module_ gobot = py::module_::import("gobot");
    return gobot.attr("app").attr("context")();
}

py::object CreateNodeScriptInstance(Node* node, const Ref<PythonScript>& script) {
    const std::string filename = script->GetPath().empty() ? "<gobot-node-script>" : script->GetPath();
    const std::string module_name = ModuleNameForScript(*script.Get());
    py::module_ sys = py::module_::import("sys");
    py::dict modules = sys.attr("modules");
    modules.attr("pop")(module_name.c_str(), py::none());

    py::module_ module = py::module_::import("types").attr("ModuleType")(module_name);
    module.attr("__file__") = filename;
    modules[module_name.c_str()] = module;

    py::module_ builtins = py::module_::import("builtins");
    py::object code = builtins.attr("compile")(script->GetSourceCode(), filename, "exec");
    builtins.attr("exec")(code, module.attr("__dict__"));

    if (!py::hasattr(module, "Script")) {
        throw std::runtime_error("Python node script '" + filename + "' must define class Script");
    }

    py::object instance = module.attr("Script")();
    if (!py::hasattr(instance, "_attach")) {
        throw std::runtime_error("Python node script '" + filename +
                                 "' Script class must inherit gobot.NodeScript");
    }

    Node* root = FindRoot(node);
    py::object root_object = root != nullptr ? MakeNodeHandle(root) : py::none();
    instance.attr("_attach")(MakeNodeHandle(node), root_object, MakeContextHandle());
    return instance;
}

bool HasCallable(const py::object& object, const char* name) {
    if (!py::hasattr(object, name)) {
        return false;
    }
    return py::isinstance<py::function>(object.attr(name)) ||
           py::hasattr(object.attr(name), "__call__");
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

        py::dict& globals = ScriptGlobals();
        globals["__name__"] = "__main__";
        globals["__file__"] = filename;

        ScopedPythonOutputCapture output_capture(filename);
        {
            py::module_ builtins = py::module_::import("builtins");
            py::object code = builtins.attr("compile")(source, filename, "exec");
            builtins.attr("exec")(code, globals);
            result.ok = true;
        }
        output_capture.Close();

        result.output = output_capture.GetOutput();
        result.error = output_capture.GetError();
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

void PythonScriptRunner::SetSceneScriptContext(EngineContext* context) {
    SceneScriptContext() = context;
}

void PythonScriptRunner::SetSceneScriptRoot(Node* root, std::uint64_t scene_epoch) {
    SceneScriptRoot() = root;
    SceneScriptEpoch() = scene_epoch;
}

void PythonScriptRunner::ClearSceneScriptContext(EngineContext* context) {
    if (SceneScriptContext() == context) {
        SceneScriptContext() = nullptr;
        SceneScriptRoot() = nullptr;
        SceneScriptEpoch() = 0;
    }
}

bool PythonScriptRunner::IsExecutingSceneScript() {
    return ExecutingSceneScript();
}

Node* PythonScriptRunner::GetExecutingSceneScriptRoot() {
    return IsExecutingSceneScript() ? SceneScriptRoot() : nullptr;
}

std::uint64_t PythonScriptRunner::GetExecutingSceneScriptEpoch() {
    return IsExecutingSceneScript() ? SceneScriptEpoch() : 0;
}

bool PythonScriptRunner::HasSceneScriptInstance(Node* node) {
    if (node == nullptr) {
        return false;
    }
    return SceneScriptInstances().find(node->GetInstanceId()) != SceneScriptInstances().end();
}

PythonExecutionResult PythonScriptRunner::AttachSceneScript(Node* node,
                                                            const Ref<PythonScript>& script) {
    PythonExecutionResult result;
    EngineContext* previous_context = nullptr;

    if (node == nullptr || !script.IsValid()) {
        result.ok = true;
        return result;
    }

    try {
        EnsureInterpreter();
        py::gil_scoped_acquire gil;
        previous_context = GetActiveAppContextOrNull();
        SetActiveAppContext(SceneScriptContext());
        AddProjectPathToSysPath(SceneScriptContext());
        py::module_::import("gobot");

        ScopedPythonOutputCapture output_capture(script->GetPath().empty() ? "<gobot-node-script>" : script->GetPath());
        DetachSceneScript(node);
        ScopedSceneScriptExecution scoped_execution;
        SceneScriptInstances()[node->GetInstanceId()] =
                SceneScriptInstance{CreateNodeScriptInstance(node, script), script->GetPath(), false};
        output_capture.Close();
        result.output = output_capture.GetOutput();
        result.error = output_capture.GetError();
        result.ok = true;
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

void PythonScriptRunner::DetachSceneScript(Node* node) {
    if (node == nullptr) {
        return;
    }
    DetachSceneScript(node->GetInstanceId());
}

void PythonScriptRunner::DetachSceneScript(ObjectID node_id) {
    if (!node_id.IsValid()) {
        return;
    }
    if (!Py_IsInitialized()) {
        SceneScriptInstances().erase(node_id);
        return;
    }
    py::gil_scoped_acquire gil;
    SceneScriptInstances().erase(node_id);
}

PythonExecutionResult PythonScriptRunner::NotifySceneScript(Node* node,
                                                            NotificationType notification,
                                                            double delta_time) {
    PythonExecutionResult result;
    EngineContext* previous_context = nullptr;
    if (node == nullptr) {
        result.ok = true;
        return result;
    }

    auto instance_iter = SceneScriptInstances().find(node->GetInstanceId());
    if (instance_iter == SceneScriptInstances().end() || instance_iter->second.disabled) {
        result.ok = true;
        return result;
    }

    try {
        EnsureInterpreter();
        py::gil_scoped_acquire gil;
        previous_context = GetActiveAppContextOrNull();
        SetActiveAppContext(SceneScriptContext());
        AddProjectPathToSysPath(SceneScriptContext());

        py::object& instance = instance_iter->second.instance;
        ScopedPythonOutputCapture output_capture(instance_iter->second.path.empty()
                                                 ? "<gobot-node-script>"
                                                 : instance_iter->second.path);
        ScopedSceneScriptExecution scoped_execution;
        switch (notification) {
            case NotificationType::Ready:
                if (HasCallable(instance, "_ready")) {
                    instance.attr("_ready")();
                }
                break;
            case NotificationType::Process:
                if (HasCallable(instance, "_process")) {
                    instance.attr("_process")(delta_time);
                }
                break;
            case NotificationType::PhysicsProcess:
                if (HasCallable(instance, "_physics_process")) {
                    instance.attr("_physics_process")(delta_time);
                }
                break;
            case NotificationType::ExitTree:
                if (HasCallable(instance, "_exit_tree")) {
                    instance.attr("_exit_tree")();
                }
                break;
            default:
                break;
        }
        output_capture.Close();
        result.output = output_capture.GetOutput();
        result.error = output_capture.GetError();
        result.ok = true;
    } catch (const py::error_already_set& error) {
        instance_iter->second.disabled = true;
        result.ok = false;
        result.error = error.what();
    } catch (const std::exception& error) {
        instance_iter->second.disabled = true;
        result.ok = false;
        result.error = error.what();
    }

    SetActiveAppContext(previous_context);
    return result;
}

void PythonScriptRunner::Shutdown() {
    if (InterpreterStartedByRunner() && Py_IsInitialized()) {
        SceneScriptInstances().clear();
        SetActiveAppContext(nullptr);
        py::finalize_interpreter();
        InterpreterStartedByRunner() = false;
    }
}

} // namespace gobot::python
