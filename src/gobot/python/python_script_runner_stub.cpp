#include "gobot/python/python_script_runner.hpp"

namespace gobot::python {

bool PythonScriptRunner::IsAvailable() {
    return false;
}

PythonExecutionResult PythonScriptRunner::ExecuteString(const std::string&,
                                                        EngineContext*,
                                                        const std::string&) {
    return {false, "", "Python script execution is not available in this build."};
}

PythonExecutionResult PythonScriptRunner::ExecuteFile(const std::string& path,
                                                      EngineContext*) {
    return {false, "", "Python script execution is not available in this build: " + path};
}

void PythonScriptRunner::SetSceneScriptContext(EngineContext*) {}

void PythonScriptRunner::SetSceneScriptRoot(Node*) {}

void PythonScriptRunner::ClearSceneScriptContext(EngineContext*) {}

bool PythonScriptRunner::HasSceneScriptInstance(Node*) {
    return false;
}

PythonExecutionResult PythonScriptRunner::AttachSceneScript(Node*, const Ref<PythonScript>&) {
    return {false, "", "Python scene scripts are not available in this build."};
}

void PythonScriptRunner::DetachSceneScript(Node*) {}

void PythonScriptRunner::DetachSceneScript(ObjectID) {}

PythonExecutionResult PythonScriptRunner::NotifySceneScript(Node*,
                                                            NotificationType,
                                                            double) {
    return {true, "", ""};
}

void PythonScriptRunner::Shutdown() {}

} // namespace gobot::python
