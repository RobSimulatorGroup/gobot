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

PythonExecutionResult PythonScriptRunner::ExecuteTick(EngineContext*,
                                                      double,
                                                      const std::string&) {
    return {true, "", ""};
}

PythonExecutionResult PythonScriptRunner::ExecutePhysicsTick(EngineContext*,
                                                             double,
                                                             const std::string&) {
    return {true, "", ""};
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

PythonExecutionResult PythonScriptRunner::NotifySceneScript(Node*,
                                                            NotificationType,
                                                            double) {
    return {true, "", ""};
}

void PythonScriptRunner::SetTickCallback(PythonTickCallback) {}

void PythonScriptRunner::SetTickClearCallback(PythonTickClearCallback) {}

bool PythonScriptRunner::HasTickCallback() {
    return false;
}

void PythonScriptRunner::ClearTickCallback() {}

void PythonScriptRunner::SetPhysicsTickCallback(PythonTickCallback) {}

void PythonScriptRunner::SetPhysicsTickClearCallback(PythonTickClearCallback) {}

bool PythonScriptRunner::HasPhysicsTickCallback() {
    return false;
}

void PythonScriptRunner::ClearPhysicsTickCallback() {}

void PythonScriptRunner::Shutdown() {}

} // namespace gobot::python
