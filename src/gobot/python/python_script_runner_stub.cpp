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
