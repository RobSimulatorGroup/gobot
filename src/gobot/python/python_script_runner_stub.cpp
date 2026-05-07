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

void PythonScriptRunner::Shutdown() {}

} // namespace gobot::python
