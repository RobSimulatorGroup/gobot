#pragma once

#include <string>

#include "gobot_export.h"

namespace gobot {

class EngineContext;

namespace python {

struct GOBOT_EXPORT PythonExecutionResult {
    bool ok{false};
    std::string output;
    std::string error;
};

using PythonTickCallback = void (*)(double delta_time);
using PythonTickClearCallback = void (*)();

class GOBOT_EXPORT PythonScriptRunner {
public:
    static bool IsAvailable();

    static PythonExecutionResult ExecuteString(const std::string& source,
                                               EngineContext* context = nullptr,
                                               const std::string& filename = "<gobot>");

    static PythonExecutionResult ExecuteFile(const std::string& path,
                                             EngineContext* context = nullptr);

    static PythonExecutionResult ExecuteTick(EngineContext* context,
                                             double delta_time,
                                             const std::string& filename = "<gobot-python-tick>");

    static PythonExecutionResult ExecutePhysicsTick(EngineContext* context,
                                                    double delta_time,
                                                    const std::string& filename = "<gobot-python-physics-tick>");

    static void SetTickCallback(PythonTickCallback callback);

    static void SetTickClearCallback(PythonTickClearCallback callback);

    static bool HasTickCallback();

    static void ClearTickCallback();

    static void SetPhysicsTickCallback(PythonTickCallback callback);

    static void SetPhysicsTickClearCallback(PythonTickClearCallback callback);

    static bool HasPhysicsTickCallback();

    static void ClearPhysicsTickCallback();

    static void Shutdown();
};

} // namespace python
} // namespace gobot
