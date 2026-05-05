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

class GOBOT_EXPORT PythonScriptRunner {
public:
    static bool IsAvailable();

    static PythonExecutionResult ExecuteString(const std::string& source,
                                               EngineContext* context = nullptr,
                                               const std::string& filename = "<gobot>");

    static PythonExecutionResult ExecuteFile(const std::string& path,
                                             EngineContext* context = nullptr);

    static void Shutdown();
};

} // namespace python
} // namespace gobot
