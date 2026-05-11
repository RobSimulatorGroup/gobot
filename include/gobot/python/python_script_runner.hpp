#pragma once

#include <cstdint>
#include <string>

#include "gobot_export.h"
#include "gobot/core/notification_enum.hpp"
#include "gobot/core/object_id.hpp"
#include "gobot/core/ref_counted.hpp"

namespace gobot {

class EngineContext;
class Node;
class PythonScript;

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

    static void SetSceneScriptContext(EngineContext* context);

    static void SetSceneScriptRoot(Node* root, std::uint64_t scene_epoch = 0);

    static void ClearSceneScriptContext(EngineContext* context);

    static bool IsExecutingSceneScript();

    static Node* GetExecutingSceneScriptRoot();

    static std::uint64_t GetExecutingSceneScriptEpoch();

    static bool HasSceneScriptInstance(Node* node);

    static PythonExecutionResult AttachSceneScript(Node* node,
                                                   const Ref<PythonScript>& script);

    static void DetachSceneScript(Node* node);

    static void DetachSceneScript(ObjectID node_id);

    static PythonExecutionResult NotifySceneScript(Node* node,
                                                   NotificationType notification,
                                                   double delta_time);

    static void Shutdown();
};

} // namespace python
} // namespace gobot
