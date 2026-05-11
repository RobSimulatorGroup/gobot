#pragma once

#include <vector>

#include "gobot/core/object_id.hpp"
#include "gobot/core/ref_counted.hpp"
#include "gobot/python/python_script_runner.hpp"

namespace gobot {

class EngineContext;
class Node;
class PythonScript;

class GOBOT_EXPORT ScenePlaySession {
public:
    ScenePlaySession() = default;

    ~ScenePlaySession();

    [[nodiscard]] bool IsRunning() const { return running_; }

    [[nodiscard]] std::size_t GetActiveScriptCount() const { return script_nodes_.size(); }

    [[nodiscard]] const std::string& GetLastError() const { return last_error_; }

    bool Start(Node* scene_root, EngineContext* context);

    void Stop();

    bool Reset(Node* scene_root, EngineContext* context);

    void NotifyProcess(double delta_time);

    void NotifyPhysicsProcess(double delta_time);

private:
    bool AttachNodeScript(Node* node);

    bool AttachNodeScriptsRecursive(Node* node);

    bool NotifyScripts(NotificationType notification, double delta_time);

    void ClearAttachedScripts(bool call_exit_tree);

private:
    bool running_{false};
    Node* scene_root_{nullptr};
    EngineContext* context_{nullptr};
    std::vector<ObjectID> script_nodes_;
    std::string last_error_;
};

} // namespace gobot
