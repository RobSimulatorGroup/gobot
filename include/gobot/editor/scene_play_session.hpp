#pragma once

#include <functional>
#include <string>
#include <utility>
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
    using ScriptOutputCallback = std::function<void(const std::string& message,
                                                    bool is_stderr,
                                                    const std::string& source)>;

    ScenePlaySession() = default;

    ~ScenePlaySession();

    [[nodiscard]] bool IsRunning() const { return running_; }

    [[nodiscard]] std::size_t GetActiveScriptCount() const { return script_nodes_.size(); }

    [[nodiscard]] const std::string& GetLastError() const { return last_error_; }

    [[nodiscard]] Node* GetRuntimeRoot() const;

    [[nodiscard]] std::uint64_t GetRuntimeSceneEpoch() const { return runtime_scene_epoch_; }

    bool Start(Node* edited_scene_root, EngineContext* context);

    void Stop();

    bool Reset(Node* edited_scene_root, EngineContext* context);

    void NotifyProcess(double delta_time);

    void NotifyPhysicsProcess(double delta_time);

    void SetScriptOutputCallback(ScriptOutputCallback callback) {
        script_output_callback_ = std::move(callback);
    }

private:
    bool CreateRuntimeScene(Node* edited_scene_root);

    bool AttachRuntimeSceneToTree(Node* edited_scene_root);

    void DestroyRuntimeScene();

    bool AttachNodeScript(Node* node);

    bool AttachNodeScriptsRecursive(Node* node);

    bool NotifyScripts(NotificationType notification, double delta_time);

    void ClearAttachedScripts(bool call_exit_tree);

private:
    bool running_{false};
    Node* edited_scene_root_{nullptr};
    Node* runtime_root_{nullptr};
    Node* runtime_holder_{nullptr};
    ObjectID runtime_root_id_{};
    ObjectID runtime_holder_id_{};
    EngineContext* context_{nullptr};
    std::uint64_t runtime_scene_epoch_{0};
    std::vector<ObjectID> script_nodes_;
    std::string last_error_;
    ScriptOutputCallback script_output_callback_;
};

} // namespace gobot
