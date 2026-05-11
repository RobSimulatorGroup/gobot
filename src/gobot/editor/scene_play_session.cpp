/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/editor/scene_play_session.hpp"

#include "gobot/core/io/python_script.hpp"
#include "gobot/core/object.hpp"
#include "gobot/log.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/scene/node.hpp"

namespace gobot {

ScenePlaySession::~ScenePlaySession() {
    Stop();
}

bool ScenePlaySession::Start(Node* scene_root, EngineContext* context) {
    if (running_) {
        return true;
    }

    last_error_.clear();
    scene_root_ = scene_root;
    context_ = context;

    if (scene_root_ == nullptr) {
        last_error_ = "Cannot start scene play session: scene root is null.";
        LOG_ERROR("{}", last_error_);
        return false;
    }

    python::PythonScriptRunner::SetSceneScriptContext(context_);
    python::PythonScriptRunner::SetSceneScriptRoot(scene_root_);

    if (!AttachNodeScriptsRecursive(scene_root_)) {
        ClearAttachedScripts(true);
        scene_root_ = nullptr;
        context_ = nullptr;
        return false;
    }

    if (!NotifyScripts(NotificationType::Ready, 0.0)) {
        ClearAttachedScripts(true);
        scene_root_ = nullptr;
        context_ = nullptr;
        return false;
    }
    running_ = true;
    LOG_INFO("Scene play session started with {} active script(s).", script_nodes_.size());
    return true;
}

void ScenePlaySession::Stop() {
    if (!running_ && script_nodes_.empty()) {
        scene_root_ = nullptr;
        context_ = nullptr;
        return;
    }

    ClearAttachedScripts(true);
    running_ = false;
    scene_root_ = nullptr;
    context_ = nullptr;
    LOG_INFO("Scene play session stopped.");
}

bool ScenePlaySession::Reset(Node* scene_root, EngineContext* context) {
    Stop();
    return Start(scene_root, context);
}

void ScenePlaySession::NotifyProcess(double delta_time) {
    if (!running_) {
        return;
    }
    NotifyScripts(NotificationType::Process, delta_time);
}

void ScenePlaySession::NotifyPhysicsProcess(double delta_time) {
    if (!running_) {
        return;
    }
    NotifyScripts(NotificationType::PhysicsProcess, delta_time);
}

bool ScenePlaySession::AttachNodeScript(Node* node) {
    if (node == nullptr) {
        return true;
    }

    Ref<PythonScript> script = node->GetScript();
    if (!script.IsValid()) {
        return true;
    }

    python::PythonExecutionResult result = python::PythonScriptRunner::AttachSceneScript(node, script);
    if (!result.ok) {
        last_error_ = "Python node script attach failed on '" + node->GetName() + "': " + result.error;
        LOG_ERROR("{}", last_error_);
        return false;
    }

    script_nodes_.push_back(node->GetInstanceId());
    return true;
}

bool ScenePlaySession::AttachNodeScriptsRecursive(Node* node) {
    if (node == nullptr) {
        return true;
    }

    if (!AttachNodeScript(node)) {
        return false;
    }

    const std::size_t child_count = node->GetChildCount();
    for (std::size_t child_index = 0; child_index < child_count; ++child_index) {
        if (!AttachNodeScriptsRecursive(node->GetChild(static_cast<int>(child_index)))) {
            return false;
        }
    }
    return true;
}

bool ScenePlaySession::NotifyScripts(NotificationType notification, double delta_time) {
    std::vector<ObjectID> remaining_nodes;
    remaining_nodes.reserve(script_nodes_.size());
    bool ok = true;

    for (ObjectID node_id : script_nodes_) {
        Node* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(node_id));
        if (node == nullptr) {
            python::PythonScriptRunner::DetachSceneScript(node_id);
            ok = false;
            continue;
        }

        python::PythonExecutionResult result =
                python::PythonScriptRunner::NotifySceneScript(node, notification, delta_time);
        if (!result.ok) {
            last_error_ = "Python node script failed on '" + node->GetName() + "': " + result.error;
            LOG_ERROR("{}", last_error_);
            python::PythonScriptRunner::DetachSceneScript(node);
            ok = false;
            continue;
        }

        remaining_nodes.push_back(node_id);
    }

    script_nodes_ = std::move(remaining_nodes);
    return ok;
}

void ScenePlaySession::ClearAttachedScripts(bool call_exit_tree) {
    for (ObjectID node_id : script_nodes_) {
        Node* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(node_id));
        if (node == nullptr) {
            python::PythonScriptRunner::DetachSceneScript(node_id);
            continue;
        }

        if (call_exit_tree) {
            python::PythonExecutionResult result =
                    python::PythonScriptRunner::NotifySceneScript(node, NotificationType::ExitTree, 0.0);
            if (!result.ok) {
                LOG_ERROR("Python node script exit failed on '{}': {}", node->GetName(), result.error);
            }
        }
        python::PythonScriptRunner::DetachSceneScript(node);
    }
    script_nodes_.clear();
}

} // namespace gobot
