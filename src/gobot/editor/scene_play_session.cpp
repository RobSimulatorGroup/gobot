/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/scene_play_session.hpp"

#include "gobot/core/io/python_script.hpp"
#include "gobot/core/object.hpp"
#include "gobot/log.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

namespace gobot {

ScenePlaySession::~ScenePlaySession() {
    Stop();
}

Node* ScenePlaySession::GetRuntimeRoot() const {
    if (runtime_root_id_ == ObjectID{}) {
        return nullptr;
    }
    return Object::PointerCastTo<Node>(ObjectDB::GetInstance(runtime_root_id_));
}

namespace {

std::uint64_t NextRuntimeSceneEpoch() {
    static std::uint64_t epoch = 1'000'000;
    ++epoch;
    if (epoch == 0) {
        epoch = 1'000'000;
    }
    return epoch;
}

struct SceneInstanceBackup {
    Node* node{nullptr};
    Ref<PackedScene> scene_instance;
};

class ScopedExpandedSceneInstances {
public:
    explicit ScopedExpandedSceneInstances(Node* root) {
        ClearSceneInstances(root);
    }

    ~ScopedExpandedSceneInstances() {
        for (auto iter = backups_.rbegin(); iter != backups_.rend(); ++iter) {
            if (iter->node != nullptr) {
                iter->node->SetSceneInstance(iter->scene_instance);
            }
        }
    }

private:
    void ClearSceneInstances(Node* node) {
        if (node == nullptr) {
            return;
        }

        Ref<PackedScene> scene_instance = node->GetSceneInstance();
        if (scene_instance.IsValid()) {
            backups_.push_back({node, scene_instance});
            node->SetSceneInstance(Ref<PackedScene>());
        }

        const std::size_t child_count = node->GetChildCount();
        for (std::size_t child_index = 0; child_index < child_count; ++child_index) {
            ClearSceneInstances(node->GetChild(static_cast<int>(child_index)));
        }
    }

    std::vector<SceneInstanceBackup> backups_;
};

void LogScriptOutput(const char* phase,
                     const Node& node,
                     const ScenePlaySession::ScriptOutputCallback& callback,
                     const python::PythonExecutionResult& result) {
    if (!result.output.empty()) {
        if (callback) {
            callback(result.output,
                     false,
                     "Python node script " + std::string(phase) + " on '" + node.GetName() + "'");
        } else {
            LOG_INFO("Python node script {} on '{}':\n{}", phase, node.GetName(), result.output);
        }
    }
    if (result.ok && !result.error.empty()) {
        if (callback) {
            callback(result.error,
                     true,
                     "Python node script " + std::string(phase) + " stderr on '" + node.GetName() + "'");
        } else {
            LOG_WARN("Python node script {} stderr on '{}':\n{}", phase, node.GetName(), result.error);
        }
    }
}

} // namespace

bool ScenePlaySession::Start(Node* edited_scene_root, EngineContext* context) {
    if (running_) {
        return true;
    }

    last_error_.clear();
    edited_scene_root_ = edited_scene_root;
    context_ = context;

    if (edited_scene_root_ == nullptr) {
        last_error_ = "Cannot start scene play session: scene root is null.";
        LOG_ERROR("{}", last_error_);
        return false;
    }

    if (!CreateRuntimeScene(edited_scene_root_)) {
        edited_scene_root_ = nullptr;
        context_ = nullptr;
        return false;
    }

    python::PythonScriptRunner::SetSceneScriptContext(context_);
    python::PythonScriptRunner::SetSceneScriptRoot(runtime_root_, runtime_scene_epoch_);

    if (!AttachNodeScriptsRecursive(runtime_root_)) {
        ClearAttachedScripts(true);
        DestroyRuntimeScene();
        runtime_scene_epoch_ = 0;
        edited_scene_root_ = nullptr;
        context_ = nullptr;
        return false;
    }

    if (!NotifyScripts(NotificationType::Ready, 0.0)) {
        ClearAttachedScripts(true);
        DestroyRuntimeScene();
        runtime_scene_epoch_ = 0;
        edited_scene_root_ = nullptr;
        context_ = nullptr;
        return false;
    }
    running_ = true;
    LOG_INFO("Scene play session started with {} active script(s).", script_nodes_.size());
    return true;
}

void ScenePlaySession::Stop() {
    runtime_root_ = GetRuntimeRoot();
    runtime_holder_ = runtime_holder_id_ == ObjectID{}
                              ? nullptr
                              : Object::PointerCastTo<Node>(ObjectDB::GetInstance(runtime_holder_id_));

    if (!running_ && script_nodes_.empty()) {
        DestroyRuntimeScene();
        edited_scene_root_ = nullptr;
        context_ = nullptr;
        runtime_scene_epoch_ = 0;
        return;
    }

    ClearAttachedScripts(true);
    running_ = false;
    DestroyRuntimeScene();
    edited_scene_root_ = nullptr;
    context_ = nullptr;
    runtime_scene_epoch_ = 0;
    LOG_INFO("Scene play session stopped.");
}

bool ScenePlaySession::Reset(Node* edited_scene_root, EngineContext* context) {
    Stop();
    return Start(edited_scene_root, context);
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

bool ScenePlaySession::CreateRuntimeScene(Node* edited_scene_root) {
    Ref<PackedScene> packed_scene = MakeRef<PackedScene>();
    {
        ScopedExpandedSceneInstances expanded_scene_instances(edited_scene_root);
        if (!packed_scene->Pack(edited_scene_root)) {
            last_error_ = "Cannot start scene play session: failed to pack edited scene.";
            LOG_ERROR("{}", last_error_);
            return false;
        }
    }

    runtime_root_ = packed_scene->Instantiate();
    if (runtime_root_ == nullptr) {
        last_error_ = "Cannot start scene play session: failed to instantiate runtime scene.";
        LOG_ERROR("{}", last_error_);
        return false;
    }
    runtime_root_id_ = runtime_root_->GetInstanceId();

    if (!AttachRuntimeSceneToTree(edited_scene_root)) {
        DestroyRuntimeScene();
        return false;
    }

    runtime_scene_epoch_ = NextRuntimeSceneEpoch();
    return true;
}

bool ScenePlaySession::AttachRuntimeSceneToTree(Node* edited_scene_root) {
    if (runtime_root_ == nullptr) {
        last_error_ = "Cannot start scene play session: runtime scene is null.";
        LOG_ERROR("{}", last_error_);
        return false;
    }

    if (edited_scene_root == nullptr || edited_scene_root->GetParent() == nullptr) {
        return true;
    }

    runtime_holder_ = Object::New<Node>();
    runtime_holder_id_ = runtime_holder_->GetInstanceId();
    runtime_holder_->SetName("__ScenePlaySessionRuntime");
    edited_scene_root->GetParent()->AddChild(runtime_holder_, false);
    runtime_holder_->AddChild(runtime_root_, false);
    return runtime_root_->IsInsideTree();
}

void ScenePlaySession::DestroyRuntimeScene() {
    runtime_holder_ = runtime_holder_id_ == ObjectID{}
                              ? nullptr
                              : Object::PointerCastTo<Node>(ObjectDB::GetInstance(runtime_holder_id_));
    runtime_root_ = runtime_root_id_ == ObjectID{}
                            ? nullptr
                            : Object::PointerCastTo<Node>(ObjectDB::GetInstance(runtime_root_id_));

    if (runtime_holder_ != nullptr) {
        if (runtime_holder_->GetParent() != nullptr) {
            runtime_holder_->GetParent()->RemoveChild(runtime_holder_);
        }
        Object::Delete(runtime_holder_);
        runtime_holder_ = nullptr;
        runtime_holder_id_ = ObjectID{};
        runtime_root_ = nullptr;
        runtime_root_id_ = ObjectID{};
        return;
    }

    if (runtime_root_ != nullptr) {
        if (runtime_root_->GetParent() != nullptr) {
            runtime_root_->GetParent()->RemoveChild(runtime_root_);
        }
        Object::Delete(runtime_root_);
    }
    runtime_root_ = nullptr;
    runtime_holder_ = nullptr;
    runtime_root_id_ = ObjectID{};
    runtime_holder_id_ = ObjectID{};
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
    LogScriptOutput("attach", *node, script_output_callback_, result);
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
        LogScriptOutput("notification", *node, script_output_callback_, result);
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
    runtime_root_ = GetRuntimeRoot();
    python::PythonScriptRunner::SetSceneScriptRoot(runtime_root_, runtime_scene_epoch_);
    for (ObjectID node_id : script_nodes_) {
        Node* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(node_id));
        if (node == nullptr) {
            python::PythonScriptRunner::DetachSceneScript(node_id);
            continue;
        }

        if (call_exit_tree) {
            python::PythonExecutionResult result =
                    python::PythonScriptRunner::NotifySceneScript(node, NotificationType::ExitTree, 0.0);
            LogScriptOutput("exit", *node, script_output_callback_, result);
            if (!result.ok) {
                LOG_ERROR("Python node script exit failed on '{}': {}", node->GetName(), result.error);
            }
        }
        python::PythonScriptRunner::DetachSceneScript(node);
    }
    script_nodes_.clear();
}

} // namespace gobot
