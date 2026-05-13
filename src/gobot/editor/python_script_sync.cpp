/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/python_script_sync.hpp"

#include "gobot/core/io/python_script.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/scene/node.hpp"

namespace gobot {

namespace {

void SyncAttachedSceneScripts(Node* node,
                              const std::string& local_path,
                              const std::string& source_code) {
    if (node == nullptr) {
        return;
    }

    Ref<PythonScript> script = node->GetScript();
    if (script.IsValid() && script->GetPath() == local_path) {
        script->SetSourceCode(source_code);
    }

    const std::size_t child_count = node->GetChildCount();
    for (std::size_t child_index = 0; child_index < child_count; ++child_index) {
        SyncAttachedSceneScripts(node->GetChild(static_cast<int>(child_index)),
                                 local_path,
                                 source_code);
    }
}

} // namespace

void SyncPythonScriptResourceSource(const std::string& local_path,
                                    const std::string& source_code,
                                    Node* scene_root) {
    if (local_path.empty()) {
        return;
    }

    Ref<Resource> cached_resource = ResourceCache::GetRef(local_path);
    Ref<PythonScript> cached_script = dynamic_pointer_cast<PythonScript>(cached_resource);
    if (cached_script.IsValid()) {
        cached_script->SetSourceCode(source_code);
    }

    SyncAttachedSceneScripts(scene_root, local_path, source_code);
}

} // namespace gobot
