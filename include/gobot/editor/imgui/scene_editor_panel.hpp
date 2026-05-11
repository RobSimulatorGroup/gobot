/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#pragma once

#include <string>

#include "gobot/scene/imgui_window.hpp"

class ImGuiTextFilter;

namespace gobot {

class GOBOT_EXPORT SceneEditorPanel : public ImGuiWindow {
public:
    SceneEditorPanel();

    ~SceneEditorPanel() override;

    void OnImGuiContent() override;

    bool DrawNode(Node* node);

private:
    Node* GetAddChildTarget(Node* scene_root) const;

    void RequestOpenAddChildDialog(Node* parent);

    void DrawAddChildDialog();

    bool CreateSelectedAddNode();

    bool IsSceneInstanceNode(Node* node) const;

    bool CanDeleteNode(Node* node) const;

    bool DeleteNode(Node* node);

    bool AttachScript(Node* node);

    bool AttachScript(Node* node, const std::string& script_path);

    bool AcceptResourceDropOnNode(Node* node);

    bool DetachScript(Node* node);

    void RequestOpenSceneInstance(const std::string& path);

    void RequestOpenScript(const std::string& path);

    void FlushPendingSceneInstanceOpen();

    void FlushPendingScriptOpen();

    ImGuiTextFilter* filter_{nullptr};
    Node* double_clicked_{nullptr};
    Node* current_{nullptr};
    Node* add_child_parent_{nullptr};
    std::string pending_open_scene_instance_path_;
    std::string pending_open_script_path_;
    bool open_add_child_dialog_{false};
    std::string add_node_search_;
    std::string selected_add_node_id_;
    bool select_up_;
    bool select_down_;

};


}
