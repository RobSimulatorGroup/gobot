/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#pragma once

#include "gobot/scene/node.hpp"
#include "gobot/editor/imgui/editor_panel.hpp"

namespace gobot {

class Node3DEditor;
class ImGuiManager;

class Editor : public Node {
    GOBCLASS(Editor, Node)
public:
    Editor();

    ~Editor() override;

    void NotificationCallBack(NotificationType notification);

    static Editor* GetInstance();

private:
    static Editor* s_singleton;

    Node3DEditor* node3d_editor_{nullptr};
    ImGuiManager* imgui_manager_{nullptr};

    std::vector<std::shared_ptr<EditorPanel>> panels_;
};


}