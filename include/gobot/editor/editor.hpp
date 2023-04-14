/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#pragma once

#include "gobot/scene/node.hpp"
#include "gobot/scene/imgui_node.hpp"

namespace ImGui {
class FileBrowser;
}

namespace gobot {

class Node3DEditor;
class ImGuiManager;

class GOBOT_EXPORT Editor : public ImGuiNode {
    GOBCLASS(Editor, Node)
public:
    Editor();

    ~Editor() override;

    void NotificationCallBack(NotificationType notification);

    static Editor* GetInstance();

    FORCE_INLINE Node3DEditor* GetNode3dEditor() { return node3d_editor_; }

    FORCE_INLINE ImGui::FileBrowser* GetFileBrowser() { return file_browser_; }

    bool Begin() override;

    void End() override;

    void OnImGuiContent() override;

    FORCE_INLINE Node* GetSelected() const { return selected_; }

    FORCE_INLINE void SetSelected(Node* selected) { selected_ = selected; }

private:
    void DrawMenuBar();

    void BeginDockSpace();

    void EndDockSpace();

private:
    static Editor* s_singleton;

    Node3DEditor* node3d_editor_{nullptr};
    ImGuiManager* imgui_manager_{nullptr};
    ImGui::FileBrowser* file_browser_{nullptr};

    Node* selected_{nullptr};

};


}