/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-20
*/

#pragma once

#include "gobot/scene/imgui_window.hpp"

namespace gobot {

class Node3D;
class EditorInspector;

class InspectorPanel : public ImGuiWindow {
    GOBCLASS(InspectorPanel, ImGuiWindow)
public:
    InspectorPanel();

    void OnImGuiContent() override;

private:
    Node3D* node_3d_{nullptr};

    // edit history of inspector
    std::vector<EditorInspector*> editor_inspectors_{};
    int current_inspector_index_{-1};
};


}